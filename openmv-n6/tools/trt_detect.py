#!/usr/bin/env python3
"""Selectable YOLOv8/YOLOv10/YOLO11 over TensorRT on the Jetson's GPU.

This is a drop-in for detect.Detector.

Why this exists: the cv2.dnn path in detect.py runs yolov4-tiny on the CPU at
68 ms per frame at 416, and it takes four of the six cores to do it. Those cores
are the same ones draining the board's CDC, and the one failure this pipeline
cannot tolerate is a host that stops reading the serial link (see the comment on
detect.THREADS). Moving the forward pass to the GPU does not just make detection
faster - it gives the cores back to the reader.

Measured on this box (Orin Nano Super, JetPack 6.2.1, TensorRT 10.3, FP16):
GPU compute 4.83 ms, end-to-end 5.19 ms including both copies. Against the
68 ms CPU number that is ~13x, and it runs on a device that was previously idle.

Model choice. The installed yolov10n is the measured default; yolov8n and
yolo11n use the same runtime contract after a static export with NMS included.
For a person at long range, the model choice is an accuracy change, not only a
speed swap. Every accepted engine emits [1,max_det,6] suppressed boxes, so raw
Ultralytics prediction planes are rejected before inference.

The input is a static 640x640. The visible frame is 640x400, so letterboxing
is pure padding at scale 1.0 - 120 rows of grey above and below, and not one
pixel resampled. The cv2.dnn path squashed 640x400 into 416x416, distorting
the aspect ratio; this path does not.

No pycuda, no cupy, no torch. Device memory is handled by calling libcudart
through ctypes, the same way fusion.c is already loaded. That is deliberate:
this host's numpy is pinned at 1.21 by the apt OpenCV build, and a pip install
that drags in numpy 2.x breaks cv2 for the whole project.
"""
import ctypes
import os
import time

import numpy as np

MODEL_DIR = os.path.expanduser("~/archive/radar/models")
MODEL_FILES = {
    "yolov10n": ("yolov10n.onnx", "yolov10n_fp16.engine"),
    # YOLOv8/11 must be exported with nms=True, batch=1, imgsz=640 so their
    # output is [1, max_det, 6], not the raw [1, 84, anchors] prediction plane.
    "yolov8n": ("yolov8n_nms.onnx", "yolov8n_nms_fp16.engine"),
    "yolo11n": ("yolo11n_nms.onnx", "yolo11n_nms_fp16.engine"),
}


def model_paths(model):
    if model not in MODEL_FILES:
        raise ValueError("unknown TensorRT detector model %r; choose %s" %
                         (model, ", ".join(sorted(MODEL_FILES))))
    onnx_name, engine_name = MODEL_FILES[model]
    return (os.path.join(MODEL_DIR, onnx_name),
            os.path.join(MODEL_DIR, engine_name))


def build_command(model):
    onnx, engine = model_paths(model)
    return ["/usr/src/tensorrt/bin/trtexec", "--onnx=" + onnx,
            "--saveEngine=" + engine, "--fp16",
            "--memPoolSize=workspace:1024", "--skipInference"]


ONNX, ENGINE = model_paths("yolov10n")
BUILD_CMD = " ".join(build_command("yolov10n"))

# Ultralytics COCO-80. Same 80 classes and the same indices as the darknet list
# in detect.COCO, but four names differ in spelling (motorbike/motorcycle,
# aeroplane/airplane, sofa/couch, pottedplant/potted_plant, tvmonitor/tv). The
# indices are what the network emits, so index 0 is 'person' either way; the
# names are kept in detect.COCO's spelling so that a --detect filter written
# against the old model still selects the same classes.
CONF = 0.25          # v10 has no NMS to clean up after it, so this is the only gate

# cudaMemcpyKind
_H2D, _D2H = 1, 2


class _Cudart:
    """The four libcudart calls this needs, and nothing else."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_size_t, ctypes.c_int,
                                             ctypes.c_void_p]
        # Pinned host memory. Pageable memory forces the driver to stage every
        # transfer through a bounce buffer, which on this board costs more than
        # the copy itself.
        self.lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.c_size_t, ctypes.c_uint]
        self.lib.cudaFreeHost.argtypes = [ctypes.c_void_p]

    def _check(self, rc, what):
        if rc != 0:
            raise RuntimeError("cuda %s failed: rc=%d" % (what, rc))

    def malloc(self, nbytes):
        p = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(p), nbytes), "malloc")
        return p

    def host_alloc(self, nbytes):
        """Returns (numpy uint8 view, raw pointer). The view aliases pinned memory."""
        p = ctypes.c_void_p()
        self._check(self.lib.cudaHostAlloc(ctypes.byref(p), nbytes, 0), "hostAlloc")
        buf = (ctypes.c_uint8 * nbytes).from_address(p.value)
        return np.frombuffer(buf, dtype=np.uint8), p

    def stream(self):
        s = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(s)), "streamCreate")
        return s

    def h2d(self, dst, src, n, stream):
        self._check(self.lib.cudaMemcpyAsync(dst, src, n, _H2D, stream), "memcpy h2d")

    def d2h(self, dst, src, n, stream):
        self._check(self.lib.cudaMemcpyAsync(dst, src, n, _D2H, stream), "memcpy d2h")

    def sync(self, stream):
        self._check(self.lib.cudaStreamSynchronize(stream), "streamSync")


def build_engine(model="yolov10n", verbose=False):
    """Build this Jetson's engine from a static, NMS-included ONNX export."""
    import subprocess
    onnx, engine = model_paths(model)
    cmd = build_command(model)
    if not os.path.isfile(onnx):
        raise FileNotFoundError(onnx)
    rc = subprocess.call(cmd,
                         stdout=None if verbose else subprocess.DEVNULL,
                         stderr=None if verbose else subprocess.DEVNULL)
    if rc != 0 or not os.path.isfile(engine):
        raise RuntimeError("trtexec failed (rc=%d): %s" %
                           (rc, " ".join(cmd)))
    return engine


def validate_io_shapes(input_shape, output_shape):
    """Reject raw or dynamic exports before their tensors reach postprocess."""
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] != 3:
        raise RuntimeError(f"expected static NCHW [1,3,H,W], got {input_shape}")
    if any(int(v) <= 0 for v in input_shape):
        raise RuntimeError(f"dynamic input is unsupported: {input_shape}")
    if (len(output_shape) != 3 or output_shape[0] != 1 or
            output_shape[2] != 6 or any(int(v) <= 0 for v in output_shape)):
        raise RuntimeError(
            f"expected NMS output [1,max_det,6], got {output_shape}; "
            "export YOLOv8/YOLO11 with nms=True, batch=1, dynamic=False")


class TrtDetector:
    """Same contract as detect.Detector: call it with a frame, get boxes back.

    Not thread-safe - one execution context, one stream, one set of buffers.
    live.py already gives the detector a thread of its own, which is the reason
    that is acceptable here.
    """

    def __init__(self, engine=None, conf=CONF, classes=None, names=None,
                 model="yolov10n"):
        import tensorrt as trt

        _onnx, default_engine = model_paths(model)
        engine = engine or default_engine
        cmd = " ".join(build_command(model))
        if not os.path.isfile(engine):
            raise FileNotFoundError(
                "%s\nbuild it with: %s" % (engine, cmd))

        self.model_name = model
        self.engine_path = engine
        self.conf = conf
        self.names = tuple(names) if names else None
        self.ms = 0.0

        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        with open(engine, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError("could not deserialize %s - a TensorRT engine is "
                               "tied to the GPU and TRT version that built it; "
                               "rebuild with: %s" % (engine, cmd))
        self._ctx = self._engine.create_execution_context()

        # Discover the bindings rather than hardcoding names, so a re-export
        # under a different exporter does not silently bind the wrong tensor.
        inputs, outputs = [], []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append((name, shape))
            else:
                outputs.append((name, shape))
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("engine needs exactly one input and one output; "
                               f"got {len(inputs)} input(s), {len(outputs)} output(s)")
        self._in_name, self._in_shape = inputs[0]
        self._out_name, self._out_shape = outputs[0]
        validate_io_shapes(self._in_shape, self._out_shape)
        _, _c, self.net_h, self.net_w = self._in_shape
        self._in_dtype = np.dtype(trt.nptype(
            self._engine.get_tensor_dtype(self._in_name)))
        self._out_dtype = np.dtype(trt.nptype(
            self._engine.get_tensor_dtype(self._out_name)))
        if self._in_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise RuntimeError(f"unsupported input dtype {self._in_dtype}")
        if self._out_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise RuntimeError(f"unsupported output dtype {self._out_dtype}")

        self.cu = _Cudart()
        self._stream = self.cu.stream()
        in_bytes = int(np.prod(self._in_shape)) * self._in_dtype.itemsize
        out_bytes = int(np.prod(self._out_shape)) * self._out_dtype.itemsize
        self._d_in = self.cu.malloc(in_bytes)
        self._d_out = self.cu.malloc(out_bytes)

        # Pinned staging buffers, viewed as the shapes the model wants. Allocated
        # once: doing this per frame would cost more than the inference.
        h_in, self._p_in = self.cu.host_alloc(in_bytes)
        h_out, self._p_out = self.cu.host_alloc(out_bytes)
        self.h_in = h_in.view(self._in_dtype).reshape(self._in_shape)
        self.h_out = h_out.view(self._out_dtype).reshape(self._out_shape)
        self._in_bytes, self._out_bytes = in_bytes, out_bytes

        self._ctx.set_tensor_address(self._in_name, int(self._d_in.value))
        self._ctx.set_tensor_address(self._out_name, int(self._d_out.value))

        # Letterbox fill. 114 is what ultralytics pads with, and the model was
        # trained against that; using black instead puts an edge in the image
        # where there is no object.
        self._pad_val = 114.0 / 255.0
        self._geom = None            # last letterbox geometry; see letterbox()

        self.classes = set(classes) if classes else None
        self._class_ids = None       # resolved lazily, needs the name table

    # -- preprocessing ----------------------------------------------------

    def letterbox(self, img):
        """HxW gray or HxWx3 BGR -> (NCHW FP16/FP32 in self.h_in, scale, padx, pady).

        Writes straight into the pinned buffer. Returns the geometry needed to
        map boxes back to the caller's pixel coordinates.
        """
        import cv2
        h, w = img.shape[:2]
        scale = min(self.net_w / w, self.net_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        padx, pady = (self.net_w - nw) // 2, (self.net_h - nh) // 2

        # The padding is the same grey every frame, so it only has to be written
        # when the frame geometry changes - which for a fixed sensor is once.
        geom = (nw, nh, padx, pady)
        if geom != self._geom:
            self.h_in[...] = self._pad_val
            self._geom = geom

        if (nw, nh) != (w, h):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dst = self.h_in[0, :, pady:pady + nh, padx:padx + nw]
        if img.ndim == 2:
            # The visible sensor is GRAYSCALE, so there is no colour to give the
            # network, and the three planes are identical. Scaling once and
            # letting numpy broadcast across them is not a micro-optimisation:
            # np.repeat to 3 channels measured 4.07 ms of a 6.32 ms preprocess,
            # which was more than the copy and a third of the inference.
            dst[...] = img.astype(np.float32) * (1.0 / 255.0)
        else:
            dst[...] = (img[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)
                        * (1.0 / 255.0))         # BGR -> RGB, HWC -> CHW
        return scale, padx, pady

    # -- inference --------------------------------------------------------

    def _name(self, cid):
        if self.names is not None and cid < len(self.names):
            return self.names[cid]
        return str(cid)

    def __call__(self, img):
        """img: HxW gray or HxWx3 BGR. Returns [{cls,conf,x,y,w,h}], newest first."""
        h, w = img.shape[:2]
        t0 = time.time()
        scale, padx, pady = self.letterbox(img)

        self.cu.h2d(self._d_in, self._p_in, self._in_bytes, self._stream)
        if not self._ctx.execute_async_v3(int(self._stream.value)):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        self.cu.d2h(self._p_out, self._d_out, self._out_bytes, self._stream)
        self.cu.sync(self._stream)
        self.ms = (time.time() - t0) * 1e3

        # [1,300,6] -> x0,y0,x1,y1,score,class in letterboxed pixels, already
        # sorted by score and already suppressed. Everything below the gate is
        # padding rows, so the first failing row ends the useful part.
        out = self.h_out[0]
        keep = out[:, 4] >= self.conf
        if self._class_ids is None and self.classes is not None:
            tbl = self.names or ()
            self._class_ids = np.array([i for i, n in enumerate(tbl)
                                        if n in self.classes], dtype=np.int32)
        if self.classes is not None:
            keep &= np.isin(out[:, 5].astype(np.int32), self._class_ids)
        sel = out[keep]
        if not len(sel):
            return []

        # Undo the letterbox. Clamp before anything downstream indexes with
        # these: an unclamped box reaches fusion_temp_region(), which would then
        # average over pixels outside the frame.
        x0 = np.clip((sel[:, 0] - padx) / scale, 0, w)
        y0 = np.clip((sel[:, 1] - pady) / scale, 0, h)
        x1 = np.clip((sel[:, 2] - padx) / scale, 0, w)
        y1 = np.clip((sel[:, 3] - pady) / scale, 0, h)

        dets = []
        for i in range(len(sel)):
            bx, by = int(x0[i]), int(y0[i])
            bw, bh = int(x1[i]) - bx, int(y1[i]) - by
            if bw <= 0 or bh <= 0:
                continue
            dets.append({"cls": self._name(int(sel[i, 5])),
                         "conf": round(float(sel[i, 4]), 3),
                         "x": bx, "y": by, "w": bw, "h": bh})
        dets.sort(key=lambda d: -d["conf"])
        return dets

    def close(self):
        cu = getattr(self, "cu", None)
        if cu is None:
            return
        for attr in ("_d_in", "_d_out"):
            p = getattr(self, attr, None)
            if p is not None:
                cu.lib.cudaFree(p)
                setattr(self, attr, None)
        for attr in ("_p_in", "_p_out"):
            p = getattr(self, attr, None)
            if p is not None:
                cu.lib.cudaFreeHost(p)
                setattr(self, attr, None)
        self.cu = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Build a registered static-NMS TensorRT detector engine")
    ap.add_argument("--build", required=True, choices=sorted(MODEL_FILES),
                    metavar="MODEL")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    print(build_engine(args.build, verbose=args.verbose))


if __name__ == "__main__":
    main()
