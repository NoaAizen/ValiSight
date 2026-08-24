"""TensorRT runtime for the trained thermal/radar person students.

Same dependency stance as trt_detect.py: tensorrt python bindings + libcudart
via ctypes - no torch, no pycuda. The engines are built on this Jetson from
the ONNX files exported by perception/export_students_onnx.py:

    /usr/src/tensorrt/bin/trtexec --onnx=thermal_student.onnx \
        --saveEngine=thermal_student.engine --fp16 --memPoolSize=workspace:256M
    /usr/src/tensorrt/bin/trtexec --onnx=radar_student.onnx \
        --saveEngine=radar_student.engine --memPoolSize=workspace:256M

Input contracts (must match the export - see export_students_onnx.py):
    thermal: thermal_seq (1,3,120,160) f32 CELSIUS [current, prev1, prev2],
             valid_prev1 (1,), valid_prev2 (1,), dt1_ms (1,)
    radar:   radar_points (1,64,6) f32 [x,y,z,velocity,snr,noise] zero-padded,
             n_radar (1,) int64
Outputs for both: scores (1,8) sigmoid confidence, boxes (1,8,4) normalized
cx,cy,w,h - thermal on the 160x120 plane, radar on the 640x400 RGB plane.

Neither class is thread-safe; give each instance one thread.
"""
import os

import numpy as np

import trt_detect  # reuse the ctypes cudart wrapper that already works here

ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "perception", "out", "gexport", "v2", "models")
MAX_DT_MS = 300.0            # training's temporal-chain window (student_data)
MAX_POINTS = 64              # radar rows per frame in the export schema


class TrtRunner:
    """Generic multi-input/multi-output TensorRT engine runner."""

    def __init__(self, engine_path):
        import tensorrt as trt
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(engine_path)
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(
                "could not deserialize %s - engines are tied to the GPU and "
                "TRT version that built them; rebuild with trtexec"
                % engine_path)
        self._ctx = self._engine.create_execution_context()
        self.cu = trt_detect._Cudart()
        self._stream = self.cu.stream()

        self.inputs, self.outputs = {}, {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            dev = self.cu.malloc(nbytes)
            host_raw, host_ptr = self.cu.host_alloc(nbytes)
            host = host_raw.view(dtype).reshape(shape)
            self._ctx.set_tensor_address(name, int(dev.value))
            record = {"shape": shape, "dtype": dtype, "nbytes": nbytes,
                      "dev": dev, "host": host, "host_ptr": host_ptr}
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = record
            else:
                self.outputs[name] = record

    def infer(self, feeds):
        """feeds: {input_name: np.ndarray}. Returns {output_name: np.ndarray}."""
        for name, rec in self.inputs.items():
            if name not in feeds:
                raise KeyError(f"missing input {name}")
            arr = np.ascontiguousarray(
                np.asarray(feeds[name], dtype=rec["dtype"]).reshape(
                    rec["shape"]))
            rec["host"][...] = arr
            self.cu.h2d(rec["dev"], rec["host_ptr"], rec["nbytes"],
                        self._stream)
        if not self._ctx.execute_async_v3(int(self._stream.value)):
            raise RuntimeError("TensorRT execute failed")
        for rec in self.outputs.values():
            self.cu.d2h(rec["host_ptr"], rec["dev"], rec["nbytes"],
                        self._stream)
        self.cu.sync(self._stream)
        return {name: rec["host"].copy()
                for name, rec in self.outputs.items()}


def _boxes_to_dets(scores, boxes, width, height, conf):
    """(8,) scores + (8,4) normalized cxcywh -> [{conf,x,y,w,h}] in pixels."""
    dets = []
    for k in range(len(scores)):
        s = float(scores[k])
        if s < conf:
            continue
        cx, cy, w, h = (float(v) for v in boxes[k])
        bw, bh = w * width, h * height
        x = cx * width - 0.5 * bw
        y = cy * height - 0.5 * bh
        x0, y0 = max(0.0, x), max(0.0, y)
        x1 = min(float(width), x + bw)
        y1 = min(float(height), y + bh)
        if x1 <= x0 or y1 <= y0:
            continue
        dets.append({"conf": round(s, 3), "x": int(x0), "y": int(y0),
                     "w": int(x1 - x0), "h": int(y1 - y0)})
    dets.sort(key=lambda d: -d["conf"])
    return dets


class ThermalToVisible:
    """Map thermal-plane boxes to visible-plane boxes through the warp LUT.

    The LUT is calib.py's (100,160,2) uint16 Q8 grid: for every 4x4 visible
    cell, the thermal pixel it samples (0xFFFF = outside the overlap). The
    student detects on the thermal plane; the operator looks at the visible
    plane - this is the same correspondence fusion.c warps pixels with, run
    in the opposite direction: a thermal box becomes the bounding box of all
    visible cells that sample from inside it.
    """

    LOW_W, LOW_H = 160, 100
    DECIMATION = 4
    INVALID = 0xFFFF

    def __init__(self, lut_path):
        lut = np.fromfile(lut_path, dtype=np.uint16).reshape(
            self.LOW_H, self.LOW_W, 2)
        self.valid = lut[..., 0] != self.INVALID
        self.tu = (lut[..., 0].astype(np.int32) + 128) >> 8
        self.tv = (lut[..., 1].astype(np.int32) + 128) >> 8

    def box(self, x, y, w, h):
        """Thermal-px box -> visible-px box, or None outside the overlap."""
        m = (self.valid
             & (self.tu >= x) & (self.tu < x + w)
             & (self.tv >= y) & (self.tv < y + h))
        if not m.any():
            return None
        ys, xs = np.nonzero(m)
        d = self.DECIMATION
        x0, y0 = int(xs.min()) * d, int(ys.min()) * d
        x1, y1 = (int(xs.max()) + 1) * d, (int(ys.max()) + 1) * d
        return x0, y0, x1 - x0, y1 - y0


class ThermalStudentTrt:
    """Person detection on raw Lepton frames.

    Feed CELSIUS float frames (raw counts * c_per_lsb + tmin - the same
    conversion the recorder metadata describes) through push(); it keeps the
    two-frame history and the training-identical validity chain: a previous
    frame counts only if it is 0 < dt <= 300 ms behind, and prev2 only if
    prev1's own link was also valid.
    """

    WIDTH, HEIGHT = 160, 120

    def __init__(self, engine=None, conf=0.5):
        self.runner = TrtRunner(
            engine or os.path.join(ENGINE_DIR, "thermal_student.engine"))
        self.conf = conf
        self.ms = 0.0
        self._hist = []            # [(frame_f32, t_ms)] newest last, len<=3
        self._prev_link_valid = False

    def push(self, celsius_frame, t_ms):
        """Add a frame (HxW float32 Celsius, monotonic ms) and detect.

        Returns [{conf,x,y,w,h}] in thermal pixels (160x120).
        """
        import time
        frame = np.asarray(celsius_frame, np.float32)
        if frame.shape != (self.HEIGHT, self.WIDTH):
            raise ValueError(f"thermal frame must be "
                             f"{self.HEIGHT}x{self.WIDTH}, got {frame.shape}")
        self._hist.append((frame, float(t_ms)))
        self._hist = self._hist[-3:]

        cur, t0 = self._hist[-1]
        seq = np.zeros((1, 3, self.HEIGHT, self.WIDTH), np.float32)
        seq[0, 0] = cur
        valid1 = valid2 = 0.0
        dt1 = 0.0
        if len(self._hist) >= 2:
            prev1, t1 = self._hist[-2]
            dt = t0 - t1
            if 0.0 < dt <= MAX_DT_MS:
                seq[0, 1] = prev1
                valid1, dt1 = 1.0, dt
                if len(self._hist) >= 3 and self._prev_link_valid:
                    seq[0, 2] = self._hist[-3][0]
                    valid2 = 1.0
        self._prev_link_valid = bool(valid1)

        t_start = time.time()
        out = self.runner.infer({
            "thermal_seq": seq,
            "valid_prev1": np.array([valid1], np.float32),
            "valid_prev2": np.array([valid2], np.float32),
            "dt1_ms": np.array([dt1], np.float32),
        })
        self.ms = (time.time() - t_start) * 1e3
        return _boxes_to_dets(out["scores"][0], out["boxes"][0],
                              self.WIDTH, self.HEIGHT, self.conf)

    def infer_raw(self, seq, valid1, valid2, dt1):
        """Stateless call for validation: seq (3,H,W) Celsius."""
        out = self.runner.infer({
            "thermal_seq": np.asarray(seq, np.float32)[None],
            "valid_prev1": np.array([valid1], np.float32),
            "valid_prev2": np.array([valid2], np.float32),
            "dt1_ms": np.array([dt1], np.float32),
        })
        return out["scores"][0], out["boxes"][0]

    def reset(self):
        self._hist.clear()
        self._prev_link_valid = False


class RadarStudentTrt:
    """Person detection from one frame of radar points.

    points: (N,6) float [x,y,z,velocity,snr,noise] in the project frame -
    the exact rows the exporter wrote into the shards' 'radar' array.
    Boxes come back in 640x400 RGB pixels.
    """

    WIDTH, HEIGHT = 640, 400

    def __init__(self, engine=None, conf=0.5):
        self.runner = TrtRunner(
            engine or os.path.join(ENGINE_DIR, "radar_student.engine"))
        self.conf = conf
        self.ms = 0.0

    def __call__(self, points):
        import time
        pts = np.zeros((1, MAX_POINTS, 6), np.float32)
        points = np.asarray(points, np.float32).reshape(-1, 6)
        n = min(len(points), MAX_POINTS)
        pts[0, :n] = points[:n]
        t_start = time.time()
        out = self.runner.infer({
            "radar_points": pts,
            "n_radar": np.array([n], np.int64),
        })
        self.ms = (time.time() - t_start) * 1e3
        return _boxes_to_dets(out["scores"][0], out["boxes"][0],
                              self.WIDTH, self.HEIGHT, self.conf)

    def infer_raw(self, points_padded, n):
        out = self.runner.infer({
            "radar_points": np.asarray(points_padded, np.float32)[None],
            "n_radar": np.array([n], np.int64),
        })
        return out["scores"][0], out["boxes"][0]
