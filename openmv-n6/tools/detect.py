#!/usr/bin/env python3
"""Object detection on the host, with a temperature attached to every box.

The point of running a detector against this pipeline is not the boxes - any
webcam gives you those. It is that each box can be handed to
fusion_temp_region(), so the answer is "person, 36.4 C peak at (312,180)" rather
than "person". That is the measurement the two cameras exist to make.

Runs on the host rather than the N6's NPU. The board can do it - the vendored
OpenMV tree has ml_yolov8 and ml_blazeface - but compiling a model for that NPU
needs stedgeai, which is x86 only, and this host is aarch64. Nothing about the
design here prevents moving it to the board later; the box-to-temperature step
is the part worth keeping either way.

Model: yolov4-tiny in darknet format, run through cv2.dnn on the CPU. Measured
on this Jetson (6 cores, OpenCV 4.8, no CUDA build): 68 ms at 416, 46 ms at 320.
The Lepton's frame period is 114 ms, so 416 fits inside a frame - but only off
the serial reader's thread, which is why live.py gives this its own.
"""
import os
import shutil
import tempfile
import time

import cv2
import numpy as np

# The bench already has these; no download and no network at run time.
MODEL_DIR = os.path.expanduser("~/archive/radar/models")
CFG = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
WEIGHTS = os.path.join(MODEL_DIR, "yolov4-tiny.weights")

# COCO-80, in the order darknet emits them. Kept here rather than in a .names
# file so a missing sidecar cannot silently shift every label by one.
COCO = (
    "person bicycle car motorbike aeroplane bus train truck boat traffic_light "
    "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
    "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair sofa "
    "pottedplant bed diningtable toilet tvmonitor laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase "
    "scissors teddy_bear hair_drier toothbrush").split()

SIZE = 416
CONF = 0.35
NMS = 0.45

# Leave the reader thread a core. cv2.dnn takes every core it is given, and the
# one thing this whole split exists to protect is the serial reader's latency:
# a host that stops draining the CDC for 500ms makes the board's out.write give
# up and discard the tail of a frame it has already announced.
THREADS = 4


class Detector:
    """yolov4-tiny over cv2.dnn. Not thread-safe; give it one thread of its own."""

    def __init__(self, size=SIZE, conf=CONF, classes=None):
        self.size, self.conf = size, conf
        # readNetFromDarknet cannot open a non-ASCII path, and it fails in a way
        # that reads as a corrupt model rather than a path problem. Stage the two
        # files somewhere ASCII and load from there.
        tmp = os.path.join(tempfile.gettempdir(), "fusion_yolo")
        os.makedirs(tmp, exist_ok=True)
        paths = []
        for src in (CFG, WEIGHTS):
            if not os.path.isfile(src):
                raise FileNotFoundError(src)
            dst = os.path.join(tmp, os.path.basename(src))
            if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
                shutil.copyfile(src, dst)
            paths.append(dst)

        cv2.setNumThreads(THREADS)
        self.net = cv2.dnn.readNetFromDarknet(*paths)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.out_names = self.net.getUnconnectedOutLayersNames()
        # None means every class. A filter is worth having because a false
        # 'diningtable' over the scene costs a box AND a temperature readout,
        # and the readout is the part that gets written down.
        self.classes = set(classes) if classes else None
        self.ms = 0.0

    def __call__(self, img):
        """img: HxW gray or HxWx3 BGR. Returns [{cls,conf,x,y,w,h}], newest first."""
        if img.ndim == 2:
            # The visible sensor is configured GRAYSCALE, so there is no colour to
            # give the network. Replicating the channel is the honest way to feed
            # it: COCO weights lose some accuracy on grey input, and inventing
            # colour would not put it back.
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        t0 = time.time()
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.size, self.size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        self.ms = (time.time() - t0) * 1e3

        boxes, confs, ids = [], [], []
        for out in outs:
            # Vectorised: the raw output is ~2500 rows and a Python loop over it
            # cost more than the forward pass on this machine.
            scores = out[:, 5:]
            cid = scores.argmax(1)
            best = scores[np.arange(len(scores)), cid]
            keep = best >= self.conf
            if self.classes is not None:
                keep &= np.isin(cid, [COCO.index(c) for c in self.classes
                                      if c in COCO])
            if not keep.any():
                continue
            sel, c, b = out[keep], cid[keep], best[keep]
            cx, cy, bw, bh = sel[:, 0] * w, sel[:, 1] * h, sel[:, 2] * w, sel[:, 3] * h
            boxes += np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1).astype(int).tolist()
            confs += b.astype(float).tolist()
            ids += c.astype(int).tolist()

        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(boxes, confs, self.conf, NMS)
        dets = []
        for i in np.array(keep).flatten():
            x, y, bw, bh = boxes[i]
            # Clamp before anything downstream indexes with these. An unclamped
            # box reaches fusion_temp_region(), which would then average over
            # pixels outside the frame.
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + bw), min(h, y + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            dets.append({"cls": COCO[ids[i]], "conf": round(confs[i], 3),
                         "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
        dets.sort(key=lambda d: -d["conf"])
        return dets


def _gpu_comes_up(timeout_s=120):
    """Can TensorRT actually start right now? Asked in a process we can lose.

    On Tegra a CUDA/nvmap allocation failure does not raise - it takes the whole
    process down with SIGSEGV. Measured 2026-08-17 on this box:

        NvMapMemAllocInternalTagged: ... error 12
        [TRT] [E] createInferRuntime: ... CUDA initialization failure with error: 2
        [TRT] [E] ... Cuda Runtime (out of memory)
        Segmentation fault (core dumped)

    A try/except around TrtDetector() cannot catch that, so live.py died at
    startup rather than falling back - the failure the fallback exists to absorb
    was the one thing it could not absorb.

    A free-memory threshold is not a usable guard either: the same failure was
    reproduced with 1386 MB MemAvailable, because nvmap needs large contiguous
    blocks out of the carveout the GPU shares with the CPU, and a box that is
    74% into swap is fragmented rather than empty. Available memory says nothing
    about contiguity. So the question is not estimated, it is asked - in a child,
    where a segfault costs an exit code instead of the viewer. Measured 0.66 s
    against live.py's ~23 s bring-up.
    """
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    code = ("import sys; sys.path.insert(0, %r); import trt_detect; "
            "d = trt_detect.TrtDetector(); d.close()" % here)
    try:
        rc = subprocess.call([sys.executable, "-c", code],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             timeout=timeout_s)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return rc == 0


def make_detector(backend="auto", size=SIZE, conf=CONF, classes=None):
    """Pick a detector. Both backends return the same [{cls,conf,x,y,w,h}].

    'auto' prefers the GPU and says so on stderr when it falls back, because a
    silent fallback here is a 10x slowdown that would otherwise be diagnosed as
    a link problem. `size` is ignored by the GPU backend: its engine is built
    for a fixed 640x640 input and changing it means rebuilding the engine.
    """
    import sys
    if backend not in ("auto", "gpu", "cpu"):
        raise ValueError("backend must be auto, gpu or cpu, not %r" % (backend,))

    if backend != "cpu":
        # Probe first. 'gpu' asks for the GPU explicitly, so it is entitled to
        # the real error rather than a fallback - but it should still get a
        # message instead of a core dump.
        if not _gpu_comes_up():
            msg = ("the GPU backend cannot start (TensorRT/CUDA failed to "
                   "initialise - usually no contiguous memory; check free -m)")
            if backend == "gpu":
                raise RuntimeError(msg)
            print("detector: %s, falling back to yolov4-tiny on the CPU" % msg,
                  file=sys.stderr)
        else:
            try:
                import trt_detect
                det = trt_detect.TrtDetector(conf=conf, classes=classes, names=COCO)
                det.backend = "gpu"
                return det
            except Exception as e:
                if backend == "gpu":
                    raise
                print("detector: no GPU backend (%s: %s), falling back to "
                      "yolov4-tiny on the CPU" % (type(e).__name__, e),
                      file=sys.stderr)

    det = Detector(size=size, conf=conf, classes=classes)
    det.backend = "cpu"
    return det


# A person is the detection this rig exists for, so it keeps a fixed green
# regardless of the warped/unwarped colour convention; the (unreg) mark on the
# temperature still carries that warning.
PERSON_COL = (0, 230, 0)

# Peak-in-box range that counts as body heat, deg C. The floor sits above a
# warm room (29-30 C has been observed here) but below bare skin; the ceiling
# rejects lamps and machines. With the unregistered warp the peak can be read
# a few pixels off the body, so the flag is a cross-check, not radiometry.
BODY_C = (31.0, 39.0)


def _dashed_rect(img, x0, y0, x1, y1, col, dash=9):
    for x in range(x0, x1, dash * 2):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), col, 2)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), col, 2)
    for y in range(y0, y1, dash * 2):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), col, 2)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), col, 2)


def annotate(img, dets, warped):
    """Draw the boxes and their readings. Modifies img in place.

    `warped` is not decoration. Without a calibrated LUT the thermal layer is
    stretched over the frame rather than registered to it, so the temperature
    inside a box belongs to whatever the stretch happened to put there. A number
    like that is worse than no number, because it looks like a measurement - so
    it is drawn in a different colour and marked, every frame, with no way to
    turn the mark off.
    """
    for d in dets:
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        hot = d.get("max_c")
        person = d["cls"] == "person"
        verified = person and bool(d.get("body_heat"))
        if person:
            col = PERSON_COL
        else:
            col = (120, 255, 120) if warped else (140, 170, 255)
        if person and not verified:
            _dashed_rect(img, x, y, x + w, y + h, col)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)

        if person:
            label = "person" if verified else "person ?"
        else:
            label = "%s %.0f%%" % (d["cls"], 100 * d["conf"])
        if d.get("radar_m") is not None:
            label += "  %.1fm" % d["radar_m"]
        if hot is not None:
            label += "  %.1fC" % hot
            if not warped:
                label += " (unreg)"
        elif d.get("no_thermal"):
            label += "  no thermal"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        mark = 16 if verified else 0                 # room for the check mark
        ty = y - 6 if y - 6 - th > 0 else y + h + th + 6
        cv2.rectangle(img, (x, ty - th - 4), (x + tw + 6 + mark, ty + 3), (0, 0, 0), -1)
        cv2.putText(img, label, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)
        if verified:
            # cv2's Hershey fonts have no U+2713, so the check is two strokes.
            cx = x + tw + 8
            cv2.line(img, (cx, ty - 4), (cx + 3, ty - 1), col, 2, cv2.LINE_AA)
            cv2.line(img, (cx + 3, ty - 1), (cx + 11, ty - th + 1), col, 2, cv2.LINE_AA)

        # The hot pixel itself, not just the box. On an inspection frame the
        # location of the peak is most of the finding - "this motor is warm" and
        # "this motor's near bearing is warm" are different reports.
        if d.get("max_x") is not None:
            cv2.drawMarker(img, (d["max_x"], d["max_y"]), (60, 220, 255),
                           cv2.MARKER_CROSS, 11, 1)
    return img
