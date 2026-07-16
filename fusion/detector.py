"""Camera detection behind one call: detect(frame, kind) -> [Box].

RGB     — YOLOv4-tiny via cv2.dnn (reuses the tested Detector from
          live_radar_camera; models/ already ships the weights, no new
          dependency). Valid in adequate light only.
Thermal — temperature-band segmentation, the host-side numpy/OpenCV port of
          thermal_heatmap_n6.py (which is the on-device MicroPython version).
          Expects a uint8 gray frame mapped MIN_C..MAX_C -> 0..255.
"""
import collections

import cv2
import numpy as np

Box = collections.namedtuple("Box", "x y w h label score")

# --- thermal segmentation (mirrors thermal_heatmap_n6.py) -----------------------
MIN_C = 15.0                    # radiometric window mapped to 0..255
MAX_C = 45.0
THERMAL_CLASSES = [             # (label, tmin_c, tmax_c); bodies ~30-38 C
    ("warm", 25.0, 30.0),
    ("human", 30.0, 38.0),
    ("hot", 38.0, 45.0),
]
THERMAL_MIN_AREA_PX = 8


def temp_to_g(t):
    t = min(max(t, MIN_C), MAX_C)
    return int((t - MIN_C) * 255.0 / (MAX_C - MIN_C))


def g_to_temp(g):
    return g * (MAX_C - MIN_C) / 255.0 + MIN_C


def thermal_detections(gray):
    """Radiometric gray frame -> [{'label','rect','cx','cy','t_mean','t_max'}]
    — same record shape as the on-device thermal_heatmap_n6.py emits."""
    out = []
    for label, lo, hi in THERMAL_CLASSES:
        mask = cv2.inRange(gray, temp_to_g(lo), temp_to_g(hi))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < THERMAL_MIN_AREA_PX:
                continue
            region = gray[y:y + h, x:x + w][
                mask[y:y + h, x:x + w] > 0]
            out.append({"label": label, "rect": (int(x), int(y),
                                                 int(w), int(h)),
                        "cx": float(cents[i][0]), "cy": float(cents[i][1]),
                        "t_mean": g_to_temp(float(region.mean())),
                        "t_max": g_to_temp(float(region.max()))})
    return out


# --- RGB (YOLO) ------------------------------------------------------------------
_rgb_net = None


def _rgb_detector():
    global _rgb_net
    if _rgb_net is None:
        from live_radar_camera import Detector   # tested YOLOv4-tiny wrapper
        _rgb_net = Detector()
    return _rgb_net


def detect(frame, kind="rgb", conf_thr=None):
    """Unified detection entry. Returns [Box(x, y, w, h, label, score)].

    kind='rgb'     -> YOLO classes (person, chair, ...). LIGHT ONLY — the
                      mode selector must not call this on dark frames.
    kind='thermal' -> temperature classes (warm / human / hot); score is the
                      blob's mean temperature normalized into its band.
    """
    if frame is None:
        return []
    if kind == "rgb":
        det = _rgb_detector()
        dets = det.detect(frame) if conf_thr is None \
            else det.detect(frame, conf_thr)
        return [Box(*d["box"], label=d["label"], score=float(d["conf"]))
                for d in dets]
    if kind == "thermal":
        boxes = []
        for d in thermal_detections(frame):
            lo, hi = next((a, b) for l, a, b in THERMAL_CLASSES
                          if l == d["label"])
            score = min(max((d["t_mean"] - lo) / max(hi - lo, 1e-6), 0.0), 1.0)
            boxes.append(Box(*d["rect"], label=d["label"], score=score))
        return boxes
    raise ValueError("unknown detector kind: %r" % (kind,))
