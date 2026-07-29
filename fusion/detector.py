"""Camera detection behind one call: detect(frame, kind) -> [Box].

RGB     — YOLOv4-tiny via cv2.dnn (reuses the tested Detector from
          live_radar_camera; models/ already ships the weights, no new
          dependency). Valid in adequate light only.
Thermal — relative-to-background blob detection (core.detect.thermal: robust
          background + contrast threshold), with the absolute temperature
          bands kept for LABELLING only. Expects a uint8 gray frame mapped
          MIN_C..MAX_C -> 0..255.
"""
import collections

import cv2
import numpy as np

from core.detect import thermal as _thermal_core

Box = collections.namedtuple("Box", "x y w h label score")

# --- thermal labelling ----------------------------------------------------------
# Detection is RELATIVE (a blob must stand clear of the frame's own
# background — core.detect.thermal), which is what survives the failure modes
# the old absolute segmentation documented: on a hot day (ambient 33-40 C)
# walls/asphalt enter the "human" band but do not stand out from background;
# a distant person falls below the band (sub-pixel fill) yet still stands
# out. The bands below only NAME a found blob by its mean temperature —
# clamped at both ends, so a cool-looking distant person stays "warm" rather
# than vanishing, and engines/exhaust above MAX_C stay "hot". True thermal
# crossover (target contrast ~= 0) still yields nothing — that is physics,
# and the honest answer is a missing box, which the uncertainty layer turns
# into a high-sigma thermal estimate (radar carries the track).
MIN_C = 15.0                    # radiometric window mapped to 0..255
MAX_C = 45.0
THERMAL_CLASSES = [             # (label, tmin_c, tmax_c); bodies ~30-38 C
    ("warm", 25.0, 30.0),
    ("human", 30.0, 38.0),
    ("hot", 38.0, 45.0),
]


def temp_to_g(t):
    t = min(max(t, MIN_C), MAX_C)
    return int((t - MIN_C) * 255.0 / (MAX_C - MIN_C))


def g_to_temp(g):
    return g * (MAX_C - MIN_C) / 255.0 + MIN_C


def _band_label(t_mean):
    """Band label for a relatively-found blob; half-open bands (exactly 30 C
    is human, not warm), clamped to the first/last band at the extremes."""
    for label, lo, hi in THERMAL_CLASSES:
        if t_mean < hi:
            return label
    return THERMAL_CLASSES[-1][0]


def thermal_detections(gray, invalid_rows=()):
    """Radiometric gray frame -> [{'label','rect','cx','cy','t_mean','t_max',
    'confidence'}] — the record shape thermal_heatmap_n6.py emits, plus the
    detection confidence. ``invalid_rows``: interpolated (synthetic) rows to
    exclude from evidence, e.g. lepton_fix.DEAD_ROWS on this module."""
    out = []
    for b in _thermal_core.detect(gray, invalid_rows):
        out.append({"label": _band_label(g_to_temp(b.mean_dn)),
                    "rect": (b.x, b.y, b.w, b.h),
                    "cx": b.cx, "cy": b.cy,
                    "t_mean": g_to_temp(b.mean_dn),
                    "t_max": g_to_temp(b.max_dn),
                    "confidence": b.confidence})
    return out


# --- RGB (YOLO) ------------------------------------------------------------------
_rgb_net = None


def _rgb_detector():
    global _rgb_net
    if _rgb_net is None:
        from live_radar_camera import Detector   # tested YOLOv4-tiny wrapper
        _rgb_net = Detector()
    return _rgb_net


def detect(frame, kind="rgb", conf_thr=None, invalid_rows=()):
    """Unified detection entry. Returns [Box(x, y, w, h, label, score)].

    kind='rgb'     -> YOLO classes (person, chair, ...). LIGHT ONLY — the
                      mode selector must not call this on dark frames.
    kind='thermal' -> temperature classes (warm / human / hot); score is the
                      detection confidence (contrast x area x valid pixels,
                      core.detect.thermal) — the value thermal_estimate()
                      consumes. invalid_rows marks interpolated rows that
                      must not count as evidence.
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
        return [Box(*d["rect"], label=d["label"], score=d["confidence"])
                for d in thermal_detections(frame, invalid_rows)]
    raise ValueError("unknown detector kind: %r" % (kind,))
