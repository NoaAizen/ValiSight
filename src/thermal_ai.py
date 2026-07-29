"""Learned person detection on the repaired thermal frame (adapter).

Stage-1 AI: the repo's existing YOLOv4-tiny (models/, COCO weights, the same
net the RGB path runs) applied to the Lepton frame. A network trained on RGB
transfers only partially to LWIR — people are person-shaped in thermal, so
it fires on clear silhouettes and misses low-contrast ones — which is
exactly why its output is MERGED with the rule-based warm-blob detector
(core.detect.merge) rather than replacing it. The merge is where the value
is: agreement -> boosted confidence + a 'person' label on the fused object.

The net sees only the real repaired frame (contrast-stretched for the net's
benefit — a deterministic per-frame linear map, not synthesis). Heavy deps
(cv2.dnn) stay in this adapter; core/ never imports them. The upgrade path
is a thermal-native net (FLIR ADAS fine-tune, TensorRT) behind this same
detect_person() seam.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

AI_CONF_THR = 0.30       # YOLO person threshold on thermal; RGB default is
                         # tuned for daylight — thermal transfer needs recall
PERSON_CLASS = "person"

_detector = None         # loaded once, lazily — import must stay cheap


class AiBox:
    """Duck-typed like fusion.detector.Box: x y w h label score."""

    __slots__ = ("x", "y", "w", "h", "label", "score", "confidence")

    def __init__(self, x, y, w, h, score):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = PERSON_CLASS
        self.score = self.confidence = score


def available():
    try:
        _load()
        return True
    except Exception:
        return False


def _load():
    global _detector
    if _detector is None:
        from live_radar_camera import Detector
        from photo_to_radar import COCO
        _detector = (Detector(), COCO)
    return _detector


def detect_person(frame_u8, conf_thr=AI_CONF_THR):
    """Repaired 160x120 gray frame -> [AiBox] of persons, frame coords.

    The frame is contrast-stretched to full range (the AGC output spans ~70
    levels; the net was trained on full-range images) and upscaled — both
    deterministic maps of real pixels.
    """
    det, coco = _load()
    f = frame_u8.astype(np.float32)
    lo, hi = np.percentile(f, [1.0, 99.0])
    f = np.clip((f - lo) * (255.0 / max(hi - lo, 1.0)), 0, 255).astype(np.uint8)
    import cv2
    up = cv2.resize(f, (f.shape[1] * 3, f.shape[0] * 3),
                    interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    out = []
    for d in det.detect(rgb, conf_thr):
        if d["label"] != PERSON_CLASS:
            continue
        x, y, w, h = d["box"]
        out.append(AiBox(int(x / 3), int(y / 3), max(1, int(w / 3)),
                         max(1, int(h / 3)), float(d["conf"])))
    return out
