"""Projection + association: camera boxes x radar points -> FusedObject.

Uses the geometric calibration (calib.json via calibrate_radar_camera):
every radar point is projected into the image; points falling inside a
detection box vote for it; the nearest-in-range point supplies range, and
the Doppler is the median over that point's range neighbourhood (a single
point's Doppler is noisy; the neighbourhood keeps it honest without letting
a background return behind the target pollute it).
"""
import math

import numpy as np

from calibrate_radar_camera import project
from radar_classify_n6 import classify_frame

DOPPLER_NEIGHBORHOOD_M = 0.75   # points within this range band share Doppler


def _range_of(p):
    return math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])


def FusedObject(label, box, range_m=None, doppler_mps=None, radar_class=None,
                n_radar_pts=0, score=None):
    """Uniform output record for every downstream consumer."""
    return {"label": label, "box": tuple(box[:4]),
            "score": score if score is not None else getattr(box, "score", None),
            "range_m": range_m, "doppler_mps": doppler_mps,
            "radar_class": radar_class, "n_radar_pts": n_radar_pts}


def _radar_class_for(point, clusters):
    """Label of the cluster whose centroid is nearest to the chosen point."""
    best, bestd = None, 1.0
    for c in clusters:
        cx, cy, cz = c["centroid"]
        d = math.sqrt((point[0] - cx) ** 2 + (point[1] - cy) ** 2
                      + (point[2] - cz) ** 2)
        if d < bestd:
            best, bestd = c, d
    return best["label"] if best else None


def fuse(boxes, points, calib, clusters=None):
    """boxes: [Box], points: [(x,y,z,v[,snr,noise])], calib: load_calib dict.

    Returns one FusedObject per box (unmatched boxes keep range_m=None —
    the caller decides how to present camera-only detections).
    clusters: optional pre-computed classify_frame output; computed here
    when omitted and needed.
    """
    if not boxes:
        return []
    if not points or calib is None:
        return [FusedObject(b.label, b) for b in boxes]

    uv, z_cam = project(points, calib)
    if clusters is None:
        clusters = classify_frame(points)

    out = []
    for b in boxes:
        inside = [i for i in range(len(points))
                  if z_cam[i] > 0
                  and b.x <= uv[i][0] <= b.x + b.w
                  and b.y <= uv[i][1] <= b.y + b.h]
        if not inside:
            out.append(FusedObject(b.label, b))
            continue
        nearest = min(inside, key=lambda i: _range_of(points[i]))
        rng = _range_of(points[nearest])
        vs = sorted(points[i][3] for i in inside
                    if abs(_range_of(points[i]) - rng)
                    <= DOPPLER_NEIGHBORHOOD_M)
        out.append(FusedObject(
            b.label, b, range_m=round(rng, 2),
            doppler_mps=round(vs[len(vs) // 2], 2),
            radar_class=_radar_class_for(points[nearest], clusters),
            n_radar_pts=len(inside)))
    return out
