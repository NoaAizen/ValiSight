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
from core.fusion import uncertainty
from radar_classify_n6 import classify_frame

DOPPLER_NEIGHBORHOOD_M = 0.75   # points within this range band share Doppler


def _range_of(p):
    return math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])


def FusedObject(label, box, range_m=None, doppler_mps=None, radar_class=None,
                n_radar_pts=0, score=None, confidence=None, uncertainty=None,
                contributing_sensor=None):
    """Uniform output record for every downstream consumer.

    ``confidence`` / ``uncertainty`` / ``contributing_sensor`` are the
    inverse-variance fusion outputs (core.fusion.uncertainty). They are
    populated only on the decision path — thermal + radar; an RGB run
    leaves them None by design."""
    return {"label": label, "box": tuple(box[:4]),
            "score": score if score is not None else getattr(box, "score", None),
            "range_m": range_m, "doppler_mps": doppler_mps,
            "radar_class": radar_class, "n_radar_pts": n_radar_pts,
            "confidence": confidence, "uncertainty": uncertainty,
            "contributing_sensor": contributing_sensor}


def _radar_cluster_for(point, clusters):
    """The cluster whose centroid is nearest to the chosen point, or None."""
    best, bestd = None, 1.0
    for c in clusters:
        cx, cy, cz = c["centroid"]
        d = math.sqrt((point[0] - cx) ** 2 + (point[1] - cy) ** 2
                      + (point[2] - cz) ** 2)
        if d < bestd:
            best, bestd = c, d
    return best


def _radar_class_for(point, clusters):
    """Label of the cluster whose centroid is nearest to the chosen point."""
    best = _radar_cluster_for(point, clusters)
    return best["label"] if best else None


def _certainty(box, cluster, kind):
    """Uncertainty-guided confidence for one fused object.

    Thermal + radar only: RGB is out of the decision path, so anything but
    kind='thermal' contributes no thermal estimate; with no radar cluster
    either, there is nothing to fuse and the fields stay None. The soft
    glint penalty lives inside radar_estimate (compact extent, velocity-
    independent) — the primary mechanism that replaced the hard
    PERSON_METAL_COST veto; the legacy matcher's cost term remains only as
    a backstop."""
    if kind != "thermal":
        return {}
    estimates = [uncertainty.thermal_estimate(box)]
    if cluster is not None and "v_abs" in cluster:
        estimates.append(uncertainty.radar_estimate(cluster))
    fused = uncertainty.combine(estimates)
    return {"confidence": round(fused.score, 3),
            "uncertainty": round(fused.sigma, 3),
            "contributing_sensor": fused.contributing_sensor}


def fuse(boxes, points, calib, clusters=None, kind=None):
    """boxes: [Box], points: [(x,y,z,v[,snr,noise])], calib: load_calib dict.

    Returns one FusedObject per box (unmatched boxes keep range_m=None —
    the caller decides how to present camera-only detections).
    clusters: optional pre-computed classify_frame output; computed here
    when omitted and needed.
    kind: the camera modality that produced ``boxes`` ('thermal' / 'rgb').
    'thermal' turns on the uncertainty-guided confidence fields on each
    fused object; RGB never does — it is not in the decision path.
    """
    if not boxes:
        return []
    if not points or calib is None:
        return [FusedObject(b.label, b, **_certainty(b, None, kind))
                for b in boxes]

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
            out.append(FusedObject(b.label, b, **_certainty(b, None, kind)))
            continue
        nearest = min(inside, key=lambda i: _range_of(points[i]))
        rng = _range_of(points[nearest])
        vs = sorted(points[i][3] for i in inside
                    if abs(_range_of(points[i]) - rng)
                    <= DOPPLER_NEIGHBORHOOD_M)
        cluster = _radar_cluster_for(points[nearest], clusters)
        out.append(FusedObject(
            b.label, b, range_m=round(rng, 2),
            doppler_mps=round(vs[len(vs) // 2], 2),
            radar_class=cluster["label"] if cluster else None,
            n_radar_pts=len(inside),
            **_certainty(b, cluster, kind)))
    return out
