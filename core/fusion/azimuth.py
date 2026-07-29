"""Azimuth-only thermal-box <-> radar-cluster association (pure).

The full fuse() path projects every radar point through a calibrated
extrinsic (calib.json). This module is the calibration-free fallback the
live overlay runs on: a pinhole azimuth from the box centre against the
cluster centroid's azimuth, greedy 1:1, then the same uncertainty-guided
inverse-variance combination the calibrated path uses. Range comes from the
matched cluster (radar owns range; a 160-px thermal image does not).

Conventions match fusion_core.CameraGeometry EXACTLY (a drift test pins
them): azimuth in degrees, + = right of centre; a cluster's camera azimuth
is atan2(-cy, cx) + yaw_offset. fusion_core cannot be imported here — it
drags cv2 and the material stack — so the three formulas are restated and
the test is the single-source-of-truth guard.

Thermal + radar only, like everything in core.fusion: RGB never enters, and
nothing here may import core.vitals.
"""
import math

from core.fusion import uncertainty

LEPTON_HFOV_DEG = 57.0   # FLIR Lepton 3.5 horizontal FOV (datasheet)
MATCH_AZ_DEG = 10.0      # mirrors fusion_core.MATCH_AZ_DEG (drift-checked)


def focal_px(width, hfov_deg=LEPTON_HFOV_DEG):
    """Pinhole focal length in pixels for a frame of this width."""
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def box_azimuth_deg(box, width, hfov_deg=LEPTON_HFOV_DEG):
    """Camera azimuth of a box centre; + = right of centre."""
    u_c = box.x + box.w / 2.0
    return math.degrees(math.atan((u_c - width / 2.0) / focal_px(width, hfov_deg)))


def box_half_width_deg(box, width, hfov_deg=LEPTON_HFOV_DEG):
    """Half the angular width of a box — widens the match gate."""
    return math.degrees(math.atan((box.w / 2.0) / focal_px(width, hfov_deg)))


def cluster_azimuth_deg(centroid, yaw_offset_deg=0.0):
    """Radar cluster centroid (x fwd, y left, z up) -> camera azimuth (deg)."""
    cx, cy = centroid[0], centroid[1]
    return math.degrees(math.atan2(-cy, max(cx, 0.01))) + yaw_offset_deg


def associate(boxes, clusters, width, hfov_deg=LEPTON_HFOV_DEG,
              max_az_deg=MATCH_AZ_DEG, yaw_offset_deg=0.0):
    """Greedy 1:1 azimuth assignment. Returns a list parallel to ``boxes``
    of the matched cluster dict or None."""
    cands = []
    for bi, b in enumerate(boxes):
        az_b = box_azimuth_deg(b, width, hfov_deg)
        gate = max_az_deg + box_half_width_deg(b, width, hfov_deg)
        for ci, c in enumerate(clusters):
            err = abs(az_b - cluster_azimuth_deg(c["centroid"], yaw_offset_deg))
            if err <= gate:
                cands.append((err, bi, ci))
    cands.sort()
    assigned = [None] * len(boxes)
    used = set()
    for _, bi, ci in cands:
        if assigned[bi] is None and ci not in used:
            assigned[bi] = clusters[ci]
            used.add(ci)
    return assigned


def fuse_thermal(boxes, clusters, width, hfov_deg=LEPTON_HFOV_DEG,
                 max_az_deg=MATCH_AZ_DEG, yaw_offset_deg=0.0):
    """Thermal boxes + radar clusters -> fused records with uncertainty.

    One record per box; unmatched boxes stay camera-only (range None) and
    their confidence is the thermal estimate alone.
    """
    matched = associate(boxes, clusters, width, hfov_deg, max_az_deg,
                        yaw_offset_deg)
    out = []
    for b, c in zip(boxes, matched):
        ests = [uncertainty.thermal_estimate(b)]
        if c is not None and "v_abs" in c:
            ests.append(uncertainty.radar_estimate(c))
        fused = uncertainty.combine(ests)
        out.append({
            "label": getattr(b, "label", None),
            "box": (b.x, b.y, b.w, b.h),
            "score": uncertainty._box_confidence(b),
            "azimuth_deg": round(box_azimuth_deg(b, width, hfov_deg), 1),
            "range_m": c["range_m"] if c else None,
            "doppler_mps": c["doppler_mps"] if c else None,
            "radar_class": c["label"] if c else None,
            "n_radar_pts": c["n_points"] if c else 0,
            "confidence": round(fused.score, 3),
            "uncertainty": round(fused.sigma, 3),
            "contributing_sensor": fused.contributing_sensor,
        })
    return out
