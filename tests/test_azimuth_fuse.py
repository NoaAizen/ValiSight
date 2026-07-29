"""core.fusion.azimuth: the calibration-free association the live overlay
runs on, plus the drift alarm pinning its geometry to fusion_core's.

Synthetic boxes and clusters with known angles go through the public
associate()/fuse_thermal() entry points; nothing internal is called.
"""
import math

import pytest

from core.fusion import azimuth as az
from core.fusion import uncertainty as unc
from fusion.detector import Box

W = 160          # Lepton frame width


def cl(x, y, label="pedestrian", n=8, v_abs=1.0, v_spread=0.6, extent=0.8):
    rng = math.sqrt(x * x + y * y)
    return {"label": label, "range_m": round(rng, 2), "doppler_mps": -v_abs,
            "extent_m": extent, "n_points": n, "v_abs": v_abs,
            "v_spread": v_spread, "extent": extent, "n": n,
            "centroid": (x, y, 0.0)}


def centered_box(score=0.9, w=30):
    return Box(int(W / 2 - w / 2), 40, w, 50, "human", score)


def test_geometry_matches_fusion_core():
    """Drift alarm: the restated formulas must equal CameraGeometry's."""
    fusion_core = pytest.importorskip("fusion_core")
    geo = fusion_core.CameraGeometry(hfov_deg=az.LEPTON_HFOV_DEG)
    assert az.MATCH_AZ_DEG == fusion_core.MATCH_AZ_DEG
    for x, y in [(3.0, 0.0), (4.0, 1.0), (2.0, -1.5), (6.0, 3.0)]:
        assert az.cluster_azimuth_deg((x, y, 0.0)) == pytest.approx(
            geo.cluster_azimuth_deg((x, y, 0.0)))
    for bx, bw in [(10, 30), (65, 30), (100, 50)]:
        b = Box(bx, 40, bw, 50, "human", 0.9)
        assert az.box_azimuth_deg(b, W) == pytest.approx(
            geo.box_azimuth_range((bx, 40, bw, 50), "person", (120, W))[0])
        assert az.box_half_width_deg(b, W) == pytest.approx(
            geo.box_half_width_deg((bx, 40, bw, 50), (120, W)))


def test_forward_cluster_matches_centered_box():
    got = az.associate([centered_box()], [cl(3.0, 0.0)], W)
    assert got[0] is not None and got[0]["range_m"] == 3.0


def test_far_off_azimuth_cluster_does_not_match():
    # 3 m forward, 2.2 m to the side -> ~36 deg off a centred box
    assert az.associate([centered_box()], [cl(3.0, 2.2)], W) == [None]


def test_greedy_assignment_prefers_smaller_error():
    left = Box(20, 40, 30, 50, "human", 0.9)      # az ~ -24 deg
    mid = centered_box()
    c_mid, c_left = cl(3.0, 0.0), cl(3.0, 1.3)    # ~ -23.4 deg (y left)
    got = az.associate([left, mid], [c_mid, c_left], W)
    assert got[0] is c_left and got[1] is c_mid


def test_fused_record_carries_uncertainty_fields():
    o = az.fuse_thermal([centered_box()], [cl(3.0, 0.0)], W)[0]
    assert o["range_m"] == 3.0 and o["radar_class"] == "pedestrian"
    assert o["confidence"] > 0.7
    assert o["contributing_sensor"] in (unc.THERMAL, unc.RADAR)
    assert o["uncertainty"] < unc.thermal_estimate(centered_box()).sigma


def test_unmatched_box_stays_thermal_only():
    o = az.fuse_thermal([centered_box()], [], W)[0]
    assert o["range_m"] is None and o["radar_class"] is None
    assert o["contributing_sensor"] == unc.THERMAL


def test_weak_box_with_good_cluster_defers_to_radar():
    o = az.fuse_thermal([centered_box(score=0.05)], [cl(3.0, 0.0)], W)[0]
    assert o["contributing_sensor"] == unc.RADAR
    assert o["confidence"] > 0.5


def test_compact_glint_is_distrusted():
    body = az.fuse_thermal([centered_box()], [cl(3.0, 0.0)], W)[0]
    glint = az.fuse_thermal([centered_box()],
                            [cl(3.0, 0.0, n=3, extent=0.15, v_spread=0.05)],
                            W)[0]
    assert glint["uncertainty"] > body["uncertainty"]
    assert glint["contributing_sensor"] == unc.THERMAL
