"""Uncertainty-guided confidence through the FULL fuse() pipeline.

Same synthetic geometry as test_fuse.py (aligned pose, centred box), but the
assertions are about the new decision-path fields: ``confidence``,
``uncertainty`` and ``contributing_sensor`` — attached only for
kind='thermal', inverse-variance-combined from the box score and the matched
radar cluster's own features. No internal function is called directly:
points go in, fused objects come out.
"""
import numpy as np

from calibrate_radar_camera import R_ALIGNED
from core.fusion import uncertainty as unc
from fusion.detector import Box
from fusion.fuse import fuse

K = np.array([[600.0, 0, 320.0],
              [0, 600.0, 240.0],
              [0, 0, 1.0]])
CALIB = {"K": K, "dist": None, "R": R_ALIGNED, "t": np.zeros((3, 1))}

STRONG_BOX = Box(280, 200, 80, 80, "human", 0.9)
WEAK_BOX = Box(280, 200, 80, 80, "human", 0.05)   # barely-there thermal blob


def pt(x, y=0.0, z=0.0, v=0.0):
    return (x, y, z, v)


def body_points(v=-1.2):
    """A walking person: 8 points, ~0.6 m extent, limb micro-Doppler."""
    return [pt(2.9 + 0.03 * i, -0.15 + 0.04 * i, -0.10 + 0.03 * i,
               v - 0.25 + 0.06 * i) for i in range(8)]


def glint_points(v=-1.2):
    """Carried metal: 3 tightly-packed specular returns at walking speed."""
    return [pt(3.0, 0.0, 0.0, v), pt(3.02, 0.01, 0.0, v),
            pt(3.01, 0.0, 0.01, v)]


def test_thermal_match_attaches_fused_confidence():
    o = fuse([STRONG_BOX], body_points(), CALIB, kind="thermal")[0]
    assert o["range_m"] is not None
    assert o["confidence"] is not None and o["confidence"] > 0.7
    assert o["contributing_sensor"] in (unc.THERMAL, unc.RADAR)
    # fusion is more certain than the thermal box alone
    th_alone = unc.thermal_estimate(STRONG_BOX)
    assert o["uncertainty"] < th_alone.sigma


def test_rgb_never_enters_the_decision_path():
    for kw in ({}, {"kind": "rgb"}):
        o = fuse([STRONG_BOX], body_points(), CALIB, **kw)[0]
        assert o["confidence"] is None
        assert o["uncertainty"] is None
        assert o["contributing_sensor"] is None


def test_weak_thermal_box_defers_to_radar_without_dropping():
    o = fuse([WEAK_BOX], body_points(), CALIB, kind="thermal")[0]
    assert o["contributing_sensor"] == unc.RADAR
    assert o["confidence"] > 0.5, "track collapsed on a weak thermal box"


def test_camera_only_box_carries_thermal_estimate():
    # points project far left of the box -> no radar vote, thermal only
    away = [pt(3.0, 0.8, 0.0, -1.0), pt(3.05, 0.85, 0.0, -1.0)]
    o = fuse([STRONG_BOX], away, CALIB, kind="thermal")[0]
    assert o["range_m"] is None
    assert o["contributing_sensor"] == unc.THERMAL
    th = unc.thermal_estimate(STRONG_BOX)
    assert abs(o["confidence"] - th.score) < 0.005
    assert abs(o["uncertainty"] - th.sigma) < 0.005


def test_moving_glint_is_distrusted_and_thermal_carries():
    body = fuse([STRONG_BOX], body_points(), CALIB, kind="thermal")[0]
    glint = fuse([STRONG_BOX], glint_points(), CALIB, kind="thermal")[0]
    assert glint["radar_class"] is not None      # the glint DID match the box
    assert glint["uncertainty"] > body["uncertainty"]
    assert glint["contributing_sensor"] == unc.THERMAL
    # the decision rides on the box score, not on the specular return
    assert abs(glint["confidence"] - STRONG_BOX.score) < 0.05
