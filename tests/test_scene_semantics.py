"""scene_semantics: ground/tree geometry rules + color confirmation."""
import numpy as np

from scene_semantics import (semantic_from_geometry, green_fraction,
                             resolve_semantics)

RADAR_HEIGHT = 1.0


def cl(centroid, ext, label="static", refl=0.0, rng=None):
    x, y, z = centroid
    return {"centroid": centroid, "ext_xyz": ext, "label": label,
            "refl_db": refl,
            "range_m": rng if rng is not None else (x * x + y * y) ** 0.5}


def test_ground_at_floor_level():
    c = cl((2.5, 0.0, -RADAR_HEIGHT), (0.8, 0.8, 0.2))
    assert semantic_from_geometry(c, RADAR_HEIGHT) == "ground"


def test_ground_not_under_the_rig():
    c = cl((0.3, 0.0, -RADAR_HEIGHT), (0.5, 0.5, 0.2))
    assert semantic_from_geometry(c, RADAR_HEIGHT) is None


def test_tall_static_column_is_tree_without_color():
    c = cl((5.0, 1.0, 0.2), (0.4, 0.4, 1.6))
    assert semantic_from_geometry(c, RADAR_HEIGHT) == "tree"


def test_medium_column_needs_color_confirmation():
    c = cl((5.0, 1.0, 0.2), (0.4, 0.4, 1.0))
    assert semantic_from_geometry(c, RADAR_HEIGHT) == "tree?"


def test_metal_pole_is_not_a_tree():
    c = cl((5.0, 1.0, 0.2), (0.4, 0.4, 1.6), refl=15.0)
    assert semantic_from_geometry(c, RADAR_HEIGHT) is None


def test_moving_cluster_gets_no_static_semantics():
    c = cl((5.0, 1.0, 0.2), (0.4, 0.4, 1.6), label="pedestrian")
    assert semantic_from_geometry(c, RADAR_HEIGHT) is None


def test_green_fraction():
    green = np.zeros((100, 100, 3), np.uint8)
    green[:] = (40, 200, 40)                    # BGR vegetation-ish
    gray = np.full((100, 100, 3), 128, np.uint8)
    assert green_fraction(green, 50, 50, 30) > 0.9
    assert green_fraction(gray, 50, 50, 30) == 0.0
    assert green_fraction(green, -100, 50, 10) == 0.0   # out of frame


def test_resolve_semantics_dark_frame_geometry_only():
    """img=None (dark): unconfirmed 'tree?' must drop, hard 'tree' stays."""
    maybe = cl((5.0, 1.0, 0.2), (0.4, 0.4, 1.0))
    sure = cl((5.0, -1.0, 0.2), (0.4, 0.4, 1.7))
    out = resolve_semantics([maybe, sure], None, 60.0, 0.0, RADAR_HEIGHT)
    assert out[0]["semantic"] is None
    assert out[1]["semantic"] == "tree"


def test_resolve_semantics_confirms_tree_by_color():
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = (40, 200, 40)                      # everything vegetation-green
    maybe = cl((5.0, 0.0, 0.2), (0.4, 0.4, 1.0))
    out = resolve_semantics([maybe], img, 60.0, 0.0, RADAR_HEIGHT)
    assert out[0]["semantic"] == "tree"
