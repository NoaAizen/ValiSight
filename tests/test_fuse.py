"""fusion.fuse: synthetic detection box + synthetic radar points -> fused
object with correct range/Doppler (acceptance test from the task brief)."""
import numpy as np

from calibrate_radar_camera import R_ALIGNED
from fusion.detector import Box
from fusion.fuse import fuse

K = np.array([[600.0, 0, 320.0],
              [0, 600.0, 240.0],
              [0, 0, 1.0]])
CALIB = {"K": K, "dist": None, "R": R_ALIGNED, "t": np.zeros((3, 1))}

# with the aligned pose, a radar point (x fwd, 0, 0) projects to the image
# centre (320, 240); y (left) moves it left in the image
CENTER_BOX = Box(280, 200, 80, 80, "person", 0.9)


def pt(x, y=0.0, z=0.0, v=0.0):
    return (x, y, z, v)


def test_point_in_box_gives_range_and_doppler():
    pts = [pt(3.0, 0.0, 0.0, -1.2), pt(3.1, 0.05, 0.0, -1.2)]
    objs = fuse([CENTER_BOX], pts, CALIB)
    assert len(objs) == 1
    o = objs[0]
    assert o["label"] == "person"
    assert abs(o["range_m"] - 3.0) < 0.05
    assert abs(o["doppler_mps"] - (-1.2)) < 0.01
    assert o["radar_class"] == "pedestrian"       # moving, small cluster
    assert o["n_radar_pts"] == 2


def test_point_outside_box_stays_camera_only():
    # ~0.8 m left at 3 m -> projects far from the centred box
    objs = fuse([CENTER_BOX], [pt(3.0, 0.8, 0.0, 1.0),
                               pt(3.0, 0.85, 0.0, 1.0)], CALIB)
    assert objs[0]["range_m"] is None
    assert objs[0]["radar_class"] is None


def test_nearest_in_range_wins():
    near = [pt(2.0, 0.0, 0.0, -0.5), pt(2.05, 0.02, 0.0, -0.5)]
    far_wall = [pt(6.0, 0.0, 0.1, 0.0), pt(6.05, 0.02, 0.1, 0.0)]
    o = fuse([CENTER_BOX], near + far_wall, CALIB)[0]
    assert abs(o["range_m"] - 2.0) < 0.05
    # the wall behind the person is in the box too, but must not pollute
    # the Doppler of the near target
    assert abs(o["doppler_mps"] - (-0.5)) < 0.01
    assert o["n_radar_pts"] == 4


def test_point_behind_camera_ignored():
    o = fuse([CENTER_BOX], [pt(-2.0), pt(-2.1)], CALIB)[0]
    assert o["range_m"] is None


def test_no_calib_degrades_to_camera_only():
    objs = fuse([CENTER_BOX], [pt(3.0), pt(3.1)], None)
    assert objs[0]["range_m"] is None and objs[0]["label"] == "person"


def test_no_boxes_no_output():
    assert fuse([], [pt(3.0)], CALIB) == []
