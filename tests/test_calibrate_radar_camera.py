"""calibrate_radar_camera: synthetic ground-truth recovery + persistence."""
import math

import numpy as np
import pytest

from calibrate_radar_camera import (calibrate_extrinsics,
                                    calibrate_intrinsics_from_points,
                                    project, save_calib, load_calib,
                                    R_ALIGNED)

K = np.array([[600.0, 0, 320.0],
              [0, 600.0, 240.0],
              [0, 0, 1.0]])


def true_pose():
    """Aligned axis permutation + a 3-deg pan + a small lever arm."""
    a = math.radians(3.0)
    pan = np.array([[math.cos(a), 0, math.sin(a)],
                    [0, 1, 0],
                    [-math.sin(a), 0, math.cos(a)]])
    R = pan @ R_ALIGNED
    t = np.array([[0.05], [-0.02], [0.01]])
    return R, t


def radar_targets(n=20):
    rng = np.random.RandomState(1)
    pts = np.column_stack([rng.uniform(1.0, 6.0, n),      # x forward
                           rng.uniform(-2.0, 2.0, n),     # y left
                           rng.uniform(-0.5, 1.0, n)])    # z up
    return pts


def test_extrinsics_recovered_from_correspondences():
    R, t = true_pose()
    pts = radar_targets()
    uv, z = project(pts, {"K": K, "dist": None, "R": R, "t": t})
    assert (z > 0).all()
    R_est, t_est, err = calibrate_extrinsics(pts, uv, K)
    assert err < 0.05                          # sub-pixel on clean data
    assert np.abs(R_est - R).max() < 1e-3
    assert np.abs(t_est - t).max() < 5e-3


def test_extrinsics_requires_enough_points():
    R, t = true_pose()
    pts = radar_targets(4)
    uv, _ = project(pts, {"K": K, "dist": None, "R": R, "t": t})
    with pytest.raises(ValueError):
        calibrate_extrinsics(pts, uv, K)


def test_project_flags_points_behind_camera():
    R, t = true_pose()
    uv, z = project([(-2.0, 0.0, 0.0, 1.5)],   # behind the radar; extra
                    {"K": K, "dist": None, "R": R, "t": t})   # doppler column
    assert z[0] < 0                            # caller must mask these


def test_save_load_roundtrip(tmp_path):
    R, t = true_pose()
    p = str(tmp_path / "calib.json")
    save_calib(p, K, np.zeros(5), R, t, reproj_err=0.7, meta={"cam": "rgb"})
    c = load_calib(p)
    assert np.allclose(c["K"], K) and np.allclose(c["R"], R)
    assert np.allclose(c["t"], t)
    assert c["reproj_err_px"] == 0.7 and c["meta"]["cam"] == "rgb"


def test_intrinsics_from_synthetic_views():
    board = (9, 6)
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * 0.025
    import cv2
    obj_pts, img_pts = [], []
    rng = np.random.RandomState(2)
    for i in range(8):
        rvec = rng.uniform(-0.3, 0.3, 3)
        tvec = np.array([rng.uniform(-0.05, 0.05),
                         rng.uniform(-0.05, 0.05),
                         rng.uniform(0.4, 0.8)])
        uv, _ = cv2.projectPoints(objp, rvec, tvec, K, np.zeros(5))
        obj_pts.append(objp)
        img_pts.append(uv.astype(np.float32))
    K_est, dist, rms, n = calibrate_intrinsics_from_points(
        obj_pts, img_pts, (640, 480))
    assert n == 8 and rms < 0.5
    assert abs(K_est[0, 0] - 600.0) / 600.0 < 0.02      # fx within 2%
