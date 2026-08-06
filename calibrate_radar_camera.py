"""
Radar <-> camera geometric calibration (OpenCV).

Two independent steps, both saved into one calib.json per camera:

  1. Intrinsics  — chessboard photos -> camera matrix K + distortion.
  2. Extrinsics  — N >= 6 (recommended 15+) correspondences between a radar
     point (x,y,z in the RADAR frame, metres) and the pixel (u,v) where that
     same target appears -> solvePnP -> R, t with  P_cam = R @ P_radar + t.

Coordinate frames:
  radar:          x = forward, y = left,  z = up          (project convention)
  camera optical: x = right,   y = down,  z = forward     (OpenCV convention)
R absorbs the axis permutation AND the mounting misalignment; t is the lever
arm between the sensors (metres). Collect correspondences across the FOV and
at 2-3 depths; accept if mean reprojection error < ~2 px, then validate at a
NEW depth that was not used for calibration.

Usage from code (the fusion package calls these):
    K, dist, rms = calibrate_intrinsics(images)
    R, t, err = calibrate_extrinsics(radar_pts, pixel_pts, K, dist)
    save_calib("calib_rgb.json", K, dist, R, t, reproj_err=err)
    calib = load_calib("calib_rgb.json")
    uv, z_cam = project(points_radar, calib)
"""
import json

import cv2
import numpy as np

DEFAULT_BOARD = (9, 6)          # inner corners of the chessboard
DEFAULT_SQUARE_M = 0.025        # square edge length (metres)


# --- intrinsics -----------------------------------------------------------------

def calibrate_intrinsics(images, board_size=DEFAULT_BOARD,
                         square_size_m=DEFAULT_SQUARE_M):
    """Chessboard images (BGR or gray) -> (K, dist, rms_px, n_used).

    Needs the full board visible in >= 5 images from varied angles/positions.
    Images where the board is not found are skipped.
    """
    obj_pts, img_pts, size = [], [], None
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0],
                           0:board_size[1]].T.reshape(-1, 2) * square_size_m
    for img in images:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, board_size)
        if not ok:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        obj_pts.append(objp)
        img_pts.append(corners)
    if len(obj_pts) < 5:
        raise ValueError("chessboard found in only %d image(s); need >= 5"
                         % len(obj_pts))
    return calibrate_intrinsics_from_points(obj_pts, img_pts, size)


def calibrate_intrinsics_from_points(obj_pts, img_pts, image_size):
    """Core of calibrate_intrinsics, exposed for synthetic tests."""
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, image_size,
                                             None, None)
    return K, dist, rms, len(obj_pts)


# --- extrinsics -----------------------------------------------------------------

def calibrate_extrinsics(radar_pts, pixel_pts, K, dist=None):
    """Radar->camera pose from point correspondences.

    radar_pts: Nx3 (x,y,z) in the RADAR frame (metres)
    pixel_pts: Nx2 (u,v) where each target appeared in the image
    Returns (R 3x3, t 3x1, mean_reproj_err_px).
    """
    radar_pts = np.asarray(radar_pts, np.float64).reshape(-1, 3)
    pixel_pts = np.asarray(pixel_pts, np.float64).reshape(-1, 2)
    if len(radar_pts) < 6:
        raise ValueError("need >= 6 correspondences (15+ recommended), got %d"
                         % len(radar_pts))
    dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64)
    ok, rvec, tvec = cv2.solvePnP(radar_pts, pixel_pts,
                                  np.asarray(K, np.float64), dist,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError("solvePnP failed on the given correspondences")
    # polish with the iterative minimizer
    rvec, tvec = cv2.solvePnPRefineLM(radar_pts, pixel_pts,
                                      np.asarray(K, np.float64), dist,
                                      rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    uv, _ = _project_arrays(radar_pts, K, dist, R, tvec)
    err = float(np.linalg.norm(uv - pixel_pts, axis=1).mean())
    return R, tvec.reshape(3, 1), err


# --- projection -----------------------------------------------------------------

def _project_arrays(points, K, dist, R, t):
    points = np.asarray(points, np.float64).reshape(-1, 3)
    rvec = cv2.Rodrigues(np.asarray(R, np.float64))[0]
    tvec = np.asarray(t, np.float64).reshape(3, 1)
    dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64)
    uv, _ = cv2.projectPoints(points, rvec, tvec,
                              np.asarray(K, np.float64), dist)
    z_cam = (np.asarray(R) @ points.T + tvec)[2]
    return uv.reshape(-1, 2), z_cam


def project(points_radar, calib):
    """Radar points -> pixel coordinates.

    points_radar: Nx(3+) — extra columns (doppler, snr...) are ignored.
    calib: dict from load_calib (or with 'K','dist','R','t' arrays).
    Returns (uv Nx2 float, z_cam N) — a point is in front of the camera and
    usable only where z_cam > 0.
    """
    pts = np.asarray([p[:3] for p in points_radar], np.float64)
    if pts.size == 0:
        return np.empty((0, 2)), np.empty(0)
    return _project_arrays(pts, calib["K"], calib.get("dist"),
                           calib["R"], calib["t"])


# --- persistence ----------------------------------------------------------------

def save_calib(path, K, dist, R, t, reproj_err=None, meta=None):
    out = {"K": np.asarray(K).tolist(),
           "dist": np.asarray(dist if dist is not None else np.zeros(5)
                              ).ravel().tolist(),
           "R": np.asarray(R).tolist(),
           "t": np.asarray(t).ravel().tolist(),
           "reproj_err_px": reproj_err,
           "meta": meta or {}}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def load_calib(path):
    """Returns {'K','dist','R','t', 'reproj_err_px', 'meta'} as numpy arrays."""
    with open(path, "r") as f:
        d = json.load(f)
    return {"K": np.asarray(d["K"], np.float64),
            "dist": np.asarray(d["dist"], np.float64),
            "R": np.asarray(d["R"], np.float64),
            "t": np.asarray(d["t"], np.float64).reshape(3, 1),
            "reproj_err_px": d.get("reproj_err_px"),
            "meta": d.get("meta", {})}


# canonical axis permutation radar->camera for a perfectly aligned mount:
# radar (fwd, left, up) -> camera (right, down, fwd)
R_ALIGNED = np.array([[0.0, -1.0, 0.0],      # cam x (right)  = -radar y (left)
                      [0.0, 0.0, -1.0],      # cam y (down)   = -radar z (up)
                      [1.0, 0.0, 0.0]])      # cam z (fwd)    =  radar x (fwd)
