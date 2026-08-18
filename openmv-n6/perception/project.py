"""Radar points -> RGB pixels, via the frozen calibration artifacts.

p_C = R @ p_R + t  (T_camera_radar.json, reduced yaw+tx fit), then the Brown
model from calib.json (f tape-fixed at 525 - never a bundle-solved f here).

What the numbers behind this are good for, verbatim from the artifact:
  - valid range 1.5-3.5 m; horizontal error ~1 px median, 35 px worst
    (left side, never in the fit set); linearity beyond 3.5 m is UNVERIFIED.
  - vertical is mechanical geometry only. Radar elevation is unusable
    (sigma ~12 deg); never gate or match on v, only on u and range.
"""
import json
import os

import cv2
import numpy as np

ART = os.path.join(os.path.dirname(__file__), '..', 'calib-artifacts')

VALID_RANGE_M = (2.0, 8.6)   # 2026-08-18 calibration (was 1.5-3.5)


class RadarProjector:
    def __init__(self, calib_json=None, extrinsics_json=None):
        with open(calib_json or os.path.join(ART, 'calib.json')) as f:
            c = json.load(f)
        with open(extrinsics_json or os.path.join(ART, 'T_camera_radar.json')) as f:
            e = json.load(f)
        self.K = np.asarray(c['K_rgb'], dtype=np.float64)
        self.dist = np.asarray(c['dist_rgb'], dtype=np.float64)
        self.R = np.asarray(e['R'], dtype=np.float64)
        self.t = np.asarray(e['t_m'], dtype=np.float64)
        self.valid_range_m = tuple(e.get('valid_range_m', VALID_RANGE_M))

    def to_camera(self, pts_R: np.ndarray) -> np.ndarray:
        """(N,3) project-frame points -> (N,3) camera-frame points."""
        return pts_R @ self.R.T + self.t

    def project(self, pts_R: np.ndarray):
        """(N,3) radar points -> (uv (N,2) float64, in_front (N,) bool).

        uv rows with in_front False are meaningless (behind the camera);
        they are left in place so indices line up with the input.
        """
        pts_R = np.asarray(pts_R, dtype=np.float64).reshape(-1, 3)
        p_c = self.to_camera(pts_R)
        in_front = p_c[:, 2] > 0.05
        uv = np.full((len(p_c), 2), np.nan)
        if in_front.any():
            proj, _ = cv2.projectPoints(
                p_c[in_front].reshape(-1, 1, 3),
                np.zeros(3), np.zeros(3), self.K, self.dist)
            uv[in_front] = proj.reshape(-1, 2)
        return uv, in_front

    def in_calibrated_band(self, pts_R: np.ndarray) -> np.ndarray:
        """Boolean mask: point range is inside the validated 1.5-3.5 m band."""
        r = np.linalg.norm(np.asarray(pts_R, dtype=np.float64), axis=-1)
        lo, hi = self.valid_range_m
        return (r >= lo) & (r <= hi)
