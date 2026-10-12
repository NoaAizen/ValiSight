#!/usr/bin/env python3
"""Compose T_thermal_radar from the two calibrations we already own.

    radar -> RGB      : calib-artifacts/T_camera_radar.json   p_C = R_cr @ p_R + t_cr
    RGB   -> thermal  : calib-artifacts/calib.json (stereo)   X_th = R_s @ X_rgb + t_s
    =>  radar -> thermal :  p_T = R_s @ (R_cr @ p_R + t_cr) + t_s
                          R_tr = R_s @ R_cr ,   t_tr = R_s @ t_cr + t_s

Per frame_conventions.txt the thermal frame is reached ONLY by this
composition, never solved independently.  No hardware needed; the output
is only as good as its two inputs (radar->RGB holdout ~1.5-1.7 deg,
stereo rms 1.1 px at 160x120 ~= 0.4 deg), so expect ~2 deg scatter.

Output: calib-artifacts/T_thermal_radar.json, same keys as
T_camera_radar.json so perception.project.RadarProjector(camera='thermal')
can load it unchanged.

Self-check: a radar point at 3 m boresight is projected into RGB (via
T_camera_radar + K_rgb) and into thermal (via the composed T + K_th); the
RGB pixel is also pushed through calib.plane_homography at 3 m - the two
thermal pixels must agree to well under a pixel (pure algebra), proving the
chain direction is right.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ART = os.path.join(ROOT, "calib-artifacts")
sys.path.insert(0, HERE)
from calib import plane_homography  # noqa: E402


def rot_to_yaw_pitch_deg(R_cam_from_radar):
    # radar x-forward maps to camera column 0; yaw = angle of that axis in cam x/z,
    # pitch = its elevation in cam y (y down -> negate).
    fwd = R_cam_from_radar[:, 0]
    yaw = np.degrees(np.arctan2(fwd[0], fwd[2]))
    pitch = np.degrees(np.arctan2(-fwd[1], np.hypot(fwd[0], fwd[2])))
    return float(yaw), float(pitch)


def main(out_path=None):
    with open(os.path.join(ART, "calib.json")) as f:
        c = json.load(f)
    with open(os.path.join(ART, "T_camera_radar.json")) as f:
        e = json.load(f)

    R_cr = np.asarray(e["R"], np.float64)
    t_cr = np.asarray(e["t_m"], np.float64)
    R_s = np.asarray(c["R"], np.float64)
    t_s = np.asarray(c["t"], np.float64)
    K_rgb = np.asarray(c["K_rgb"], np.float64)
    d_rgb = np.asarray(c["dist_rgb"], np.float64)
    K_th = np.asarray(c["K_th"], np.float64)
    d_th = np.asarray(c["dist_th"], np.float64)

    R_tr = R_s @ R_cr
    t_tr = R_s @ t_cr + t_s

    # ---- self-check: the two routes to a thermal pixel must agree ----
    Z = 3.0
    p_R = np.array([[Z, 0.0, 0.0]])                      # 3 m boresight, radar frame
    p_C = p_R @ R_cr.T + t_cr
    uv_rgb, _ = cv2.projectPoints(p_C.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_rgb, d_rgb)
    p_T = p_R @ R_tr.T + t_tr
    uv_th, _ = cv2.projectPoints(p_T.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_th, d_th)
    # route 2: ideal RGB pixel -> homography at the point's RGB depth -> ideal thermal pixel
    ideal_rgb = cv2.undistortPoints(uv_rgb, K_rgb, d_rgb, P=K_rgb).reshape(-1, 2)
    H = plane_homography(K_rgb, K_th, R_s, t_s.reshape(3, 1), float(p_C[0, 2]))
    h = H @ np.array([ideal_rgb[0, 0], ideal_rgb[0, 1], 1.0])
    ideal_th_H = h[:2] / h[2]
    ideal_th = cv2.undistortPoints(uv_th, K_th, d_th, P=K_th).reshape(-1, 2)[0]
    chain_err_px = float(np.linalg.norm(ideal_th - ideal_th_H))
    assert chain_err_px < 0.05, "chain self-check failed: %.3f px" % chain_err_px

    yaw, pitch = rot_to_yaw_pitch_deg(R_tr)
    out = {
        "R": R_tr.tolist(),
        "t_m": t_tr.tolist(),
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "camera": "thermal",
        "image_size": [160, 120],
        "convention": "p_T = R @ p_R + t; radar x fwd y left z up; thermal x right y down z fwd (OpenCV)",
        "formulation": "COMPOSED, not solved: T_thermal_radar = T_thermal_rgb (calib.json stereo R,t) o T_rgb_radar (T_camera_radar.json). "
                       "Inherits both error budgets; radar z/elevation remains unusable exactly as in T_camera_radar.json.",
        "valid_range_m": e.get("valid_range_m", [2.0, 8.6]),
        "sources": {
            "T_camera_radar": {"source": e.get("source"), "yaw_deg": e.get("yaw_deg"),
                               "holdout_du_median_px": e.get("holdout_du_median_px"),
                               "radar_cfg_sha256": e.get("radar_cfg_sha256")},
            "stereo_rgb_thermal": {"file": "calib.json", "rms_stereo_px": c.get("rms_stereo"),
                                   "rms_th_px": c.get("rms_th"), "baseline_mm": c.get("baseline_mm"),
                                   "t_m": t_s.tolist()},
        },
        "self_check_chain_err_px": chain_err_px,
        "example_3m_boresight": {"rgb_uv": uv_rgb.reshape(2).tolist(), "thermal_uv": uv_th.reshape(2).tolist()},
        "expected_scatter_deg_note": "radar->RGB holdout ~1.5-1.7 deg (az quantisation) + stereo ~0.4 deg at 160x120 -> ~2 deg; "
                                     "1 thermal px = 0.36 deg (f_th~159 px).",
        "K_source": "calib-artifacts/calib.json K_th/dist_th (Hagai, 2026-08-18)",
        "dv_note": e.get("dv_note"),
    }
    out_path = out_path or os.path.join(ART, "T_thermal_radar.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)

    print("T_thermal_radar written ->", out_path)
    print("  yaw %.2f deg  pitch %.2f deg   t = [%.1f, %.1f, %.1f] mm (thermal frame: right, down, fwd)"
          % (yaw, pitch, *(t_tr * 1000)))
    print("  3 m boresight radar point -> RGB px (%.1f, %.1f)  thermal px (%.1f, %.1f)   chain check %.4f px"
          % (*uv_rgb.reshape(2), *uv_th.reshape(2), chain_err_px))
    # where does a 3 m target land across the thermal frame, every 10 deg of radar az
    print("  radar az (deg) -> thermal u at 3 m: " + "  ".join(
        "%+d:%.0f" % (a, cv2.projectPoints(
            (np.array([[Z * np.cos(np.radians(a)), -Z * np.sin(np.radians(a)), 0.0]]) @ R_tr.T + t_tr).reshape(-1, 1, 3),
            np.zeros(3), np.zeros(3), K_th, d_th)[0].reshape(2)[0]) for a in (-20, -10, 0, 10, 20)))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
