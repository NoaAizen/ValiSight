#!/usr/bin/env python3
"""Calibrate the PAG7936 / Lepton pair and emit the warp LUT that fusion.c consumes.

Approach: calibrate each camera's intrinsics and distortion, stereo-calibrate for
the rigid transform between them, then derive the plane-induced homography
analytically for whatever working distance we want:

    H = K_th @ (R - t @ n.T / Z) @ inv(K_rgb),     n = [0,0,1]

This matters. Fitting a homography from one shot pins you to the distance you
happened to shoot at; deriving it from (R,t) lets 0.8m - the optimal calibration
distance for a 12mm baseline over a 0.5-2m working range - be chosen after the
fact, and makes a distance-parameterised LUT free if it is ever wanted.

The LUT folds in both lens distortions, which a homography alone cannot express:

    low-res grid pixel
      -> full-res RGB pixel (accounting for the 4x decimation's half-pixel offset)
      -> undistort with (K_rgb, d_rgb)          [ideal RGB]
      -> H                                       [ideal thermal]
      -> re-apply (K_th, d_th) distortion        [raw thermal pixel, what we sample]

Usage:
    ./calib.py detect  <pairs_dir> -o corners.json
    ./calib.py solve   corners.json -o calib.json
    ./calib.py lut     calib.json  -o warp.lut [--distance 0.8]
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

# Board geometry, matching the verified hardware.
RGB_W, RGB_H = 640, 400
TH_W, TH_H = 160, 120
LOW_W, LOW_H = 160, 100
DECIMATION = RGB_W // LOW_W

FUSION_INVALID = 0xFFFF
DEFAULT_DISTANCE_M = 0.8


# ------------------------------------------------------------------ geometry


def distort_points(norm, K, dist):
    """Ideal normalised coords -> distorted pixel coords.

    cv2.undistortPoints inverts this; OpenCV has no public forward equivalent for
    loose points, so it is written out. Plain Brown-Conrady: 3 radial + 2
    tangential, which is all a lens this short needs.
    """
    norm = np.asarray(norm, np.float64).reshape(-1, 2)
    d = np.zeros(5)
    d[:len(np.ravel(dist))] = np.ravel(dist)[:5]
    k1, k2, p1, p2, k3 = d

    x, y = norm[:, 0], norm[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2

    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    return np.stack([u, v], -1)


def plane_homography(K_rgb, K_th, R, t, distance_m):
    """Homography induced by a fronto-parallel plane at `distance_m`, ideal->ideal.

    stereoCalibrate gives X_th = R @ X_rgb + t. On a plane with unit normal n at
    distance d, n'X_rgb / d == 1, so t can be written as t (n'X_rgb)/d and

        X_th = (R + t n'/d) X_rgb

    The sign is *plus*. The minus form that appears in most references assumes t
    points the other way (camera 2 -> camera 1); using it here puts the thermal
    layer off by 2|t|f/Z - 3.4 thermal px at 0.5m, which is larger than the entire
    registration budget the 12mm baseline buys.
    """
    n = np.array([[0.0], [0.0], [1.0]])
    t = np.asarray(t, np.float64).reshape(3, 1)
    return K_th @ (R + (t @ n.T) / float(distance_m)) @ np.linalg.inv(K_rgb)


def grid_to_rgb_pixels():
    """Low-res grid coords -> the full-res RGB pixel each one is the average of.

    Grid cell k covers columns [k*D, k*D+D-1], so its centre is k*D + (D-1)/2.
    fusion.c's grid_pos() assumes exactly this; the two must agree or the whole
    thermal layer shifts by half a grid cell.
    """
    gx, gy = np.meshgrid(np.arange(LOW_W), np.arange(LOW_H))
    off = (DECIMATION - 1) / 2.0
    return np.stack([gx * DECIMATION + off, gy * DECIMATION + off], -1).reshape(-1, 2)


def build_lut(calib, distance_m=DEFAULT_DISTANCE_M):
    """Produce the uint16 Q8 warp table fusion.c loads with --warp."""
    K_rgb = np.array(calib["K_rgb"])
    d_rgb = np.array(calib["dist_rgb"])
    K_th = np.array(calib["K_th"])
    d_th = np.array(calib["dist_th"])
    R = np.array(calib["R"])
    t = np.array(calib["t"])

    H = plane_homography(K_rgb, K_th, R, t, distance_m)

    pts = grid_to_rgb_pixels().astype(np.float64)
    ideal = cv2.undistortPoints(pts.reshape(-1, 1, 2), K_rgb, d_rgb, P=K_rgb).reshape(-1, 2)

    hom = np.concatenate([ideal, np.ones((len(ideal), 1))], 1) @ H.T
    w = hom[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    ideal_th = hom[:, :2] / w

    norm_th = np.stack([(ideal_th[:, 0] - K_th[0, 2]) / K_th[0, 0],
                        (ideal_th[:, 1] - K_th[1, 2]) / K_th[1, 1]], -1)
    raw = distort_points(norm_th, K_th, d_th)

    return pack_lut(raw)


def pack_lut(raw_th_pixels):
    """Q8 pack + out-of-frame masking, shared by the real and the reference path."""
    lut = np.full((LOW_H * LOW_W, 2), FUSION_INVALID, np.uint16)
    u, v = raw_th_pixels[:, 0], raw_th_pixels[:, 1]

    ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u <= TH_W - 1) & (v <= TH_H - 1)
    lut[ok, 0] = np.round(u[ok] * 256).astype(np.uint16)
    lut[ok, 1] = np.round(v[ok] * 256).astype(np.uint16)
    return lut.reshape(LOW_H, LOW_W, 2)


# ------------------------------------------------------------------ detection


def detect_corners(img, pattern):
    """Checkerboard corners, sub-pixel refined. Returns None if not found."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(img, pattern, flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    return cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), crit).reshape(-1, 2)


def detect_dir(pairs_dir, pattern, square_m):
    """Scan a capture directory for pairs where the board is visible in BOTH frames."""
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m

    out = {"pattern": list(pattern), "square_m": square_m, "views": []}
    for jp in sorted(glob.glob(os.path.join(pairs_dir, "*.json"))):
        stem = jp[:-5]
        rgb_p, th_p = stem + "_rgb0.raw", stem + "_thermal.raw"
        if not os.path.exists(rgb_p):
            rgb_p = stem + "_rgb.raw"
        if not (os.path.exists(rgb_p) and os.path.exists(th_p)):
            continue

        rgb = np.fromfile(rgb_p, np.uint8).reshape(RGB_H, RGB_W)
        th = np.fromfile(th_p, np.uint8).reshape(TH_H, TH_W)
        # the thermal frame is small and low-contrast; upscaling before detection
        # measurably improves the hit rate, and cornerSubPix is scale-aware
        th_big = cv2.resize(th, (TH_W * 4, TH_H * 4), interpolation=cv2.INTER_CUBIC)

        c_rgb = detect_corners(rgb, pattern)
        c_th = detect_corners(th_big, pattern)
        name = os.path.basename(stem)
        if c_rgb is None or c_th is None:
            print("  %-10s skip (rgb=%s thermal=%s)" %
                  (name, c_rgb is not None, c_th is not None), file=sys.stderr)
            continue

        out["views"].append({"name": name,
                             "rgb": c_rgb.tolist(),
                             "thermal": (c_th / 4.0).tolist()})
        print("  %-10s ok" % name, file=sys.stderr)

    return out


# ------------------------------------------------------------------ solve


def solve(corners, fix_principal=False):
    pattern = tuple(corners["pattern"])
    square = corners["square_m"]
    views = corners["views"]
    if len(views) < 6:
        raise SystemExit("need at least 6 usable views, have %d" % len(views))

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square

    obj = [objp] * len(views)
    p_rgb = [np.asarray(v["rgb"], np.float32).reshape(-1, 1, 2) for v in views]
    p_th = [np.asarray(v["thermal"], np.float32).reshape(-1, 1, 2) for v in views]

    flags = cv2.CALIB_FIX_PRINCIPAL_POINT if fix_principal else 0
    # k3 is not identifiable from a handful of views on a lens this short and
    # mostly just absorbs noise into the corners
    flags_th = flags | cv2.CALIB_FIX_K3

    e1, K_rgb, d_rgb, _, _ = cv2.calibrateCamera(obj, p_rgb, (RGB_W, RGB_H), None, None,
                                                 flags=flags)
    e2, K_th, d_th, _, _ = cv2.calibrateCamera(obj, p_th, (TH_W, TH_H), None, None,
                                               flags=flags_th)

    e3, K_rgb, d_rgb, K_th, d_th, R, t, _, _ = cv2.stereoCalibrate(
        obj, p_rgb, p_th, K_rgb, d_rgb, K_th, d_th, (RGB_W, RGB_H),
        flags=cv2.CALIB_FIX_INTRINSIC)

    baseline_mm = float(np.linalg.norm(t) * 1000.0)
    return {
        "K_rgb": K_rgb.tolist(), "dist_rgb": np.ravel(d_rgb).tolist(),
        "K_th": K_th.tolist(), "dist_th": np.ravel(d_th).tolist(),
        "R": R.tolist(), "t": np.ravel(t).tolist(),
        "rms_rgb": float(e1), "rms_th": float(e2), "rms_stereo": float(e3),
        "baseline_mm": baseline_mm, "views": len(views),
    }


# ------------------------------------------------------------------ cli


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect")
    d.add_argument("pairs_dir")
    d.add_argument("-o", "--out", default="corners.json")
    d.add_argument("--pattern", default="7x5", help="inner corners, e.g. 7x5")
    d.add_argument("--square", type=float, default=0.030, help="square size in metres")

    s = sub.add_parser("solve")
    s.add_argument("corners", default="corners.json")
    s.add_argument("-o", "--out", default="calib.json")
    s.add_argument("--fix-principal", action="store_true")

    l = sub.add_parser("lut")
    l.add_argument("calib", default="calib.json")
    l.add_argument("-o", "--out", default="warp.lut")
    l.add_argument("--distance", type=float, default=DEFAULT_DISTANCE_M)

    a = ap.parse_args()

    if a.cmd == "detect":
        pat = tuple(int(v) for v in a.pattern.split("x"))
        res = detect_dir(a.pairs_dir, pat, a.square)
        json.dump(res, open(a.out, "w"), indent=1)
        print("%d usable view(s) -> %s" % (len(res["views"]), a.out))

    elif a.cmd == "solve":
        res = solve(json.load(open(a.corners)), a.fix_principal)
        json.dump(res, open(a.out, "w"), indent=2)
        print("views=%d  rms rgb=%.3f th=%.3f stereo=%.3f px\nbaseline=%.1f mm -> %s" % (
            res["views"], res["rms_rgb"], res["rms_th"], res["rms_stereo"],
            res["baseline_mm"], a.out))
        if not 5.0 < res["baseline_mm"] < 40.0:
            print("warning: recovered baseline %.1fmm is implausible for this rig; "
                  "check the square size and pattern" % res["baseline_mm"], file=sys.stderr)

    elif a.cmd == "lut":
        lut = build_lut(json.load(open(a.calib)), a.distance)
        lut.tofile(a.out)
        cov = 100.0 * np.mean(lut[..., 0] != FUSION_INVALID)
        print("warp LUT %dx%d at %.2fm, %.1f%% thermal coverage -> %s (%d B)" % (
            LOW_W, LOW_H, a.distance, cov, a.out, lut.nbytes))

    return 0


if __name__ == "__main__":
    sys.exit(main())
