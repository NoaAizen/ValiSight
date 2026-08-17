#!/usr/bin/env python3
"""Calibrate the PAG7936 / Lepton pair and emit the warp LUT that fusion.c consumes.

Approach: calibrate each camera's intrinsics and distortion, stereo-calibrate for
the rigid transform between them, then derive the plane-induced homography
analytically for whatever working distance we want:

    H = K_th @ (R + t @ n.T / Z) @ inv(K_rgb),     n = [0,0,1]

PLUS, not minus. The subtracting form appears in most references and assumes t
points the other way; with the sign flipped the error is 2|t|f/Z, which is 3.4
thermal pixels at 0.5 m -- more than the entire registration budget the 12 mm
baseline buys, and it reads as "the fusion is misaligned, maybe we need a range
sensor". plane_homography() below has always been right; this line was not.

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


# Preprocessing ladder, in the order that was measured to help on this rig.
# The foil target is the reason it exists: crinkled aluminium is a superb LWIR
# checker (emissivity ~0.05, so it mirrors the cold ceiling) and a poor visible
# one -- every crease is a specular highlight, so the "white" squares are not
# uniform quads and SB rejects them. A 3x3 blur before CLAHE kills the creases
# without moving the corners. Measured on captures/foil_preview2: the raw frame
# yields nothing at any pattern size, `blur3+clahe2` yields a full grid.
def _ladder(img):
    clahe = cv2.createCLAHE
    yield 'raw', img
    yield 'clahe2', clahe(2.0, (8, 8)).apply(img)
    yield 'blur3+clahe2', clahe(2.0, (8, 8)).apply(cv2.GaussianBlur(img, (3, 3), 0))
    yield 'blur5+clahe3', clahe(3.0, (8, 8)).apply(cv2.GaussianBlur(img, (5, 5), 0))
    yield 'bilat+clahe2', clahe(2.0, (8, 8)).apply(cv2.bilateralFilter(img, 9, 60, 60))


def find_pattern(img, pattern):
    """First hit from the preprocessing ladder. (corners, variant_name) or None."""
    sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE
    for name, im in _ladder(img):
        ok, corners = cv2.findChessboardCornersSB(im, pattern, sb_flags)
        if ok:
            return corners.reshape(-1, 2), name
    return None


def subgrid_risk(img, pattern, variant=None):
    """True if a LARGER grid than `pattern` is also findable in this image.

    THE CHECK THAT MATTERS, and the reason it is separate from detection.

    findChessboardCorners asks "is there a pattern-sized grid here", not "is
    this THE grid". On a board with more inner corners than the declared
    pattern it happily locks onto a sub-window, and it may lock onto a
    different sub-window in each camera -- offset by one square, say. That
    offset is a pure translation in the target plane, and a homography absorbs
    a planar translation EXACTLY. So the residuals stay sub-pixel, every
    quality metric passes, and the solved (R,t) is wrong by one square width.

    Inverted contrast makes it likelier rather than less: foil reads bright in
    the visible and dark in LWIR, so the two detectors do not even start from
    the same square.

    No amount of solver cleverness recovers from this; the fix is a board whose
    full inner grid IS the declared pattern. This function refuses the view
    instead of letting it through.

    `variant` restricts the search to the preprocessing that found the pattern
    in the first place. That is both faster and the sharper question: whether
    the SAME view of the image also contains a larger grid. Running the whole
    ladder here would flag boards that only reveal an extra row under contrast
    settings the real detection never used.
    """
    sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE
    if variant is not None:
        img = dict(_ladder(img))[variant]
    pw, ph = pattern
    for bigger in ((pw + 1, ph), (pw, ph + 1), (pw + 1, ph + 1)):
        if variant is not None:
            if cv2.findChessboardCornersSB(img, bigger, sb_flags)[0]:
                return True
        elif find_pattern(img, bigger) is not None:
            return True
    return False


def detect_corners(img, pattern):
    """Checkerboard corners, sub-pixel refined. Returns None if not found.

    SB detector first: it survives small squares, cluttered backgrounds and
    the blurry 4x thermal upscale far better than the classic detector, and
    its corners are already subpixel. Classic path kept as fallback.
    """
    sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE
    hit = find_pattern(img, pattern)
    if hit is not None:
        return hit[0]

    # Windowed fallback. Verified on this rig: a board that fails detection in
    # the full 640x400 frame (bright glass, people, hot doors dominate the
    # normalisation) is found cleanly when the search is restricted to a
    # window around it. Overlapping windows, coarse-to-fine, first hit wins.
    h, w = img.shape[:2]
    for fy, fx in ((0.75, 0.6), (0.6, 0.45)):
        wh, ww = int(h * fy), int(w * fx)
        for y0 in range(0, h - wh + 1, max(1, (h - wh) // 2 or 1)):
            for x0 in range(0, w - ww + 1, max(1, (w - ww) // 2 or 1)):
                win = np.ascontiguousarray(img[y0:y0 + wh, x0:x0 + ww])
                # The ladder here too, not just on the full frame. Measured on
                # the foil board: the full frame yields nothing at any variant
                # (the backlit windows dominate CLAHE's normalisation), while
                # a window around the board plus blur3+clahe2 yields the grid.
                # Restricting the search and fixing the local contrast are two
                # different repairs and this target needs both.
                hit = find_pattern(win, pattern)
                if hit is not None:
                    return hit[0] + np.float32([x0, y0])

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
        if c_rgb is None:
            print("  %-10s skip (rgb=%s thermal=%s)" %
                  (name, c_rgb is not None, c_th is not None), file=sys.stderr)
            continue

        # A stereo view is only usable if BOTH detections are the whole board.
        # Checked per modality because the board can be fully visible to one
        # camera and clipped for the other, and a clipped board is exactly the
        # sub-window case. Rejecting to rgb-only is safe: the view still
        # constrains the RGB intrinsics, it just cannot constrain (R,t).
        if c_th is not None:
            risky = [m for m, im in (("rgb", rgb), ("thermal", th_big))
                     if subgrid_risk(im, pattern)]
            if risky:
                print("  %-10s rgb-only  SUBGRID RISK in %s: a larger grid is "
                      "also findable, so the %s detection is a sub-window at an "
                      "unverifiable position" % (name, "+".join(risky), pattern),
                      file=sys.stderr)
                c_th = None

        # A view the thermal cannot see still constrains the RGB intrinsics.
        # (This board's varnish hides the squares in LWIR, so whole sessions
        # can be rgb-only; solve() calibrates stereo from the subset that has
        # both, when that subset is big enough.)
        out["views"].append({"name": name,
                             "rgb": c_rgb.tolist(),
                             "thermal": (c_th / 4.0).tolist() if c_th is not None else None})
        print("  %-10s %s" % (name, "ok" if c_th is not None else "rgb-only"),
              file=sys.stderr)

    return out


# ------------------------------------------------------------------ solve


def solve(corners, fix_principal=False, fix_f=None):
    pattern = tuple(corners["pattern"])
    square = corners["square_m"]
    views = corners["views"]
    stereo_views = [v for v in views if v.get("thermal") is not None]
    if len(views) < 6:
        raise SystemExit("need at least 6 usable views, have %d" % len(views))

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square

    flags = cv2.CALIB_FIX_PRINCIPAL_POINT if fix_principal else 0
    # k3 is not identifiable from a handful of views on a lens this short and
    # mostly just absorbs noise into the corners
    flags_th = flags | cv2.CALIB_FIX_K3

    obj = [objp] * len(views)
    p_rgb = [np.asarray(v["rgb"], np.float32).reshape(-1, 1, 2) for v in views]
    K0 = None
    if fix_f:
        # f from a physical measurement (board at tape-measured distances)
        # outranks the bundle: with mostly fronto-parallel views the bundle's
        # f is degenerate with tvec.z and converges wrong with excellent rms
        # (measured on this rig: bundle said 445 px, tape said 521-529 px).
        # Fix f, let the bundle solve only principal point and distortion.
        K0 = np.array([[fix_f, 0, RGB_W / 2.0], [0, fix_f, RGB_H / 2.0], [0, 0, 1.0]])
        # k3 with mostly-frontal views is pure overfit: freeing it moved rms
        # 0.396 -> 0.393 while swinging (k2, k3) from (-0.04, 0) to (0.52, -0.79).
        flags |= (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH |
                  cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3)
    e1, K_rgb, d_rgb, _, _ = cv2.calibrateCamera(obj, p_rgb, (RGB_W, RGB_H), K0, None,
                                                 flags=flags)

    out = {
        "K_rgb": K_rgb.tolist(), "dist_rgb": np.ravel(d_rgb).tolist(),
        "rms_rgb": float(e1), "views": len(views),
        "stereo_views": len(stereo_views),
    }
    if fix_f:
        out["f_fixed_px"] = float(fix_f)

    # Thermal intrinsics and the RGB->thermal extrinsic need views the thermal
    # actually saw. With fewer than 6 the stereo problem is under-constrained;
    # emit an rgb-only calibration rather than a garbage transform.
    if len(stereo_views) >= 6:
        obj_s = [objp] * len(stereo_views)
        ps_rgb = [np.asarray(v["rgb"], np.float32).reshape(-1, 1, 2) for v in stereo_views]
        ps_th = [np.asarray(v["thermal"], np.float32).reshape(-1, 1, 2) for v in stereo_views]

        e2, K_th, d_th, _, _ = cv2.calibrateCamera(obj_s, ps_th, (TH_W, TH_H), None, None,
                                                   flags=flags_th)
        e3, K_rgb, d_rgb, K_th, d_th, R, t, _, _ = cv2.stereoCalibrate(
            obj_s, ps_rgb, ps_th, K_rgb, d_rgb, K_th, d_th, (RGB_W, RGB_H),
            flags=cv2.CALIB_FIX_INTRINSIC)

        out.update({
            "K_th": K_th.tolist(), "dist_th": np.ravel(d_th).tolist(),
            "R": R.tolist(), "t": np.ravel(t).tolist(),
            "rms_th": float(e2), "rms_stereo": float(e3),
            "baseline_mm": float(np.linalg.norm(t) * 1000.0),
        })
    else:
        print("only %d stereo view(s): emitting RGB-only calibration "
              "(no thermal intrinsics / R,t / LUT input)" % len(stereo_views),
              file=sys.stderr)
    return out


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
    s.add_argument("--fix-f", type=float, default=None,
                   help="fix RGB focal length (px) from a physical measurement")

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
        res = solve(json.load(open(a.corners)), a.fix_principal, a.fix_f)
        json.dump(res, open(a.out, "w"), indent=2)
        if "rms_stereo" in res:
            print("views=%d  rms rgb=%.3f th=%.3f stereo=%.3f px\nbaseline=%.1f mm -> %s" % (
                res["views"], res["rms_rgb"], res["rms_th"], res["rms_stereo"],
                res["baseline_mm"], a.out))
            if not 5.0 < res["baseline_mm"] < 40.0:
                print("warning: recovered baseline %.1fmm is implausible for this rig; "
                      "check the square size and pattern" % res["baseline_mm"], file=sys.stderr)
        else:
            print("views=%d (rgb-only)  rms rgb=%.3f px -> %s" % (
                res["views"], res["rms_rgb"], a.out))

    elif a.cmd == "lut":
        lut = build_lut(json.load(open(a.calib)), a.distance)
        lut.tofile(a.out)
        cov = 100.0 * np.mean(lut[..., 0] != FUSION_INVALID)
        print("warp LUT %dx%d at %.2fm, %.1f%% thermal coverage -> %s (%d B)" % (
            LOW_W, LOW_H, a.distance, cov, a.out, lut.nbytes))

    return 0


if __name__ == "__main__":
    sys.exit(main())
