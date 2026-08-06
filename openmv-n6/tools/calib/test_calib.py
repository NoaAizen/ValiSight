#!/usr/bin/env python3
"""Validate the calibration chain against known ground truth.

Two independent things are checked, because agreeing with yourself proves
nothing:

  A. build_lut() goes pixel -> undistort -> homography -> redistort. The
     reference here instead ray-traces: pixel -> ray -> intersect the plane ->
     transform into the thermal camera -> project. Different algebra, same
     answer required. This is what would catch a wrong sign in
     H = K(R - t n'/Z)K', or a transposed rotation.

  B. solve() is fed corners projected from known parameters, and the LUT built
     from what it recovers is compared against the LUT from the truth. This
     exercises calibrateCamera -> stereoCalibrate -> homography -> LUT end to
     end, with sub-pixel noise on the corners so it is not a free pass.
"""
import sys

import cv2
import numpy as np

import calib

FAILS = []

# Ground truth. RGB is ~70 deg horizontal; the thermal matches a Lepton 3.5's
# 57 deg (fx = 80/tan(28.5) = 147.3). Baseline is the measured 12mm, mostly in x.
K_RGB = np.array([[460.0, 0, 320.0], [0, 460.0, 200.0], [0, 0, 1.0]])
D_RGB = np.array([-0.280, 0.090, 0.0005, -0.0003, 0.0])
K_TH = np.array([[147.3, 0, 80.0], [0, 147.3, 60.0], [0, 0, 1.0]])
D_TH = np.array([-0.120, 0.050, 0.0, 0.0, 0.0])

_rvec = np.array([0.006, -0.009, 0.002])          # slight mechanical misalignment
R_TRUE = cv2.Rodrigues(_rvec)[0]
T_TRUE = np.array([-0.0120, 0.0008, 0.0004])      # metres, thermal w.r.t. rgb

TRUTH = {"K_rgb": K_RGB.tolist(), "dist_rgb": D_RGB.tolist(),
         "K_th": K_TH.tolist(), "dist_th": D_TH.tolist(),
         "R": R_TRUE.tolist(), "t": T_TRUE.tolist()}

PATTERN = (7, 5)
SQUARE = 0.040


def check(name, ok, detail=""):
    print("  %-56s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def raytrace_lut(distance_m):
    """Reference LUT by ray-plane intersection - no homography anywhere."""
    pts = calib.grid_to_rgb_pixels().astype(np.float64)

    # distorted pixel -> ideal normalised ray
    norm = cv2.undistortPoints(pts.reshape(-1, 1, 2), K_RGB, D_RGB).reshape(-1, 2)

    # the plane z = distance in the RGB camera frame
    X = np.concatenate([norm * distance_m, np.full((len(norm), 1), distance_m)], 1)

    # into the thermal camera: X_th = R X_rgb + t
    X_th = X @ R_TRUE.T + T_TRUE.reshape(1, 3)

    z = X_th[:, 2:3].copy()
    z[np.abs(z) < 1e-12] = 1e-12
    norm_th = X_th[:, :2] / z

    return calib.pack_lut(calib.distort_points(norm_th, K_TH, D_TH))


def lut_delta(a, b):
    """Median / max disagreement in thermal pixels over commonly-valid entries."""
    va = a[..., 0] != calib.FUSION_INVALID
    vb = b[..., 0] != calib.FUSION_INVALID
    both = va & vb
    if both.sum() == 0:
        return None, None, 0.0, 0.0
    da = a[..., :2][both].astype(np.float64) / 256.0
    db = b[..., :2][both].astype(np.float64) / 256.0
    err = np.linalg.norm(da - db, axis=-1)
    agree = 100.0 * both.sum() / max(va.sum(), vb.sum())
    return float(np.median(err)), float(err.max()), agree, both.sum()


def make_views(n=16, seed=3):
    """Board poses spread over the working range, projected into both cameras."""
    rng = np.random.default_rng(seed)
    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float64)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE

    views = []
    for i in range(n):
        z = 0.35 + 0.35 * (i / max(1, n - 1))
        rvec = rng.uniform(-0.35, 0.35, 3)
        tvec = np.array([-SQUARE * PATTERN[0] / 2 + rng.uniform(-0.03, 0.03),
                         -SQUARE * PATTERN[1] / 2 + rng.uniform(-0.03, 0.03), z])

        p_rgb, _ = cv2.projectPoints(objp, rvec, tvec, K_RGB, D_RGB)

        R1 = cv2.Rodrigues(rvec)[0]
        R2 = R_TRUE @ R1
        t2 = R_TRUE @ tvec.reshape(3, 1) + T_TRUE.reshape(3, 1)
        p_th, _ = cv2.projectPoints(objp, cv2.Rodrigues(R2)[0], t2, K_TH, D_TH)

        p_rgb = p_rgb.reshape(-1, 2) + rng.normal(0, 0.10, (len(objp), 2))
        p_th = p_th.reshape(-1, 2) + rng.normal(0, 0.10, (len(objp), 2))

        if (p_rgb[:, 0].min() < 4 or p_rgb[:, 0].max() > calib.RGB_W - 5 or
                p_rgb[:, 1].min() < 4 or p_rgb[:, 1].max() > calib.RGB_H - 5 or
                p_th[:, 0].min() < 2 or p_th[:, 0].max() > calib.TH_W - 3 or
                p_th[:, 1].min() < 2 or p_th[:, 1].max() > calib.TH_H - 3):
            continue
        views.append({"name": "v%02d" % i, "rgb": p_rgb.tolist(), "thermal": p_th.tolist()})
    return {"pattern": list(PATTERN), "square_m": SQUARE, "views": views}


def main():
    print("A. homography LUT vs independent ray-trace")
    for dist in (0.5, 0.8, 2.0):
        med, mx, agree, n = lut_delta(calib.build_lut(TRUTH, dist), raytrace_lut(dist))
        check("Z=%.1fm  median error" % dist, med is not None and med < 0.02,
              "median=%.4f max=%.4f px over %d entries" % (med, mx, n))
        check("Z=%.1fm  coverage masks agree" % dist, agree > 99.0, "%.1f%%" % agree)

    print("\nB. recovered calibration vs ground truth")
    corners = make_views()
    print("  %d usable synthetic views" % len(corners["views"]))
    got = calib.solve(corners)

    print("  rms rgb=%.3f th=%.3f stereo=%.3f px" %
          (got["rms_rgb"], got["rms_th"], got["rms_stereo"]))
    print("  baseline recovered %.2f mm (true %.2f mm)" %
          (got["baseline_mm"], np.linalg.norm(T_TRUE) * 1000))

    check("reprojection rms is sane", max(got["rms_rgb"], got["rms_th"]) < 0.6,
          "rgb=%.3f th=%.3f" % (got["rms_rgb"], got["rms_th"]))
    check("baseline within 1mm",
          abs(got["baseline_mm"] - np.linalg.norm(T_TRUE) * 1000) < 1.0,
          "%.2f vs %.2f mm" % (got["baseline_mm"], np.linalg.norm(T_TRUE) * 1000))

    med, mx, agree, n = lut_delta(calib.build_lut(got, 0.8), calib.build_lut(TRUTH, 0.8))
    # The whole registration budget at 12mm is +-1.32 thermal px; the calibration
    # itself must consume well under half of it to be worth anything.
    check("LUT median error < 0.3 thermal px", med is not None and med < 0.30,
          "median=%.3f max=%.3f px" % (med, mx))
    check("LUT max error < 1.0 thermal px", mx is not None and mx < 1.0,
          "max=%.3f px" % mx)

    print()
    if FAILS:
        print("FAILED: %s" % ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
