#!/usr/bin/env python3
"""Prove the radar->camera extrinsic chain against known ground truth, no hardware.

Nothing a radar extrinsic gets wrong announces itself. A frame swap, a sign, a
fit on six clustered points - all of them converge, all of them report a small
reprojection error, and all of them are wrong at a distance nobody stood at. So
every check here is built around a truth that is known in advance:

  A. the coordinate chain, radar frame -> camera frame -> pixel, with each axis
     tested on its own so a swap moves the answer somewhere visible;
  B. project() refusing to invent a pixel - behind the camera, and past the
     radius where the distortion polynomial folds an off-axis target back into
     the frame;
  C. recovery of a known (R, t) from correspondences with sub-pixel noise;
  D. the same with the radar's REAL angular noise applied in the radar domain,
     which is the honest test of whether the reported sigmas mean anything;
  E. the quality gate rejecting the four degenerate sets it exists for, including
     a set that fits beautifully and is in the wrong coordinate frame;
  F. leave-one-distance-out validation showing an error the in-sample residual
     does not;
  G. persistence, the refusal to load an ungated fit, and staleness;
  H. the subcommands, because that is the only way the tool is ever run.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radar_extrinsics as rx  # noqa: E402

FAILS = []

# The visible camera from test_calib.py, so both calibration tools are checked
# against the same imaginary lens.
K = np.array([[460.0, 0, 320.0], [0, 460.0, 200.0], [0, 0, 1.0]])
D = np.array([-0.280, 0.090, 0.0005, -0.0003, 0.0])
SIZE = (640, 400)

# A plausible bracket: the radar sits 25 mm right of the camera, 60 mm below it
# and 10 mm further back, with a couple of degrees of mounting error. t is the
# radar origin expressed in the camera frame, so "below" is +y (y is down).
T_TRUE = np.array([0.025, 0.060, -0.010])
R_TRUE = cv2.Rodrigues(np.array([0.010, -0.020, 0.008]))[0] @ rx.R_CANONICAL

# Tripod stations, not a continuum - which is how the depth-bin gate is meant to
# see the data, and why a "many depths" claim from a single station is catchable.
STATIONS = (0.8, 1.5, 2.5, 3.8)


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------ synthesis


def radar_grid(n, stations=STATIONS, az_deg=31.0, el_deg=20.0, seed=5):
    """Reflector positions in the PROJECT radar frame: x fwd, y left, z up."""
    g = np.random.default_rng(seed)
    r = g.choice(np.asarray(stations, float), n)
    az = np.radians(g.uniform(-az_deg, az_deg, n))
    el = np.radians(g.uniform(-el_deg, el_deg, n))
    return np.stack([r * np.cos(el) * np.cos(az),
                     r * np.cos(el) * np.sin(az),
                     r * np.sin(el)], -1)


def to_pixels(points_radar, R=R_TRUE, t=T_TRUE):
    uv, _, _ = rx.project(points_radar, R, t, K, D)
    return uv


def perturb_angles(points, sigma_az_deg, sigma_el_deg, seed=11):
    """Move the RADAR points, not the pixels.

    This is the direction the error really runs: the operator clicks the
    reflector in the image to a fraction of a pixel, and the radar reports its
    bearing to a degree. Noise added to the pixels instead would make the fit
    look like a camera problem and would hide the axis asymmetry entirely.
    """
    g = np.random.default_rng(seed)
    p = np.asarray(points, float)
    r = np.linalg.norm(p, axis=1)
    az = np.arctan2(p[:, 1], p[:, 0]) + np.radians(g.normal(0, sigma_az_deg, len(p)))
    el = np.arcsin(np.clip(p[:, 2] / r, -1, 1)) + np.radians(
        g.normal(0, sigma_el_deg, len(p)))
    return np.stack([r * np.cos(el) * np.cos(az),
                     r * np.cos(el) * np.sin(az),
                     r * np.sin(el)], -1)


def dataset(n=140, seed=5, pixel_noise=0.3, angular_noise=None, margin=6,
            stations=STATIONS, az_deg=31.0, el_deg=20.0, range_bias_m=0.0):
    """(radar Nx3, pixel Nx2) with everything off the image discarded."""
    truth = radar_grid(n, stations, az_deg, el_deg, seed)
    pix = to_pixels(truth)
    keep = (np.isfinite(pix).all(1) & (pix[:, 0] > margin) & (pix[:, 0] < SIZE[0] - margin)
            & (pix[:, 1] > margin) & (pix[:, 1] < SIZE[1] - margin))
    truth, pix = truth[keep], pix[keep]

    radar = truth
    if angular_noise:
        radar = perturb_angles(radar, angular_noise[0], angular_noise[1], seed + 1)
    if range_bias_m:
        # A residual range bias is radial, and radial is exactly what solvePnP
        # cannot tell from a lever arm along the line of sight.
        r = np.linalg.norm(radar, axis=1, keepdims=True)
        radar = radar * (1.0 + range_bias_m / r)
    if pixel_noise:
        pix = pix + np.random.default_rng(seed + 2).normal(0, pixel_noise, pix.shape)
    return radar, pix


def corr_file(path, radar, pix, camera="rgb"):
    doc = {"camera": camera, "image_size": list(SIZE),
           "correspondences": [{"name": "p%03d" % i, "radar": list(map(float, r)),
                                "pixel": list(map(float, p))}
                               for i, (r, p) in enumerate(zip(radar, pix))]}
    with open(path, "w") as fh:
        json.dump(doc, fh)
    return doc


# ------------------------------------------------------------------ A. the chain


def coordinate_chain():
    print("A. radar frame -> camera frame -> pixel")

    got = rx.verify_convention_against_mmwave()
    check("R_CANONICAL agrees with mmwave.ti_to_project",
          got is not False, "mmwave.py not on the path" if got is None else "checked")

    # One axis at a time, at the ideal mount so there is nothing to hide behind.
    # If any of the three signs is flipped these land in the opposite half of the
    # image, which is exactly the "visibly different answer" a swap must give.
    zero = np.zeros(3)
    uv, z, st = rx.project([[2.0, 0.0, 0.0]], rx.R_CANONICAL, zero, K, None, SIZE)
    check("straight ahead lands on the principal point",
          np.allclose(uv[0], [320.0, 200.0], atol=1e-6) and abs(z[0] - 2.0) < 1e-9,
          "uv=(%.1f, %.1f) z=%.2f m" % (uv[0, 0], uv[0, 1], z[0]))

    uv, _, _ = rx.project([[2.0, 0.5, 0.0]], rx.R_CANONICAL, zero, K, None, SIZE)
    check("radar +y (left) goes to the LEFT of the image", uv[0, 0] < 320.0,
          "u=%.1f" % uv[0, 0])

    uv, _, _ = rx.project([[2.0, 0.0, 0.5]], rx.R_CANONICAL, zero, K, None, SIZE)
    check("radar +z (up) goes to the TOP of the image", uv[0, 1] < 200.0,
          "v=%.1f" % uv[0, 1])

    # The swap that actually happens: TI's frame fed in as if it were the project
    # frame. Same point, and it has to land somewhere else - if it did not, no
    # test anywhere could catch the mix-up.
    p = np.array([2.0, 0.5, 0.3])
    p_ti = np.array([-p[1], p[0], p[2]])           # project -> TI, a 90 deg turn about z
    a, _, _ = rx.project([p], rx.R_CANONICAL, zero, K, None, SIZE)
    b, _, _ = rx.project([p_ti], rx.R_CANONICAL, zero, K, None, SIZE)
    check("the same point in TI's frame projects somewhere else entirely",
          not np.isfinite(b).all() or np.linalg.norm(a[0] - b[0]) > 100.0,
          "%.0f px apart" % (np.linalg.norm(a[0] - b[0]) if np.isfinite(b).all() else -1))

    # and against OpenCV, so the hand-written frame maths is not marking its own work
    pts = radar_grid(40, seed=2)
    uv, _, _ = rx.project(pts, R_TRUE, T_TRUE, K, D)
    ref, _ = cv2.projectPoints(pts, cv2.Rodrigues(R_TRUE)[0], T_TRUE, K, D)
    check("in-front projection matches cv2.projectPoints",
          np.nanmax(np.abs(uv - ref.reshape(-1, 2))) < 1e-9,
          "max delta %.2e px" % np.nanmax(np.abs(uv - ref.reshape(-1, 2))))


def refusals():
    print("\nB. project() refuses to invent a pixel")

    behind = np.array([[-2.0, 0.3, 0.1]])
    naive, _ = cv2.projectPoints(behind, cv2.Rodrigues(rx.R_CANONICAL)[0],
                                 np.zeros(3), K, D)
    naive = naive.reshape(2)
    uv, z, st = rx.project(behind, rx.R_CANONICAL, np.zeros(3), K, None, SIZE)
    check("a target BEHIND the camera would otherwise get a real-looking pixel",
          0 <= naive[0] < SIZE[0] and 0 <= naive[1] < SIZE[1],
          "cv2 alone says (%.0f, %.0f)" % (naive[0], naive[1]))
    check("...and is reported as behind, with no pixel at all",
          st[0] == rx.PROJ_BEHIND and not np.isfinite(uv[0]).any() and z[0] < 0,
          "z=%.2f m" % z[0])

    # 60 deg off axis with a single strong k1: the radial polynomial has already
    # turned over, so the "distorted" pixel folds back into the frame.
    fold_d = np.array([-0.280, 0.0, 0.0, 0.0, 0.0])
    off = np.array([[1.0, 1.732, 0.0]])
    naive, _ = cv2.projectPoints(off, cv2.Rodrigues(rx.R_CANONICAL)[0], np.zeros(3),
                                 K, fold_d)
    naive = naive.reshape(2)
    uv, _, st = rx.project(off, rx.R_CANONICAL, np.zeros(3), K, fold_d, SIZE)
    check("a 60 deg off-axis target folds into the frame under Brown-Conrady",
          0 <= naive[0] < SIZE[0] and 0 <= naive[1] < SIZE[1],
          "cv2 alone says (%.0f, %.0f), r_valid=%.2f"
          % (naive[0], naive[1], rx.distortion_valid_radius(fold_d)))
    check("...and is reported unmodelled rather than drawn on the scene",
          st[0] == rx.PROJ_UNMODELLED and not np.isfinite(uv[0]).any())

    # Outside the frame but geometrically real: the coordinate must survive
    # intact. Clamping it to the border would put a target that is off to the
    # left onto the leftmost column, which reads as a detection at the edge.
    edge = np.array([[2.0, 2.0, 0.0]])          # 45 deg to the left, well off frame
    uv, _, st = rx.project(edge, rx.R_CANONICAL, np.zeros(3), K, None, SIZE)
    check("an out-of-frame target keeps its true coordinate, unclamped",
          st[0] == rx.PROJ_OUTSIDE and np.isfinite(uv[0]).all() and uv[0, 0] < -1.0,
          "u=%.1f" % uv[0, 0])
    check("without an image size, out-of-frame is not decidable and is not claimed",
          rx.project(edge, rx.R_CANONICAL, np.zeros(3), K, None)[2][0] == rx.PROJ_OK)

    # a detection with no elevation is a line, not a point
    boxes, ok = rx.project_unknown_elevation([[2.0, 0.2]], R_TRUE, T_TRUE, K, D, SIZE)
    u_span, v_span = boxes[0, 1] - boxes[0, 0], boxes[0, 3] - boxes[0, 2]
    check("an elevation-less detection is a vertical span, not a chosen height",
          ok[0] and v_span > 20 * max(u_span, 1e-6) and v_span > 100,
          "u span %.1f px, v span %.1f px" % (u_span, v_span))


# ------------------------------------------------------------------ C/D. recovery


def recover_subpixel():
    print("\nC. recovery from a known (R, t), sub-pixel noise")
    radar, pix = dataset(pixel_noise=0.3)
    R, t, stats = rx.solve(radar, pix, K, D, image_size=SIZE)
    g = rx.gate(radar, pix, stats, SIZE)
    print(rx.summarise(stats, g))

    ang = rx.rotation_angle_deg(R, R_TRUE)
    dt_mm = np.linalg.norm(t - T_TRUE) * 1000.0
    check("rotation recovered to better than 0.1 deg", ang < 0.1, "%.4f deg" % ang)
    check("lever arm recovered to better than 3 mm", dt_mm < 3.0,
          "%.2f mm off (%.1f vs %.1f mm)"
          % (dt_mm, stats["lever_arm_mm"], np.linalg.norm(T_TRUE) * 1000))
    check("mount tilt reads as a bracket, not a frame error",
          stats["mount_tilt_deg"] < 5.0, "%.2f deg" % stats["mount_tilt_deg"])
    check("residuals are reported per correspondence, not just averaged",
          len(stats["per_point"]) == stats["n_inliers"]
          and all(p["px"] is not None for p in stats["per_point"]),
          "%d entries" % len(stats["per_point"]))
    check("a clean set passes the gate", g["passed"], "; ".join(g["failures"]))

    # a planted blunder - a reflector clicked on the wrong target - must be thrown
    # out rather than shared over every other correspondence
    bad_pix = pix.copy()
    bad_pix[[3, 17, 40]] += np.array([90.0, -70.0])
    R2, t2, s2 = rx.solve(radar, bad_pix, K, D, image_size=SIZE)
    inl = np.array(s2["inlier_mask"])
    check("RANSAC rejects planted blunders and keeps the rest",
          (not inl[[3, 17, 40]].any()) and inl.sum() >= len(radar) - 6,
          "%d/%d kept" % (inl.sum(), len(radar)))
    check("...and the pose survives them",
          rx.rotation_angle_deg(R2, R_TRUE) < 0.2
          and np.linalg.norm(t2 - T_TRUE) * 1000 < 5.0,
          "%.3f deg, %.2f mm" % (rx.rotation_angle_deg(R2, R_TRUE),
                                 np.linalg.norm(t2 - T_TRUE) * 1000))
    return radar, pix


def recover_radar_noise():
    """The honest version: the radar is a degree wrong in azimuth and three in
    elevation, and the pixels are exact.

    The point of this one is not that (R, t) comes out perfect - it cannot. It is
    that the residual the tool reports lands where the noise model says it should.
    A tool that quotes "1.4 px" against a sensor whose own bearing noise is worth
    8 px of image is quoting the optimiser, not the calibration.
    """
    print("\nD. recovery with the radar's real angular noise")
    sa, se = rx.RADAR_SIGMA_AZ_DEG, rx.RADAR_SIGMA_EL_DEG
    radar, pix = dataset(n=220, pixel_noise=0.2, angular_noise=(sa, se))
    R, t, stats = rx.solve(radar, pix, K, D, image_size=SIZE)
    g = rx.gate(radar, pix, stats, SIZE)
    print(rx.summarise(stats, g))

    su, sv = rx.pixel_sigma(K)
    check("the pixel noise floor follows from the angular one",
          abs(su - 460 * np.radians(sa)) < 1e-6 and abs(sv - 460 * np.radians(se)) < 1e-6,
          "%.1f px in u, %.1f px in v" % (su, sv))
    med = stats["sigma"]["median"]
    # chi with two degrees of freedom has median 1.18; anything near it means the
    # whitening and the sigmas describe the same world
    check("median residual is about one radar sigma, as the model predicts",
          0.6 < med < 1.8, "%.2f sigma (%.1f px)" % (med, stats["px"]["median"]))
    check("the v residual is several times the u residual, as the array geometry says",
          stats["dv_px"]["median"] > 2.0 * stats["du_px"]["median"],
          "du %.1f px, dv %.1f px" % (stats["du_px"]["median"], stats["dv_px"]["median"]))

    ang = rx.rotation_angle_deg(R, R_TRUE)
    check("rotation still averages down to a fraction of a degree", ang < 0.5,
          "%.3f deg" % ang)
    check("the fit is accepted, with the residual it deserves", g["passed"],
          "; ".join(g["failures"]))
    # The translation is the weak one, and the bootstrap has to say so out loud
    # rather than let a millimetre-precise t be quoted from degree-precise data.
    unc = stats["uncertainty"]
    clean = rx.solve(*dataset(n=220, pixel_noise=0.2), K, D, image_size=SIZE)[2]
    check("bootstrap reports a real uncertainty on the lever arm",
          unc is not None and 0.5 < unc["lever_arm_std_mm"] < 30.0
          and unc["rotation_std_deg"] < 1.0,
          "lever +-%.1f mm, rotation +-%.3f deg"
          % (unc["lever_arm_std_mm"], unc["rotation_std_deg"]))
    check("...and it grows with the radar noise instead of quoting the optimiser",
          unc["lever_arm_std_mm"] > 5.0 * clean["uncertainty"]["lever_arm_std_mm"],
          "+-%.2f mm at radar noise vs +-%.2f mm at sub-pixel noise"
          % (unc["lever_arm_std_mm"], clean["uncertainty"]["lever_arm_std_mm"]))


# ------------------------------------------------------------------ E. the gate


def gate_rejects():
    print("\nE. the gate refuses what it exists to refuse")

    def fit_and_gate(radar, pix, **kw):
        R, t, s = rx.solve(radar, pix, K, D, image_size=SIZE, bootstrap_n=0, **kw)
        return R, t, s, rx.gate(radar, pix, s, SIZE)

    def why(g, word):
        return any(word in f for f in g["failures"])

    # 1. six points is enough to solve and nowhere near enough to trust
    radar, pix = dataset(n=9)
    _, _, s, g = fit_and_gate(radar[:6], pix[:6])
    check("6 correspondences are rejected on count",
          not g["passed"] and why(g, "inlier correspondences"),
          "%d failure(s)" % len(g["failures"]))

    # 2. clustered in the image: a perfect fit over one twentieth of the frame
    radar, pix = dataset(n=400, az_deg=3.0, el_deg=2.0)
    _, _, s, g = fit_and_gate(radar, pix)
    check("points clustered in the image are rejected on spread",
          not g["passed"] and (why(g, "image width") or why(g, "frame area")),
          "median %.2f px, hull %.2f%%"
          % (s["px"]["median"], 100 * g["metrics"]["hull_frac"]))

    # 3. one distance: the fit is exact there and unconstrained everywhere else
    radar, pix = dataset(n=200, stations=(1.5,))
    _, _, s, g = fit_and_gate(radar, pix)
    check("a single distance is rejected on depth spread",
          not g["passed"] and (why(g, "distinct distance") or why(g, "ratio")
                               or why(g, "depth span")),
          "median %.2f px at one station" % s["px"]["median"])

    # 4. a metre-long lever arm is not two boxes on a bracket
    radar, pix = dataset(n=160)
    far = np.array([0.9, 0.35, -0.2])
    pix_far = to_pixels(radar, R_TRUE, far)
    ok = np.isfinite(pix_far).all(1)
    _, t, s, g = fit_and_gate(radar[ok], pix_far[ok])
    check("an implausible lever arm is rejected on physics",
          not g["passed"] and why(g, "lever arm"),
          "%.0f mm recovered" % s["lever_arm_mm"])

    # 5. THE one. Radar points handed over in TI's frame instead of the project
    #    frame. It fits to the same residual as the correct set - the reprojection
    #    error is blind to it - and only the mount angle gives it away.
    radar, pix = dataset(n=160)
    radar_ti = np.stack([-radar[:, 1], radar[:, 0], radar[:, 2]], -1)
    _, _, s_ok, g_ok = fit_and_gate(radar, pix)
    _, _, s_ti, g_ti = fit_and_gate(radar_ti, pix)
    check("the wrong-frame fit has just as good a residual as the right one",
          abs(s_ti["px"]["median"] - s_ok["px"]["median"]) < 0.15,
          "%.3f px wrong-frame vs %.3f px right" % (s_ti["px"]["median"],
                                                    s_ok["px"]["median"]))
    check("...and is rejected anyway, on the mount angle",
          not g_ti["passed"] and why(g_ti, "coordinate-frame error"),
          "%.1f deg from canonical" % s_ti["mount_tilt_deg"])

    # 6. Warnings, which must fire without failing the fit. A lever arm that runs
    #    along the optical axis is the ambiguous one: it is what an absorbed radar
    #    range bias looks like, and also what a radar genuinely mounted behind the
    #    camera looks like. Fitting a real setback is the way to exercise it,
    #    because a bias only gets absorbed when the targets are too clustered for
    #    the spread gate to allow the fit at all.
    radar, _ = dataset(n=160)
    setback = np.array([0.025, 0.060, -0.085])
    pix_set = to_pixels(radar, R_TRUE, setback)
    ok = np.isfinite(pix_set).all(1)
    _, t, s, g = fit_and_gate(radar[ok], pix_set[ok])
    check("a lever arm along the optical axis is flagged as bias-ambiguous",
          g["passed"] and any("range bias" in w for w in g["warnings"]),
          "t_z = %.0f mm" % s["t_mm"][2])

    # and a wide, multi-depth set does NOT absorb a range bias into t - which is
    # why the held-out check below, not the gate, is what catches one
    radar, pix = dataset(n=160, range_bias_m=0.06)
    _, t, s, g = fit_and_gate(radar, pix)
    check("a 60 mm range bias does not hide in t when the set is well spread",
          abs(s["t_mm"][2] - T_TRUE[2] * 1000) < 25.0,
          "t_z = %.0f mm against a true %.0f mm" % (s["t_mm"][2], T_TRUE[2] * 1000))


# ------------------------------------------------------------------ F. held out


def held_out():
    print("\nF. held-out validation at a distance the fit never saw")

    radar, pix = dataset(n=200, pixel_noise=0.3)
    rows = rx.holdout_by_depth(radar, pix, K, D, SIZE)
    depths = [r["depth_m"] for r in rows]
    check("every station is held out in turn", len(rows) == len(STATIONS),
          "tested at %s m" % ", ".join("%.1f" % d for d in depths))
    worst = max(rows, key=lambda r: r["held_median_px"])
    for r in rows:
        print("    %.1f m: fit %.2f px on %d, held-out %.2f px / %.2f sigma on %d"
              % (r["depth_m"], r["fit_median_px"], r["n_fit"],
                 r["held_median_px"], r["held_median_sigma"], r["n_test"]))
    check("clean data extrapolates in depth", worst["held_median_px"] < 1.5,
          "worst %.2f px at %.1f m" % (worst["held_median_px"], worst["depth_m"]))

    # Teeth. A radial range bias costs f*|t_perp|*bias/z^2 pixels, so it is a
    # CLOSE-range error: it hides at 3.8 m and shows at 0.8 m. Fit on the far
    # stations, where it is invisible, and the near station reports it - which is
    # the whole reason a held-out distance is checked rather than a held-out
    # subset of the same distance.
    radar, pix = dataset(n=200, pixel_noise=0.3, range_bias_m=0.06)
    rows = rx.holdout_by_depth(radar, pix, K, D, SIZE)
    near = min(rows, key=lambda r: r["depth_m"])
    for r in rows:
        print("    %.1f m: fit %.2f px, held-out %.2f px  (with a 60 mm range bias)"
              % (r["depth_m"], r["fit_median_px"], r["held_median_px"]))
    check("a range bias invisible in-sample shows up at the held-out near station",
          near["held_median_px"] > 2.5 * near["fit_median_px"],
          "%.2f px held-out vs %.2f px in-sample at %.1f m"
          % (near["held_median_px"], near["fit_median_px"], near["depth_m"]))


# ------------------------------------------------------------------ G. the file


def persistence():
    print("\nG. the calib file, and knowing when it is stale")
    tmp = tempfile.mkdtemp(prefix="radar_extrinsics_")
    try:
        radar, pix = dataset(n=160, pixel_noise=0.3)
        R, t, stats = rx.solve(radar, pix, K, D, image_size=SIZE)
        g = rx.gate(radar, pix, stats, SIZE)

        intr = os.path.join(tmp, "calib.json")
        with open(intr, "w") as fh:
            json.dump({"K_rgb": K.tolist(), "dist_rgb": D.tolist(),
                       "K_th": K.tolist(), "dist_th": D.tolist()}, fh)
        Kr, Dr = rx.intrinsics_from_calib(intr, "rgb")
        check("intrinsics are read out of calib.py's own file",
              np.allclose(Kr, K) and np.allclose(Dr, D))

        path = os.path.join(tmp, "radar_calib.json")
        rx.save_calib(path, R, t, K, D, SIZE, stats, g, camera="rgb",
                      meta={"correspondence_path": "corr.json", "intrinsics_path": intr,
                            "rig_id": "bench-A", "mount_token": "2026-08-09-bolt1"})
        cal = rx.load_calib(path)

        check("R and t survive the round trip exactly",
              np.allclose(cal["R"], R, atol=0) and np.allclose(cal["t"], t, atol=0))
        check("the file carries the range it was collected over",
              cal["working_range_m"][0] < 0.9 and cal["working_range_m"][1] > 3.5,
              "%.2f..%.2f m" % tuple(cal["working_range_m"]))
        check("the file carries per-correspondence residuals, not just a mean",
              len(cal["per_correspondence"]) == stats["n_inliers"]
              and "residual_px" in cal and cal["n_correspondences"] == len(radar))
        check("the depth assumption is stated in the file itself",
              "full 3D" in cal["depth_assumption"] and "working_range_m"
              in cal["depth_assumption"])
        check("the frame convention travels with the numbers",
              "forward" in cal["frames"]["radar"] and "P_cam" in cal["frames"]["transform"])

        # projection through the loaded file, including the range window
        uv, z, st = rx.project_calib(np.array([[2.0, 0.1, 0.0], [9.0, 0.2, 0.0]]), cal)
        check("a detection past the fitted range is marked, not quietly used",
              st[0] == rx.PROJ_OK and st[1] == rx.PROJ_OUT_OF_RANGE,
              "%.2f..%.2f m fitted" % tuple(cal["working_range_m"]))

        check("nothing is stale straight after writing",
              rx.stale_reasons(cal, intr, "2026-08-09-bolt1") == [],
              str(rx.stale_reasons(cal, intr, "2026-08-09-bolt1")))
        with open(intr, "a") as fh:
            fh.write("\n")                       # stand-in for a re-run of calib.py
        reasons = rx.stale_reasons(cal, intr, "2026-08-09-bolt1")
        check("re-solved intrinsics invalidate the extrinsics",
              any("re-solved" in r for r in reasons), "; ".join(reasons))
        reasons = rx.stale_reasons(cal, None, "2026-08-10-bolt2")
        check("a new mount token invalidates them too",
              any("assembly has changed" in r for r in reasons), "; ".join(reasons))

        # a failed fit must not be loadable by accident
        bad = os.path.join(tmp, "bad.json")
        radar_s, pix_s = dataset(n=200, stations=(1.5,))
        Rb, tb, sb = rx.solve(radar_s, pix_s, K, D, image_size=SIZE, bootstrap_n=0)
        gb = rx.gate(radar_s, pix_s, sb, SIZE)
        rx.save_calib(bad, Rb, tb, K, D, SIZE, sb, gb, camera="rgb")
        try:
            rx.load_calib(bad)
            check("a fit that failed its gate cannot be loaded by accident", False)
        except ValueError as e:
            check("a fit that failed its gate cannot be loaded by accident",
                  "quality gate" in str(e))
        check("...unless the caller says so in as many words",
              rx.load_calib(bad, allow_failed_gate=True)["gate"]["passed"] is False)

        # a hand-edited R is not a rotation and must not load
        with open(path) as fh:
            doc = json.load(fh)
        doc["R"][0][0] += 0.2
        edited = os.path.join(tmp, "edited.json")
        with open(edited, "w") as fh:
            json.dump(doc, fh)
        try:
            rx.load_calib(edited)
            check("an edited R is caught on load", False)
        except ValueError as e:
            check("an edited R is caught on load", "not a rotation" in str(e))

        # the overlap ROI
        mask, cov = rx.fov_overlap_mask(R, t, K, D, SIZE, (0.5, 4.0))
        check("the radar cone covers this camera's whole FOV, so the limit is range",
              cov > 0.99 and mask.shape == (SIZE[1], SIZE[0]),
              "%.1f%% of the frame" % (100 * cov))
        narrow, cov_n = rx.fov_overlap_mask(R, t, K, D, SIZE, (0.5, 4.0),
                                            az_limit_deg=15.0, el_limit_deg=10.0)
        rows = np.nonzero(narrow.any(1))[0]
        check("a narrow radar FOV masks the rest of the frame instead of stretching",
              0.05 < cov_n < 0.5 and narrow[SIZE[1] // 2, SIZE[0] // 2] == 1
              and narrow[0, 0] == 0,
              "%.1f%% of the frame, rows %d..%d" % (100 * cov_n, rows[0], rows[-1]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def command_line():
    """Drive the subcommands as a shell would.

    A calibration tool is only ever run through its CLI, and a format string that
    raises there wastes a capture session that cannot be repeated once the
    reflector and the tripod have been put away.
    """
    print("\nH. the command line")
    tmp = tempfile.mkdtemp(prefix="radar_extrinsics_cli_")
    try:
        intr = os.path.join(tmp, "calib.json")
        with open(intr, "w") as fh:
            json.dump({"K_rgb": K.tolist(), "dist_rgb": D.tolist(),
                       "K_th": K.tolist(), "dist_th": D.tolist()}, fh)
        good = os.path.join(tmp, "corr.json")
        corr_file(good, *dataset(n=160, pixel_noise=0.3))
        out = os.path.join(tmp, "radar_calib.json")

        def run(*argv):
            r = subprocess.run([sys.executable, os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "radar_extrinsics.py")]
                + list(argv), capture_output=True, text=True)
            return r.returncode, r.stdout + r.stderr

        rc, log = run("solve", good, "-i", intr, "-c", "rgb", "-o", out,
                      "--mount-token", "cli-token", "--rig-id", "bench-A")
        check("solve writes a calib and reports it", rc == 0 and os.path.exists(out)
              and "lever arm" in log and "held-out" in log, log.strip().splitlines()[-1])

        rc, log = run("check", out, "-i", intr, "--mount-token", "cli-token")
        check("check reads it back clean", rc == 0 and "PASSED" in log,
              log.strip().splitlines()[0])
        rc, log = run("check", out, "-i", intr, "--mount-token", "someone-refitted-it")
        check("check exits non-zero once the mount token moves",
              rc == 1 and "assembly has changed" in log)

        rc, log = run("validate", out, good)
        check("validate reports residuals against an existing calib",
              rc == 0 and "median" in log, log.strip().splitlines()[-1])
        rc, log = run("holdout", good, "-i", intr)
        check("holdout runs from the command line",
              rc == 0 and log.count("held-out") == len(STATIONS))

        png = os.path.join(tmp, "roi.png")
        rc, log = run("roi", out, "-o", png)
        check("roi writes an overlap mask", rc == 0 and os.path.exists(png)
              and cv2.imread(png, cv2.IMREAD_GRAYSCALE).shape == (SIZE[1], SIZE[0]),
              log.strip().splitlines()[0])

        bad = os.path.join(tmp, "one_depth.json")
        corr_file(bad, *dataset(n=200, stations=(1.5,)))
        rejected = os.path.join(tmp, "rejected.json")
        rc, log = run("solve", bad, "-i", intr, "-o", rejected)
        check("solve refuses to write a rejected fit",
              rc == 1 and not os.path.exists(rejected) and "refusing" in log)
        rc, log = run("solve", bad, "-i", intr, "-o", rejected, "--force")
        check("--force writes it, marked failed, and still exits non-zero",
              rc == 1 and os.path.exists(rejected) and "MARKED FAILED" in log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    coordinate_chain()
    refusals()
    recover_subpixel()
    recover_radar_noise()
    gate_rejects()
    held_out()
    persistence()
    command_line()

    print()
    if FAILS:
        print("FAILED (%d): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("all radar extrinsic checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
