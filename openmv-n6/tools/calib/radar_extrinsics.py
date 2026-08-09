#!/usr/bin/env python3
"""Solve the IWR1843 -> camera rigid transform and project radar points into pixels.

The camera pair already has calib.py. This is the third sensor, and it is a
different problem: there is no chessboard the radar can see, so the input is
hand-made correspondences - a corner reflector on a tripod, its radar detection
and the pixel it appears at - and the geometry is full 3D rather than a plane
homography. Nothing here is depth-limited the way the thermal warp is: given
(R, t) a radar point carries its own range, so the projection is exact at every
distance. What IS depth-limited is the evidence: a fit is only trustworthy over
the range of distances the correspondences were collected at, and that range is
written into the calib file so a consumer can refuse work outside it.

Frames, stated here rather than inherited, because a mix-up does not raise:

    radar   x = forward, y = left, z = up      (the project frame mmwave.py returns;
                                                TI's own is x = right, y = forward)
    camera  x = right,   y = down, z = forward (OpenCV optical)
    transform   P_cam = R @ P_radar + t,  t = the radar origin in camera coordinates

R therefore contains BOTH the fixed axis permutation (R_CANONICAL below) and the
few degrees of real mounting error. A convention slip looks exactly like a 90 deg
mount and fits with a small residual - which is why the quality gate measures the
angle between the recovered R and R_CANONICAL and refuses anything far from it.

Weighting: the radar's angular accuracy is the error budget, not the camera's.
A cross-range error of r*sigma_az metres at range r projects to roughly
f*sigma_az pixels REGARDLESS of r (the r cancels against z_cam), so the natural
unit is pixels after all - but not isotropically: this radar's elevation array is
two elements against eight in azimuth, so sigma_el >> sigma_az and a pixel of v
error is much weaker evidence than a pixel of u error. solve() whitens by
scaling the image axes, which is exactly a diagonal rescale of K, so the PnP cost
becomes a proper chi-square and the RANSAC threshold means the same thing on both
axes. Residuals are reported in both raw pixels and in radar sigmas.

Usage:
    ./radar_extrinsics.py solve    corr.json -i calib.json -c rgb -o radar.json
    ./radar_extrinsics.py holdout  corr.json -i calib.json -c rgb
    ./radar_extrinsics.py validate radar.json corr.json
    ./radar_extrinsics.py check    radar.json [-i calib.json]
    ./radar_extrinsics.py roi      radar.json -o roi.png [--range 2.0]

Correspondence file (JSON), the format `solve` and `validate` read:
    {"camera": "rgb", "image_size": [640, 400],
     "correspondences": [{"name": "s01_p1", "radar": [x, y, z], "pixel": [u, v]}, ...]}
"""
import argparse
import getpass
import hashlib
import json
import os
import platform
import socket
import sys
import time

import cv2
import numpy as np

SCHEMA = "radar_extrinsics/1"

# The mount with zero misalignment: radar (fwd, left, up) -> camera (right, down,
# fwd). Written out rather than derived so the three sign choices are readable.
R_CANONICAL = np.array([[0.0, -1.0, 0.0],     # cam x (right)   = -radar y (left)
                        [0.0, 0.0, -1.0],     # cam y (down)    = -radar z (up)
                        [1.0, 0.0, 0.0]])     # cam z (forward) =  radar x (forward)

FRAMES = {
    "radar": "x=forward, y=left, z=up (project frame; mmwave.py convention='project')",
    "camera": "x=right, y=down, z=forward (OpenCV optical)",
    "transform": "P_cam = R @ P_radar + t; t is the radar origin in camera coordinates",
}

# Radar angular accuracy, 1 sigma, for a strong static target. These are NOMINAL
# - roughly a tenth of the beamwidth of the 8-element azimuth virtual array and
# of the 2-element elevation pair - not measured on this rig. They set the whole
# residual scale, so measure them (repeat detections of a fixed reflector) and
# replace them; every reported "sigma" number moves with these two numbers.
RADAR_SIGMA_AZ_DEG = 1.0
RADAR_SIGMA_EL_DEG = 3.0

# radar/configs/radar_10hz.cfg: aoaFovCfg -1 -60 60 -30 30, cfarFovCfg range gate
# 0.35..5.0 m, and the 175 kHz IF high-pass that blinds everything under 0.375 m
# no matter what the range gate says.
RADAR_AZ_LIMIT_DEG = 60.0
RADAR_EL_LIMIT_DEG = 30.0
RADAR_MIN_RANGE_M = 0.375
RADAR_MAX_RANGE_M = 5.0

# project() status codes. A consumer must branch on these; the uv value alone is
# not self-describing, which is the whole point of not clamping.
PROJ_OK = 0            # in front, distortion model valid, inside the image
PROJ_OUTSIDE = 1       # same, but outside the image - uv is real and UNCLAMPED
PROJ_BEHIND = 2        # z_cam <= 0; uv is NaN, there is no such pixel
PROJ_UNMODELLED = 3    # past the radius where Brown-Conrady stops being monotonic
PROJ_OUT_OF_RANGE = 4  # outside the range window the calibration was fitted over

MIN_Z_M = 1e-4         # anything closer than this to the pupil is "behind"

DEPTH_BIN_M = 0.5      # tripod stations, not a continuum: bin before counting depths

GATE = {
    "min_correspondences": 12,   # 6 is enough to solve and nowhere near enough to trust
    "min_inlier_frac": 0.60,
    "min_u_span_frac": 0.40,     # spread across the image, not a cluster at the centre
    "min_v_span_frac": 0.20,     # looser than u: the radar's elevation is coarse and
                                 # targets tend to sit near tripod height
    "min_hull_frac": 0.06,
    "min_depth_bins": 3,
    "min_depth_ratio": 1.6,      # far/near; depth is what separates t from R
    "min_depth_span_m": 0.60,
    "max_median_sigma": 2.0,     # residual measured against the radar's own noise
    "max_p95_sigma": 5.0,
    "max_median_px": 30.0,       # backstop: a wrong K can make sigmas meaninglessly large
    "min_lever_mm": 2.0,
    "max_lever_mm": 300.0,       # two boxes on one bracket: centimetres, not metres
    "max_mount_tilt_deg": 25.0,  # past this it is a frame convention error, not a mount
}

WARN_LEVER_MM = (10.0, 200.0)
WARN_TZ_MM = 30.0        # along-axis lever arm that is really an uncalibrated range bias
WARN_LEVER_STD_MM = 15.0
WARN_EL_SPAN_DEG = 10.0


# ------------------------------------------------------------------ geometry


def _pts3(a):
    return np.ascontiguousarray(np.asarray(a, np.float64).reshape(-1, 3))


def _pts2(a):
    return np.ascontiguousarray(np.asarray(a, np.float64).reshape(-1, 2))


def _dist5(dist):
    """Distortion coefficients as OpenCV wants them; None means a pinhole."""
    if dist is None:
        return np.zeros(5)
    return np.ascontiguousarray(np.ravel(np.asarray(dist, np.float64)))


def rotation_angle_deg(A, B):
    """Angle of the rotation that takes A to B, in degrees."""
    c = (np.trace(np.asarray(A, np.float64) @ np.asarray(B, np.float64).T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def mount_tilt_deg(R):
    """How far the recovered mount is from the ideal axis permutation.

    This is the one number that separates "the bracket is 2 degrees off" from
    "the radar points were handed over in TI's frame". Both fit; only one is a
    mount.
    """
    return rotation_angle_deg(R, R_CANONICAL)


def verify_convention_against_mmwave():
    """Cross-check R_CANONICAL against the parser that produces the points.

    The convention is stated in this file on purpose - a calibration tool that
    cannot be read without opening the driver is a tool nobody reads. But the two
    must agree, so if radar/mmwave.py is importable, ask it: feed ti_to_project a
    TI-frame unit vector pointing forward and confirm R_CANONICAL sends the result
    down the camera's +z. Returns None when mmwave.py is not on the path, which is
    the normal case for a host-side tool run from a copy of this directory.
    """
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    path = os.path.normpath(os.path.join(here, "..", "..", "radar"))
    if not os.path.exists(os.path.join(path, "mmwave.py")):
        return None
    saved = list(sys.path)
    try:
        sys.path.insert(0, path)
        import mmwave  # noqa: E402
    except ImportError:
        return None
    finally:
        sys.path[:] = saved

    checks = []
    for ti, cam_expected in (((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),    # TI forward -> cam +z
                             ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),    # TI right   -> cam +x
                             ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0))):  # TI up      -> cam -y
        p = np.array(mmwave.ti_to_project(*ti), np.float64)
        checks.append(bool(np.allclose(R_CANONICAL @ p, np.array(cam_expected), atol=1e-12)))
    return all(checks)


def distortion_valid_radius(dist, r_max=3.0):
    """Largest normalised radius where r -> r*radial(r) is still increasing.

    Beyond it the Brown-Conrady polynomial folds: a target 60 deg off axis comes
    back with a plausible in-frame pixel. cv2.projectPoints will hand you that
    pixel without comment, which is how a radar detection well outside the
    camera's cone ends up drawn on top of the scene.
    """
    d = np.zeros(5)
    v = _dist5(dist)
    d[:min(5, len(v))] = v[:5]
    k1, k2, _, _, k3 = d
    r = np.linspace(1e-6, r_max, 4000)
    f = r * (1.0 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    bad = np.nonzero(np.diff(f) <= 0)[0]
    return float(r[bad[0]]) if len(bad) else float(r_max)


def project(points_radar, R, t, K, dist=None, image_size=None):
    """Radar points -> (uv, z_cam, status). Nothing is clamped, ever.

    uv is NaN wherever no pixel exists (behind the camera, or past the radius
    where the distortion model folds); it is the true, possibly far out-of-frame
    coordinate where a pixel does exist but lies outside the sensor. status says
    which, using the PROJ_* codes.

    image_size=(W, H) is what makes PROJ_OUTSIDE decidable. Without it every
    point in front is reported PROJ_OK, so pass it - load_calib carries it for
    exactly this reason.
    """
    P = _pts3(points_radar)
    R = np.asarray(R, np.float64).reshape(3, 3)
    t = np.asarray(t, np.float64).reshape(3)
    K = np.asarray(K, np.float64).reshape(3, 3)
    dist = _dist5(dist)

    n = len(P)
    uv = np.full((n, 2), np.nan)
    status = np.full(n, PROJ_BEHIND, np.int32)
    if n == 0:
        return uv, np.zeros(0), status

    cam = P @ R.T + t
    z = cam[:, 2].copy()
    front = z > MIN_Z_M
    if not front.any():
        return uv, z, status

    norm = cam[front, :2] / z[front, None]
    # Project only the points in front. cv2.projectPoints divides by z whatever
    # its sign, so a target behind the sensor comes back point-mirrored into the
    # frame - a real detection at a real pixel that is simply not there.
    proj, _ = cv2.projectPoints(np.ascontiguousarray(cam[front]),
                                np.zeros(3), np.zeros(3), K, dist)
    sub = np.full((int(front.sum()), 2), np.nan)
    sub_status = np.full(int(front.sum()), PROJ_OK, np.int32)

    r_ok = np.linalg.norm(norm, axis=1) <= distortion_valid_radius(dist)
    sub[r_ok] = proj.reshape(-1, 2)[r_ok]
    sub_status[~r_ok] = PROJ_UNMODELLED

    if image_size is not None:
        w, h = int(image_size[0]), int(image_size[1])
        # Judged on the undistorted pixel as well: inside the fold radius the two
        # agree, and outside it the distorted one is meaningless anyway.
        ideal = np.stack([K[0, 0] * norm[:, 0] + K[0, 1] * norm[:, 1] + K[0, 2],
                          K[1, 1] * norm[:, 1] + K[1, 2]], -1)
        inside = (np.isfinite(sub).all(1)
                  & (sub[:, 0] >= 0) & (sub[:, 0] <= w - 1)
                  & (sub[:, 1] >= 0) & (sub[:, 1] <= h - 1)
                  & (ideal[:, 0] >= 0) & (ideal[:, 0] <= w - 1)
                  & (ideal[:, 1] >= 0) & (ideal[:, 1] <= h - 1))
        sub_status[(sub_status == PROJ_OK) & ~inside] = PROJ_OUTSIDE

    uv[front] = sub
    status[front] = sub_status
    return uv, z, status


def project_calib(points_radar, calib):
    """project() plus the range window the calibration was actually fitted over.

    A detection at 8 m still projects to a pixel; that pixel has never been
    checked against anything. It is marked PROJ_OUT_OF_RANGE rather than dropped,
    so a caller can choose to draw it dimmed instead of pretending it is as good
    as the rest.
    """
    uv, z, status = project(points_radar, calib["R"], calib["t"], calib["K"],
                            calib.get("dist"), calib.get("image_size"))
    lo, hi = calib.get("working_range_m") or (RADAR_MIN_RANGE_M, RADAR_MAX_RANGE_M)
    rng = np.linalg.norm(_pts3(points_radar), axis=1)
    out = (rng < lo) | (rng > hi)
    status[out & (status == PROJ_OK)] = PROJ_OUT_OF_RANGE
    return uv, z, status


def project_unknown_elevation(points_xy_radar, R, t, K, dist=None, image_size=None,
                              z_span_m=(-1.0, 1.5), samples=17):
    """A detection with no elevation is a LINE in the image, not a point.

    The IWR1843's elevation estimate comes from a two-element array and is absent
    entirely for any detection the AoA stage did not resolve in that dimension.
    Choosing a height silently (z=0 is the usual choice) puts the marker on the
    floor at 4 m and on someone's chest at 0.8 m. This returns the bounding box
    of every pixel the detection could occupy for z anywhere in z_span_m, and a
    validity flag; a caller that needs a single point must state its own height
    assumption rather than inherit one from here.
    """
    xy = np.asarray(points_xy_radar, np.float64).reshape(-1, 2)
    zs = np.linspace(z_span_m[0], z_span_m[1], samples)
    boxes = np.full((len(xy), 4), np.nan)
    ok = np.zeros(len(xy), bool)
    for i, (x, y) in enumerate(xy):
        pts = np.stack([np.full(samples, x), np.full(samples, y), zs], -1)
        uv, _, st = project(pts, R, t, K, dist, image_size)
        good = np.isfinite(uv).all(1) & (st != PROJ_BEHIND) & (st != PROJ_UNMODELLED)
        if not good.any():
            continue
        g = uv[good]
        boxes[i] = [g[:, 0].min(), g[:, 0].max(), g[:, 1].min(), g[:, 1].max()]
        ok[i] = True
    return boxes, ok


# ------------------------------------------------------------------ noise model


def pixel_sigma(K, sigma_az_deg=RADAR_SIGMA_AZ_DEG, sigma_el_deg=RADAR_SIGMA_EL_DEG):
    """Radar angular sigma -> the pixel sigma it produces, per image axis.

    A cross-range error of r*sigma at range r lands at f*r*sigma/z_cam pixels, and
    for a lever arm of centimetres z_cam == r to a part in fifty, so the r cancels:
    the expected pixel residual is f*sigma at every distance. That is why an
    unweighted pixel cost is defensible here at all - what is NOT defensible is
    treating u and v alike, since sigma_el is several times sigma_az.

    Assumes the near-canonical mount, i.e. radar azimuth drives u and elevation
    drives v. Off by a mount tilt of a few degrees, which is far inside the
    accuracy of the sigmas themselves.
    """
    K = np.asarray(K, np.float64)
    return (float(K[0, 0] * np.radians(sigma_az_deg)),
            float(K[1, 1] * np.radians(sigma_el_deg)))


def _whiten(K, pixels, sigma_uv):
    """Scale the image axes so one unit is one radar sigma on that axis.

    Scaling u by a and v by b is exactly K -> diag(a, b, 1) @ K, and distortion
    lives in normalised coordinates upstream of K, so this is an exact change of
    variables rather than an approximation: PnP in the whitened frame minimises
    the true Mahalanobis cost and the RANSAC threshold is in sigmas on both axes.
    """
    a, b = 1.0 / sigma_uv[0], 1.0 / sigma_uv[1]
    S = np.diag([a, b, 1.0])
    return S @ np.asarray(K, np.float64), pixels * np.array([a, b])


def residual_stats(radar_points, pixel_points, R, t, K, dist=None,
                   sigma_uv=None, image_size=None):
    """Per-correspondence residuals in pixels AND in radar sigmas.

    Per-correspondence, not a mean: a mean of 3 px hides one point at 40 px, and
    that one point is either a mis-clicked pixel or a ghost detection - both of
    which are worth finding before they are averaged away.
    """
    radar = _pts3(radar_points)
    pix = _pts2(pixel_points)
    if sigma_uv is None:
        sigma_uv = pixel_sigma(K)
    uv, z, status = project(radar, R, t, K, dist, image_size)

    d = uv - pix
    err = np.linalg.norm(d, axis=1)
    norm = np.linalg.norm(d / np.asarray(sigma_uv), axis=1)
    fin = np.isfinite(err)

    def stat(v):
        v = v[fin]
        if not len(v):
            return {"median": None, "mean": None, "p90": None, "p95": None, "max": None}
        return {"median": float(np.median(v)), "mean": float(v.mean()),
                "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
                "max": float(v.max())}

    return {
        "n": int(len(radar)),
        "n_unprojectable": int((~fin).sum()),
        "px": stat(err),
        "sigma": stat(norm),
        # Split by axis because the two axes carry different information: a u-only
        # residual pattern is an azimuth/rotation problem, a v-only one is the
        # elevation array doing what it does.
        "du_px": stat(np.abs(d[:, 0])), "dv_px": stat(np.abs(d[:, 1])),
        "per_point": [
            {"px": (float(err[i]) if fin[i] else None),
             "sigma": (float(norm[i]) if fin[i] else None),
             "du": (float(d[i, 0]) if fin[i] else None),
             "dv": (float(d[i, 1]) if fin[i] else None),
             "range_m": float(np.linalg.norm(radar[i])),
             "z_cam_m": float(z[i]), "status": int(status[i])}
            for i in range(len(radar))],
        "sigma_uv_px": [float(sigma_uv[0]), float(sigma_uv[1])],
    }


# ------------------------------------------------------------------ solve


def _pnp(radar, pix, Kw, dist, rvec=None, tvec=None):
    """SQPNP then LM, in the whitened frame. Returns fresh (rvec, tvec).

    The copies are not defensive tidiness: solvePnPRefineLM takes rvec and tvec
    as InputOutputArrays and writes through them, so passing the same seed pose
    into a loop silently turns independent fits into a chain, and returning the
    result unpacked hands every caller a view of one buffer. The bootstrap did
    both, and reported a lever-arm uncertainty of exactly zero.
    """
    if rvec is None:
        ok, rvec, tvec = cv2.solvePnP(radar.reshape(-1, 1, 3), pix.reshape(-1, 1, 2),
                                      Kw, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            raise RuntimeError("solvePnP failed on %d correspondences" % len(radar))
    rvec, tvec = cv2.solvePnPRefineLM(
        radar.reshape(-1, 1, 3), pix.reshape(-1, 1, 2), Kw, dist,
        np.array(rvec, np.float64).reshape(3, 1),
        np.array(tvec, np.float64).reshape(3, 1))
    return np.array(rvec, np.float64), np.array(tvec, np.float64)


def solve(radar_points, pixel_points, K, dist=None,
          sigma_az_deg=RADAR_SIGMA_AZ_DEG, sigma_el_deg=RADAR_SIGMA_EL_DEG,
          ransac_sigma=3.0, image_size=None, bootstrap_n=64, seed=1):
    """Correspondences -> (R, t, stats). P_cam = R @ P_radar + t.

    RANSAC first because a hand-built correspondence set always contains a ghost:
    a multipath return, a tripod leg, or the pixel of the reflector clicked one
    frame after it was moved. Refinement then runs on the inliers only, so the
    reported residual describes the fit rather than the outlier.

    The RANSAC threshold is in radar sigmas, not pixels, and it is applied in the
    whitened frame - 3 sigma on an axis where sigma is 24 px and on one where it
    is 8 px, instead of one pixel threshold that silently discards a third of the
    honest elevation measurements or admits every azimuth blunder.
    """
    radar = _pts3(radar_points)
    pix = _pts2(pixel_points)
    if len(radar) != len(pix):
        raise ValueError("%d radar points against %d pixels" % (len(radar), len(pix)))
    if len(radar) < 6:
        raise ValueError("solvePnP needs >= 6 correspondences, got %d" % len(radar))
    K = np.asarray(K, np.float64).reshape(3, 3)
    dist = _dist5(dist)

    sigma_uv = pixel_sigma(K, sigma_az_deg, sigma_el_deg)
    Kw, pixw = _whiten(K, pix, sigma_uv)

    # RANSAC draws its own random subsets; without a fixed seed the same input
    # gives a slightly different calibration every run, which makes any argument
    # about whether a rig moved unanswerable.
    cv2.setRNGSeed(int(seed))
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        radar.reshape(-1, 1, 3), pixw.reshape(-1, 1, 2), Kw, dist,
        reprojectionError=float(ransac_sigma), iterationsCount=2000,
        confidence=0.9999, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError("solvePnPRansac found no consistent pose")

    inliers = np.zeros(len(radar), bool)
    inliers[np.ravel(inl).astype(int) if inl is not None else np.arange(len(radar))] = True
    if inliers.sum() >= 6:
        rvec, tvec = _pnp(radar[inliers], pixw[inliers], Kw, dist, rvec, tvec)

    R = cv2.Rodrigues(rvec)[0]
    t = np.ravel(tvec).astype(np.float64)

    stats = residual_stats(radar[inliers], pix[inliers], R, t, K, dist,
                           sigma_uv, image_size)
    stats["all_points"] = residual_stats(radar, pix, R, t, K, dist, sigma_uv, image_size)
    stats["n_correspondences"] = int(len(radar))
    stats["n_inliers"] = int(inliers.sum())
    stats["inlier_mask"] = inliers.tolist()
    stats["lever_arm_mm"] = float(np.linalg.norm(t) * 1000.0)
    stats["t_mm"] = (t * 1000.0).tolist()
    stats["mount_tilt_deg"] = mount_tilt_deg(R)
    stats["radar_noise_model"] = {"sigma_az_deg": sigma_az_deg,
                                  "sigma_el_deg": sigma_el_deg,
                                  "sigma_u_px": sigma_uv[0], "sigma_v_px": sigma_uv[1]}

    rng = np.linalg.norm(radar[inliers], axis=1)
    stats["working_range_m"] = [float(rng.min()), float(rng.max())]
    stats["depth_bins_m"] = sorted(set(float(np.round(r / DEPTH_BIN_M) * DEPTH_BIN_M)
                                       for r in rng))
    az = np.degrees(np.arctan2(radar[inliers, 1], radar[inliers, 0]))
    el = np.degrees(np.arcsin(np.clip(radar[inliers, 2] / np.maximum(rng, 1e-9), -1, 1)))
    stats["az_span_deg"] = float(az.max() - az.min())
    stats["el_span_deg"] = float(el.max() - el.min())

    if bootstrap_n:
        stats["uncertainty"] = _bootstrap(radar[inliers], pixw[inliers], Kw, dist,
                                          rvec, tvec, bootstrap_n, seed)
    return R, t, stats


def _bootstrap(radar, pixw, Kw, dist, rvec, tvec, n, seed):
    """Resample the inliers to say how well t is actually pinned down.

    Worth the milliseconds: the parallax that determines t is f*|t|/z pixels -
    about 15 px at 2 m for a 65 mm lever arm - against a per-point noise of a
    similar size. The rotation averages down with sqrt(N) and comes out tight;
    the translation frequently does not, and a calibration that reports t to the
    millimetre when the data only supports a centimetre is the kind of number
    that later gets defended in a meeting.
    """
    gen = np.random.default_rng(seed)
    ts, angs = [], []
    R0 = cv2.Rodrigues(rvec)[0]
    for _ in range(int(n)):
        idx = gen.integers(0, len(radar), len(radar))
        if len(np.unique(idx)) < 6:
            continue
        try:
            rv, tv = _pnp(radar[idx], pixw[idx], Kw, dist, rvec, tvec)
        except cv2.error:
            continue
        ts.append(np.ravel(tv).copy())
        angs.append(rotation_angle_deg(cv2.Rodrigues(rv)[0], R0))
    if len(ts) < 8:
        return None
    ts = np.array(ts)
    return {"n": len(ts),
            "t_std_mm": (ts.std(0) * 1000.0).tolist(),
            "lever_arm_std_mm": float(np.linalg.norm(ts, axis=1).std() * 1000.0),
            "rotation_std_deg": float(np.std(angs))}


# ------------------------------------------------------------------ quality gate


def _spread(pixel_points, image_size):
    pix = _pts2(pixel_points)
    w, h = float(image_size[0]), float(image_size[1])
    hull = cv2.convexHull(pix.astype(np.float32).reshape(-1, 1, 2))
    return {"u_span_frac": float((pix[:, 0].max() - pix[:, 0].min()) / w),
            "v_span_frac": float((pix[:, 1].max() - pix[:, 1].min()) / h),
            "hull_frac": float(cv2.contourArea(hull) / (w * h))}


def gate(radar_points, pixel_points, stats, image_size, limits=None):
    """Refuse a fit rather than return a bad one.

    Every limit here exists because the fit that violates it still converges and
    still reports a small residual. Six points clustered at one distance produce
    a beautiful reprojection error and a transform that is wrong everywhere the
    reflector was not standing.

    Returns {"passed", "failures", "warnings", "metrics", "limits"}. Failures are
    reasons to redo the collection; warnings are things to write down.
    """
    lim = dict(GATE)
    lim.update(limits or {})
    radar = _pts3(radar_points)
    pix = _pts2(pixel_points)
    inl = np.array(stats["inlier_mask"], bool)

    f, w = [], []
    m = {}

    m["n_inliers"] = int(inl.sum())
    m["inlier_frac"] = float(inl.mean())
    if m["n_inliers"] < lim["min_correspondences"]:
        f.append("only %d inlier correspondences, need %d"
                 % (m["n_inliers"], lim["min_correspondences"]))
    if m["inlier_frac"] < lim["min_inlier_frac"]:
        f.append("RANSAC kept only %.0f%% of the correspondences; the set has more "
                 "ghosts than fit" % (100 * m["inlier_frac"]))

    m.update(_spread(pix[inl] if inl.any() else pix, image_size))
    if m["u_span_frac"] < lim["min_u_span_frac"]:
        f.append("points span %.0f%% of image width, need %.0f%%"
                 % (100 * m["u_span_frac"], 100 * lim["min_u_span_frac"]))
    if m["v_span_frac"] < lim["min_v_span_frac"]:
        f.append("points span %.0f%% of image height, need %.0f%%"
                 % (100 * m["v_span_frac"], 100 * lim["min_v_span_frac"]))
    if m["hull_frac"] < lim["min_hull_frac"]:
        f.append("points cover %.1f%% of the frame area, need %.1f%%"
                 % (100 * m["hull_frac"], 100 * lim["min_hull_frac"]))

    rng = np.linalg.norm(radar[inl], axis=1) if inl.any() else np.linalg.norm(radar, axis=1)
    m["n_depth_bins"] = len(stats.get("depth_bins_m") or [])
    m["depth_ratio"] = float(rng.max() / max(rng.min(), 1e-6))
    m["depth_span_m"] = float(rng.max() - rng.min())
    # Depth spread is what separates the rotation from the lever arm. At one
    # distance the two trade off almost freely and the fit is exact there and
    # nowhere else - the single most common way a radar extrinsic is wrong.
    if m["n_depth_bins"] < lim["min_depth_bins"]:
        f.append("correspondences at %d distinct distance(s), need %d"
                 % (m["n_depth_bins"], lim["min_depth_bins"]))
    if m["depth_ratio"] < lim["min_depth_ratio"]:
        f.append("far/near distance ratio %.2f, need %.2f"
                 % (m["depth_ratio"], lim["min_depth_ratio"]))
    if m["depth_span_m"] < lim["min_depth_span_m"]:
        f.append("depth span %.2f m, need %.2f m"
                 % (m["depth_span_m"], lim["min_depth_span_m"]))

    m["median_px"] = stats["px"]["median"]
    m["median_sigma"] = stats["sigma"]["median"]
    m["p95_sigma"] = stats["sigma"]["p95"]
    if m["median_sigma"] is None:
        f.append("no correspondence projects to a pixel at all")
    else:
        if m["median_sigma"] > lim["max_median_sigma"]:
            f.append("median residual %.2f radar sigma, limit %.2f"
                     % (m["median_sigma"], lim["max_median_sigma"]))
        if m["p95_sigma"] > lim["max_p95_sigma"]:
            f.append("p95 residual %.2f radar sigma, limit %.2f"
                     % (m["p95_sigma"], lim["max_p95_sigma"]))
        if m["median_px"] > lim["max_median_px"]:
            f.append("median residual %.1f px, limit %.1f (a residual this large in "
                     "absolute terms usually means the intrinsics are not this camera's)"
                     % (m["median_px"], lim["max_median_px"]))

    m["lever_arm_mm"] = stats["lever_arm_mm"]
    m["mount_tilt_deg"] = stats["mount_tilt_deg"]
    if not lim["min_lever_mm"] <= m["lever_arm_mm"] <= lim["max_lever_mm"]:
        f.append("lever arm %.0f mm is not physically plausible for two sensors on "
                 "one bracket (allowed %.0f..%.0f mm)"
                 % (m["lever_arm_mm"], lim["min_lever_mm"], lim["max_lever_mm"]))
    if m["mount_tilt_deg"] > lim["max_mount_tilt_deg"]:
        # The classic: radar points handed over in TI's frame instead of the
        # project frame is a 90 deg rotation about z, and it fits perfectly.
        f.append("recovered mount is %.1f deg from the canonical axis permutation "
                 "(limit %.1f): this is a coordinate-frame error, not a bracket"
                 % (m["mount_tilt_deg"], lim["max_mount_tilt_deg"]))

    if m["lever_arm_mm"] < WARN_LEVER_MM[0] or m["lever_arm_mm"] > WARN_LEVER_MM[1]:
        w.append("lever arm %.0f mm is at the edge of plausible; measure it with a "
                 "ruler and compare" % m["lever_arm_mm"])
    tz = stats["t_mm"][2]
    if abs(tz) > WARN_TZ_MM:
        # The fit cannot tell a radar genuinely set back from the camera plane
        # from an uncalibrated range bias absorbed along the line of sight, and
        # the two behave differently at other distances. Only a ruler can.
        w.append("t_z = %.0f mm along the optical axis. Either the radar really "
                 "sits that far behind the camera, or an uncalibrated radar range "
                 "bias has been absorbed as a fake lever arm - the fit cannot tell "
                 "these apart. Measure it, and run compRangeBiasAndRxChanPhase "
                 "before trusting close-range parallax" % tz)
    unc = stats.get("uncertainty")
    if unc and unc["lever_arm_std_mm"] > WARN_LEVER_STD_MM:
        w.append("bootstrap says the lever arm is only known to +-%.0f mm; do not "
                 "quote t to the millimetre and expect close-range parallax to hold"
                 % unc["lever_arm_std_mm"])
    if stats.get("el_span_deg", 0) < WARN_EL_SPAN_DEG:
        w.append("targets span only %.1f deg in elevation; rotation about the "
                 "camera x axis is weakly observed" % stats.get("el_span_deg", 0))
    if rng.min() < RADAR_MIN_RANGE_M:
        w.append("a correspondence at %.2f m is inside the 0.375 m IF high-pass "
                 "blind zone; the radar cannot have detected it cleanly" % rng.min())
    if rng.max() > RADAR_MAX_RANGE_M:
        w.append("a correspondence at %.2f m is past the %.1f m cfarFovCfg range gate"
                 % (rng.max(), RADAR_MAX_RANGE_M))

    return {"passed": not f, "failures": f, "warnings": w, "metrics": m, "limits": lim}


# ------------------------------------------------------------------ held-out check


def holdout_by_depth(radar_points, pixel_points, K, dist=None, image_size=None,
                     bin_m=DEPTH_BIN_M, **kw):
    """Leave one distance out, fit on the rest, report the error where it was blind.

    The in-sample residual answers "did the optimiser converge". This answers the
    only question that matters downstream: does the transform still hold at a
    distance it has never seen? Extrapolating in depth is where a lever arm error
    shows up, because the parallax term is f*|t|/z - largest exactly where a
    close-range inspection pipeline works.
    """
    radar = _pts3(radar_points)
    pix = _pts2(pixel_points)
    rng = np.linalg.norm(radar, axis=1)
    key = np.round(rng / bin_m).astype(int)

    out = []
    for k in sorted(set(key.tolist())):
        test = key == k
        fit = ~test
        if fit.sum() < 6 or test.sum() < 2:
            continue
        try:
            R, t, st = solve(radar[fit], pix[fit], K, dist, image_size=image_size,
                             bootstrap_n=0, **kw)
        except (ValueError, RuntimeError, cv2.error) as e:
            out.append({"depth_m": float(k * bin_m), "n_test": int(test.sum()),
                        "error": str(e)})
            continue
        nm = st["radar_noise_model"]
        held = residual_stats(radar[test], pix[test], R, t, K, dist,
                              (nm["sigma_u_px"], nm["sigma_v_px"]), image_size)
        out.append({
            "depth_m": float(k * bin_m),
            "range_m": [float(rng[test].min()), float(rng[test].max())],
            "n_fit": int(fit.sum()), "n_test": int(test.sum()),
            "fit_median_px": st["px"]["median"], "fit_median_sigma": st["sigma"]["median"],
            "held_median_px": held["px"]["median"], "held_max_px": held["px"]["max"],
            "held_median_sigma": held["sigma"]["median"],
            "lever_arm_mm": st["lever_arm_mm"],
            # A calibration whose rotation swings when one distance is removed is
            # being driven by that distance, whatever its residual says.
            "mount_tilt_deg": st["mount_tilt_deg"],
        })
    return out


# ------------------------------------------------------------------ fov overlap


def fov_overlap_mask(R, t, K, dist=None, image_size=(640, 400),
                     range_m=(RADAR_MIN_RANGE_M, RADAR_MAX_RANGE_M),
                     az_limit_deg=RADAR_AZ_LIMIT_DEG, el_limit_deg=RADAR_EL_LIMIT_DEG,
                     samples=9, step=2):
    """Pixels the radar can ever illuminate: the fusable region, as a mask.

    Everything outside it is camera-only and must be masked, not extrapolated -
    a radar overlay drawn over the whole frame implies coverage the sensor does
    not have. The mask is range-dependent because the lever arm shifts the cone
    slightly; it is computed as the union over the range window, i.e. "could ever
    be seen", which is the permissive reading. Sample a single range for the
    strict one.
    """
    w, h = int(image_size[0]), int(image_size[1])
    gu, gv = np.meshgrid(np.arange(0, w, step, np.float32),
                         np.arange(0, h, step, np.float32))
    pix = np.stack([gu.ravel(), gv.ravel()], -1)
    rays = cv2.undistortPoints(pix.reshape(-1, 1, 2), np.asarray(K, np.float64),
                               _dist5(dist)).reshape(-1, 2)
    d = np.concatenate([rays, np.ones((len(rays), 1))], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    R = np.asarray(R, np.float64).reshape(3, 3)
    t = np.asarray(t, np.float64).reshape(3)
    small = np.zeros(len(rays), bool)
    for s in np.linspace(range_m[0], range_m[1], samples):
        # inverse of P_cam = R P_radar + t, written for row vectors:
        # P_radar = R.T (P_cam - t)  ==  (P_cam - t) @ R
        p = (d * s - t) @ R
        r = np.linalg.norm(p, axis=1)
        az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        el = np.degrees(np.arcsin(np.clip(p[:, 2] / np.maximum(r, 1e-9), -1, 1)))
        small |= ((np.abs(az) <= az_limit_deg) & (np.abs(el) <= el_limit_deg)
                  & (r >= range_m[0]) & (r <= range_m[1]))

    mask = cv2.resize(small.reshape(gu.shape).astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)
    return mask, float(mask.mean())


# ------------------------------------------------------------------ persistence


def _sha256_file(path):
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def provenance(correspondence_path=None, correspondences=None,
               intrinsics_path=None, rig_id=None, mount_token=None, notes=None):
    """Enough to tell whether this file still describes the hardware in front of you.

    The three digests are the load-bearing part. `intrinsics_sha256` matters most:
    R and t are only meaningful against the K they were fitted with, so a re-run
    of calib.py silently invalidates this file, and nothing about the numbers
    themselves would show it. `mount_token` is the manual half - extrinsics are a
    property of the assembly, so anyone who unbolts anything sets a new token and
    every calib carrying the old one is stale by definition.
    """
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "user": (os.environ.get("USER") or getpass.getuser() or "?"),
        "tool": os.path.basename(__file__ if "__file__" in globals() else "?"),
        "tool_sha256": _sha256_file(__file__) if "__file__" in globals() else None,
        "versions": {"python": platform.python_version(),
                     "cv2": cv2.__version__, "numpy": np.__version__},
        "correspondences_file": correspondence_path,
        "correspondences_sha256": (_sha256_obj(correspondences)
                                   if correspondences is not None else None),
        "intrinsics_file": intrinsics_path,
        "intrinsics_sha256": _sha256_file(intrinsics_path) if intrinsics_path else None,
        "rig_id": rig_id,
        "mount_token": mount_token,
        "notes": notes,
        "invalidated_by": [
            "any mechanical change: unbolting, re-seating or re-aiming either sensor",
            "re-running calib.py (K and dist change, R and t do not follow)",
            "a radar config change that moves the range bias "
            "(compRangeBiasAndRxChanPhase, profileCfg)",
            "a large change in radar die temperature vs collection, which moves "
            "range bias and with it the apparent lever arm",
        ],
    }


def save_calib(path, R, t, K, dist, image_size, stats, gate_result,
               camera=None, holdout=None, meta=None):
    """Write the calib file. Same shape and spirit as calib.py's: flat JSON,
    lists not arrays, every number that a reader would otherwise have to guess.

    The per-correspondence residuals are kept, not just the summary, because the
    question asked six months later is always "which point was bad", and by then
    the correspondence file is gone.
    """
    out = {
        "schema": SCHEMA,
        "frames": FRAMES,
        "camera": camera,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "K": np.asarray(K, np.float64).tolist(),
        "dist": _dist5(dist).tolist(),
        "R": np.asarray(R, np.float64).tolist(),
        "t": np.ravel(np.asarray(t, np.float64)).tolist(),
        "lever_arm_mm": stats["lever_arm_mm"],
        "t_mm": stats["t_mm"],
        "mount_tilt_deg": stats["mount_tilt_deg"],
        "n_correspondences": stats["n_correspondences"],
        "n_inliers": stats["n_inliers"],
        "inlier_mask": stats["inlier_mask"],
        "residual_px": {k: stats["px"][k] for k in ("median", "mean", "p90", "p95", "max")},
        "residual_sigma": {k: stats["sigma"][k]
                           for k in ("median", "mean", "p90", "p95", "max")},
        "residual_du_px": stats["du_px"], "residual_dv_px": stats["dv_px"],
        "per_correspondence": stats["per_point"],
        "radar_noise_model": stats["radar_noise_model"],
        "working_range_m": stats["working_range_m"],
        "depth_bins_m": stats["depth_bins_m"],
        "az_span_deg": stats["az_span_deg"], "el_span_deg": stats["el_span_deg"],
        "uncertainty": stats.get("uncertainty"),
        "gate": gate_result,
        "holdout": holdout,
        # Not a plane homography: (R, t) with a per-point range is exact at any
        # distance. The limit is evidential, and it is working_range_m above.
        "depth_assumption": "none - full 3D projection; validated only over "
                            "working_range_m, extrapolation beyond it is unchecked",
        "provenance": provenance(**(meta or {})),
    }
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def load_calib(path, allow_failed_gate=False):
    """Read a calib file back as numpy, refusing one that never passed its gate.

    A calibration that a human overrode with --force is exactly the one that ends
    up loaded by a script that never saw the warning. Refusing here means the
    override has to be repeated at the point of use.
    """
    with open(path, "r") as fh:
        d = json.load(fh)
    if d.get("schema") != SCHEMA:
        raise ValueError("%s: schema %r, expected %r" % (path, d.get("schema"), SCHEMA))
    g = d.get("gate") or {}
    if not g.get("passed", False) and not allow_failed_gate:
        raise ValueError("%s did not pass its quality gate and must not be used: %s"
                         % (path, "; ".join(g.get("failures") or ["no gate recorded"])))
    for key in ("K", "R", "t", "dist", "image_size"):
        if key not in d:
            raise ValueError("%s: missing %r" % (path, key))
    d["K"] = np.asarray(d["K"], np.float64).reshape(3, 3)
    d["R"] = np.asarray(d["R"], np.float64).reshape(3, 3)
    d["t"] = np.asarray(d["t"], np.float64).reshape(3)
    d["dist"] = np.asarray(d["dist"], np.float64)
    d["image_size"] = tuple(int(v) for v in d["image_size"])

    # A file can be well-formed and still not describe a rotation, e.g. after a
    # hand edit. Catch it here rather than as a slow drift in the overlay.
    if abs(np.linalg.det(d["R"]) - 1.0) > 1e-6 or \
            np.abs(d["R"] @ d["R"].T - np.eye(3)).max() > 1e-6:
        raise ValueError("%s: R is not a rotation matrix" % path)
    return d


def stale_reasons(calib, intrinsics_path=None, mount_token=None):
    """Why this calibration might no longer describe the hardware. Empty is good."""
    p = calib.get("provenance") or {}
    out = []
    if intrinsics_path:
        now = _sha256_file(intrinsics_path)
        was = p.get("intrinsics_sha256")
        if now is None:
            out.append("intrinsics file %s is gone" % intrinsics_path)
        elif was and now != was:
            out.append("intrinsics %s have been re-solved since this fit (%s -> %s); "
                       "R and t are meaningless against a different K"
                       % (intrinsics_path, (was or "?")[:8], now[:8]))
        elif not was:
            out.append("no intrinsics digest recorded, so a re-solve cannot be detected")
    if mount_token is not None and p.get("mount_token") != mount_token:
        out.append("mount token %r, rig now says %r: the assembly has changed"
                   % (p.get("mount_token"), mount_token))
    if not p.get("mount_token"):
        out.append("no mount token recorded; a mechanical change cannot be detected")
    if "__file__" in globals():
        now = _sha256_file(__file__)
        if p.get("tool_sha256") and now and now != p["tool_sha256"]:
            out.append("written by a different version of radar_extrinsics.py")
    return out


def load_correspondences(path):
    """-> (radar Nx3, pixel Nx2, meta). Radar points must be in the PROJECT frame."""
    with open(path, "r") as fh:
        d = json.load(fh)
    cs = d["correspondences"]
    radar = _pts3([c["radar"] for c in cs])
    pix = _pts2([c["pixel"] for c in cs])
    meta = {k: v for k, v in d.items() if k != "correspondences"}
    if "image_size" not in meta:
        raise ValueError("%s: image_size is required - the spread gate cannot be "
                         "evaluated without it" % path)
    return radar, pix, meta, d


def intrinsics_from_calib(path, camera):
    """Pull (K, dist, image_size) out of a calib.py calib.json."""
    with open(path, "r") as fh:
        d = json.load(fh)
    if camera == "rgb":
        return (np.asarray(d["K_rgb"], np.float64), np.asarray(d["dist_rgb"], np.float64))
    if camera in ("thermal", "th"):
        return (np.asarray(d["K_th"], np.float64), np.asarray(d["dist_th"], np.float64))
    raise ValueError("camera must be rgb or thermal, got %r" % camera)


# ------------------------------------------------------------------ reporting


def summarise(stats, gate_result, prefix="  "):
    lines = []
    lines.append("%sinliers %d/%d, lever arm %.1f mm, mount tilt %.2f deg"
                 % (prefix, stats["n_inliers"], stats["n_correspondences"],
                    stats["lever_arm_mm"], stats["mount_tilt_deg"]))
    lines.append("%st = [%.1f, %.1f, %.1f] mm (radar origin in camera frame)"
                 % (prefix, *stats["t_mm"]))
    nm = stats["radar_noise_model"]
    lines.append("%sresidual median %.2f px / %.2f sigma, p95 %.2f px / %.2f sigma, "
                 "max %.2f px" % (prefix, stats["px"]["median"], stats["sigma"]["median"],
                                  stats["px"]["p95"], stats["sigma"]["p95"],
                                  stats["px"]["max"]))
    lines.append("%s1 sigma = %.1f px in u, %.1f px in v (radar az %.1f deg, el %.1f deg)"
                 % (prefix, nm["sigma_u_px"], nm["sigma_v_px"],
                    nm["sigma_az_deg"], nm["sigma_el_deg"]))
    lines.append("%srange %.2f..%.2f m over %d bin(s), az span %.0f deg, el span %.0f deg"
                 % (prefix, stats["working_range_m"][0], stats["working_range_m"][1],
                    len(stats["depth_bins_m"]), stats["az_span_deg"], stats["el_span_deg"]))
    u = stats.get("uncertainty")
    if u:
        lines.append("%sbootstrap: lever arm +-%.1f mm, rotation +-%.3f deg"
                     % (prefix, u["lever_arm_std_mm"], u["rotation_std_deg"]))
    for wmsg in gate_result["warnings"]:
        lines.append("%swarning: %s" % (prefix, wmsg))
    for fmsg in gate_result["failures"]:
        lines.append("%sREJECT: %s" % (prefix, fmsg))
    return "\n".join(lines)


# ------------------------------------------------------------------ cli


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_intrinsics(p):
        p.add_argument("-i", "--intrinsics", required=True,
                       help="calib.json from calib.py")
        p.add_argument("-c", "--camera", default="rgb", choices=["rgb", "thermal"])
        p.add_argument("--sigma-az", type=float, default=RADAR_SIGMA_AZ_DEG)
        p.add_argument("--sigma-el", type=float, default=RADAR_SIGMA_EL_DEG)

    s = sub.add_parser("solve")
    s.add_argument("correspondences")
    s.add_argument("-o", "--out", default="radar_calib.json")
    add_intrinsics(s)
    s.add_argument("--rig-id", default=None)
    s.add_argument("--mount-token", default=None,
                   help="a string that changes whenever anything is unbolted")
    s.add_argument("--notes", default=None)
    s.add_argument("--no-holdout", action="store_true")
    s.add_argument("--force", action="store_true",
                   help="write a calib that failed the gate, marked as failed")

    h = sub.add_parser("holdout")
    h.add_argument("correspondences")
    add_intrinsics(h)

    v = sub.add_parser("validate")
    v.add_argument("calib")
    v.add_argument("correspondences")

    c = sub.add_parser("check")
    c.add_argument("calib")
    c.add_argument("-i", "--intrinsics", default=None)
    c.add_argument("--mount-token", default=None)

    r = sub.add_parser("roi")
    r.add_argument("calib")
    r.add_argument("-o", "--out", default="radar_roi.png")
    r.add_argument("--range", type=float, nargs=2, default=None,
                   help="range window in metres (default: the calib's own)")

    a = ap.parse_args()

    if a.cmd in ("solve", "holdout"):
        radar, pix, meta, raw = load_correspondences(a.correspondences)
        K, dist = intrinsics_from_calib(a.intrinsics, a.camera)
        size = meta["image_size"]

        if a.cmd == "holdout":
            rows = holdout_by_depth(radar, pix, K, dist, size,
                                    sigma_az_deg=a.sigma_az, sigma_el_deg=a.sigma_el)
            print("leave-one-distance-out (fit blind to the tested distance)")
            for row in rows:
                if "error" in row:
                    print("  %.1f m  n=%-3d  FAILED: %s"
                          % (row["depth_m"], row["n_test"], row["error"]))
                    continue
                print("  %.2f-%.2f m  fit n=%-3d (%.2f px) -> held-out n=%-3d median "
                      "%.2f px / %.2f sigma, max %.2f px, lever %.0f mm"
                      % (row["range_m"][0], row["range_m"][1], row["n_fit"],
                         row["fit_median_px"], row["n_test"], row["held_median_px"],
                         row["held_median_sigma"], row["held_max_px"],
                         row["lever_arm_mm"]))
            return 0

        R, t, stats = solve(radar, pix, K, dist, sigma_az_deg=a.sigma_az,
                            sigma_el_deg=a.sigma_el, image_size=size)
        g = gate(radar, pix, stats, size)
        print(summarise(stats, g))

        hold = None
        if not a.no_holdout:
            hold = holdout_by_depth(radar, pix, K, dist, size,
                                    sigma_az_deg=a.sigma_az, sigma_el_deg=a.sigma_el)
            worst = max((r for r in hold if "held_median_px" in r),
                        key=lambda r: r["held_median_px"], default=None)
            if worst:
                print("  worst held-out distance %.2f-%.2f m: median %.2f px / %.2f "
                      "sigma (in-sample there: %.2f px)"
                      % (worst["range_m"][0], worst["range_m"][1],
                         worst["held_median_px"], worst["held_median_sigma"],
                         worst["fit_median_px"]))

        if not g["passed"] and not a.force:
            print("refusing to write a calibration that failed its gate "
                  "(--force writes it marked as failed)", file=sys.stderr)
            return 1

        save_calib(a.out, R, t, K, dist, size, stats, g, camera=a.camera, holdout=hold,
                   meta={"correspondence_path": a.correspondences, "correspondences": raw,
                         "intrinsics_path": a.intrinsics, "rig_id": a.rig_id,
                         "mount_token": a.mount_token, "notes": a.notes})
        print("-> %s%s" % (a.out, "" if g["passed"] else "  (MARKED FAILED)"))
        return 0 if g["passed"] else 1

    if a.cmd == "validate":
        cal = load_calib(a.calib, allow_failed_gate=True)
        radar, pix, meta, _ = load_correspondences(a.correspondences)
        nm = cal["radar_noise_model"]
        st = residual_stats(radar, pix, cal["R"], cal["t"], cal["K"], cal["dist"],
                            (nm["sigma_u_px"], nm["sigma_v_px"]), cal["image_size"])
        rng = np.linalg.norm(radar, axis=1)
        lo, hi = cal["working_range_m"]
        outside = int(((rng < lo) | (rng > hi)).sum())
        print("%d correspondence(s), %.2f..%.2f m (%d outside the fitted %.2f..%.2f m)"
              % (len(radar), rng.min(), rng.max(), outside, lo, hi))
        print("  median %.2f px / %.2f sigma, p95 %.2f px, max %.2f px"
              % (st["px"]["median"], st["sigma"]["median"], st["px"]["p95"],
                 st["px"]["max"]))
        for i, p in enumerate(st["per_point"]):
            if p["px"] is None or p["sigma"] > 3.0:
                print("    #%d at %.2f m: %s" % (i, p["range_m"],
                      "does not project" if p["px"] is None
                      else "%.1f px / %.1f sigma" % (p["px"], p["sigma"])))
        for reason in stale_reasons(cal):
            print("  stale: %s" % reason)
        return 0

    if a.cmd == "check":
        cal = load_calib(a.calib, allow_failed_gate=True)
        g = cal.get("gate") or {}
        print("%s: %s, %d/%d inliers, median %.2f px, lever %.1f mm, tilt %.2f deg"
              % (a.calib, "PASSED" if g.get("passed") else "FAILED GATE",
                 cal.get("n_inliers", 0), cal.get("n_correspondences", 0),
                 (cal.get("residual_px") or {}).get("median") or float("nan"),
                 cal.get("lever_arm_mm") or float("nan"),
                 cal.get("mount_tilt_deg") or float("nan")))
        print("  fitted over %.2f..%.2f m; %s"
              % (cal["working_range_m"][0], cal["working_range_m"][1],
                 cal["depth_assumption"]))
        for f in (g.get("failures") or []):
            print("  REJECT: %s" % f)
        for wmsg in (g.get("warnings") or []):
            print("  warning: %s" % wmsg)
        reasons = stale_reasons(cal, a.intrinsics, a.mount_token)
        for reason in reasons:
            print("  stale: %s" % reason)
        return 1 if (reasons or not g.get("passed")) else 0

    if a.cmd == "roi":
        cal = load_calib(a.calib, allow_failed_gate=True)
        rng = tuple(a.range) if a.range else tuple(cal["working_range_m"])
        mask, cov = fov_overlap_mask(cal["R"], cal["t"], cal["K"], cal["dist"],
                                     cal["image_size"], rng)
        cv2.imwrite(a.out, mask * 255)
        print("radar/camera overlap over %.2f..%.2f m: %.1f%% of the frame -> %s"
              % (rng[0], rng[1], 100 * cov, a.out))
        if cov > 0.999:
            print("  (the radar cone contains the whole camera FOV at this range; "
                  "the real limit here is range, not angle)")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
