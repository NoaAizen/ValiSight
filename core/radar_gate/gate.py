"""Drop the radar's manufactured returns, keep the relevant ones (pure).

The IWR1843 point cloud is roughly half fictitious (corpus: fict% ~59% for the
IWR1843 point cloud). Those ghosts cost the N6 compute and USB bandwidth and
pollute clustering. This module removes the most-likely-fake points cheaply,
on-device, and *reports how much it cut* — the efficiency win — while staying
honest about the trade-off: every gate that drops a ghost can also drop a real
weak target (corpus: CFAR already drives missed%). Keep the gates conservative.

Three gates, cheapest first:
  1. absolute reflectivity — drop points whose snr+noise (the ABSOLUTE level
     that follows the range equation, per iwr1843_uart) is below a floor. Weak
     specks near the noise floor are the classic manufactured returns.
  2. FOV — drop points outside the relevant range/azimuth window.
  3. isolated-point rejection — drop points with no neighbour within the cluster
     epsilon; a lone return is almost always a sidelobe/multipath ghost.
"""
import math
from collections import namedtuple

# Absolute received level (snr+noise, dB) below which a point is treated as a
# manufactured/noise-floor return. Conservative — raise to cut more, at the cost
# of missing weak real targets.
MIN_ABS_DB = 8.0
# Only apply the reflectivity gate within this range. Beyond ~4 m the CFAR-
# censored return keeps only 1-2 dB of usable dynamic range (radar_material),
# so an absolute-level gate out there would delete real distant weak targets
# (survivor bias). Farther points pass the reflectivity gate untouched.
REFL_MAX_RANGE_M = 4.0
# Relevant zone (matches cfarFovCfg / aoaFovCfg in the .cfg).
R_MIN_M = 0.25
R_MAX_M = 9.0
AZ_MAX_DEG = 60.0
# Isolation test: a point with no neighbour within this radius is a ghost.
# 1.2 m is the cluster epsilon the corpus settled on.
NEIGHBOR_EPS_M = 1.2
MIN_NEIGHBORS = 1

ReductionReport = namedtuple(
    "ReductionReport",
    ["n_in", "n_out", "dropped_weak", "dropped_fov", "dropped_isolated",
     "reduction_ratio"])


def _range_az(p):
    x, y, z = p[0], p[1], p[2]
    rng = math.sqrt(x * x + y * y + z * z)
    az = math.degrees(math.atan2(-y, x))   # project convention: right -> +az
    return rng, az


def gate_points(points, snr=None, noise=None,
                min_abs_db=MIN_ABS_DB, r_min=R_MIN_M, r_max=R_MAX_M,
                az_max_deg=AZ_MAX_DEG, neighbor_eps=NEIGHBOR_EPS_M,
                min_neighbors=MIN_NEIGHBORS, refl_max_range=REFL_MAX_RANGE_M):
    """Filter a frame's points, returning ``(kept_points, ReductionReport)``.

    ``points`` are ``(x_fwd, y_left, z_up, v[, ...])``. ``snr``/``noise`` are the
    parallel per-point CFAR side-info lists (dB); when absent, the reflectivity
    gate is skipped (it cannot be applied honestly without the levels). The
    reflectivity gate is applied only within ``refl_max_range`` (see the
    REFL_MAX_RANGE_M note).
    """
    n_in = len(points)
    if n_in == 0:
        return [], ReductionReport(0, 0, 0, 0, 0, 0.0)

    have_levels = snr is not None and noise is not None
    dropped_weak = dropped_fov = 0
    stage1 = []
    for i, p in enumerate(points):
        rng, az = _range_az(p)
        # reflectivity gate: near range only, where snr+noise is discriminative
        if have_levels and rng <= refl_max_range \
                and i < len(snr) and i < len(noise):
            if snr[i] + noise[i] < min_abs_db:
                dropped_weak += 1
                continue
        if not (r_min <= rng <= r_max) or abs(az) > az_max_deg:
            dropped_fov += 1
            continue
        stage1.append(p)

    kept = []
    dropped_isolated = 0
    eps2 = neighbor_eps * neighbor_eps
    m = len(stage1)
    for i in range(m):
        xi, yi, zi = stage1[i][0], stage1[i][1], stage1[i][2]
        neighbors = 0
        for j in range(m):
            if i == j:
                continue
            dx = xi - stage1[j][0]
            dy = yi - stage1[j][1]
            dz = zi - stage1[j][2]
            if dx * dx + dy * dy + dz * dz <= eps2:
                neighbors += 1
                if neighbors >= min_neighbors:
                    break
        if neighbors >= min_neighbors:
            kept.append(stage1[i])
        else:
            dropped_isolated += 1

    n_out = len(kept)
    ratio = 1.0 - (n_out / n_in)
    return kept, ReductionReport(n_in, n_out, dropped_weak, dropped_fov,
                                 dropped_isolated, round(ratio, 3))
