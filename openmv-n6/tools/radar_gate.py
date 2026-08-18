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

WHAT ACTUALLY HAPPENS ON MEASURED HARDWARE. Read this before trusting the three
paragraphs above. Over 60144 points of session 20260728_203833:

    dropped_weak 0   dropped_fov 0   dropped_isolated 19995   reduction 0.3325

**Gates 1 and 2 have never fired and, as configured, cannot.** Gate 1 compares
snr+noise against MIN_ABS_DB = 8.0 while the observed levels span 58.5-97.1 dB;
the closest any point came to the threshold was 50 dB above it. Gate 2 duplicates
the on-device `cfarFovCfg 0.25 9.0` / `aoaFovCfg -60 60`, which the radar has
already applied before the point reaches the UART. This module advertises three
gates and ships one. 100% of the reduction is gate 3.

Gate 1's premise is also wrong, and it is worth knowing why rather than just
retuning it. Inverting the CFAR threshold (alpha = N(Pfa^(-1/N) - 1)) for the
stock `cfarCfg ... 15 ...` gives Pfa ~ 3e-6 per cell, about 0.01 thermal false
alarms per frame across 4096 cells — under 0.2% of a 6-point cloud. Whatever the
fictitious returns are, they are not noise-floor detections: they are multipath
and sidelobes. That is geometry, so no absolute-level threshold reaches them,
which is exactly what the measurement shows.

AND GATE 3 IS NOT MEASURED TO REMOVE GHOSTS EITHER. On that session it removed
two scatterers whose range was stable to 0.9 mm and 2.8 mm, present in 97% and
94% of 9752 frames. They are 1.35 m apart and their nearest other neighbour is
2.9-4.6 m away, so each reads as "isolated". Keep rate by range band: 98-99%
inside 4 m, and 0.8% in the 4-6 m band where those two live. It also *creates*
flicker rather than removing it -- a scatterer present in all 9752 frames picks
up 134 appear/disappear transitions once its survival depends on a neighbour
also being detected that frame.

NEIGHBOR_EPS_M = 1.2 sits in the empty gap between measured inter-scatterer
distances of 1.13 m and 1.35 m. It is not on a plateau, it is on a cliff face:
+/-0.1 m moves ~30% of the cloud. Its value is decided by one room's furniture.

None of this makes the gate wrong -- it makes it UNTESTED. The corpus fict%~59%
came from a multipath-rich corridor; the session above is a small static room
with the far wall at 5.03 m. Settling whether the 33% is ever really ghosts
needs a corridor recording with a surveyed scatterer layout, not a new constant.
Until then, do not tune these thresholds on desk data, and note that the
odometry path deliberately bypasses this module entirely (see radar_static).
"""
import math
from collections import namedtuple

# Absolute received level (snr+noise, dB) below which a point is treated as a
# manufactured/noise-floor return.
# INERT as configured: measured levels are 58.5-97.1 dB, so this has never
# fired. Left at 8.0 deliberately rather than retuned -- the sweep found no
# operating point. The first threshold with any bite (86 dB) deletes 30.7% of
# the near-zone cloud and empties 966 of 9752 frames, and level's ability to
# predict a ghost is inconsistent in sign across range bands (AUC 0.70, 0.48,
# 0.75, 0.83, 0.03, 0.10 for the 0-1..5-6 m bands -- inverted where it is
# strongest). snr+noise on this data reads out the CFAR noise floor, which is
# itself a function of range, not target reflectivity. If a reflectivity gate is
# ever wanted, `snr` alone is the range-compensated quantity and the defensible
# thing to threshold; this session gives no evidence it would discriminate either.
MIN_ABS_DB = 8.0
# Only apply the reflectivity gate within this range. Beyond ~4 m the CFAR-
# censored return keeps only 1-2 dB of usable dynamic range, so an absolute-level
# gate out there would delete real distant weak targets (survivor bias). Farther
# points pass the reflectivity gate untouched. (The range restriction is measured
# and correct; it is moot only because the threshold above never fires at all.)
REFL_MAX_RANGE_M = 4.0
# Relevant zone. Duplicates cfarFovCfg / aoaFovCfg in the .cfg, which the radar
# has already enforced before the point reaches the UART -- hence dropped_fov 0.
# Kept as a guard against a .cfg whose FOV lines drift from these values, not as
# a working filter. R_MAX_M = 9.0 sits inside the unambiguous range (10.46 m on
# every current config), so it is not admitting folded returns.
R_MIN_M = 0.25
R_MAX_M = 9.0
AZ_MAX_DEG = 60.0
# Isolation test: a point with no neighbour within this radius is a ghost.
# 1.2 m is the cluster epsilon the corpus settled on. See the docstring: this is
# the only gate that fires, it is on a cliff rather than a plateau, and on desk
# data every point it removed was real. Do not tune it without a corridor
# recording.
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
