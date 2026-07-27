"""
Material estimation from radar reflectivity — metal vs fabric (IWR1843).

How it works (v2 — empirical, replaces the broken 40*log10(R) model):

  score_db = signal - baseline(range)          # via radar_calibration.Baseline

signal is the absolute level snr+noise (TLV 7) when available, else raw snr.
baseline(range) is what this rig typically reports at that range, measured
from its own session logs (python radar_calibration.py). Score 0 = a typical
static return; metal glints stand out ABOVE the local norm at any range.

Why v1 failed: the demo firmware only outputs CFAR *detections*, so reported
SNR hugs the CFAR floor at every range (~12-15 dB from 0.4 to 12 m). Adding
40log10(R) turned the metric into a pure distance function — 96.8% of v1
material labels were predictable from range alone (fabric = near, metal = far,
including trees at 7.6 m labelled "metal").

Classification (on the cluster's PEAK score — metal announces itself with a
few strong specular glints; the median dilutes them with edge points):

  peak >= METAL_DB    -> "metal"    (strong glint above the local norm)
  peak <= FABRIC_DB   -> "fabric"   (even the best return is near the floor)
  in between          -> "mid"      (wood, drywall, plastic, body)
  range > max_range   -> "unknown"  (in our session logs, beyond ~4 m the
                                     CFAR-censored returns keep only ~1-2 dB
                                     of usable dynamic range — honest answer
                                     is "can't tell", not a guess)

HONESTY CAVEAT: a monostatic amplitude measurement senses a REFLECTIVITY
class, not a material. "metal" really means "strong specular/retro reflector"
— smooth dielectrics at normal incidence (glass, wet wall) can score like
metal, and a human body at 77 GHz returns fairly strongly too. The 10/4 dB
thresholds are uncalibrated placeholders until measured against known objects
(see the calibrate note below). Keep the class names as UI shorthand only.

Calibrate: point the rig at a known metal and a known fabric object at 1-3 m,
run live_radar_camera.py --debug-refl, read the peak scores, set
--metal-db/--fabric-db. MCU-friendly: pure math + one JSON table.
"""
import math

from radar_calibration import Baseline

METAL_DB = 10.0            # peak score (dB above baseline) at/above -> metal
FABRIC_DB = 4.0            # peak score at/below -> fabric
MATERIAL_MAX_RANGE_M = 4.0 # beyond this the material call is "unknown"

# --- mixed-cluster splitting (person holding/next to a metal object) ---------
# A metal object touching a person merges into one cluster (CLUSTER_EPS is
# 1.2 m), and its glint then paints the whole person "[metal]". The score
# distribution inside such a cluster is strongly bimodal — a compact glint
# core >= METAL_DB embedded in weak body returns — so it can be split.
SPLIT_MIN_GLINT_PTS = 3    # glint core points needed (one outlier != object)
SPLIT_MIN_BODY_PTS = 4     # remaining points must still form a real body
SPLIT_CORE_RADIUS_M = 0.6  # glint core must be compact, not scattered speckle

_baseline = None


def get_baseline():
    """Shared lazily-loaded Baseline (flat fallback if not built yet)."""
    global _baseline
    if _baseline is None:
        _baseline = Baseline.load()
    return _baseline


def point_scores(grp, baseline=None):
    """Range-normalized score (dB) per cluster point.

    grp: points as (x, y, z, v, snr_db[, noise_db]) tuples; points without an
    SNR field are skipped."""
    bl = baseline or get_baseline()
    out = []
    for p in grp:
        if len(p) < 5 or p[4] is None:
            continue
        r = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
        noise = p[5] if len(p) > 5 else None
        out.append(bl.score(r, p[4], noise))
    return out


def peak_score(scores):
    """Robust peak: 2nd-highest when there are enough points (a lone outlier
    point should not flip a cluster to metal), else the max."""
    if not scores:
        return None
    s = sorted(scores)
    return s[-2] if len(s) >= 4 else s[-1]


def material(score_db, metal_db=METAL_DB, fabric_db=FABRIC_DB,
             range_m=None, max_range=MATERIAL_MAX_RANGE_M):
    """Peak score (+ range gate) -> material class string."""
    if score_db is None:
        return "unknown"
    if range_m is not None and range_m > max_range:
        return "unknown"
    if score_db >= metal_db:
        return "metal"
    if score_db <= fabric_db:
        return "fabric"
    return "mid"


def annotate_clusters(clusters, metal_db=METAL_DB, fabric_db=FABRIC_DB,
                      baseline=None, max_range=MATERIAL_MAX_RANGE_M):
    """Add 'refl_db' (peak range-normalized score) and 'material' to clusters
    from classify_frame(..., include_points=True). Returns the list for
    chaining."""
    bl = baseline or get_baseline()
    for c in clusters:
        pk = peak_score(point_scores(c.get("points", ()), bl))
        c["refl_db"] = None if pk is None else round(pk, 1)
        c["material"] = material(pk, metal_db, fabric_db,
                                 c.get("range_m"), max_range)
    return clusters


def split_mixed_clusters(clusters, metal_db=METAL_DB, fabric_db=FABRIC_DB,
                         baseline=None, max_range=MATERIAL_MAX_RANGE_M):
    """Split clusters that hide a metal object inside a soft body.

    For each annotated cluster (needs 'points'): score every point; if a
    compact core of >= SPLIT_MIN_GLINT_PTS points sits at metal level while
    the remaining points read soft (median <= fabric_db), emit TWO clusters —
    the glint core and the body — each re-classified and re-annotated.
    Everything else passes through untouched. Run after annotate_clusters().
    """
    from radar_classify_n6 import cluster_dict
    bl = baseline or get_baseline()
    out = []
    for c in clusters:
        pts = c.get("points")
        if not pts or c.get("range_m", 1e9) > max_range:
            out.append(c)
            continue
        glint, rest, rest_scores = [], [], []
        for p in pts:
            s = None
            if len(p) >= 5 and p[4] is not None:
                r = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
                s = bl.score(r, p[4], p[5] if len(p) > 5 else None)
            if s is not None and s >= metal_db:
                glint.append(p)
            else:
                rest.append(p)
                if s is not None:
                    rest_scores.append(s)
        if (len(glint) < SPLIT_MIN_GLINT_PTS
                or len(rest) < SPLIT_MIN_BODY_PTS or not rest_scores):
            out.append(c)
            continue
        rest_scores.sort()
        if rest_scores[len(rest_scores) // 2] > fabric_db:
            out.append(c)          # body isn't soft — not the mixed case
            continue
        gx = sum(p[0] for p in glint) / len(glint)
        gy = sum(p[1] for p in glint) / len(glint)
        gz = sum(p[2] for p in glint) / len(glint)
        if any(math.sqrt((p[0] - gx) ** 2 + (p[1] - gy) ** 2 +
                         (p[2] - gz) ** 2) > SPLIT_CORE_RADIUS_M
               for p in glint):
            out.append(c)          # scattered speckle, not one object
            continue
        sub = [cluster_dict(glint, include_points=True),
               cluster_dict(rest, include_points=True)]
        out.extend(annotate_clusters(sub, metal_db, fabric_db, bl, max_range))
    return out
