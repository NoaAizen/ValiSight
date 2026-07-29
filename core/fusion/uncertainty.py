"""Uncertainty-guided thermal + radar fusion.

Each sensor reports a per-detection confidence (score in [0,1]) AND an
uncertainty (sigma > 0, smaller = more trusted); fusion combines them by
inverse variance instead of a hard rule. The *concept* is adapted from
Cai et al., "Robust Human Detection under Visual Degradation via Thermal and
mmWave Radar Fusion" (EWSN '23, arXiv:2307.03623) — their extractor is a
learned Bayesian network, ours is an analytic combiner over hand-defined
per-sensor uncertainties, appropriate to the current rule-based core and
swappable for a learned estimator later.

Each sigma term is grounded in a documented failure mode of that sensor:

  thermal  — a missing or weak detector box IS thermal's failure signature
             (flat temperature field, low contrast, occlusion), so sigma
             rises as box confidence falls, and is very large with no box.
  radar    — sigma rises with sparse support (low ``n``), with high
             ``v_spread`` (multipath clutter smearing the target), and for
             static/aliased returns (|v_abs| below the classifier's
             STATIC_V, which cannot confirm a moving person). A compact
             spatial extent with few-to-moderate points is treated as a
             specular metal glint and given a load-bearing sigma penalty —
             this is the soft replacement for a hard metal veto.

The glint term is deliberately velocity-INDEPENDENT. An earlier version
gated it on |v_abs| < STATIC_V, and a mutation test proved the penalty was
then redundant with the static + sparsity terms. A metal object held by a
walking person moves at walking speed; keying the glint on compact extent
(plus a point-count cap) keeps the penalty load-bearing for exactly that
case.

Pure module: stdlib only, importable with no hardware attached, and it must
never import core.vital_signs (the perception path is firewalled from it).

Inputs come from the real pipeline types through the accessor helpers at the
bottom: the radar cluster is the feature dict produced by
``radar_classify_n6.features()`` (keys ``n``, ``v_spread``, ``v_abs``,
``extent``; the ``cluster_dict()`` aliases ``n_points`` / ``extent_m`` are
also accepted, as are attribute-style objects), and the thermal box is
anything carrying a ``confidence`` / ``score`` / ``conf`` field, or None
when the detector produced nothing.
"""
import math
from dataclasses import dataclass

THERMAL = "thermal"
RADAR = "radar"

# --- tunables ----------------------------------------------------------------
# ANALYTIC SCAFFOLD. Every value below is a hand-set starting point for the
# rule-based core, NOT a number from the paper and NOT calibrated against a
# dataset. Retune freely; the auditor and unit tests pin behaviour (ordering,
# margins, exact fusion math), not these literals.

THERMAL_SIGMA_FLOOR = 0.10   # sigma of a fully confident box
THERMAL_SIGMA_SLOPE = 0.90   # sigma added as confidence falls to 0
THERMAL_SIGMA_BLIND = 8.0    # no box at all: thermal is effectively blind

RADAR_SIGMA_FLOOR = 0.15     # sigma of a dense, moving, clean cluster
SPARSITY_FULL_SUPPORT_N = 8  # points at/above this count as full support
SPARSITY_SIGMA_WEIGHT = 0.6  # sigma added as n falls from full support to 0
CLUTTER_VSPREAD_ONSET = 1.0  # m/s: v_spread above this reads as multipath smear
CLUTTER_SIGMA_WEIGHT = 0.5   # sigma added per m/s of v_spread past the onset
RADAR_STATIC_V = 0.25        # m/s: mirrors src/radar_classify_n6.STATIC_V —
                             # kept in sync by the unit test, not by import,
                             # so this module stays importable anywhere
STATIC_SIGMA_PENALTY = 0.9   # static/aliased return cannot confirm a person
GLINT_MAX_EXTENT_M = 0.35    # compact extent: specular metal glint signature
GLINT_MAX_POINTS = 12        # glints are few-to-moderate points, never dense
GLINT_SIGMA_PENALTY = 1.5    # load-bearing: replaces the hard metal veto

RADAR_SCORE_BASE = 0.35      # a cluster existing at all is weak evidence
RADAR_SCORE_MOVING_BONUS = 0.25
LIMB_VSPREAD = 0.5           # m/s: micro-Doppler spread hinting at limbs
RADAR_SCORE_LIMB_BONUS = 0.20
RADAR_SCORE_SUPPORT_BONUS = 0.20


@dataclass
class SensorEstimate:
    """One sensor's opinion: score in [0,1], sigma > 0 (smaller = more
    trusted), and which sensor produced it."""
    score: float
    sigma: float
    sensor: str


@dataclass
class FusedEstimate:
    """Inverse-variance combination of per-sensor estimates."""
    score: float
    sigma: float
    contributing_sensor: str


def thermal_estimate(box):
    """Estimate from the thermal detector box (or None when there is none).

    A missing or weak box is thermal's own failure signature, so it maps to
    low score / high sigma rather than to an error.
    """
    conf = _box_confidence(box)
    if conf is None:
        return SensorEstimate(score=0.0, sigma=THERMAL_SIGMA_BLIND,
                              sensor=THERMAL)
    conf = _clamp01(conf)
    sigma = THERMAL_SIGMA_FLOOR + THERMAL_SIGMA_SLOPE * (1.0 - conf)
    return SensorEstimate(score=conf, sigma=sigma, sensor=THERMAL)


def radar_estimate(cluster):
    """Estimate from a radar cluster's classifier features.

    Sigma terms, one per failure mode; see the module docstring. The glint
    term MUST stay velocity-independent — do not gate it on RADAR_STATIC_V.
    """
    n = _cluster_field(cluster, "n")
    v_spread = _cluster_field(cluster, "v_spread")
    v_abs = abs(_cluster_field(cluster, "v_abs"))
    extent = _cluster_field(cluster, "extent")

    sigma = RADAR_SIGMA_FLOOR
    sigma += SPARSITY_SIGMA_WEIGHT * max(0.0, 1.0 - n / SPARSITY_FULL_SUPPORT_N)
    sigma += CLUTTER_SIGMA_WEIGHT * max(0.0, v_spread - CLUTTER_VSPREAD_ONSET)
    moving = v_abs >= RADAR_STATIC_V
    if not moving:
        sigma += STATIC_SIGMA_PENALTY
    if extent <= GLINT_MAX_EXTENT_M and n <= GLINT_MAX_POINTS:
        sigma += GLINT_SIGMA_PENALTY

    score = RADAR_SCORE_BASE
    if moving:
        score += RADAR_SCORE_MOVING_BONUS
    if v_spread >= LIMB_VSPREAD:
        score += RADAR_SCORE_LIMB_BONUS
    score += RADAR_SCORE_SUPPORT_BONUS * min(1.0, n / SPARSITY_FULL_SUPPORT_N)
    return SensorEstimate(score=_clamp01(score), sigma=sigma, sensor=RADAR)


def combine(estimates):
    """Inverse-variance fusion of any number of SensorEstimates.

    precision_i = 1/sigma_i^2; fused sigma = sqrt(1/sum(precision));
    fused score = precision-weighted mean of the scores;
    contributing_sensor = the modality with the highest precision.
    """
    ests = [e for e in estimates if e is not None]
    if not ests:
        raise ValueError("combine() needs at least one estimate")
    for e in ests:
        if not (e.sigma > 0.0 and math.isfinite(e.sigma)):
            raise ValueError("sigma must be finite and > 0, got %r" % (e.sigma,))
    total_precision = sum(1.0 / (e.sigma * e.sigma) for e in ests)
    weighted_sum = sum(e.score / (e.sigma * e.sigma) for e in ests)
    best = max(ests, key=lambda e: 1.0 / (e.sigma * e.sigma))
    sigma = math.sqrt(1.0 / total_precision)
    score = weighted_sum / total_precision
    contributing = best.sensor
    return FusedEstimate(score=score, sigma=sigma,
                         contributing_sensor=contributing)


# --- accessor seam -----------------------------------------------------------
# All field access to the pipeline's Box / Cluster shapes goes through these
# two helpers, so swapping the upstream types touches nothing above.

_BOX_CONF_KEYS = ("confidence", "score", "conf")
_CLUSTER_KEYS = {
    "n": ("n", "n_points"),
    "v_spread": ("v_spread",),
    "v_abs": ("v_abs",),
    "extent": ("extent", "extent_m"),
}


def _box_confidence(box):
    """Detector-box confidence, or None when there is no usable box."""
    if box is None:
        return None
    for key in _BOX_CONF_KEYS:
        if isinstance(box, dict):
            if key in box and box[key] is not None:
                return float(box[key])
        elif getattr(box, key, None) is not None:
            return float(getattr(box, key))
    return None


def _cluster_field(cluster, name):
    for key in _CLUSTER_KEYS[name]:
        if isinstance(cluster, dict):
            if key in cluster:
                return float(cluster[key])
        elif hasattr(cluster, key):
            return float(getattr(cluster, key))
    raise KeyError("cluster has no %r (tried %s)" % (name, _CLUSTER_KEYS[name]))


def _clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
