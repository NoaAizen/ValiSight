"""Transparent thermal/radar decision baseline.

This is intentionally not a learned fusion layer. It is the deployable
baseline that a future reliability gate must beat on an independent test set.
Inputs must already be calibrated probabilities from the two independent
detectors; stale or invalid evidence is removed before any arithmetic.
"""
from dataclasses import dataclass, field
from math import exp, log
from typing import Optional, Tuple


@dataclass(frozen=True)
class Evidence:
    probability: float
    age_ms: float
    valid: bool = True
    caveats: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    label: str                 # PERSON / NO_PERSON / UNKNOWN
    probability: Optional[float]
    provenance: str            # BOTH / THERMAL_ONLY / RADAR_ONLY / NONE
    caveats: Tuple[str, ...] = field(default_factory=tuple)


def _usable(evidence, max_age_ms):
    if evidence is None:
        return False, 'missing'
    if not evidence.valid:
        return False, 'invalid'
    if evidence.age_ms < 0 or evidence.age_ms > max_age_ms:
        return False, 'stale'
    if not 0.0 <= evidence.probability <= 1.0:
        return False, 'uncalibrated-range'
    return True, None


def _logit(p):
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def _label(p, positive_threshold, negative_threshold):
    if p >= positive_threshold:
        return 'PERSON'
    if p <= negative_threshold:
        return 'NO_PERSON'
    return 'UNKNOWN'


def decide(thermal: Optional[Evidence], radar: Optional[Evidence],
           max_age_ms=300.0, positive_threshold=0.70,
           negative_threshold=0.30, contradiction_high=0.75,
           contradiction_low=0.25):
    """Fuse two calibrated probabilities with explicit degradation.

    With both sensors, equal-weight log-odds are the simple baseline. A strong
    disagreement returns UNKNOWN; it is not averaged into a confident lie.
    With one usable sensor, its own calibrated answer is returned and marked
    single-modality. A future model may learn reliability weights, but must
    preserve these validity and age gates.
    """
    t_ok, t_reason = _usable(thermal, max_age_ms)
    r_ok, r_reason = _usable(radar, max_age_ms)
    caveats = []
    if thermal is not None:
        caveats.extend(thermal.caveats)
    if radar is not None:
        caveats.extend(radar.caveats)
    if not t_ok:
        caveats.append(f'thermal-{t_reason}')
    if not r_ok:
        caveats.append(f'radar-{r_reason}')

    if not t_ok and not r_ok:
        return Decision('UNKNOWN', None, 'NONE', tuple(caveats))

    if t_ok and not r_ok:
        p = float(thermal.probability)
        caveats.append('single-modality')
        return Decision(_label(p, positive_threshold, negative_threshold),
                        p, 'THERMAL_ONLY', tuple(caveats))

    if r_ok and not t_ok:
        p = float(radar.probability)
        caveats.append('single-modality')
        return Decision(_label(p, positive_threshold, negative_threshold),
                        p, 'RADAR_ONLY', tuple(caveats))

    tp, rp = float(thermal.probability), float(radar.probability)
    disagree = ((tp >= contradiction_high and rp <= contradiction_low) or
                (rp >= contradiction_high and tp <= contradiction_low))
    if disagree:
        caveats.append('sensor-contradiction')
        return Decision('UNKNOWN', None, 'BOTH', tuple(caveats))

    p = _sigmoid(_logit(tp) + _logit(rp))
    return Decision(_label(p, positive_threshold, negative_threshold),
                    p, 'BOTH', tuple(caveats))
