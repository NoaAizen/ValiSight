#!/usr/bin/env python3
"""Navigation between map fixes: how position degrades, and how it is recovered.

Two halves of one loop. ``propagation`` says how fast a fix decays once the map
stops correcting it, which is what decides how often a correction is needed.
``heading`` recovers one of the four numbers a fix consists of by matching an
image against what the map predicts, which is what supplies the correction.

``identification`` supplies the constants the first of those runs on, fitted
from a recording off the rig rather than assumed, so a decay rate is a
measurement instead of a prediction.
"""

from .heading import (
    HeadingCandidate,
    HeadingFix,
    HeadingMatcher,
    HeadingMatchError,
    angular_difference_deg,
    bearings_from_columns,
    bearings_from_view,
)
from .identification import (
    Hold,
    Recording,
    RecordingError,
    allan_deviation,
    angle_random_walk_dps_sqrt_s,
    identification_checks,
    identify,
    load_recording,
)
from .propagation import (
    RIG_ERROR_MODEL,
    STANDARD_GRAVITY,
    DeadReckoner,
    DriftBudget,
    DriftTerm,
    ImuErrorModel,
    NavState,
    NoiseTerm,
    PropagationError,
    SpeedAiding,
    advance,
    propagate,
)

__all__ = [
    "DeadReckoner",
    "DriftBudget",
    "DriftTerm",
    "HeadingCandidate",
    "HeadingFix",
    "HeadingMatchError",
    "HeadingMatcher",
    "Hold",
    "ImuErrorModel",
    "NavState",
    "NoiseTerm",
    "PropagationError",
    "RIG_ERROR_MODEL",
    "Recording",
    "RecordingError",
    "STANDARD_GRAVITY",
    "SpeedAiding",
    "advance",
    "allan_deviation",
    "angle_random_walk_dps_sqrt_s",
    "angular_difference_deg",
    "bearings_from_columns",
    "bearings_from_view",
    "identification_checks",
    "identify",
    "load_recording",
    "propagate",
]
