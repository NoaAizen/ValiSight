#!/usr/bin/env python3
"""Navigation between map fixes: how position degrades, and how it is recovered.

Two halves of one loop. ``propagation`` says how fast a fix decays once the map
stops correcting it, which is what decides how often a correction is needed.
``heading`` recovers one of the four numbers a fix consists of by matching an
image against what the map predicts, which is what supplies the correction.
"""

from .heading import HeadingFix, HeadingMatcher, bearings_from_columns
from .propagation import (
    DriftBudget,
    DriftTerm,
    ImuErrorModel,
    NavState,
    NoiseTerm,
    SpeedAiding,
)

__all__ = [
    "DriftBudget",
    "DriftTerm",
    "HeadingFix",
    "HeadingMatcher",
    "ImuErrorModel",
    "NavState",
    "NoiseTerm",
    "SpeedAiding",
    "bearings_from_columns",
]
