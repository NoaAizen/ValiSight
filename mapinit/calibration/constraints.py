#!/usr/bin/env python3
"""The extension point for rig calibration.

A constraint is one source of information about the rig. Each validates that it
is usable and contributes observations to the calibration problem. Adding a new
source later — IMU biases, lever-arm, camera boresight — means writing one
subclass and passing it to CalibrationStage. No stage or runner code changes.

Constraints deliberately do not solve. Validation runs during initialization,
where the useful question is whether the calibration *can* converge, not what
it converges to. Catching an unusable constraint here costs nothing; discovering
it inside a solver costs a run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List

from ..check import Check


@dataclass(frozen=True)
class Observation:
    """One measurement contributed to the calibration problem.

    ``kind`` names what was measured so a solver can dispatch on it, and
    ``sigma`` carries the measurement uncertainty in the observation's own
    units, which is what lets observations of different types be weighted
    against each other.
    """

    kind: str
    source: str
    sigma: float
    payload: Dict[str, Any] = field(default_factory=dict)


class CalibrationConstraint(ABC):
    """Base class for one source of calibration information."""

    #: Stable identifier used in check names and reports.
    name: ClassVar[str] = "constraint"

    @abstractmethod
    def validate(self) -> List[Check]:
        """Report whether this constraint is usable, without solving anything."""

    @abstractmethod
    def observations(self) -> List[Observation]:
        """The observations this constraint contributes to the calibration."""

    def check_name(self, suffix: str) -> str:
        """Namespace a check so its origin is unambiguous in a combined report."""
        return f"{self.name}.{suffix}"
