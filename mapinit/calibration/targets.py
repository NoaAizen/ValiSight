#!/usr/bin/env python3
"""Surveyed control targets used to constrain rig calibration.

Targets are points whose position is known to survey accuracy. Observing them
replaces the far larger set of observations a self-calibration would need,
because each target contributes an absolute constraint rather than a relative
one.

Two things make a target set unusable, and both are cheap to detect before any
solver runs:

* Targets surveyed less accurately than claimed. The accuracy budget assumes a
  particular sigma; a target that misses it silently widens the solution.
* A degenerate layout. If the targets share a height, or fall along a line,
  elevation is barely observable no matter how many targets there are or how
  precisely they were surveyed. A solver will still return an answer — it just
  carries an elevation uncertainty large enough to dominate the vertical error
  budget, which is exactly the failure this stage exists to prevent.

The thresholds below are engineering defaults, not physical constants. They are
constructor arguments so that a deployment states its own budget explicitly
rather than inheriting an assumption buried in this file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, Sequence, Tuple

from ..check import Check
from .constraints import CalibrationConstraint, Observation


@dataclass(frozen=True)
class SurveyedTarget:
    """A control point in a local East-North-Up frame, in meters."""

    name: str
    east_m: float
    north_m: float
    up_m: float
    sigma_m: float

    @classmethod
    def from_dict(cls, raw: dict) -> "SurveyedTarget":
        try:
            return cls(
                name=str(raw["name"]),
                east_m=float(raw["east_m"]),
                north_m=float(raw["north_m"]),
                up_m=float(raw["up_m"]),
                sigma_m=float(raw["sigma_m"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed target entry {raw!r}: {exc}") from exc

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.east_m, self.north_m, self.up_m)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(a, b)


def _max_perpendicular_offset(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Return (baseline length, largest offset from that baseline) in plan view.

    The baseline is the most widely separated pair of points; the offset is how
    far the remaining points depart from the line through it. A ratio near zero
    means the layout is collinear.
    """
    if len(points) < 3:
        return (0.0, 0.0)

    # Widest pair defines the baseline, so the ratio is not inflated by a short one
    p_a, p_b = max(
        ((p, q) for i, p in enumerate(points) for q in points[i + 1:]),
        key=lambda pair: _distance(pair[0], pair[1]),
    )
    baseline = _distance(p_a, p_b)
    if baseline == 0.0:
        return (0.0, 0.0)

    (ax, ay), (bx, by) = p_a, p_b
    dx, dy = bx - ax, by - ay
    # Perpendicular distance via the 2D cross product, normalised by the baseline
    offset = max(abs((px - ax) * dy - (py - ay) * dx) / baseline for px, py in points)
    return (baseline, offset)


class SurveyedTargetConstraint(CalibrationConstraint):
    """Validates a set of surveyed targets before it is used for calibration."""

    name: ClassVar[str] = "surveyed_targets"

    def __init__(
        self,
        targets: Sequence[SurveyedTarget],
        rig_position: Optional[Tuple[float, float, float]] = None,
        max_sigma_m: float = 0.005,
        min_targets: int = 6,
        min_vertical_spread_m: float = 1.0,
        min_collinearity_ratio: float = 0.10,
        min_elevation_spread_deg: float = 15.0,
    ) -> None:
        self.targets = list(targets)
        self.rig_position = rig_position
        self.max_sigma_m = max_sigma_m
        self.min_targets = min_targets
        self.min_vertical_spread_m = min_vertical_spread_m
        self.min_collinearity_ratio = min_collinearity_ratio
        self.min_elevation_spread_deg = min_elevation_spread_deg

    @classmethod
    def from_json(cls, path: Path, **overrides) -> "SurveyedTargetConstraint":
        """Load targets from a JSON file.

        Expected shape::

            {
              "rig_position": [0.0, 0.0, 1.5],
              "targets": [
                {"name": "T1", "east_m": 12.0, "north_m": 3.0,
                 "up_m": 4.5, "sigma_m": 0.003}
              ]
            }
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Surveyed target file not found: {path}")

        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Surveyed target file {path} is not valid JSON: {exc}") from exc

        entries = raw.get("targets")
        if not isinstance(entries, list):
            raise ValueError(f"Surveyed target file {path} has no 'targets' list.")

        rig = raw.get("rig_position")
        if rig is not None:
            rig = tuple(float(v) for v in rig)

        return cls(
            targets=[SurveyedTarget.from_dict(e) for e in entries],
            rig_position=rig,
            **overrides,
        )

    def elevation_angles_deg(self) -> List[float]:
        """Elevation angle from the rig to each target, in degrees."""
        if self.rig_position is None:
            return []
        rx, ry, rz = self.rig_position
        angles = []
        for target in self.targets:
            de, dn, du = target.east_m - rx, target.north_m - ry, target.up_m - rz
            horizontal = math.hypot(de, dn)
            if horizontal == 0.0 and du == 0.0:
                continue
            angles.append(math.degrees(math.atan2(du, horizontal)))
        return angles

    def validate(self) -> List[Check]:
        checks: List[Check] = [
            Check.in_range(
                self.check_name("count"),
                len(self.targets),
                self.min_targets,
                float("inf"),
                unit="targets",
                detail=f"{len(self.targets)} targets, minimum {self.min_targets}",
            )
        ]

        if not self.targets:
            return checks

        # 1. Survey accuracy: every target must meet the budget, and the report
        #    names the worst offender rather than only saying the set failed
        worst = max(self.targets, key=lambda t: t.sigma_m)
        checks.append(
            Check.in_range(
                self.check_name("sigma"),
                worst.sigma_m,
                0.0,
                self.max_sigma_m,
                unit="m",
                detail=(
                    f"worst target {worst.name!r} sigma {worst.sigma_m * 1000:.1f} mm, "
                    f"budget {self.max_sigma_m * 1000:.1f} mm"
                ),
            )
        )

        # 2. Vertical spread: targets at a single height leave elevation weakly
        #    observable regardless of how many there are
        ups = [t.up_m for t in self.targets]
        vertical_spread = max(ups) - min(ups)
        checks.append(
            Check.in_range(
                self.check_name("vertical_spread"),
                vertical_spread,
                self.min_vertical_spread_m,
                float("inf"),
                unit="m",
                detail=f"targets span {vertical_spread:.2f} m vertically",
            )
        )

        # 3. Plan-view geometry: a collinear layout is rank-deficient in bearing
        baseline, offset = _max_perpendicular_offset([(t.east_m, t.north_m) for t in self.targets])
        ratio = offset / baseline if baseline > 0 else 0.0
        checks.append(
            Check.in_range(
                self.check_name("collinearity"),
                ratio,
                self.min_collinearity_ratio,
                float("inf"),
                detail=(
                    f"largest departure from the {baseline:.1f} m baseline is "
                    f"{offset:.2f} m (ratio {ratio:.3f})"
                ),
            )
        )

        # 4. Elevation angle spread, when the rig position is known. This is the
        #    most direct predictor of how well elevation will be determined.
        angles = self.elevation_angles_deg()
        if angles:
            spread = max(angles) - min(angles)
            checks.append(
                Check.in_range(
                    self.check_name("elevation_spread"),
                    spread,
                    self.min_elevation_spread_deg,
                    float("inf"),
                    unit="deg",
                    detail=(
                        f"elevation angles span {spread:.1f} deg "
                        f"({min(angles):.1f} to {max(angles):.1f})"
                    ),
                )
            )

        return checks

    def observations(self) -> List[Observation]:
        return [
            Observation(
                kind="surveyed_point",
                source=f"{self.name}:{target.name}",
                sigma=target.sigma_m,
                payload={"position_enu_m": target.position},
            )
            for target in self.targets
        ]
