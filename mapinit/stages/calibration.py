#!/usr/bin/env python3
"""Stage 3: validate that the rig calibration inputs can produce a solution.

The stage owns no calibration knowledge of its own. It holds a list of
CalibrationConstraint objects, asks each whether it is usable, and aggregates
the answers. Adding a new source of calibration information later means writing
one constraint class and passing it here — this file does not change.

Validation only. Whether the calibration converges, and to what, belongs to a
solver; whether it *can* converge is answerable now, and is what initialization
needs to know before committing to a run.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..calibration.constraints import CalibrationConstraint
from ..check import Check
from ..context import InitContext
from ..stage import InitStage


class CalibrationStage(InitStage):
    name = "calibration"

    def __init__(self, constraints: Sequence[CalibrationConstraint] = ()) -> None:
        self.constraints = list(constraints)

    def skip_reason(self, ctx: InitContext) -> Optional[str]:
        if not self.constraints:
            return (
                "no calibration constraints configured; pass them to "
                "CalibrationStage(constraints=[...]) to enable this stage"
            )
        return None

    def execute(self, ctx: InitContext) -> Tuple[List[Check], Dict[str, Any]]:
        checks: List[Check] = []
        observation_count = 0

        for constraint in self.constraints:
            checks.extend(constraint.validate())
            observation_count += len(constraint.observations())

        checks.append(
            Check.in_range(
                "calibration.observations",
                observation_count,
                1,
                float("inf"),
                unit="observations",
                detail=(
                    f"{observation_count} observations from "
                    f"{len(self.constraints)} constraint(s)"
                ),
            )
        )

        data: Dict[str, Any] = {
            "observation_count": observation_count,
            "constraints": [c.name for c in self.constraints],
        }

        # Constraints that solve as well as validate publish their result, so a
        # consumer takes the transform from the report rather than reaching back
        # into the constraint object it passed in
        for constraint in self.constraints:
            solution = getattr(constraint, "solution", None)
            if solution is not None:
                data[f"{constraint.name}_solution"] = solution

        return checks, data
