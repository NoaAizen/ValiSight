#!/usr/bin/env python3
"""Stage 4: solve position against the map priors. Interface only.

This is the product the earlier stages exist to protect: an initial pose,
expressed relative to known maps, with an uncertainty a consumer can reason
about. The algorithm is not designed yet, so the stage is declared and left
unregistered in the default pipeline.

It is here rather than absent so the shape of the output is fixed now — a pose
plus its covariance, not a bare position — and so the seam it plugs into is
exercised by the same machinery as every other stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..check import Check
from ..context import InitContext
from ..stage import InitStage


@dataclass(frozen=True)
class InitialPose:
    """A map-relative pose estimate with the uncertainty that qualifies it."""

    latitude: float
    longitude: float
    #: Orthometric height, so it shares a datum with GLO-30 and surveyed targets
    height_m: float
    yaw_deg: float
    #: Standard deviations, in meters and degrees
    sigma_horizontal_m: float
    sigma_vertical_m: float
    sigma_yaw_deg: float


class PoseInitStage(InitStage):
    name = "pose_init"

    def skip_reason(self, ctx: InitContext) -> Optional[str]:
        return "map-relative pose solving is not implemented yet"

    def solve(self, ctx: InitContext) -> InitialPose:
        """Estimate the initial pose by registering observations against the priors."""
        raise NotImplementedError(
            "PoseInitStage.solve is not implemented. The registration algorithm "
            "against GLO-30 / Overture has not been designed yet."
        )

    def execute(self, ctx: InitContext) -> Tuple[List[Check], Dict[str, Any]]:
        pose = self.solve(ctx)
        return [Check.that("pose_init.solved", True, f"pose fixed at {pose}")], {"pose": pose}
