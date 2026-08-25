#!/usr/bin/env python3
"""Stage 4: solve position against the map priors.

This is the product the earlier stages exist to protect: an initial pose,
expressed relative to known maps, with an uncertainty a consumer can reason
about.

v0 (2026-08-25): radar wall returns matched against building footprints
(``mapinit.nav.walls``). The stage runs when it is given observations — a
``PosePrior`` (GNSS, manual pin, or dead reckoning) and the static returns
from ``radar_detections_all()`` — and reports itself skipped otherwise, so a
run without a radar still says what it did not do rather than failing.

Height is not solved here: it is taken from the priors stage (ground + rig
height above ground), because two-element radar elevation cannot argue
with a DEM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..check import Check
from ..context import InitContext
from ..stage import InitStage
from ..nav.walls import PosePrior, WallFix, WallMatcher, WallReturn


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


@dataclass(frozen=True)
class PoseObservations:
    """What the pose stage needs from the rig: where it thinks it is, and walls."""

    prior: PosePrior
    #: Static returns from ``radar_detections_all()``, already filtered
    #: (``mapinit.nav.walls.static_returns``)
    wall_returns: Tuple[WallReturn, ...]
    rig_height_agl_m: float = 1.5


class PoseInitStage(InitStage):
    name = "pose_init"

    def __init__(self, observations: Optional[PoseObservations] = None,
                 matcher: Optional[WallMatcher] = None) -> None:
        self.observations = observations
        self.matcher = matcher or WallMatcher()

    def skip_reason(self, ctx: InitContext) -> Optional[str]:
        if self.observations is None:
            return ("no observations: pass PoseObservations(prior, wall_returns) "
                    "to solve position from radar wall returns")
        return None

    def solve(self, ctx: InitContext) -> Tuple[InitialPose, WallFix]:
        """Register the wall returns against the footprints the priors stage found."""
        from ..geo.buildings import BuildingLayer
        from pathlib import Path

        obs = self.observations
        assert obs is not None
        overture = ctx.output("priors", "overture_path")
        if overture is None:
            raise RuntimeError("priors stage published no overture_path; pose_init needs footprints")
        layer = BuildingLayer.from_geojson(Path(overture))
        fix = self.matcher.match(list(obs.wall_returns), layer, obs.prior)

        ground = ctx.output("priors", "ground_elevation_m")
        alt = ctx.output("priors", "ego_altitude_prior")
        sigma_v = float(getattr(alt, "sigma_m", 5.0)) if alt is not None else 5.0
        height = (float(ground) if ground is not None else 0.0) + obs.rig_height_agl_m
        pose = InitialPose(
            latitude=fix.latitude, longitude=fix.longitude, height_m=height,
            yaw_deg=fix.heading_deg,
            sigma_horizontal_m=max(fix.sigma_east_m, fix.sigma_north_m),
            sigma_vertical_m=sigma_v, sigma_yaw_deg=fix.sigma_heading_deg,
        )
        return pose, fix

    def execute(self, ctx: InitContext) -> Tuple[List[Check], Dict[str, Any]]:
        pose, fix = self.solve(ctx)
        checks = list(fix.checks)
        checks.append(Check.that("pose_init.accepted", fix.accepted, str(fix)))
        return checks, {"pose": pose, "fix": fix}
