#!/usr/bin/env python3
"""Stage 2: confirm the map priors covering this point are available offline.

GLO-30 and Overture are cached locally and serve as a ground plane and an
ego-altitude prior. At 30 m posting they are not fine enough to support
obstacle-level residuals, so nothing downstream should treat them as such.

The stage resolves the specific tile covering the point rather than checking
that a directory is non-empty. A tile that does not cover the point is the same
failure as no tile at all, and only the point-in-tile lookup can tell them apart.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..check import Check
from ..context import InitContext
from ..geo.dem import DemSampler
from ..geo.providers import BasePriorDataProvider, LocalFilePriorProvider
from ..stage import InitStage


class PriorsStage(InitStage):
    name = "priors"

    def __init__(self, provider: BasePriorDataProvider | None = None) -> None:
        self.provider = provider

    def _resolve_provider(self, ctx: InitContext) -> BasePriorDataProvider:
        return (
            self.provider
            or ctx.prior_provider
            or LocalFilePriorProvider(ctx.priors_dir, tile_size_deg=ctx.tile_size_deg)
        )

    def execute(self, ctx: InitContext) -> Tuple[List[Check], Dict[str, Any]]:
        provider = self._resolve_provider(ctx)

        # Raises TileNotFound / AmbiguousTiles / FileNotFoundError on failure
        priors = provider.get_priors(ctx.latitude, ctx.longitude)

        point = f"lat={ctx.latitude}, lon={ctx.longitude}"
        checks = [
            Check.that(
                "priors.dem_covers_point",
                True,
                f"GLO-30 tile {priors.glo30_path.name} covers {point}",
            ),
            Check.that(
                "priors.vector_covers_point",
                True,
                f"Overture layer {priors.overture_path.name} covers {point}",
            ),
        ]

        data = {
            "glo30_path": priors.glo30_path,
            "overture_path": priors.overture_path,
            "provider": type(provider).__name__,
        }

        # A tile that covers the point on paper but yields no height there is
        # still unusable, so the guard reads a value rather than trusting the
        # filename. Raises DemUnavailable if the raster cannot answer.
        with DemSampler(priors.glo30_path) as dem:
            ground = dem.ground_elevation(ctx.latitude, ctx.longitude)
            checks.append(
                Check.in_range(
                    "priors.dem_yields_elevation",
                    ground.ground_m,
                    -450.0,   # Dead Sea shore, the lowest land on Earth
                    9000.0,
                    unit="m",
                    detail=f"ground plane at {point}: {ground}",
                )
            )
            data["ground_elevation_m"] = ground.ground_m
            data["surface_elevation_m"] = ground.surface_m
            data["dem_posting_m"] = dem.posting_m

            # Publish the geoid-corrected altitude prior when the geoid stage
            # already ran, so a consumer never has to pair them up itself
            undulation = ctx.output("geoid", "undulation_m")
            if undulation is not None:
                prior = dem.ego_altitude_prior(
                    ctx.latitude, ctx.longitude, geoid_undulation_m=undulation
                )
                data["ego_altitude_prior"] = prior
                checks.append(
                    Check.that("priors.ego_altitude_prior", True, str(prior))
                )

        return checks, data
