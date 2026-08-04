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

        return checks, {
            "glo30_path": priors.glo30_path,
            "overture_path": priors.overture_path,
            "provider": type(provider).__name__,
        }
