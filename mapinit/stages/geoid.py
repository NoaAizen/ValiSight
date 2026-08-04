#!/usr/bin/env python3
"""Stage 1: prove the vertical datum is live before anything depends on it.

This runs first on purpose. GNSS reports height above the ellipsoid; surveyed
targets and GLO-30 elevations are orthometric, referenced to EGM2008. Without a
working geoid the two are offset by the undulation at that point — roughly 20 m
in Israel — and that offset is indistinguishable from a calibration error.

PROJ makes the failure quiet rather than loud: with no vertical shift grid it
falls back to a ballpark transformation that returns the input height unchanged
instead of raising. GeoidModel rejects that state at construction, so reaching
the checks below means a real grid produced the number.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..check import Check
from ..context import GLOBAL_GEOID_BOUND_M, InitContext
from ..geo.geoid import GeoidModel
from ..stage import InitStage


class GeoidStage(InitStage):
    name = "geoid"

    def execute(self, ctx: InitContext) -> Tuple[List[Check], Dict[str, Any]]:
        # Raises GeoidGridUnavailable if PROJ has no real grid to work from
        model = GeoidModel(extra_data_dir=ctx.data_dir)

        checks: List[Check] = [
            Check.that(
                "geoid.grid_live",
                True,
                f"PROJ resolved a non-ballpark transform: {model.description}",
            )
        ]

        undulation = model.undulation(ctx.latitude, ctx.longitude)

        # Physical plausibility, always enforced and independent of location:
        # EGM2008 undulation spans roughly -107 m to +86 m worldwide.
        checks.append(
            Check.in_range(
                "geoid.global_bound",
                abs(undulation),
                0.0,
                GLOBAL_GEOID_BOUND_M,
                unit="m",
                detail=f"N = {undulation:+.3f} m, within the global EGM2008 envelope",
            )
        )

        # Regional plausibility, only when the caller stated an expectation.
        # Keeping this optional is deliberate: a default range baked in here
        # would be an assumption about the operating area that no output reveals.
        if ctx.expected_geoid_range is not None:
            low, high = ctx.expected_geoid_range
            checks.append(
                Check.in_range(
                    "geoid.expected_range",
                    undulation,
                    low,
                    high,
                    unit="m",
                    detail=(
                        f"N = {undulation:+.3f} m against the caller-supplied range "
                        f"[{low:.2f}, {high:.2f}] m for lat={ctx.latitude}, lon={ctx.longitude}"
                    ),
                )
            )

        return checks, {
            "undulation_m": undulation,
            "transform": model.description,
        }
