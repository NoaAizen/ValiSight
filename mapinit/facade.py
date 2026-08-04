#!/usr/bin/env python3
"""The object consuming code holds onto.

Everything else in this package is internal detail. A consumer constructs one
MapInitializer, and gets three things from it:

* ``run()``               - guarded initialization, returning every check
* ``separation_probe``    - the geoid callable other modules inject
* ``priors()``            - the map layers covering the point

Construction is cheap and never touches disk. The geoid grid is opened on first
use, so holding a MapInitializer costs nothing until something is asked of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from .context import InitContext
from .runner import InitializationPipeline, InitReport

if TYPE_CHECKING:
    from .calibration.constraints import CalibrationConstraint
    from .geo.geoid import GeoidModel
    from .geo.providers import BasePriorDataProvider, PriorPaths


class MapInitializer:
    """Initializes position relative to known maps, with checks at every stage."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        repo_dir: Optional[Path] = None,
        expected_geoid_range: Optional[Tuple[float, float]] = None,
        prior_provider: Optional["BasePriorDataProvider"] = None,
        constraints: "Optional[list[CalibrationConstraint]]" = None,
    ) -> None:
        self.context = InitContext(
            latitude=latitude,
            longitude=longitude,
            expected_geoid_range=expected_geoid_range,
            prior_provider=prior_provider,
            **({"repo_dir": repo_dir} if repo_dir is not None else {}),
        )
        self._constraints = list(constraints or [])
        self._geoid: Optional["GeoidModel"] = None

    # -- geoid ------------------------------------------------------------

    @property
    def geoid(self) -> "GeoidModel":
        """The geoid model, opened on first access.

        Raises GeoidGridUnavailable if PROJ has no real vertical shift grid,
        rather than returning the unchanged height PROJ would fall back to.
        """
        if self._geoid is None:
            from .geo.geoid import GeoidModel

            self._geoid = GeoidModel(extra_data_dir=self.context.data_dir)
        return self._geoid

    def separation_probe(self, lon: float, lat: float) -> float:
        """Geoid-ellipsoid separation at a point, in meters, positive above.

        Argument order is **(lon, lat)** to match the probe signature that
        liveness invariants inject — note it is the reverse of the (lat, lon)
        order used everywhere else in this package. Getting it backwards inside
        Israel silently returns another location's undulation, a few meters off,
        which reads as a calibration error rather than a coordinate mistake.

        Raises rather than returning a fallback when no grid is loaded, so an
        invariant that treats a raising probe as "not live" reports correctly.
        """
        return self.geoid.undulation(latitude=lat, longitude=lon)

    def geoid_separation_m(self, latitude: float, longitude: float) -> float:
        """Separation at a point, in this package's own (lat, lon) order."""
        return self.geoid.undulation(latitude=latitude, longitude=longitude)

    def orthometric_height_m(self, ellipsoidal_height_m: float) -> float:
        """Convert an ellipsoidal height at this point to an orthometric one."""
        return self.geoid.orthometric_height(
            self.context.latitude, self.context.longitude, ellipsoidal_height_m
        )

    # -- priors -----------------------------------------------------------

    def priors(self) -> "PriorPaths":
        """Resolve the DEM and vector layers covering this point.

        Raises TileNotFound when no tile covers it, rather than returning an
        arbitrary neighbouring tile.
        """
        from .geo.providers import LocalFilePriorProvider

        provider = self.context.prior_provider or LocalFilePriorProvider(
            self.context.priors_dir, tile_size_deg=self.context.tile_size_deg
        )
        return provider.get_priors(self.context.latitude, self.context.longitude)

    # -- full run ---------------------------------------------------------

    def pipeline(self) -> InitializationPipeline:
        """The stages that will run, in order. Append to extend."""
        from .stages.calibration import CalibrationStage
        from .stages.geoid import GeoidStage
        from .stages.priors import PriorsStage

        return InitializationPipeline(
            [GeoidStage(), PriorsStage(), CalibrationStage(self._constraints)]
        )

    def run(self, fail_fast: bool = True) -> InitReport:
        """Run every initialization stage and return the collected checks.

        ``fail_fast=False`` runs all stages regardless of failure, so one call
        surfaces every problem instead of one per fix.
        """
        return self.pipeline().run(self.context, fail_fast=fail_fast)
