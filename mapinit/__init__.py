"""Map-relative position initialization with explicit checks at every stage.

Consuming code needs one object::

    from mapinit import MapInitializer

    init = MapInitializer(latitude=31.7683, longitude=35.2137)

    report = init.run()                 # guarded startup, every check reported
    init.separation_probe               # geoid callable, signature (lon, lat)
    init.priors()                       # DEM and vector layers for this point

Everything else this package offers is catalogued in ``mapinit.api``, which is
the source of the names below rather than a description of them: ``__all__``
and the lazy-import table are both built from it, so a public name always
carries a summary of what it is and a note on what it is for. To read the
catalogue::

    python -c "import mapinit; print(mapinit.describe())"
    python -c "import mapinit; print(mapinit.describe('navigation'))"

The failure types are exported alongside the rest. A consumer is expected to
catch them rather than to receive a fallback value, so it must be able to write
``except TileNotFound`` without reaching into ``mapinit.geo.*``; the returned
records are exported for the same reason, so consuming code can annotate what
it holds.

**Nothing is imported until it is named.** Every export resolves on first
attribute access, so importing this package costs nothing and pulls in neither
pyproj nor the raster stack unless something under ``mapinit.geo`` is actually
touched. A process that only wants ``DeadReckoner`` never opens a GeoTIFF
library, and one that only wants ``Check`` -- which is how perception vendors
this core -- pays for nothing at all.
"""

from typing import TYPE_CHECKING

from .api import AREAS, PUBLIC_API, Export, describe, exports_by_area, extra_for

if TYPE_CHECKING:  # for type checkers and editors only; never executed
    from .calibration.constraints import CalibrationConstraint, Observation
    from .calibration.imu import (
        CalibrationDependencyMissing,
        ImuCalibrationFailed,
        ImuCameraConstraint,
        ImuCameraSolution,
        RigOrientation,
        solve_imu_camera,
        tilt_separation_deg,
    )
    from .calibration.targets import SurveyedTarget, SurveyedTargetConstraint
    from .check import Check, CheckFailed
    from .context import GLOBAL_GEOID_BOUND_M, InitContext
    from .geo.buildings import Building, BuildingLayer, HeightSource, resolve_height
    from .geo.dem import (
        DemSampler,
        DemUnavailable,
        DependencyMissing,
        EgoAltitudePrior,
        GroundEstimate,
        ScaleNotSupported,
    )
    from .geo.geoid import (
        EGM2008_GRID_NAME,
        EGM2008_GRID_URL,
        GeoidGridUnavailable,
        GeoidModel,
    )
    from .geo.providers import (
        BasePriorDataProvider,
        DatabasePriorProvider,
        LocalFilePriorProvider,
        PriorPaths,
    )
    from .geo.tiles import (
        AmbiguousTiles,
        TileBounds,
        TileNotFound,
        parse_tile_bounds,
        select_tile,
    )
    from .geo.view import (
        PredictedView,
        SkylinePoint,
        VerticalEdge,
        ViewPredictor,
        bearing_and_range,
        relative_bearing,
    )
    from .map_initializer import MapInitializer
    from .nav.heading import (
        HeadingCandidate,
        HeadingFix,
        HeadingMatcher,
        HeadingMatchError,
        angular_difference_deg,
        bearings_from_columns,
        bearings_from_view,
    )
    from .nav.identification import (
        Hold,
        Recording,
        RecordingError,
        allan_deviation,
        angle_random_walk_dps_sqrt_s,
        identification_checks,
        identify,
        load_recording,
    )
    from .nav.propagation import (
        RIG_ERROR_MODEL,
        STANDARD_GRAVITY,
        DeadReckoner,
        DriftBudget,
        DriftTerm,
        ImuErrorModel,
        NavState,
        NoiseTerm,
        PropagationError,
        SpeedAiding,
        advance,
        propagate,
    )
    from .runner import InitializationPipeline, InitReport
    from .stage import InitStage, StageResult, StageStatus

#: Public name -> the module that defines it, built from the catalogue so the
#: two can never disagree. Resolved on first attribute access.
_LAZY = {export.name: export.module for export in PUBLIC_API}

#: The catalogue's own entry points, which are not themselves catalogued: a
#: description of the surface is not part of the surface it describes.
_CATALOGUE = ["AREAS", "Export", "PUBLIC_API", "describe", "exports_by_area", "extra_for"]

__all__ = sorted(_LAZY) + _CATALOGUE


def __getattr__(name: str):
    """Import a public name on first use, keeping every dependency out of import."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. "
            f"Run mapinit.describe() for the public surface."
        )

    from importlib import import_module

    try:
        resolved = import_module(module, __name__)
    except ImportError as exc:
        # A public name is not allowed to fail with somebody else's import
        # error. Half this package installs with no dependencies at all, so
        # reaching for the other half without its extra is an ordinary thing
        # to do by accident, and the message has to say which extra rather
        # than naming a third-party module the caller never asked for.
        extra = extra_for(name)
        if extra is None:
            raise
        raise ImportError(
            f"mapinit.{name} needs the {extra!r} extra, which is not installed. "
            f"Install it with: pip install 'mapinit[{extra}]'  "
            f"(from a checkout: pip install -e '.[{extra}]')"
        ) from exc

    return getattr(resolved, name)


def __dir__():
    return sorted(__all__)
