"""Map-relative position initialization with explicit checks at every stage.

Consuming code needs one object::

    from mapinit import MapInitializer

    init = MapInitializer(latitude=31.7683, longitude=35.2137)

    report = init.run()                 # guarded startup, every check reported
    init.separation_probe               # geoid callable, signature (lon, lat)
    init.priors()                       # DEM and vector layers for this point

The failure types are exported alongside it. A consumer is expected to catch
them rather than to receive a fallback value, so it must be able to write
``except TileNotFound`` without reaching into ``mapinit.geo.*``; the returned
records are exported for the same reason, so consuming code can annotate what
it holds.

Everything below MapInitializer is internal. Importing this package does not
require pyproj: the geoid grid is opened on first use, and every name that
lives under ``mapinit.geo`` is resolved lazily, because that subpackage imports
pyproj when it is first touched.
"""

from typing import TYPE_CHECKING

from .check import Check, CheckFailed
from .context import GLOBAL_GEOID_BOUND_M, InitContext
from .map_initializer import MapInitializer
from .runner import InitializationPipeline, InitReport
from .stage import InitStage, StageResult, StageStatus

if TYPE_CHECKING:  # for type checkers and editors only; never executed
    from .geo.dem import EgoAltitudePrior, GroundEstimate
    from .geo.geoid import GeoidGridUnavailable
    from .geo.providers import BasePriorDataProvider, PriorPaths
    from .geo.tiles import TileNotFound

#: Names re-exported from ``mapinit.geo``, mapped to the module that defines
#: them. Resolved on first attribute access rather than at import, so that
#: naming one of these does not pull pyproj into a process that never uses it.
_LAZY = {
    "BasePriorDataProvider": ".geo.providers",
    "EgoAltitudePrior": ".geo.dem",
    "GeoidGridUnavailable": ".geo.geoid",
    "GroundEstimate": ".geo.dem",
    "PriorPaths": ".geo.providers",
    "TileNotFound": ".geo.tiles",
}

__all__ = [
    # The public surface
    "MapInitializer",
    # Failures a consumer catches. Returning a fallback instead of raising one
    # of these is the break that moves positions without failing a test.
    "GeoidGridUnavailable",
    "TileNotFound",
    # Records handed back, exported so consumers can name their own types
    "EgoAltitudePrior",
    "GroundEstimate",
    "PriorPaths",
    # Extension points, for adding stages or constraints
    "BasePriorDataProvider",
    "Check",
    "CheckFailed",
    "InitStage",
    "StageResult",
    "StageStatus",
    # Internals, exposed for tests and advanced wiring
    "GLOBAL_GEOID_BOUND_M",
    "InitContext",
    "InitializationPipeline",
    "InitReport",
]


def __getattr__(name: str):
    """Import a ``mapinit.geo`` name on first use, keeping pyproj out of import."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
