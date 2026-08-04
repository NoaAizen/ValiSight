"""Map-relative position initialization with explicit checks at every stage.

Consuming code needs one object::

    from mapinit import MapInitializer

    init = MapInitializer(latitude=31.7683, longitude=35.2137)

    report = init.run()                 # guarded startup, every check reported
    init.separation_probe               # geoid callable, signature (lon, lat)
    init.priors()                       # DEM and vector layers for this point

Everything below MapInitializer is internal. Importing this package does not
require pyproj; the grid is opened on first use.
"""

from .check import Check, CheckFailed
from .context import GLOBAL_GEOID_BOUND_M, InitContext
from .map_initializer import MapInitializer
from .runner import InitializationPipeline, InitReport
from .stage import InitStage, StageResult, StageStatus

__all__ = [
    # The public surface
    "MapInitializer",
    # Extension points, for adding stages or constraints
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
