"""Map-relative position initialization with explicit checks at every stage.

Typical use::

    from mapinit import InitContext, InitializationPipeline

    ctx = InitContext(latitude=31.7683, longitude=35.2137)
    report = InitializationPipeline.default().run(ctx)
    print(report.summary())

Stage implementations are imported lazily by ``InitializationPipeline.default``
so that importing this package does not require pyproj.
"""

from .check import Check, CheckFailed
from .context import GLOBAL_GEOID_BOUND_M, InitContext
from .runner import InitializationPipeline, InitReport
from .stage import InitStage, StageResult, StageStatus

__all__ = [
    "Check",
    "CheckFailed",
    "GLOBAL_GEOID_BOUND_M",
    "InitContext",
    "InitializationPipeline",
    "InitReport",
    "InitStage",
    "StageResult",
    "StageStatus",
]
