#!/usr/bin/env python3
"""Shared state passed through the initialization pipeline.

Holds the point being initialized, where data lives, and the caller's explicit
expectations. Stages read their configuration from here and publish outputs
back, so a stage never reaches into another stage directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .geo.providers import BasePriorDataProvider
    from .stage import StageResult

#: Physical bound on EGM2008 geoid undulation anywhere on Earth (roughly
#: -107 m south of India to +86 m near Iceland). Exceeding it means the
#: transform is wrong, not that the location is unusual.
GLOBAL_GEOID_BOUND_M = 120.0


@dataclass
class InitContext:
    """Inputs, configuration, and accumulated results for one initialization run."""

    latitude: float
    longitude: float
    repo_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    #: Optional regional expectation for the geoid undulation. Left as None the
    #: pipeline only enforces the global physical bound; supplying a range is an
    #: explicit assertion about the operating area, and is reported as such.
    expected_geoid_range: Optional[Tuple[float, float]] = None

    #: Edge length of one prior tile, in degrees. GLO-30 ships 1-degree tiles.
    tile_size_deg: float = 1.0

    #: Overrides the default local-disk provider; the seam for a database source.
    prior_provider: Optional["BasePriorDataProvider"] = None

    #: Populated by the runner as stages complete, keyed by stage name.
    results: Dict[str, "StageResult"] = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return self.repo_dir / "data"

    @property
    def priors_dir(self) -> Path:
        return self.data_dir / "priors"

    def output(self, stage: str, key: str, default: Any = None) -> Any:
        """Read a value published by an earlier stage."""
        result = self.results.get(stage)
        if result is None:
            return default
        return result.data.get(key, default)
