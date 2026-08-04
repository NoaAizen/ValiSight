"""Shared fixtures for the initialization test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapinit.calibration import SurveyedTarget  # noqa: E402
from mapinit.geo.geoid import EGM2008_GRID_NAME  # noqa: E402

#: Tiles used across the prior-selection tests, as SW-corner tags.
TILE_TAGS = ["N30_00_E034_00", "N31_00_E035_00", "N32_00_E035_00", "S01_00_W002_00"]

#: Jerusalem, the reference point this project was built around.
JERUSALEM = (31.7683, 35.2137)

#: Geoid undulation at JERUSALEM, in meters. Cross-checked in two independent
#: ways: PROJ against us_nga_egm08_25.tif, and a hand-written PGM reader against
#: the GeographicLib egm2008-5.pgm distribution. They agreed to 3 mm.
JERUSALEM_UNDULATION_M = 19.758

#: Sizes above the provider's corruption floors (50 KiB DEM, 10 KiB vector).
DEM_BYTES = 60_000
VECTOR_BYTES = 20_000


def grid_available() -> bool:
    """Whether PROJ can find the EGM2008 grid this repo expects."""
    return (REPO_ROOT / "data" / EGM2008_GRID_NAME).is_file()


#: Marks tests that cannot run without the ~77 MB vertical shift grid.
requires_geoid_grid = pytest.mark.skipif(
    not grid_available(),
    reason=(
        f"data/{EGM2008_GRID_NAME} is absent. Fetch it with: "
        f"curl -o data/{EGM2008_GRID_NAME} https://cdn.proj.org/{EGM2008_GRID_NAME}"
    ),
)


def write_priors(root: Path, dem_names: Sequence[str], vector_names: Sequence[str]) -> Path:
    """Build a priors tree with files large enough to pass the integrity floors."""
    glo30 = root / "glo30"
    overture = root / "overture"
    glo30.mkdir(parents=True, exist_ok=True)
    overture.mkdir(parents=True, exist_ok=True)

    for name in dem_names:
        (glo30 / name).write_bytes(b"\0" * DEM_BYTES)
    for name in vector_names:
        (overture / name).write_bytes(b"\0" * VECTOR_BYTES)
    return root


@pytest.fixture
def tiled_priors(tmp_path: Path) -> Path:
    """A priors tree covering TILE_TAGS, named the way Copernicus names tiles."""
    return write_priors(
        tmp_path / "priors",
        [f"Copernicus_DSM_COG_10_{tag}_DEM.tif" for tag in TILE_TAGS],
        [f"overture_{tag}_buildings.parquet" for tag in TILE_TAGS],
    )


def make_targets(
    count: int = 8,
    sigma_m: float = 0.003,
    vertical_spread_m: float = 6.0,
    collinear: bool = False,
) -> List[SurveyedTarget]:
    """Build a target set, optionally degenerate, for the calibration checks.

    The default layout spirals inward as it rises, so near targets are high and
    far targets are low. Range and height both varying is what produces a wide
    elevation spread: a ring at constant radius spans only atan(spread/radius),
    which is a narrow band however tall the targets are.
    """
    import math

    targets = []
    for i in range(count):
        fraction = i / max(count - 1, 1)
        if collinear:
            # All targets on the East axis: no bearing diversity at all
            east, north = 10.0 + 40.0 * fraction, 0.0
        else:
            angle = 2 * math.pi * i / count
            radius = 45.0 - 35.0 * fraction
            east, north = radius * math.cos(angle), radius * math.sin(angle)
        targets.append(
            SurveyedTarget(
                name=f"T{i + 1}",
                east_m=east,
                north_m=north,
                up_m=vertical_spread_m * fraction,
                sigma_m=sigma_m,
            )
        )
    return targets
