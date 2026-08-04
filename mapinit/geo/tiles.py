#!/usr/bin/env python3
"""Resolve which geospatial tile file covers a given point.

Tiled datasets such as Copernicus GLO-30 encode the south-west corner of each
tile in its filename, so a point-in-tile lookup can be done without opening the
raster. This mirrors the spatial query a database-backed provider would run, so
swapping LocalFilePriorProvider for DatabasePriorProvider does not change which
tile a given coordinate resolves to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

#: Matches the SW-corner tag in tiled dataset filenames. Handles both the
#: Copernicus form (``N31_00_E035_00``) and the bare form (``N31E035``).
_TILE_TAG = re.compile(
    r"(?P<ns>[NS])(?P<lat>\d{2})(?:_(?P<lat_min>\d{2}))?"
    r"_?(?P<ew>[EW])(?P<lon>\d{3})(?:_(?P<lon_min>\d{2}))?",
    re.IGNORECASE,
)


class TileNotFound(FileNotFoundError):
    """Raised when no tile in a directory covers the requested point."""


class AmbiguousTiles(RuntimeError):
    """Raised when several untagged candidate files make the choice arbitrary."""


@dataclass(frozen=True)
class TileBounds:
    """Geographic extent of a single tile, in decimal degrees."""

    min_lat: float
    min_lon: float
    size_deg: float

    @property
    def max_lat(self) -> float:
        return self.min_lat + self.size_deg

    @property
    def max_lon(self) -> float:
        return self.min_lon + self.size_deg

    def contains(self, latitude: float, longitude: float) -> bool:
        """Half-open on the upper edge so neighbouring tiles never both match."""
        return (
            self.min_lat <= latitude < self.max_lat
            and self.min_lon <= longitude < self.max_lon
        )

    def __str__(self) -> str:
        return f"[{self.min_lat:.0f}..{self.max_lat:.0f}N, {self.min_lon:.0f}..{self.max_lon:.0f}E]"


def parse_tile_bounds(path: Path, size_deg: float = 1.0) -> Optional[TileBounds]:
    """Extract tile bounds from a filename, or None if it carries no tile tag."""
    match = _TILE_TAG.search(path.name)
    if match is None:
        return None

    latitude = int(match.group("lat")) + int(match.group("lat_min") or 0) / 60.0
    longitude = int(match.group("lon")) + int(match.group("lon_min") or 0) / 60.0

    # Tile tags name the south-west corner, so the hemisphere sign applies directly
    if match.group("ns").upper() == "S":
        latitude = -latitude
    if match.group("ew").upper() == "W":
        longitude = -longitude

    return TileBounds(min_lat=latitude, min_lon=longitude, size_deg=size_deg)


def select_tile(
    candidates: Iterable[Path],
    latitude: float,
    longitude: float,
    dataset: str,
    size_deg: float = 1.0,
) -> Path:
    """Return the single file covering the point.

    Falls back to an untagged file only when it is the sole candidate, so an
    untiled global export still works while a directory of several untagged
    files raises instead of silently picking one.
    """
    # Sort so that any tie is resolved identically across runs and filesystems
    files: List[Path] = sorted(candidates)
    if not files:
        raise TileNotFound(f"No {dataset} file available to cover lat={latitude}, lon={longitude}.")

    tagged = [(f, b) for f in files if (b := parse_tile_bounds(f, size_deg)) is not None]

    if not tagged:
        if len(files) == 1:
            return files[0]
        raise AmbiguousTiles(
            f"{len(files)} {dataset} files carry no tile tag (e.g. 'N31_00_E035_00'), "
            f"so the tile covering lat={latitude}, lon={longitude} cannot be determined: "
            f"{', '.join(f.name for f in files)}"
        )

    matches = [f for f, bounds in tagged if bounds.contains(latitude, longitude)]
    if matches:
        return matches[0]

    available = ", ".join(f"{f.name} {bounds}" for f, bounds in tagged)
    raise TileNotFound(
        f"No {dataset} tile covers lat={latitude}, lon={longitude}. Available: {available}"
    )
