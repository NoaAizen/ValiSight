#!/usr/bin/env python3
"""Building footprints with heights, ready to extrude into a 3D scene.

A footprint without a height is a line on the ground. To seat a camera image
onto a three-dimensional world every building needs a roof height, and the
sources disagree about how confidently they can supply one:

* **published** - Overture carries a height, from survey or imagery. Trust it.
* **from_floors** - only a storey count is known, so the height is that count
  times a nominal storey height. Good to a metre or two.
* **from_dsm** - neither is known, so the height is measured off the DEM as the
  surface standing above the surrounding ground. Only works for buildings large
  enough to register at 30 m posting.
* **assumed** - nothing worked, and a class-based default stands in.

Every building carries which of these produced its height. That distinction is
the whole point: a renderer can draw an assumed height differently, and a pose
solver can refuse to match against one. Collapsing them into a single number
would hide that most of a city is a guess.

The DSM path carries a trap of its own. GLO-30 is a surface model, so the value
over a building already includes the building. The roof height is therefore an
assignment -- roof sits at ground plus height -- and never an addition on top of
a DSM reading (CLAUDE.md trap #8).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

#: Nominal storey height in meters, used when only a floor count is known.
METRES_PER_FLOOR = 3.2

#: Fallbacks by building class, in meters, for footprints nothing else can
#: resolve. Deliberately conservative: under-drawing a building is a missing
#: feature, over-drawing one is a false landmark that a pose solver may latch on.
ASSUMED_HEIGHT_M: Dict[str, float] = {
    "house": 6.0,
    "detached": 6.0,
    "residential": 9.0,
    "apartments": 15.0,
    "commercial": 9.0,
    "retail": 6.0,
    "industrial": 8.0,
    "school": 9.0,
    "hospital": 20.0,
    "church": 12.0,
    "synagogue": 12.0,
    "mosque": 12.0,
    "garage": 3.0,
    "shed": 3.0,
    "roof": 3.0,
}
DEFAULT_ASSUMED_HEIGHT_M = 8.0

#: A building shorter than this is noise, taller than this is a data error.
MIN_PLAUSIBLE_HEIGHT_M = 2.0
MAX_PLAUSIBLE_HEIGHT_M = 200.0

#: A DSM at 30 m posting cannot see a footprint much smaller than one cell.
#: Below this area the DSM path is not attempted at all.
MIN_DSM_FOOTPRINT_M2 = 900.0


class HeightSource(str, Enum):
    PUBLISHED = "published"
    FROM_FLOORS = "from_floors"
    FROM_DSM = "from_dsm"
    ASSUMED = "assumed"

    @property
    def is_measured(self) -> bool:
        """Whether this height came from data rather than from a default."""
        return self in (HeightSource.PUBLISHED, HeightSource.FROM_FLOORS, HeightSource.FROM_DSM)


@dataclass(frozen=True)
class Building:
    """One footprint with everything needed to extrude it."""

    identifier: str
    #: Outer ring as (longitude, latitude) pairs, closed.
    ring: Tuple[Tuple[float, float], ...]
    height_m: float
    source: HeightSource
    name: Optional[str] = None
    building_class: Optional[str] = None
    #: Ground elevation under the footprint, orthometric, when known.
    ground_m: Optional[float] = None

    @property
    def centroid(self) -> Tuple[float, float]:
        """Area centroid of the ring, as (longitude, latitude)."""
        origin_x, origin_y, local = _localise(self.ring)
        if local is None:
            return (origin_x, origin_y)

        area = cx = cy = 0.0
        for (x0, y0), (x1, y1) in zip(local, local[1:] + local[:1]):
            cross = x0 * y1 - x1 * y0
            area += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross

        if abs(area) < 1e-20:  # degenerate ring: fall back to the mean vertex
            return (
                origin_x + sum(p[0] for p in local) / len(local),
                origin_y + sum(p[1] for p in local) / len(local),
            )
        return (origin_x + cx / (3.0 * area), origin_y + cy / (3.0 * area))

    @property
    def footprint_m2(self) -> float:
        """Ground area in square meters, via the shoelace formula."""
        origin_x, origin_y, local = _localise(self.ring)
        if local is None or len(local) < 3:
            return 0.0

        # Convert the local offsets to meters before summing, so the scale
        # factor is applied to small numbers rather than to absolute longitudes
        x_scale = 111_320.0 * math.cos(math.radians(origin_y))
        y_scale = 111_320.0
        area = 0.0
        for (x0, y0), (x1, y1) in zip(local, local[1:] + local[:1]):
            area += (x0 * x_scale) * (y1 * y_scale) - (x1 * x_scale) * (y0 * y_scale)
        return abs(area) / 2.0

    @property
    def roof_m(self) -> Optional[float]:
        """Absolute orthometric height of the roof.

        Assignment, not addition: the roof is the ground plus the building's
        own height. Adding a height to a DSM reading, which already contains the
        building, counts it twice (trap #8).
        """
        if self.ground_m is None:
            return None
        return self.ground_m + self.height_m


def _plausible(height: Optional[float]) -> bool:
    return height is not None and MIN_PLAUSIBLE_HEIGHT_M <= height <= MAX_PLAUSIBLE_HEIGHT_M


def resolve_height(
    published_m: Optional[float],
    num_floors: Optional[int],
    dsm_height_m: Optional[float] = None,
    building_class: Optional[str] = None,
    metres_per_floor: float = METRES_PER_FLOOR,
) -> Tuple[float, HeightSource]:
    """Best available height for one footprint, and where it came from.

    Ordered by how directly each source measured the building. Implausible
    values fall through to the next source rather than being clamped, since a
    height of 0.3 m or 900 m is a data error and clamping would disguise it as
    a measurement.
    """
    if _plausible(published_m):
        return (float(published_m), HeightSource.PUBLISHED)

    if num_floors is not None and num_floors > 0:
        from_floors = float(num_floors) * metres_per_floor
        if _plausible(from_floors):
            return (from_floors, HeightSource.FROM_FLOORS)

    if _plausible(dsm_height_m):
        return (float(dsm_height_m), HeightSource.FROM_DSM)

    key = (building_class or "").lower()
    return (ASSUMED_HEIGHT_M.get(key, DEFAULT_ASSUMED_HEIGHT_M), HeightSource.ASSUMED)


class BuildingLayer:
    """A cached set of footprints, with the heights needed to extrude them."""

    def __init__(self, buildings: Sequence[Building]) -> None:
        self.buildings = list(buildings)

    def __len__(self) -> int:
        return len(self.buildings)

    def __iter__(self) -> Iterator[Building]:
        return iter(self.buildings)

    @classmethod
    def from_geojson(cls, path: Path) -> "BuildingLayer":
        """Load a footprint file written by tools/fetch_overture.py or fetch_priors.py.

        Both layouts are accepted: Overture publishes ``height_m`` directly,
        while the OpenStreetMap stand-in carries ``height`` and ``levels`` as
        strings. Heights are resolved here rather than at fetch time so the
        rules live in one place.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Building layer not found: {path}")

        raw = json.loads(path.read_text())
        buildings = []
        for feature in raw.get("features", []):
            geometry = feature.get("geometry") or {}
            ring = _outer_ring(geometry)
            if ring is None:
                continue

            properties = feature.get("properties", {})
            height, source = resolve_height(
                published_m=_as_float(properties.get("height_m") or properties.get("height")),
                num_floors=_as_int(properties.get("num_floors") or properties.get("levels")),
                building_class=properties.get("class"),
            )
            buildings.append(Building(
                identifier=str(properties.get("id", "")),
                ring=ring,
                height_m=height,
                source=source,
                name=properties.get("name"),
                building_class=properties.get("class"),
            ))
        return cls(buildings)

    def coverage(self) -> Dict[str, int]:
        """How many buildings got their height from each source."""
        counts = {source.value: 0 for source in HeightSource}
        for building in self.buildings:
            counts[building.source.value] += 1
        return counts

    def measured_fraction(self) -> float:
        """Share of buildings whose height came from data rather than a default."""
        if not self.buildings:
            return 0.0
        measured = sum(1 for b in self.buildings if b.source.is_measured)
        return measured / len(self.buildings)

    def near(self, latitude: float, longitude: float, radius_m: float) -> List[Building]:
        """Buildings whose centroid falls within a radius, nearest first."""
        metres_per_deg_lat = 111_320.0
        metres_per_deg_lon = metres_per_deg_lat * math.cos(math.radians(latitude))

        found = []
        for building in self.buildings:
            lon, lat = building.centroid
            distance = math.hypot(
                (lat - latitude) * metres_per_deg_lat,
                (lon - longitude) * metres_per_deg_lon,
            )
            if distance <= radius_m:
                found.append((distance, building))
        return [building for _, building in sorted(found, key=lambda pair: pair[0])]

    def fill_heights_from_dsm(self, dem, max_buildings: Optional[int] = None) -> Dict[str, int]:
        """Measure heights off the DSM for buildings that have no better source.

        Only attempted for footprints large enough to register at the DEM's
        posting; a house smaller than one cell contributes no signal, and a
        number derived from it would be interpolation noise wearing the costume
        of a measurement.

        Returns how many were filled, skipped as too small, and left alone.
        """
        from .dem import DemUnavailable

        stats = {"filled": 0, "too_small": 0, "already_known": 0, "no_data": 0}
        updated = []
        attempted = 0

        for building in self.buildings:
            if building.source is not HeightSource.ASSUMED:
                stats["already_known"] += 1
                updated.append(building)
                continue
            if building.footprint_m2 < MIN_DSM_FOOTPRINT_M2:
                stats["too_small"] += 1
                updated.append(building)
                continue
            if max_buildings is not None and attempted >= max_buildings:
                updated.append(building)
                continue

            attempted += 1
            longitude, latitude = building.centroid
            try:
                estimate = dem.ground_elevation(latitude, longitude)
            except DemUnavailable:
                stats["no_data"] += 1
                updated.append(building)
                continue

            height, source = resolve_height(
                published_m=None,
                num_floors=None,
                dsm_height_m=estimate.structure_m,
                building_class=building.building_class,
            )
            if source is HeightSource.FROM_DSM:
                stats["filled"] += 1
            updated.append(Building(
                identifier=building.identifier,
                ring=building.ring,
                height_m=height,
                source=source,
                name=building.name,
                building_class=building.building_class,
                ground_m=estimate.ground_m,
            ))

        self.buildings = updated
        return stats


def _localise(ring):
    """Shift a ring to a local origin, returning (origin_x, origin_y, offsets).

    Building rings span a fraction of a degree while sitting at absolute
    coordinates two orders of magnitude larger, so shoelace terms become
    differences of nearly equal numbers and lose most of their significant
    digits. Working relative to the first vertex keeps the arithmetic on the
    small numbers where the precision actually is; without it the centroid of a
    40 m square lands about 20 cm off its own centre.
    """
    if not ring:
        return (0.0, 0.0, None)
    closed = ring[:-1] if ring[0] == ring[-1] else ring
    if not closed:
        return (ring[0][0], ring[0][1], None)

    origin_x, origin_y = closed[0]
    return (origin_x, origin_y, [(x - origin_x, y - origin_y) for x, y in closed])


def _ring_area(ring) -> float:
    """Unsigned shoelace area of a ring, in square degrees, for comparison only."""
    _, _, local = _localise(ring)
    if local is None or len(local) < 3:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(local, local[1:] + local[:1]):
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def _outer_ring(geometry: dict) -> Optional[Tuple[Tuple[float, float], ...]]:
    """Outer ring of a Polygon, or the largest part of a MultiPolygon."""
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if not coordinates:
        return None

    if kind == "Polygon":
        ring = coordinates[0]
    elif kind == "MultiPolygon":
        # Largest part by area stands in for the building's mass. Vertex count
        # would be wrong: a small ornate wing can carry more points than the
        # main block it is attached to.
        parts = [polygon[0] for polygon in coordinates if polygon and polygon[0]]
        ring = max(parts, key=_ring_area, default=None)
    else:
        return None

    if not ring or len(ring) < 4:
        return None
    return tuple((float(point[0]), float(point[1])) for point in ring)


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
