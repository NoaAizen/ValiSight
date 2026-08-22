#!/usr/bin/env python3
"""Predict what a camera at a known pose should see of the map.

This is the piece that makes an image comparable with a map. Given where the
camera is and which way it faces, it produces the two features an outdoor scene
reduces to:

* **vertical edges** — building corners. A corner is a vertical line in the
  world and projects to a vertical line in the image, and its horizontal
  position depends only on the footprint and the viewer's pose. Height decides
  where the line stops, not where it is, which is why footprints alone carry
  the signal even though almost no building in the cache has a published
  height.
* **the skyline** — the highest terrain along each bearing. Where there are no
  buildings, and about half of the operating area has none within reach, the
  ridgeline is what remains. A 30 m posting resolves nothing at obstacle scale
  but resolves a ridge a kilometre off perfectly well.

Both are produced as a function of bearing, because bearing is what an image
measures well and what heading is read from. Nothing here needs the IMU: the
prediction is made for a heading, and the heading that best matches an image is
the answer.

Occlusion between buildings is decided by footprint alone. At street level a
nearer building spanning the same bearing hides a farther one whatever their
heights are, so the missing height data costs nothing here either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .buildings import Building, BuildingLayer
    from .dem import DemSampler

#: Bearing resolution of a prediction, in degrees. Matched to the Lepton, whose
#: measured 53.4 degrees across 160 columns give 0.33 degrees per pixel; finer
#: than the image can resolve buys nothing.
DEFAULT_BEARING_STEP_DEG = 0.25

#: Default field of view, taken from the rig's own calibration rather than the
#: Lepton datasheet. The two differ by 3.6 degrees, which is eleven pixels.
DEFAULT_HFOV_DEG = 53.4

#: How far to look. Beyond this a building edge is a fraction of a pixel wide
#: and the DEM's own error dominates the elevation angle.
DEFAULT_MAX_RANGE_M = 400.0
DEFAULT_SKYLINE_RANGE_M = 6000.0

#: Where the skyline march begins. Nearer than this the DEM is reporting
#: rooftops, and at 30 m posting a rooftop is a smear rather than a silhouette.
#: Buildings inside this radius are the edge detector's job.
DEFAULT_SKYLINE_MIN_RANGE_M = 200.0

#: Step along a ray when marching for the skyline. Half the DEM posting, so no
#: cell is stepped over.
SKYLINE_STEP_M = 15.0

METRES_PER_DEG_LAT = 111_320.0


@dataclass(frozen=True)
class VerticalEdge:
    """A building corner, as the camera would see it."""

    #: Bearing from the viewer, degrees clockwise from north.
    bearing_deg: float
    #: Bearing relative to where the camera is pointing; 0 is frame centre.
    relative_deg: float
    range_m: float
    building_id: str
    building_name: Optional[str] = None
    #: True when a nearer footprint spans this bearing and hides the corner.
    occluded: bool = False

    def pixel_column(self, width_px: int, hfov_deg: float) -> float:
        """Where this edge falls across the frame, in pixels from the left."""
        focal = (width_px / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
        return width_px / 2.0 + focal * math.tan(math.radians(self.relative_deg))


@dataclass(frozen=True)
class SkylinePoint:
    """The highest terrain along one bearing, and how far up it sits."""

    bearing_deg: float
    relative_deg: float
    #: Elevation angle above horizontal, degrees. Negative looks down.
    elevation_deg: float
    range_m: float
    terrain_m: float


@dataclass
class PredictedView:
    """What the map says should be in frame from one pose."""

    latitude: float
    longitude: float
    #: Camera height above ground, meters.
    height_agl_m: float
    heading_deg: float
    hfov_deg: float

    edges: List[VerticalEdge] = field(default_factory=list)
    skyline: List[SkylinePoint] = field(default_factory=list)
    #: Ground elevation under the viewer, orthometric.
    ground_m: float = 0.0

    @property
    def visible_edges(self) -> List[VerticalEdge]:
        return [edge for edge in self.edges if not edge.occluded]

    def edge_bearings(self) -> List[float]:
        """Relative bearings of visible edges, sorted across the frame."""
        return sorted(edge.relative_deg for edge in self.visible_edges)

    def __str__(self) -> str:
        visible = len(self.visible_edges)
        horizon = (
            f"{min(p.elevation_deg for p in self.skyline):+.1f} to "
            f"{max(p.elevation_deg for p in self.skyline):+.1f} deg"
            if self.skyline else "not computed"
        )
        return (
            f"view from {self.latitude:.5f}, {self.longitude:.5f} "
            f"at {self.height_agl_m:.1f} m AGL, heading {self.heading_deg:.1f} deg\n"
            f"  {visible} visible building edges of {len(self.edges)} in the "
            f"{self.hfov_deg:.0f} deg field\n"
            f"  skyline spans {horizon}"
        )


def _scales(latitude: float) -> Tuple[float, float]:
    """Meters per degree of latitude and of longitude at this latitude."""
    return (METRES_PER_DEG_LAT, METRES_PER_DEG_LAT * math.cos(math.radians(latitude)))


def bearing_and_range(
    latitude: float, longitude: float, to_lat: float, to_lon: float
) -> Tuple[float, float]:
    """Bearing in degrees clockwise from north, and ground distance in meters."""
    lat_scale, lon_scale = _scales(latitude)
    east = (to_lon - longitude) * lon_scale
    north = (to_lat - latitude) * lat_scale
    return (math.degrees(math.atan2(east, north)) % 360.0, math.hypot(east, north))


def relative_bearing(bearing_deg: float, heading_deg: float) -> float:
    """Fold a bearing into (-180, 180] relative to where the camera points."""
    return (bearing_deg - heading_deg + 180.0) % 360.0 - 180.0


class ViewPredictor:
    """Turns a pose into the map features that should appear in frame."""

    def __init__(
        self,
        buildings: "BuildingLayer",
        dem: Optional["DemSampler"] = None,
        bearing_step_deg: float = DEFAULT_BEARING_STEP_DEG,
    ) -> None:
        self.buildings = buildings
        self.dem = dem
        self.bearing_step_deg = bearing_step_deg

    # -- building edges ---------------------------------------------------

    def _corner_edges(
        self,
        latitude: float,
        longitude: float,
        heading_deg: float,
        hfov_deg: float,
        max_range_m: float,
    ) -> List[VerticalEdge]:
        """Every building corner inside the field, nearest first."""
        half_fov = hfov_deg / 2.0
        edges = []

        for building in self.buildings.near(latitude, longitude, max_range_m):
            # A closed ring repeats its first vertex; counting it twice would
            # double an edge that is only there once
            ring = building.ring[:-1] if building.ring[0] == building.ring[-1] else building.ring
            for corner_lon, corner_lat in ring:
                bearing, distance = bearing_and_range(latitude, longitude, corner_lat, corner_lon)
                if distance < 1.0 or distance > max_range_m:
                    continue
                relative = relative_bearing(bearing, heading_deg)
                if abs(relative) > half_fov:
                    continue
                edges.append(VerticalEdge(
                    bearing_deg=bearing,
                    relative_deg=relative,
                    range_m=distance,
                    building_id=building.identifier,
                    building_name=building.name,
                ))

        return sorted(edges, key=lambda edge: edge.range_m)

    def _mark_occluded(
        self,
        latitude: float,
        longitude: float,
        edges: Sequence[VerticalEdge],
        max_range_m: float,
    ) -> List[VerticalEdge]:
        """Hide corners that a nearer footprint stands in front of.

        Decided from footprints alone. At street level a nearer building
        spanning the same bearing blocks the view along it regardless of how
        tall either one is, so this needs none of the height data the cache
        mostly lacks.
        """
        # Bearing span each building occupies, and how near its closest point is
        spans = []
        for building in self.buildings.near(latitude, longitude, max_range_m):
            ring = building.ring[:-1] if building.ring[0] == building.ring[-1] else building.ring
            bearings, ranges = [], []
            for corner_lon, corner_lat in ring:
                bearing, distance = bearing_and_range(latitude, longitude, corner_lat, corner_lon)
                bearings.append(bearing)
                ranges.append(distance)
            if not bearings or min(ranges) < 1.0:
                continue

            # Work relative to the first corner so a span crossing due north
            # does not appear to wrap the long way round
            reference = bearings[0]
            offsets = [((b - reference + 180.0) % 360.0) - 180.0 for b in bearings]
            spans.append((
                building.identifier,
                reference + min(offsets),
                reference + max(offsets),
                min(ranges),
                max(ranges),
            ))

        marked = []
        for edge in edges:
            occluded = False
            for identifier, low, high, nearest, farthest in spans:
                if identifier == edge.building_id or nearest >= edge.range_m:
                    continue
                offset = ((edge.bearing_deg - low + 180.0) % 360.0) - 180.0
                if 0.0 <= offset <= (high - low) and farthest < edge.range_m:
                    occluded = True
                    break
            marked.append(VerticalEdge(
                bearing_deg=edge.bearing_deg,
                relative_deg=edge.relative_deg,
                range_m=edge.range_m,
                building_id=edge.building_id,
                building_name=edge.building_name,
                occluded=occluded,
            ))
        return marked

    # -- skyline ----------------------------------------------------------

    def _skyline(
        self,
        latitude: float,
        longitude: float,
        eye_m: float,
        heading_deg: float,
        hfov_deg: float,
        max_range_m: float,
        min_range_m: float,
    ) -> List[SkylinePoint]:
        """Highest terrain along each bearing, as an elevation angle.

        Marching outward and keeping the largest elevation angle finds the
        horizon rather than the farthest ground: a near ridge hides everything
        behind it, and it is the ridge that appears against the sky.
        """
        if self.dem is None:
            return []

        import numpy as np

        lat_scale, lon_scale = _scales(latitude)
        half_fov = hfov_deg / 2.0

        relatives = np.arange(-half_fov, half_fov + 1e-9, self.bearing_step_deg)
        # Start beyond the near field. Within it the DEM is reporting rooftops
        # rather than terrain, at a posting far too coarse to render a building
        # silhouette; those bearings are covered by the edge detector instead.
        distances = np.arange(min_range_m, max_range_m + 1e-9, SKYLINE_STEP_M)

        bearings = (heading_deg + relatives) % 360.0
        east = np.sin(np.radians(bearings))[:, None]
        north = np.cos(np.radians(bearings))[:, None]
        spans = distances[None, :]

        latitudes = latitude + (north * spans) / lat_scale
        longitudes = longitude + (east * spans) / lon_scale

        terrain = self.dem.sample_many(latitudes.ravel(), longitudes.ravel())
        terrain = terrain.reshape(latitudes.shape)

        with np.errstate(invalid="ignore"):
            angles = np.degrees(np.arctan2(terrain - eye_m, spans))
        angles = np.where(np.isnan(angles), -90.0, angles)

        # The horizon is the highest angle along a ray, not the farthest ground:
        # a near ridge hides everything behind it and is what meets the sky
        peak = np.argmax(angles, axis=1)
        rows = np.arange(len(relatives))

        return [
            SkylinePoint(
                bearing_deg=float(bearings[i]),
                relative_deg=float(relatives[i]),
                elevation_deg=float(angles[i, peak[i]]),
                range_m=float(distances[peak[i]]),
                terrain_m=float(terrain[i, peak[i]]) if not math.isnan(terrain[i, peak[i]]) else 0.0,
            )
            for i in rows
        ]

    # -- the prediction ---------------------------------------------------

    def predict(
        self,
        latitude: float,
        longitude: float,
        heading_deg: float,
        height_agl_m: float = 1.5,
        hfov_deg: float = DEFAULT_HFOV_DEG,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
        skyline_range_m: float = DEFAULT_SKYLINE_RANGE_M,
        skyline_min_range_m: float = DEFAULT_SKYLINE_MIN_RANGE_M,
        with_skyline: bool = True,
    ) -> PredictedView:
        """What the map says is in frame from this pose."""
        ground = 0.0
        if self.dem is not None:
            ground = self.dem.ground_elevation(latitude, longitude).ground_m

        edges = self._corner_edges(latitude, longitude, heading_deg, hfov_deg, max_range_m)
        edges = self._mark_occluded(latitude, longitude, edges, max_range_m)

        skyline = []
        if with_skyline and self.dem is not None:
            skyline = self._skyline(
                latitude, longitude, ground + height_agl_m,
                heading_deg, hfov_deg, skyline_range_m, skyline_min_range_m,
            )

        return PredictedView(
            latitude=latitude,
            longitude=longitude,
            height_agl_m=height_agl_m,
            heading_deg=heading_deg,
            hfov_deg=hfov_deg,
            edges=edges,
            skyline=skyline,
            ground_m=ground,
        )

    def sweep_headings(
        self,
        latitude: float,
        longitude: float,
        step_deg: float = 5.0,
        height_agl_m: float = 1.5,
        hfov_deg: float = DEFAULT_HFOV_DEG,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
    ) -> List[PredictedView]:
        """Predict the view for every heading around the compass.

        This is how a heading is recovered from a single photograph: predict all
        of them and keep whichever matches. It also answers whether recovery is
        possible at all — if every heading from a spot looks alike, no matcher
        will separate them.

        Which corners exist and what hides them does not depend on where the
        camera points, only on where it stands, so that work is done once and
        each heading is then a filter over the result. Recomputing it per
        heading costs a second each and yields exactly the same edges.
        """
        # Bearings and occlusion are heading-independent: compute over the full
        # circle once
        all_edges = self._corner_edges(
            latitude, longitude, heading_deg=0.0, hfov_deg=360.0, max_range_m=max_range_m
        )
        all_edges = self._mark_occluded(latitude, longitude, all_edges, max_range_m)

        ground = 0.0
        if self.dem is not None:
            ground = self.dem.ground_elevation(latitude, longitude).ground_m

        half_fov = hfov_deg / 2.0
        views = []
        for index in range(int(round(360.0 / step_deg))):
            heading = step_deg * index
            in_frame = []
            for edge in all_edges:
                relative = relative_bearing(edge.bearing_deg, heading)
                if abs(relative) > half_fov:
                    continue
                in_frame.append(VerticalEdge(
                    bearing_deg=edge.bearing_deg,
                    relative_deg=relative,
                    range_m=edge.range_m,
                    building_id=edge.building_id,
                    building_name=edge.building_name,
                    occluded=edge.occluded,
                ))
            views.append(PredictedView(
                latitude=latitude,
                longitude=longitude,
                height_agl_m=height_agl_m,
                heading_deg=heading,
                hfov_deg=hfov_deg,
                edges=in_frame,
                ground_m=ground,
            ))
        return views
