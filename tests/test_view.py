"""Tests for predicting what a camera should see of the map.

The prediction is what an image gets compared against, so an error here is
invisible until it silently reports the wrong heading. These pin the geometry
against cases whose answer can be worked out by hand, and the invariants that
must hold whatever the data.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conftest import write_dem  # noqa: E402
from mapinit.geo.buildings import BuildingLayer  # noqa: E402
from mapinit.geo.dem import DemSampler  # noqa: E402
from mapinit.geo.view import (  # noqa: E402
    ViewPredictor,
    bearing_and_range,
    relative_bearing,
)

ORIGIN_LAT, ORIGIN_LON = 31.5, 35.5


def offset(east_m: float, north_m: float):
    """A point a given number of metres east and north of the origin."""
    lat_scale = 111_320.0
    lon_scale = lat_scale * math.cos(math.radians(ORIGIN_LAT))
    return (ORIGIN_LON + east_m / lon_scale, ORIGIN_LAT + north_m / lat_scale)


def square_at(east_m: float, north_m: float, side_m: float = 20.0):
    """A closed square footprint centred on an offset from the origin."""
    half = side_m / 2
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    ring = [offset(east_m + dx, north_m + dy) for dx, dy in corners]
    return ring + [ring[0]]


def layer(*footprints, **properties):
    """A BuildingLayer built in memory from squares."""
    import tempfile

    features = []
    for index, ring in enumerate(footprints):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring]]},
            "properties": {"id": f"b{index}", "height_m": 12.0, **properties},
        })
    handle = tempfile.NamedTemporaryFile("w", suffix=".geojson", delete=False)
    json.dump({"type": "FeatureCollection", "features": features}, handle)
    handle.close()
    return BuildingLayer.from_geojson(Path(handle.name))


# --------------------------------------------------------------------------
# Bearings
# --------------------------------------------------------------------------


@pytest.mark.parametrize("east, north, expected", [
    pytest.param(0, 100, 0.0, id="due-north"),
    pytest.param(100, 0, 90.0, id="due-east"),
    pytest.param(0, -100, 180.0, id="due-south"),
    pytest.param(-100, 0, 270.0, id="due-west"),
])
def test_bearing_is_clockwise_from_north(east, north, expected):
    lon, lat = offset(east, north)
    bearing, distance = bearing_and_range(ORIGIN_LAT, ORIGIN_LON, lat, lon)
    assert bearing == pytest.approx(expected, abs=0.5)
    assert distance == pytest.approx(100.0, rel=0.01)


def test_relative_bearing_folds_across_north():
    """359 degrees seen from a heading of 1 is two degrees left, not 358 right."""
    assert relative_bearing(359.0, 1.0) == pytest.approx(-2.0)
    assert relative_bearing(1.0, 359.0) == pytest.approx(2.0)
    assert relative_bearing(90.0, 90.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Which corners land in frame
# --------------------------------------------------------------------------


def test_a_building_dead_ahead_appears_near_frame_centre():
    predictor = ViewPredictor(layer(square_at(0, 60)))
    view = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0, with_skyline=False)

    assert len(view.visible_edges) == 4
    assert min(abs(edge.relative_deg) for edge in view.visible_edges) < 12.0


def test_a_building_behind_the_camera_is_not_in_frame():
    predictor = ViewPredictor(layer(square_at(0, 60)))
    view = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=180.0, with_skyline=False)
    assert view.visible_edges == []


def test_turning_the_camera_moves_edges_across_the_frame_by_the_same_angle():
    """A ten degree turn shifts every edge ten degrees the other way.

    This is the relationship heading recovery depends on, so it is worth
    pinning directly rather than trusting it to fall out of the projection.
    """
    predictor = ViewPredictor(layer(square_at(0, 80)))
    straight = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0, with_skyline=False)
    turned = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=10.0, with_skyline=False)

    for before, after in zip(straight.edge_bearings(), turned.edge_bearings()):
        assert after == pytest.approx(before - 10.0, abs=0.01)


def test_a_closed_ring_does_not_report_its_first_corner_twice():
    predictor = ViewPredictor(layer(square_at(0, 60)))
    view = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0, hfov_deg=180.0, with_skyline=False)
    assert len(view.edges) == 4, "a square has four corners, not five"


def test_edges_beyond_the_range_limit_are_dropped():
    predictor = ViewPredictor(layer(square_at(0, 60), square_at(0, 500)))
    view = predictor.predict(
        ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0, max_range_m=200.0, with_skyline=False
    )
    assert all(edge.range_m <= 200.0 for edge in view.edges)
    assert len(view.edges) == 4


def test_pixel_column_places_frame_centre_at_the_middle():
    predictor = ViewPredictor(layer(square_at(0, 100)))
    view = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0, with_skyline=False)
    centre = min(view.visible_edges, key=lambda e: abs(e.relative_deg))
    assert centre.pixel_column(160, 57.0) == pytest.approx(80.0, abs=20.0)


def test_an_edge_at_the_field_limit_lands_at_the_frame_border():
    """Half the field of view maps to the edge of the image."""
    from mapinit.geo.view import VerticalEdge

    edge = VerticalEdge(bearing_deg=0.0, relative_deg=28.5, range_m=50.0, building_id="x")
    assert edge.pixel_column(160, 57.0) == pytest.approx(160.0, abs=1.0)


# --------------------------------------------------------------------------
# Occlusion
# --------------------------------------------------------------------------


def test_a_nearer_building_hides_one_directly_behind_it():
    """Decided from footprints alone, which is why missing heights cost nothing."""
    predictor = ViewPredictor(layer(square_at(0, 40), square_at(0, 120)))
    view = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0, with_skyline=False)

    far = [e for e in view.edges if e.range_m > 80]
    assert far, "the far building should be in frame at all"
    assert all(edge.occluded for edge in far), "it stands directly behind the near one"
    assert all(not edge.occluded for edge in view.edges if e_range(edge) < 80)


def e_range(edge):
    return edge.range_m


def test_buildings_side_by_side_do_not_hide_each_other():
    predictor = ViewPredictor(layer(square_at(-40, 80), square_at(40, 80)))
    view = predictor.predict(
        ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0, hfov_deg=120.0, with_skyline=False
    )
    assert len(view.visible_edges) == 8


def test_a_building_never_occludes_itself():
    predictor = ViewPredictor(layer(square_at(0, 30)))
    view = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0, with_skyline=False)
    assert all(not edge.occluded for edge in view.edges)


# --------------------------------------------------------------------------
# The skyline
# --------------------------------------------------------------------------


@pytest.fixture
def dem(tmp_path):
    with DemSampler(write_dem(tmp_path / "dem.tif", 31.0, 35.0)) as sampler:
        yield sampler


def test_skyline_is_computed_for_every_bearing_in_the_field(dem):
    predictor = ViewPredictor(layer(), dem, bearing_step_deg=1.0)
    view = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0, hfov_deg=57.0)
    assert len(view.skyline) == pytest.approx(58, abs=1)
    assert all(-90 <= point.elevation_deg <= 90 for point in view.skyline)


def test_skyline_bearings_span_the_field_symmetrically(dem):
    predictor = ViewPredictor(layer(), dem, bearing_step_deg=1.0)
    view = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=90.0, hfov_deg=40.0)
    relatives = [point.relative_deg for point in view.skyline]
    assert min(relatives) == pytest.approx(-20.0, abs=0.5)
    assert max(relatives) == pytest.approx(20.0, abs=0.5)


def test_skyline_skips_the_near_field_where_the_dem_reports_rooftops(dem):
    """Starting the march at the viewer's feet finds a roof, not a ridge.

    At 30 m posting a building silhouette is a smear, and it sits at a steep
    angle that would swamp any real horizon behind it.
    """
    predictor = ViewPredictor(layer(), dem, bearing_step_deg=5.0)
    view = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0)
    assert all(point.range_m >= 200.0 for point in view.skyline if point.range_m > 0)


def test_a_higher_camera_sees_a_lower_horizon(dem):
    predictor = ViewPredictor(layer(), dem, bearing_step_deg=5.0)
    low = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0, height_agl_m=1.5)
    high = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0, height_agl_m=50.0)

    low_mean = sum(p.elevation_deg for p in low.skyline) / len(low.skyline)
    high_mean = sum(p.elevation_deg for p in high.skyline) / len(high.skyline)
    assert high_mean < low_mean


def test_no_dem_means_no_skyline_rather_than_a_wrong_one():
    predictor = ViewPredictor(layer(square_at(0, 60)), dem=None)
    view = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=0.0)
    assert view.skyline == []
    assert view.visible_edges, "edges still work without terrain"


# --------------------------------------------------------------------------
# The heading sweep
# --------------------------------------------------------------------------


def test_sweep_covers_the_whole_compass():
    predictor = ViewPredictor(layer(square_at(0, 60), square_at(60, 0), square_at(-60, -30)))
    views = predictor.sweep_headings(ORIGIN_LAT, ORIGIN_LON, step_deg=10.0)
    assert len(views) == 36
    assert [v.heading_deg for v in views][:3] == [0.0, 10.0, 20.0]


def test_sweep_agrees_with_predicting_each_heading_separately():
    """The fast path must produce exactly what the slow one does.

    Corners and occlusion do not depend on where the camera points, so the
    sweep computes them once. That optimisation is only safe if it changes
    nothing, which is what this checks.
    """
    buildings = layer(square_at(0, 60), square_at(50, 40), square_at(-45, 55))
    predictor = ViewPredictor(buildings)

    swept = {v.heading_deg: v for v in predictor.sweep_headings(ORIGIN_LAT, ORIGIN_LON, step_deg=30.0)}
    for heading, fast in swept.items():
        slow = predictor.predict(ORIGIN_LAT, ORIGIN_LON, heading_deg=heading, with_skyline=False)
        assert fast.edge_bearings() == pytest.approx(slow.edge_bearings(), abs=1e-9)


def test_every_building_is_seen_from_some_heading():
    predictor = ViewPredictor(layer(square_at(0, 60), square_at(60, 0), square_at(0, -60)))
    views = predictor.sweep_headings(ORIGIN_LAT, ORIGIN_LON, step_deg=15.0)
    seen = {edge.building_id for view in views for edge in view.visible_edges}
    assert seen == {"b0", "b1", "b2"}


def test_headings_are_distinguishable_by_their_edge_pattern():
    """If every heading looked alike, no matcher could recover one."""
    predictor = ViewPredictor(layer(square_at(0, 60), square_at(70, 20), square_at(-50, -40)))
    views = predictor.sweep_headings(ORIGIN_LAT, ORIGIN_LON, step_deg=30.0)

    patterns = [tuple(round(b, 1) for b in view.edge_bearings()) for view in views]
    assert len(set(patterns)) > len(patterns) / 2, "too many headings look identical"
