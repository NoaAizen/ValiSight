"""Tests for building footprints and the heights used to extrude them.

The height of a building comes from one of four places, and which one it came
from matters as much as the number. These tests pin the precedence between
them, the refusal to dress a guess up as a measurement, and the assignment
rule that keeps a DSM reading from counting a building twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conftest import write_dem  # noqa: E402
from mapinit.geo.buildings import (  # noqa: E402
    ASSUMED_HEIGHT_M,
    DEFAULT_ASSUMED_HEIGHT_M,
    MAX_PLAUSIBLE_HEIGHT_M,
    METRES_PER_FLOOR,
    MIN_DSM_FOOTPRINT_M2,
    Building,
    BuildingLayer,
    HeightSource,
    resolve_height,
)
from mapinit.geo.dem import DemSampler  # noqa: E402


def square(centre_lon: float, centre_lat: float, side_m: float):
    """A closed square ring of roughly the requested side, in degrees."""
    import math

    half_lat = (side_m / 2) / 111_320.0
    half_lon = (side_m / 2) / (111_320.0 * math.cos(math.radians(centre_lat)))
    return (
        (centre_lon - half_lon, centre_lat - half_lat),
        (centre_lon + half_lon, centre_lat - half_lat),
        (centre_lon + half_lon, centre_lat + half_lat),
        (centre_lon - half_lon, centre_lat + half_lat),
        (centre_lon - half_lon, centre_lat - half_lat),
    )


# --------------------------------------------------------------------------
# Height precedence
# --------------------------------------------------------------------------


def test_published_height_wins_over_everything():
    height, source = resolve_height(
        published_m=18.5, num_floors=3, dsm_height_m=9.0, building_class="house"
    )
    assert (height, source) == (18.5, HeightSource.PUBLISHED)


def test_floor_count_is_used_when_no_height_is_published():
    height, source = resolve_height(published_m=None, num_floors=4, dsm_height_m=30.0)
    assert source is HeightSource.FROM_FLOORS
    assert height == pytest.approx(4 * METRES_PER_FLOOR)


def test_dsm_is_the_third_choice():
    height, source = resolve_height(published_m=None, num_floors=None, dsm_height_m=11.4)
    assert (height, source) == (11.4, HeightSource.FROM_DSM)


def test_class_default_is_the_last_resort():
    height, source = resolve_height(
        published_m=None, num_floors=None, dsm_height_m=None, building_class="apartments"
    )
    assert source is HeightSource.ASSUMED
    assert height == ASSUMED_HEIGHT_M["apartments"]


def test_unknown_class_falls_back_to_a_generic_default():
    height, source = resolve_height(None, None, None, building_class="pagoda")
    assert (height, source) == (DEFAULT_ASSUMED_HEIGHT_M, HeightSource.ASSUMED)


@pytest.mark.parametrize("bad", [0.0, 0.3, -5.0, MAX_PLAUSIBLE_HEIGHT_M + 1, 9000.0])
def test_implausible_published_heights_fall_through_rather_than_being_clamped(bad):
    """A 900 m building is a data error, and clamping would hide that.

    Falling through means the result is labelled by whatever source actually
    supplied it, so nothing downstream mistakes a repaired value for a reading.
    """
    height, source = resolve_height(published_m=bad, num_floors=5, dsm_height_m=None)
    assert source is HeightSource.FROM_FLOORS
    assert height == pytest.approx(5 * METRES_PER_FLOOR)


def test_zero_floors_is_not_a_building_height():
    _, source = resolve_height(published_m=None, num_floors=0, dsm_height_m=None)
    assert source is HeightSource.ASSUMED


def test_only_measured_sources_report_as_measured():
    assert HeightSource.PUBLISHED.is_measured
    assert HeightSource.FROM_FLOORS.is_measured
    assert HeightSource.FROM_DSM.is_measured
    assert not HeightSource.ASSUMED.is_measured


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_footprint_area_matches_the_square_it_was_built_from():
    building = Building("b1", square(35.19, 31.76, 30.0), 9.0, HeightSource.PUBLISHED)
    assert building.footprint_m2 == pytest.approx(900.0, rel=0.02)


def test_centroid_sits_at_the_middle_of_a_square():
    building = Building("b1", square(35.19, 31.76, 40.0), 9.0, HeightSource.PUBLISHED)
    longitude, latitude = building.centroid
    assert longitude == pytest.approx(35.19, abs=1e-6)
    assert latitude == pytest.approx(31.76, abs=1e-6)


def test_degenerate_ring_does_not_divide_by_zero():
    collapsed = ((35.19, 31.76), (35.19, 31.76), (35.19, 31.76), (35.19, 31.76))
    building = Building("b1", collapsed, 9.0, HeightSource.ASSUMED)
    assert building.centroid == pytest.approx((35.19, 31.76))
    assert building.footprint_m2 == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# Trap #8: assignment, not addition
# --------------------------------------------------------------------------


def test_roof_is_ground_plus_height_not_a_dsm_reading_plus_height():
    """The DSM over a building already contains the building.

    Ground under this footprint is 802.1 m and the building is 7.4 m, so the
    roof is 809.5 m. Adding 7.4 to the DSM surface, which is already 809.5,
    would put the roof at 816.9 and count the building twice.
    """
    building = Building(
        "b1", square(35.1889, 31.7614, 30.0), 7.4, HeightSource.FROM_DSM, ground_m=802.1
    )
    assert building.roof_m == pytest.approx(809.5)


def test_roof_is_unknown_until_the_ground_is_known():
    building = Building("b1", square(35.19, 31.76, 30.0), 9.0, HeightSource.PUBLISHED)
    assert building.roof_m is None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def write_layer(path: Path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def feature(ring, **properties):
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring]]},
        "properties": properties,
    }


def test_overture_and_openstreetmap_layouts_both_load(tmp_path):
    """Overture publishes height_m as a number; the OSM stand-in uses strings."""
    path = write_layer(tmp_path / "b.geojson", [
        feature(square(35.19, 31.76, 30), id="ovt", height_m=21.0, num_floors=7),
        feature(square(35.20, 31.76, 30), id="osm", height="12.5", levels="4"),
    ])
    layer = BuildingLayer.from_geojson(path)

    assert len(layer) == 2
    by_id = {b.identifier: b for b in layer}
    assert by_id["ovt"].height_m == pytest.approx(21.0)
    assert by_id["osm"].height_m == pytest.approx(12.5)
    assert all(b.source is HeightSource.PUBLISHED for b in layer)


def test_multipolygon_uses_its_largest_part(tmp_path):
    path = tmp_path / "multi.geojson"
    big = [list(p) for p in square(35.19, 31.76, 60)]
    small = [list(p) for p in square(35.21, 31.76, 10)]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "geometry": {"type": "MultiPolygon", "coordinates": [[small], [big]]},
        "properties": {"id": "m1", "height_m": 10.0},
    }]}))

    layer = BuildingLayer.from_geojson(path)
    assert len(layer) == 1
    assert layer.buildings[0].footprint_m2 == pytest.approx(3600.0, rel=0.05)


def test_features_without_usable_geometry_are_skipped(tmp_path):
    path = tmp_path / "mixed.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [35.19, 31.76]},
         "properties": {"id": "point"}},
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[35.19, 31.76]]]},
         "properties": {"id": "too-few-vertices"}},
        feature(square(35.19, 31.76, 30), id="good", height_m=9.0),
    ]}))

    layer = BuildingLayer.from_geojson(path)
    assert [b.identifier for b in layer] == ["good"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BuildingLayer.from_geojson(tmp_path / "absent.geojson")


# --------------------------------------------------------------------------
# Coverage reporting
# --------------------------------------------------------------------------


def test_coverage_counts_every_source(tmp_path):
    path = write_layer(tmp_path / "b.geojson", [
        feature(square(35.19, 31.760, 30), id="a", height_m=20.0),
        feature(square(35.19, 31.761, 30), id="b", num_floors=3),
        feature(square(35.19, 31.762, 30), id="c", **{"class": "house"}),
        feature(square(35.19, 31.763, 30), id="d"),
    ])
    layer = BuildingLayer.from_geojson(path)

    assert layer.coverage() == {
        "published": 1, "from_floors": 1, "from_dsm": 0, "assumed": 2
    }
    assert layer.measured_fraction() == pytest.approx(0.5)


def test_measured_fraction_of_an_empty_layer_is_zero(tmp_path):
    layer = BuildingLayer.from_geojson(write_layer(tmp_path / "empty.geojson", []))
    assert layer.measured_fraction() == 0.0


def test_near_returns_buildings_sorted_by_distance(tmp_path):
    path = write_layer(tmp_path / "b.geojson", [
        feature(square(35.1889, 31.7614, 20), id="here", height_m=9.0),
        feature(square(35.1900, 31.7614, 20), id="near", height_m=9.0),
        feature(square(35.2500, 31.7614, 20), id="far", height_m=9.0),
    ])
    layer = BuildingLayer.from_geojson(path)

    found = layer.near(31.7614, 35.1889, radius_m=500)
    assert [b.identifier for b in found] == ["here", "near"]


# --------------------------------------------------------------------------
# Filling from the DSM
# --------------------------------------------------------------------------


@pytest.fixture
def dem(tmp_path):
    with DemSampler(write_dem(tmp_path / "dem.tif", 31.0, 35.0)) as sampler:
        yield sampler


def test_small_footprints_are_not_measured_from_a_30_metre_dem(dem, tmp_path):
    """A house smaller than one DEM cell contributes no signal.

    Jerusalem's median footprint is under 300 square metres, so this path
    declines for most of the city rather than returning interpolation noise
    dressed as a measurement.
    """
    path = write_layer(tmp_path / "b.geojson", [
        feature(square(35.5, 31.5, 15), id="house"),        # 225 m2
        feature(square(35.5, 31.6, 40), id="block"),        # 1600 m2
    ])
    layer = BuildingLayer.from_geojson(path)
    stats = layer.fill_heights_from_dsm(dem)

    assert stats["too_small"] == 1
    by_id = {b.identifier: b for b in layer}
    assert by_id["house"].source is HeightSource.ASSUMED
    assert by_id["house"].ground_m is None


def test_buildings_that_already_have_a_height_are_left_alone(dem, tmp_path):
    path = write_layer(tmp_path / "b.geojson", [
        feature(square(35.5, 31.5, 60), id="known", height_m=24.0),
    ])
    layer = BuildingLayer.from_geojson(path)
    stats = layer.fill_heights_from_dsm(dem)

    assert stats["already_known"] == 1 and stats["filled"] == 0
    assert layer.buildings[0].height_m == pytest.approx(24.0)


def test_the_dsm_threshold_is_about_one_cell(dem):
    """900 square metres is a 30 by 30 metre footprint: exactly one DEM cell."""
    assert MIN_DSM_FOOTPRINT_M2 == pytest.approx(30.0 * 30.0)
