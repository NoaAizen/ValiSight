"""Validation against the real Jerusalem prior cache.

Everywhere else the tests build their own rasters, which proves the code does
what it was written to do but not that it does the right thing to real data.
These run against the actual cache: the Copernicus GLO-30 tile covering
Jerusalem and the OpenStreetMap building extract for the city.

Populate it once with::

    python tools/fetch_priors.py --jerusalem

Everything here skips cleanly when the cache is absent, since it is gitignored
and nobody should have to download 68 MB to run the suite.

Expected elevations are taken from the DEM itself rather than from an outside
source, so these are consistency and regression checks, not an independent
accuracy assessment of GLO-30.
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

from mapinit import MapInitializer  # noqa: E402
from mapinit.geo.dem import DemSampler, ScaleNotSupported  # noqa: E402
from mapinit.geo.geoid import EGM2008_GRID_NAME  # noqa: E402

PRIORS = REPO_ROOT / "data" / "priors"
DEM_PATH = PRIORS / "glo30" / "Copernicus_DSM_COG_10_N31_00_E035_00_DEM.tif"
VECTOR_PATH = PRIORS / "overture" / "overture_N31_00_E035_00_buildings.geojson"

requires_cache = pytest.mark.skipif(
    not (DEM_PATH.is_file() and VECTOR_PATH.is_file()),
    reason="Jerusalem prior cache absent. Populate it with: python tools/fetch_priors.py --jerusalem",
)
requires_geoid = pytest.mark.skipif(
    not (REPO_ROOT / "data" / EGM2008_GRID_NAME).is_file(),
    reason=f"data/{EGM2008_GRID_NAME} absent",
)

#: Jerusalem neighbourhoods the rig is expected to operate in, with the ground
#: elevation the DEM reports for each. Bounds are wide enough to tolerate the
#: sampling method but tight enough to catch a coordinate mix-up.
NEIGHBOURHOODS = [
    pytest.param(31.7614, 35.1889, 790, 820, id="givat-mordechai"),
    pytest.param(31.7590, 35.1900, 750, 785, id="bet-vegan"),
    pytest.param(31.7530, 35.1790, 695, 730, id="kiryat-hayovel"),
    pytest.param(31.7690, 35.1870, 755, 790, id="ramat-bet-hakerem"),
    pytest.param(31.76465, 35.19134, 720, 760, id="givat-ram-campus"),
]


@pytest.fixture(scope="module")
def dem():
    with DemSampler(DEM_PATH) as sampler:
        yield sampler


@pytest.fixture(scope="module")
def buildings():
    return json.loads(VECTOR_PATH.read_text())


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


@requires_cache
def test_dem_covers_the_whole_of_jerusalem(dem):
    """The tile spans a full degree, so the city sits well inside it."""
    for name, (latitude, longitude) in {
        "north": (31.87, 35.22), "south": (31.71, 35.20),
        "east": (31.78, 35.28), "west": (31.77, 35.15),
    }.items():
        assert dem.covers(latitude, longitude), f"{name} edge of Jerusalem not covered"


@requires_cache
@pytest.mark.parametrize("latitude, longitude, low, high", NEIGHBOURHOODS)
def test_neighbourhood_elevation_is_plausible(dem, latitude, longitude, low, high):
    estimate = dem.ground_elevation(latitude, longitude)
    assert low <= estimate.ground_m <= high, f"got {estimate}"


@requires_cache
@pytest.mark.parametrize("latitude, longitude, low, high", NEIGHBOURHOODS)
def test_built_up_areas_show_structure_above_ground(dem, latitude, longitude, low, high):
    """GLO-30 is a DSM, so in a built-up neighbourhood the surface stands proud.

    Zero separation would mean the ring estimate is reading the same pixels as
    the point, which is the failure the ring exists to avoid.
    """
    estimate = dem.ground_elevation(latitude, longitude)
    assert estimate.structure_m > 0.0
    assert estimate.structure_m < 60.0, "more than this is not a building, it is a slope"


@requires_cache
def test_the_vector_layer_records_its_true_extent(buildings):
    """The tile tag in the filename says one degree; the bbox says what is real."""
    minimum_longitude, minimum_latitude, maximum_longitude, maximum_latitude = buildings["bbox"]
    assert minimum_latitude < 31.7590 < maximum_latitude, "Bet VeGan outside the extract"
    assert minimum_longitude < 35.1889 < maximum_longitude, "Givat Mordechai outside the extract"
    # Real extent is a city, not the full degree the filename implies
    assert (maximum_latitude - minimum_latitude) < 1.0
    assert buildings["properties"]["feature_count"] == len(buildings["features"])


@requires_cache
@pytest.mark.parametrize("latitude, longitude, low, high", NEIGHBOURHOODS)
def test_every_neighbourhood_has_buildings_nearby(buildings, latitude, longitude, low, high):
    nearby = sum(
        1 for feature in buildings["features"]
        if math.hypot(
            (feature["geometry"]["coordinates"][0][0][1] - latitude) * 111_000,
            (feature["geometry"]["coordinates"][0][0][0] - longitude) * 94_000,
        ) < 600
    )
    assert nearby > 50, f"only {nearby} buildings within 600 m"


@requires_cache
def test_building_polygons_are_closed_rings(buildings):
    for feature in buildings["features"][:500]:
        ring = feature["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1], "GeoJSON polygons must close"
        assert len(ring) >= 4


# --------------------------------------------------------------------------
# Terrain relationships
# --------------------------------------------------------------------------


@requires_cache
def test_relative_heights_between_neighbourhoods_are_right(dem):
    """Kiryat HaYovel sits in a valley below the ridge neighbourhoods.

    An east-west or sign mix-up in the sampling would scramble this ordering
    while leaving every individual elevation inside its plausible band.
    """
    givat_mordechai = dem.ground_elevation(31.7614, 35.1889).ground_m
    bet_vegan = dem.ground_elevation(31.7590, 35.1900).ground_m
    kiryat_hayovel = dem.ground_elevation(31.7530, 35.1790).ground_m

    assert kiryat_hayovel < bet_vegan
    assert kiryat_hayovel < givat_mordechai


@requires_cache
def test_sampling_is_continuous_across_a_neighbourhood(dem):
    """Steps of a few metres must not jump metres in height.

    Nearest-neighbour sampling would produce a staircase with 30 m treads; this
    is what pins the interpolation as bilinear on real terrain.
    """
    heights = [
        dem.surface_elevation_m(31.7614 + i * 0.00005, 35.1889) for i in range(10)
    ]
    steps = [abs(b - a) for a, b in zip(heights, heights[1:])]
    assert max(steps) < 2.0, f"discontinuous sampling: steps {steps}"
    assert any(step > 0 for step in steps), "constant output is not interpolation"


@requires_cache
def test_ground_plane_over_a_jerusalem_ridge_is_sloped_but_planar(dem):
    slope_percent, aspect_deg, residual_m = dem.ground_plane(31.7614, 35.1889, radius_m=200.0)
    assert 0.0 < slope_percent < 60.0, "Jerusalem is hilly, not vertical"
    assert 0.0 <= aspect_deg < 360.0
    assert residual_m < 30.0, "a plane should still roughly describe a hillside"


@requires_cache
def test_obstacle_scale_is_refused_on_real_data_too(dem):
    """Trap #10, against the actual tile: 1.2 m is forty times the posting."""
    with pytest.raises(ScaleNotSupported):
        dem.assert_scale_supported(1.2)


# --------------------------------------------------------------------------
# The full pipeline, on real data
# --------------------------------------------------------------------------


@requires_cache
@requires_geoid
@pytest.mark.parametrize("latitude, longitude, low, high", NEIGHBOURHOODS)
def test_initialization_succeeds_across_jerusalem(latitude, longitude, low, high):
    report = MapInitializer(
        latitude=latitude, longitude=longitude, expected_geoid_range=(19.0, 21.0)
    ).run()
    assert report.ok, report.summary()


@requires_cache
@requires_geoid
def test_ego_altitude_prior_relates_both_height_systems_on_real_terrain():
    """h = H + N, with every term from measured data rather than assumption."""
    initializer = MapInitializer(latitude=31.7614, longitude=35.1889)
    prior = initializer.ego_altitude_prior(rig_height_agl_m=1.5)

    assert 19.0 < prior.geoid_undulation_m < 21.0, "Jerusalem undulation"
    assert prior.ellipsoidal_m == pytest.approx(
        prior.orthometric_m + prior.geoid_undulation_m
    )
    # Ignoring the geoid would bias ego altitude by the undulation itself
    assert prior.ellipsoidal_m - prior.orthometric_m > 19.0
    assert prior.sigma_m >= 4.0, "cannot be tighter than the DEM's own accuracy"


@requires_cache
@requires_geoid
def test_the_same_tile_serves_every_neighbourhood():
    """One 1-degree tile covers the whole operating area; no seams to cross."""
    tiles = {
        MapInitializer(latitude=lat, longitude=lon).priors().glo30_path.name
        for lat, lon in [(p.values[0], p.values[1]) for p in NEIGHBOURHOODS]
    }
    assert len(tiles) == 1, f"expected one tile, got {tiles}"
