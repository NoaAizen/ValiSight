"""End-to-end tests for map-relative position initialization.

Organised by what each group protects:

* tile selection      - the requested point picks the tile, not file order
* untagged files      - an ambiguous directory raises instead of guessing
* geoid               - the vertical datum is real and correctly signed
* regression          - a missing grid raises; it never yields a number
* calibration         - unusable target sets are rejected before any solver
* pipeline            - stage order, fail-fast, skipping, and report shape
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from conftest import (
    JERUSALEM,
    JERUSALEM_UNDULATION_M,
    make_targets,
    requires_geoid_grid,
    write_priors,
)
from mapinit import Check, InitContext, InitializationPipeline, MapInitializer, StageStatus
from mapinit.calibration import SurveyedTargetConstraint
from mapinit.geo.dem import DemSampler, DemUnavailable, ScaleNotSupported
from mapinit.geo.geoid import GeoidGridUnavailable, GeoidModel
from mapinit.geo.providers import LocalFilePriorProvider
from mapinit.geo.tiles import AmbiguousTiles, TileNotFound, parse_tile_bounds, select_tile
from mapinit.stage import InitStage, StageResult
from mapinit.stages import CalibrationStage, GeoidStage, PoseInitStage, PriorsStage

# --------------------------------------------------------------------------
# Tile selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latitude, longitude, expected_tag",
    [
        pytest.param(*JERUSALEM, "N31_00_E035_00", id="jerusalem"),
        pytest.param(30.5, 34.9, "N30_00_E034_00", id="negev"),
        pytest.param(32.0, 35.0, "N32_00_E035_00", id="on-tile-corner"),
        pytest.param(-0.5, -1.5, "S01_00_W002_00", id="south-west-hemisphere"),
    ],
)
def test_point_selects_covering_tile(tiled_priors, latitude, longitude, expected_tag):
    priors = LocalFilePriorProvider(tiled_priors).get_priors(latitude, longitude)
    assert expected_tag in priors.glo30_path.name
    assert expected_tag in priors.overture_path.name


def test_point_outside_every_tile_raises(tiled_priors):
    with pytest.raises(TileNotFound) as excinfo:
        LocalFilePriorProvider(tiled_priors).get_priors(48.85, 2.35)  # Paris
    # The message must name the point and what was available, or it is not actionable
    assert "48.85" in str(excinfo.value)
    assert "Copernicus" in str(excinfo.value)


def test_tile_bounds_are_half_open_so_neighbours_never_overlap():
    lower = parse_tile_bounds(Path("Copernicus_DSM_COG_10_N31_00_E035_00_DEM.tif"))
    upper = parse_tile_bounds(Path("Copernicus_DSM_COG_10_N32_00_E035_00_DEM.tif"))
    assert lower.contains(31.9999, 35.5)
    assert not lower.contains(32.0, 35.5)
    assert upper.contains(32.0, 35.5)


def test_southern_hemisphere_tag_gives_negative_bounds():
    bounds = parse_tile_bounds(Path("Copernicus_DSM_COG_10_S01_00_W002_00_DEM.tif"))
    assert bounds.min_lat == -1.0
    assert bounds.min_lon == -2.0


def test_bare_tile_tag_form_is_understood():
    bounds = parse_tile_bounds(Path("N31E035.tif"))
    assert (bounds.min_lat, bounds.min_lon) == (31.0, 35.0)


# --------------------------------------------------------------------------
# Untagged files
# --------------------------------------------------------------------------


def test_single_untagged_file_is_accepted_as_global_export(tmp_path):
    priors_dir = write_priors(tmp_path / "priors", ["global_dem.tif"], ["all_buildings.parquet"])
    priors = LocalFilePriorProvider(priors_dir).get_priors(*JERUSALEM)
    assert priors.glo30_path.name == "global_dem.tif"


def test_several_untagged_files_raise_instead_of_guessing(tmp_path):
    priors_dir = write_priors(tmp_path / "priors", ["dem_a.tif", "dem_b.tif"], ["v.parquet"])
    with pytest.raises(AmbiguousTiles):
        LocalFilePriorProvider(priors_dir).get_priors(*JERUSALEM)


def test_selection_is_deterministic_across_calls(tiled_priors):
    provider = LocalFilePriorProvider(tiled_priors)
    names = {provider.get_priors(*JERUSALEM).glo30_path.name for _ in range(5)}
    assert len(names) == 1, "tile choice must not depend on filesystem iteration order"


def test_empty_directory_raises(tmp_path):
    priors_dir = write_priors(tmp_path / "priors", [], [])
    with pytest.raises(TileNotFound):
        LocalFilePriorProvider(priors_dir).get_priors(*JERUSALEM)


def test_select_tile_prefers_tagged_match_over_untagged_neighbour(tmp_path):
    candidates = [tmp_path / "Copernicus_DSM_COG_10_N31_00_E035_00_DEM.tif", tmp_path / "misc.tif"]
    assert select_tile(candidates, *JERUSALEM, dataset="DEM").name.startswith("Copernicus")


# --------------------------------------------------------------------------
# Geoid
# --------------------------------------------------------------------------


@requires_geoid_grid
def test_undulation_matches_independently_verified_value():
    model = GeoidModel(extra_data_dir=Path(__file__).resolve().parent.parent / "data")
    assert model.undulation(*JERUSALEM) == pytest.approx(JERUSALEM_UNDULATION_M, abs=0.01)


@requires_geoid_grid
def test_undulation_sign_convention_is_positive_above_ellipsoid():
    """N is positive where the geoid sits above the ellipsoid.

    PROJ returns the orthometric height of h=0, which is -N. Getting this
    backwards flips a ~20 m correction into a ~40 m error.
    """
    model = GeoidModel(extra_data_dir=Path(__file__).resolve().parent.parent / "data")
    undulation = model.undulation(*JERUSALEM)
    orthometric = model.orthometric_height(*JERUSALEM, ellipsoidal_height_m=100.0)
    assert undulation > 0
    assert orthometric == pytest.approx(100.0 - undulation, abs=1e-6)


@requires_geoid_grid
@pytest.mark.parametrize(
    "latitude, longitude, expected",
    [
        pytest.param(0.0, 0.0, 17.23, id="null-island"),
        pytest.param(51.4778, -0.0015, 45.90, id="greenwich"),
    ],
)
def test_undulation_matches_published_values(latitude, longitude, expected):
    model = GeoidModel(extra_data_dir=Path(__file__).resolve().parent.parent / "data")
    assert model.undulation(latitude, longitude) == pytest.approx(expected, abs=0.05)


@requires_geoid_grid
def test_geoid_stage_publishes_undulation_and_transform():
    ctx = InitContext(latitude=JERUSALEM[0], longitude=JERUSALEM[1])
    result = GeoidStage().run(ctx)
    assert result.status is StageStatus.OK
    assert result.data["undulation_m"] == pytest.approx(JERUSALEM_UNDULATION_M, abs=0.01)
    assert "ballpark" not in result.data["transform"].lower()


@requires_geoid_grid
def test_caller_supplied_range_is_checked_and_reported():
    ctx = InitContext(*JERUSALEM, expected_geoid_range=(17.0, 18.0))
    result = GeoidStage().run(ctx)
    # The range that shipped originally excluded the correct answer; it must fail
    assert result.status is StageStatus.FAILED
    failed = [c for c in result.failed_checks if c.name == "geoid.expected_range"]
    assert failed and failed[0].measured == pytest.approx(JERUSALEM_UNDULATION_M, abs=0.01)
    assert failed[0].expected == (17.0, 18.0)


@requires_geoid_grid
def test_no_expected_range_means_only_the_global_bound_applies():
    ctx = InitContext(*JERUSALEM)
    result = GeoidStage().run(ctx)
    assert result.status is StageStatus.OK
    assert not any(c.name == "geoid.expected_range" for c in result.checks)


# --------------------------------------------------------------------------
# Regression: the failure this package was built around
# --------------------------------------------------------------------------


def test_missing_grid_raises_and_never_returns_a_number(tmp_path, monkeypatch):
    """A geoid with no grid must fail loudly.

    The original defect swallowed the error and substituted a hardcoded 17.54 m,
    so the guard passed while reporting a number that was both fabricated and
    wrong. Pointing PROJ at an empty directory reproduces the missing-grid state.
    """
    monkeypatch.setenv("PROJ_DATA", str(tmp_path))
    monkeypatch.setenv("PROJ_LIB", str(tmp_path))

    import subprocess
    import sys

    # A subprocess is required: PROJ caches its grid search path per process,
    # so the environment override cannot take effect in the running interpreter.
    script = (
        "from mapinit.geo.geoid import GeoidModel, GeoidGridUnavailable\n"
        "try:\n"
        "    value = GeoidModel().undulation(31.7683, 35.2137)\n"
        "except GeoidGridUnavailable as exc:\n"
        "    print('RAISED')\n"
        "else:\n"
        "    print('RETURNED', value)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        env={"PATH": "/usr/bin:/bin", "PROJ_DATA": str(tmp_path), "HOME": str(tmp_path)},
    )
    assert "RAISED" in completed.stdout, (
        f"missing grid must raise, got: {completed.stdout}{completed.stderr}"
    )
    assert "17.54" not in completed.stdout


def test_geoid_error_message_says_how_to_fix_it(tmp_path):
    from mapinit.geo.geoid import EGM2008_GRID_NAME, EGM2008_GRID_URL

    error = GeoidGridUnavailable(f"install {EGM2008_GRID_NAME} from {EGM2008_GRID_URL}")
    assert "us_nga_egm08_25.tif" in str(error)
    assert "cdn.proj.org" in str(error)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def _named(checks, suffix):
    return next(c for c in checks if c.name == f"surveyed_targets.{suffix}")


def test_well_conditioned_target_set_passes_every_check():
    constraint = SurveyedTargetConstraint(make_targets(), rig_position=(0.0, 0.0, 1.5))
    checks = constraint.validate()
    assert all(c.passed for c in checks), [str(c) for c in checks if not c.passed]


def test_target_surveyed_worse_than_the_budget_fails():
    targets = make_targets()
    targets[3] = type(targets[3])(**{**targets[3].__dict__, "sigma_m": 0.012})
    checks = SurveyedTargetConstraint(targets).validate()
    sigma_check = _named(checks, "sigma")
    assert not sigma_check.passed
    assert sigma_check.measured == pytest.approx(0.012)
    assert targets[3].name in sigma_check.detail, "the report must name the offending target"


def test_too_few_targets_fails():
    checks = SurveyedTargetConstraint(make_targets(count=3), min_targets=6).validate()
    assert not _named(checks, "count").passed


def test_targets_at_one_height_fail_vertical_spread():
    checks = SurveyedTargetConstraint(make_targets(vertical_spread_m=0.0)).validate()
    assert not _named(checks, "vertical_spread").passed


def test_collinear_targets_fail_even_when_numerous_and_precise():
    """Bearing diversity is not something more targets can substitute for."""
    constraint = SurveyedTargetConstraint(make_targets(count=40, sigma_m=0.001, collinear=True))
    checks = constraint.validate()
    assert not _named(checks, "collinearity").passed
    assert _named(checks, "count").passed
    assert _named(checks, "sigma").passed


def test_targets_in_a_narrow_elevation_band_fail():
    """The layout that produces the large elevation uncertainty this stage guards against."""
    targets = make_targets(count=10, vertical_spread_m=0.4)
    constraint = SurveyedTargetConstraint(targets, rig_position=(0.0, 0.0, 1.5))
    spread = max(constraint.elevation_angles_deg()) - min(constraint.elevation_angles_deg())
    assert spread < 15.0
    assert not _named(constraint.validate(), "elevation_spread").passed


def test_elevation_spread_is_omitted_when_the_rig_position_is_unknown():
    checks = SurveyedTargetConstraint(make_targets(), rig_position=None).validate()
    assert not any(c.name == "surveyed_targets.elevation_spread" for c in checks)


def test_observations_carry_position_and_sigma():
    observations = SurveyedTargetConstraint(make_targets(count=5)).observations()
    assert len(observations) == 5
    assert all(o.sigma == pytest.approx(0.003) for o in observations)
    assert all(len(o.payload["position_enu_m"]) == 3 for o in observations)


def test_targets_load_from_json(tmp_path):
    import json

    path = tmp_path / "targets.json"
    path.write_text(json.dumps({
        "rig_position": [0.0, 0.0, 1.5],
        "targets": [
            {"name": t.name, "east_m": t.east_m, "north_m": t.north_m,
             "up_m": t.up_m, "sigma_m": t.sigma_m}
            for t in make_targets()
        ],
    }))
    constraint = SurveyedTargetConstraint.from_json(path)
    assert len(constraint.targets) == 8
    assert constraint.rig_position == (0.0, 0.0, 1.5)
    assert all(c.passed for c in constraint.validate())


def test_malformed_target_file_raises_with_context(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"targets": [{"name": "T1"}]}')
    with pytest.raises(ValueError, match="Malformed target entry"):
        SurveyedTargetConstraint.from_json(path)


# --------------------------------------------------------------------------
# Custom constraint: the extension point the calibration is built around
# --------------------------------------------------------------------------


def test_a_new_constraint_type_needs_no_stage_changes():
    """Adding calibration information means one class, wired in at the caller."""
    from mapinit.calibration.constraints import CalibrationConstraint, Observation

    class LeverArmConstraint(CalibrationConstraint):
        name = "lever_arm"

        def validate(self):
            return [Check.that(self.check_name("measured"), True, "lever arm surveyed")]

        def observations(self):
            return [Observation(kind="lever_arm", source=self.name, sigma=0.002)]

    ctx = InitContext(*JERUSALEM)
    result = CalibrationStage([LeverArmConstraint()]).run(ctx)

    assert result.status is StageStatus.OK
    assert result.data["constraints"] == ["lever_arm"]
    assert result.data["observation_count"] == 1
    assert any(c.name == "lever_arm.measured" for c in result.checks)


def test_calibration_skips_when_no_constraints_are_configured():
    """Unconfigured is neither a pass nor a failure, and must not read as either."""
    result = CalibrationStage([]).run(InitContext(*JERUSALEM))
    assert result.status is StageStatus.SKIPPED
    assert not result.failed
    assert "no calibration constraints configured" in result.checks[0].detail


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


class _StubStage(InitStage):
    """Records that it ran, and fails on demand."""

    def __init__(self, name: str, should_pass: bool = True) -> None:
        self.name = name
        self.should_pass = should_pass
        self.ran = False

    def execute(self, ctx):
        self.ran = True
        return [Check.that(f"{self.name}.stub", self.should_pass, "stub")], {"ran": True}


def test_default_pipeline_runs_the_geoid_guard_first():
    """The cheapest check, and the one that invalidates everything downstream."""
    names = [stage.name for stage in InitializationPipeline.default().stages]
    assert names == ["geoid", "priors", "calibration"]


def test_fail_fast_stops_at_the_first_failure():
    first, second = _StubStage("first", should_pass=False), _StubStage("second")
    report = InitializationPipeline([first, second]).run(InitContext(*JERUSALEM))
    assert first.ran and not second.ran
    assert not report.ok
    assert len(report.results) == 1


def test_diagnose_mode_runs_every_stage_and_collects_all_failures():
    stages = [_StubStage("first", should_pass=False), _StubStage("second", should_pass=False)]
    report = InitializationPipeline(stages).run(InitContext(*JERUSALEM), fail_fast=False)
    assert all(stage.ran for stage in stages)
    assert len(report.failed_checks) == 2
    assert not report.ok


def test_stage_outputs_are_visible_to_later_stages():
    ctx = InitContext(*JERUSALEM)
    InitializationPipeline([_StubStage("first")]).run(ctx)
    assert ctx.output("first", "ran") is True
    assert ctx.output("first", "absent", default="fallback") == "fallback"


def test_an_exception_becomes_a_failed_result_not_a_crash():
    class Exploding(InitStage):
        name = "exploding"

        def execute(self, ctx):
            raise RuntimeError("grid on fire")

    report = InitializationPipeline([Exploding()]).run(InitContext(*JERUSALEM))
    assert not report.ok
    result = report.stage("exploding")
    assert isinstance(result.error, RuntimeError)
    assert "grid on fire" in result.summary()


def test_a_failing_check_is_carried_into_the_report():
    from mapinit.check import CheckFailed

    class Rejecting(InitStage):
        name = "rejecting"

        def execute(self, ctx):
            Check.in_range("rejecting.value", 42.0, 0.0, 10.0, unit="m").raise_if_failed()
            return [], {}

    result = InitializationPipeline([Rejecting()]).run(InitContext(*JERUSALEM)).stage("rejecting")
    assert isinstance(result.error, CheckFailed)
    assert result.failed_checks[0].measured == 42.0
    assert result.failed_checks[0].expected == (0.0, 10.0)


def test_skipped_stages_do_not_fail_the_run():
    report = InitializationPipeline([_StubStage("ok"), PoseInitStage()]).run(InitContext(*JERUSALEM))
    assert report.ok
    assert report.stage("pose_init").status is StageStatus.SKIPPED


def test_pose_solve_is_explicit_about_being_unimplemented():
    with pytest.raises(NotImplementedError, match="not been designed"):
        PoseInitStage().solve(InitContext(*JERUSALEM))


def test_report_summary_names_the_failure():
    report = InitializationPipeline([_StubStage("broken", should_pass=False)]).run(InitContext(*JERUSALEM))
    summary = report.summary()
    assert "FAILED" in summary
    assert "broken.stub" in summary


def test_unknown_stage_lookup_raises():
    report = InitializationPipeline([_StubStage("only")]).run(InitContext(*JERUSALEM))
    with pytest.raises(KeyError, match="missing"):
        report.stage("missing")


@requires_geoid_grid
def test_full_pipeline_succeeds_with_geoid_and_priors_present(tiled_priors):
    ctx = InitContext(*JERUSALEM, prior_provider=LocalFilePriorProvider(tiled_priors))
    report = InitializationPipeline([GeoidStage(), PriorsStage(), CalibrationStage()]).run(ctx)

    assert report.ok, report.summary()
    assert report.stage("geoid").data["undulation_m"] == pytest.approx(JERUSALEM_UNDULATION_M, abs=0.01)
    assert "N31_00_E035_00" in report.stage("priors").data["glo30_path"].name
    assert report.stage("calibration").status is StageStatus.SKIPPED


@requires_geoid_grid
def test_diagnose_reports_priors_failure_without_hiding_a_good_geoid(tmp_path):
    """One diagnostic run must show both what works and what does not."""
    ctx = InitContext(*JERUSALEM, prior_provider=LocalFilePriorProvider(tmp_path / "absent"))
    report = InitializationPipeline([GeoidStage(), PriorsStage()]).run(ctx, fail_fast=False)

    assert report.stage("geoid").status is StageStatus.OK
    assert report.stage("priors").status is StageStatus.FAILED
    assert not report.ok


# --------------------------------------------------------------------------
# Check semantics
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# DEM sampling: the ground plane and ego-altitude prior
# --------------------------------------------------------------------------


def _dem_path(tiled_priors):
    return tiled_priors / "glo30" / "Copernicus_DSM_COG_10_N31_00_E035_00_DEM.tif"


def test_dem_reports_its_own_coverage(tiled_priors):
    with DemSampler(_dem_path(tiled_priors)) as dem:
        assert dem.covers(*JERUSALEM)
        assert not dem.covers(48.85, 2.35)  # Paris


def test_sampling_outside_the_tile_raises(tiled_priors):
    with DemSampler(_dem_path(tiled_priors)) as dem:
        with pytest.raises(DemUnavailable, match="outside"):
            dem.surface_elevation_m(48.85, 2.35)


def test_missing_dem_file_raises(tmp_path):
    with pytest.raises(DemUnavailable, match="not found"):
        DemSampler(tmp_path / "absent.tif")


def test_bilinear_sampling_varies_smoothly_across_a_pixel(tiled_priors):
    """Interpolation, not nearest neighbour: adjacent points must differ."""
    with DemSampler(_dem_path(tiled_priors)) as dem:
        a = dem.surface_elevation_m(31.5000, 35.5000)
        b = dem.surface_elevation_m(31.5000, 35.5010)
        assert a != b
        assert abs(a - b) < 1.0, "a fraction of a pixel must not jump"


def test_obstacle_scale_queries_are_refused(tiled_priors):
    """Trap #10: 30 m posting cannot support a 1.2 m clustering epsilon."""
    with DemSampler(_dem_path(tiled_priors)) as dem:
        with pytest.raises(ScaleNotSupported, match="finer than"):
            dem.assert_scale_supported(1.2)
        dem.assert_scale_supported(100.0)  # coarse enough, must not raise


def test_ground_is_estimated_from_the_ring_not_the_point(tiled_priors):
    """Trap #8: GLO-30 is a DSM, so the value overhead may be a roof.

    The synthetic tile carries a central bump standing above its surroundings.
    Sampled at the peak, ground must come out below the surface, which only
    happens if the estimate uses the ring rather than the point.
    """
    with DemSampler(_dem_path(tiled_priors)) as dem:
        estimate = dem.ground_elevation(31.5, 35.5, ring_radius_m=2000.0)
        assert estimate.surface_m > estimate.ground_m
        assert estimate.structure_m == pytest.approx(estimate.surface_m - estimate.ground_m)
        assert estimate.ring_samples > 0


def test_ground_plane_fit_reports_slope_and_residual(tiled_priors):
    with DemSampler(_dem_path(tiled_priors)) as dem:
        slope, aspect, residual = dem.ground_plane(31.2, 35.2, radius_m=2000.0)
        assert slope >= 0.0
        assert 0.0 <= aspect < 360.0
        assert residual >= 0.0


def test_nodata_does_not_leak_into_an_elevation(tmp_path):
    """A nodata neighbour must never be interpolated as if it were a height."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / "holed.tif"
    data = np.full((40, 40), 700.0, dtype="float32")
    data[20:24, 20:24] = -32767.0
    with rasterio.open(
        path, "w", driver="GTiff", height=40, width=40, count=1, dtype="float32",
        crs="EPSG:4326", nodata=-32767.0,
        transform=from_bounds(35.0, 31.0, 36.0, 32.0, 40, 40),
    ) as dst:
        dst.write(data, 1)

    with DemSampler(path) as dem:
        # Beside the hole, where a naive read would blend -32767 into the result
        assert dem.surface_elevation_m(31.5, 35.48) == pytest.approx(700.0, abs=1.0)


def test_all_nodata_raises_rather_than_returning_the_sentinel(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / "empty.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=20, width=20, count=1, dtype="float32",
        crs="EPSG:4326", nodata=-32767.0,
        transform=from_bounds(35.0, 31.0, 36.0, 32.0, 20, 20),
    ) as dst:
        dst.write(np.full((20, 20), -32767.0, dtype="float32"), 1)

    with DemSampler(path) as dem:
        with pytest.raises(DemUnavailable, match="no data"):
            dem.surface_elevation_m(31.5, 35.5)


def test_ego_altitude_prior_carries_both_height_systems(tiled_priors):
    """The pair must stay consistent: h = H + N, with N never assumed."""
    with DemSampler(_dem_path(tiled_priors)) as dem:
        prior = dem.ego_altitude_prior(
            31.5, 35.5, geoid_undulation_m=19.8, rig_height_agl_m=1.5
        )
        assert prior.ellipsoidal_m == pytest.approx(prior.orthometric_m + 19.8)
        assert prior.orthometric_m == pytest.approx(prior.ground.ground_m + 1.5)
        assert prior.sigma_m >= 4.0, "cannot be tighter than the DEM's own accuracy"


def test_priors_stage_reports_a_sampled_elevation(tiled_priors):
    ctx = InitContext(*JERUSALEM, prior_provider=LocalFilePriorProvider(tiled_priors))
    result = PriorsStage().run(ctx)

    assert result.status is StageStatus.OK
    assert "ground_elevation_m" in result.data
    assert any(c.name == "priors.dem_yields_elevation" for c in result.checks)


def test_priors_stage_pairs_the_prior_with_the_geoid_when_it_ran(tiled_priors):
    """The altitude prior must reach consumers already geoid-corrected."""
    ctx = InitContext(*JERUSALEM, prior_provider=LocalFilePriorProvider(tiled_priors))
    ctx.results["geoid"] = StageResult("geoid", StageStatus.OK, [], {"undulation_m": 19.8})

    result = PriorsStage().run(ctx)
    prior = result.data["ego_altitude_prior"]
    assert prior.geoid_undulation_m == pytest.approx(19.8)
    assert prior.ellipsoidal_m == pytest.approx(prior.orthometric_m + 19.8)


def test_priors_stage_fails_when_the_dem_is_not_a_raster(tmp_path):
    """A file with the right name and size but no readable raster must fail."""
    priors_dir = tmp_path / "priors"
    (priors_dir / "glo30").mkdir(parents=True)
    (priors_dir / "overture").mkdir(parents=True)
    (priors_dir / "glo30" / "Copernicus_DSM_COG_10_N31_00_E035_00_DEM.tif").write_bytes(
        b"\0" * 60_000
    )
    (priors_dir / "overture" / "overture_N31_00_E035_00_b.parquet").write_bytes(b"\0" * 20_000)

    ctx = InitContext(*JERUSALEM, prior_provider=LocalFilePriorProvider(priors_dir))
    assert PriorsStage().run(ctx).status is StageStatus.FAILED


# --------------------------------------------------------------------------
# MapInitializer: the surface consuming code holds onto
# --------------------------------------------------------------------------


def test_constructing_the_initializer_touches_no_disk(monkeypatch):
    """Holding a MapInitializer must cost nothing until something is asked."""
    import mapinit.geo.geoid as geoid_module

    def explode(*args, **kwargs):
        raise AssertionError("the geoid grid was opened during construction")

    monkeypatch.setattr(geoid_module, "GeoidModel", explode)
    MapInitializer(*JERUSALEM)  # must not raise


@requires_geoid_grid
def test_separation_probe_takes_lon_lat_in_that_order():
    """The probe signature is (lon, lat) — the reverse of the rest of the package."""
    init = MapInitializer(*JERUSALEM)
    latitude, longitude = JERUSALEM

    assert init.separation_probe(longitude, latitude) == pytest.approx(
        init.geoid_separation_m(latitude, longitude)
    )


@requires_geoid_grid
def test_swapping_probe_arguments_gives_a_different_answer():
    """Why the argument order matters: the mistake is silent, not loud.

    Jerusalem's coordinates transposed land in the Mediterranean off Libya.
    Both calls succeed and both return a plausible undulation, so nothing
    surfaces the error except the value being several meters off — which reads
    as a calibration bias rather than a coordinate bug.
    """
    init = MapInitializer(*JERUSALEM)
    latitude, longitude = JERUSALEM

    correct = init.separation_probe(longitude, latitude)
    swapped = init.separation_probe(latitude, longitude)
    assert abs(correct - swapped) > 1.0


@requires_geoid_grid
def test_probe_satisfies_an_injected_liveness_invariant():
    """Stands in for core.physics_invariants.geoid.geoid_is_live.

    That invariant lives in a pure layer and cannot load a grid itself; it
    injects a probe and asks whether the undulation is non-zero where it must
    be. This package supplies that probe.
    """

    def geoid_is_live(separation_probe, ref_lon=34.8, ref_lat=32.1, min_separation_m=5.0):
        try:
            separation = separation_probe(ref_lon, ref_lat)
        except Exception:
            return False, "probe raised: grid not usable"
        if separation is None or abs(separation) < min_separation_m:
            return False, "separation ~0: grid did not load"
        return True, f"grid live: {separation:.3f} m"

    ok, reason = geoid_is_live(MapInitializer(*JERUSALEM).separation_probe)
    assert ok, reason
    assert "18.9" in reason, "the hagai reference point measures 18.916 m, not 17-18 m"


@requires_geoid_grid
def test_orthometric_conversion_uses_the_initializer_point():
    init = MapInitializer(*JERUSALEM)
    assert init.orthometric_height_m(100.0) == pytest.approx(
        100.0 - JERUSALEM_UNDULATION_M, abs=0.01
    )


@requires_geoid_grid
def test_initializer_resolves_priors_for_its_own_point(tiled_priors):
    init = MapInitializer(*JERUSALEM, prior_provider=LocalFilePriorProvider(tiled_priors))
    assert "N31_00_E035_00" in init.priors().glo30_path.name


@requires_geoid_grid
def test_initializer_run_matches_the_pipeline_it_exposes(tiled_priors):
    init = MapInitializer(*JERUSALEM, prior_provider=LocalFilePriorProvider(tiled_priors))
    report = init.run()

    assert report.ok, report.summary()
    assert [s.name for s in init.pipeline().stages] == ["geoid", "priors", "calibration"]
    assert report.stage("geoid").data["undulation_m"] == pytest.approx(
        JERUSALEM_UNDULATION_M, abs=0.01
    )


def test_initializer_carries_calibration_constraints_into_the_run():
    from mapinit.calibration import SurveyedTargetConstraint

    init = MapInitializer(*JERUSALEM, constraints=[SurveyedTargetConstraint(make_targets())])
    assert init.pipeline().stages[-1].constraints, "constraints must reach CalibrationStage"


# --------------------------------------------------------------------------
# Check semantics
# --------------------------------------------------------------------------


def test_check_records_measurement_and_expectation():
    check = Check.in_range("depth", 5.0, 0.0, 10.0, unit="m")
    assert check.passed and check.measured == 5.0 and check.expected == (0.0, 10.0)
    assert "5.000 m" in check.detail


def test_infinite_upper_bound_is_supported_for_minimum_only_checks():
    assert Check.in_range("count", 12, 6, math.inf).passed
    assert not Check.in_range("count", 3, 6, math.inf).passed
