"""The pose stage inside the pipeline: skipped without observations, solved with.

Uses a synthetic footprint file and a local prior provider so it needs no
geoid grid or DEM; the geoid stage fails or is absent and the pose stage
must still run on what the priors stage published.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from mapinit import (  # noqa: E402
    InitialPose,
    MapInitializer,
    PoseObservations,
    PosePrior,
    WallReturn,
)
from mapinit.nav.walls import local_scales  # noqa: E402
from mapinit.stages.pose import PoseInitStage  # noqa: E402
from mapinit.context import InitContext  # noqa: E402
from mapinit.stage import StageResult, StageStatus  # noqa: E402

LAT, LON = 31.7683, 35.2137


def _geojson(path: Path) -> Path:
    lon_scale, lat_scale = local_scales(LAT)

    def ring(x0, y0, x1, y1):
        return [[LON + x / lon_scale, LAT + y / lat_scale]
                for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]

    boxes = [(-30, -20, -6, 12), (-30, 23, -6, 40), (6, -20, 30, 4), (6, 8, 30, 40), (-30, 42, 30, 56)]
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": f"b{i}", "height": 9.0},
         "geometry": {"type": "Polygon", "coordinates": [ring(*b)]}}
        for i, b in enumerate(boxes)]}
    path.write_text(json.dumps(fc))
    return path


def _returns():
    from tests.test_walls import cast_returns, street
    return tuple(cast_returns(street(), 1.0, 2.0, 30.0, noise_m=0.05))


def _run_pose_stage(tmp_path, observations):
    """Drive the stage directly with a priors result, as the runner would."""
    ctx = InitContext(latitude=LAT, longitude=LON, repo_dir=tmp_path)
    ctx.results["priors"] = StageResult(
        stage="priors", status=StageStatus.OK, checks=[],
        data={"overture_path": _geojson(tmp_path / "b.geojson"),
              "ground_elevation_m": 778.0, "ego_altitude_prior": None})
    return PoseInitStage(observations).run(ctx)


def test_without_observations_the_stage_is_skipped_and_says_why(tmp_path):
    result = _run_pose_stage(tmp_path, None)
    assert result.status == StageStatus.SKIPPED
    assert "PoseObservations" in result.summary()


def test_with_observations_the_stage_publishes_a_pose_with_sigmas(tmp_path):
    prior = PosePrior(LAT, LON, 34.0, 3.0, 4.0, source="manual_pin")
    result = _run_pose_stage(tmp_path, PoseObservations(prior, _returns(), rig_height_agl_m=1.2))
    assert result.status == StageStatus.OK, result.summary()
    pose = result.data["pose"]
    assert isinstance(pose, InitialPose)
    lon_scale, lat_scale = local_scales(LAT)
    assert abs((pose.longitude - LON) * lon_scale - 1.0) < 0.5
    assert abs((pose.latitude - LAT) * lat_scale - 2.0) < 0.5
    assert abs(pose.yaw_deg - 30.0) < 1.0
    assert pose.height_m == pytest.approx(778.0 + 1.2)
    assert 0 < pose.sigma_horizontal_m < 1.5 and 0 < pose.sigma_yaw_deg < 3.0
    assert any(c.name == "pose_init.accepted" and c.passed for c in result.checks)


def test_a_rejected_fix_fails_the_stage_rather_than_publishing_a_guess(tmp_path):
    prior = PosePrior(LAT, LON, 34.0, 3.0, 4.0)
    few = tuple(WallReturn(5.0, 0.0) for _ in range(3))
    result = _run_pose_stage(tmp_path, PoseObservations(prior, few))
    assert result.status == StageStatus.FAILED
    assert any(c.name == "walls.returns" and not c.passed for c in result.checks)


def test_map_initializer_accepts_observations_and_registers_the_stage():
    init = MapInitializer(LAT, LON, pose_observations=PoseObservations(
        PosePrior(LAT, LON, 0.0, 3.0, 3.0), _returns()))
    names = [stage.name for stage in init.pipeline().stages]
    assert names[-1] == "pose_init"
