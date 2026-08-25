"""Consumer-contract tests for the Yael map initialization boundary."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pytest

from perception import map_api
from perception.map_api import (
    MapAPIContractError,
    MapAPIUnavailable,
    MapInitRequest,
    MapInitializationAPI,
    SCHEMA_VERSION,
)


class Status(str, Enum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    measured: float | None = None
    expected: tuple[float, float] | None = None
    unit: str = ""


@dataclass
class Altitude:
    orthometric_m: float = 801.2
    ellipsoidal_m: float = 821.0
    geoid_undulation_m: float = 19.8
    sigma_m: float = 4.2


@dataclass
class Stage:
    stage: str
    status: Status
    checks: list[Check]
    data: dict = field(default_factory=dict)
    error: BaseException | None = None


class Report:
    def __init__(self, results, ok=True):
        self.results = results
        self.ok = ok

    def summary(self):
        return "Initialization succeeded" if self.ok else "Initialization FAILED"


class FakeInitializer:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(("init", kwargs))

    def run(self, fail_fast=True):
        self.calls.append(("run", fail_fast))
        return Report([
            Stage("geoid", Status.OK, [
                Check("geoid.global_bound", True, "in range", 19.8, (0.0, math.inf), "m")
            ], {"undulation_m": 19.8, "transform": "EGM2008"}),
            Stage("priors", Status.OK, [], {
                "glo30_path": Path("/data/N31E035.tif"),
                "overture_path": Path("/data/N31E035.parquet"),
                "provider": "LocalFilePriorProvider",
                "ground_elevation_m": 801.2,
                "surface_elevation_m": 811.0,
                "dem_posting_m": 30.0,
                "ego_altitude_prior": Altitude(),
            }),
            Stage("calibration", Status.SKIPPED, [
                Check("calibration.configured", True, "skipped: no constraints")
            ]),
        ])


def test_success_is_stable_strict_json():
    FakeInitializer.calls.clear()
    request = MapInitRequest(31.7683, 35.2137, (19.0, 20.5), Path("/yael"))
    result = MapInitializationAPI(FakeInitializer).initialize(request, fail_fast=False)

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["ok"] is True
    assert result["location"] == {"latitude_deg": 31.7683, "longitude_deg": 35.2137}
    assert result["geoid"]["undulation_m"] == 19.8
    assert result["priors"]["glo30_path"] == "/data/N31E035.tif"
    assert result["ego_altitude_prior"]["ellipsoidal_m"] == 821.0
    assert result["stages"][2]["status"] == "skipped"
    assert result["stages"][0]["checks"][0]["expected"] == [0.0, None]
    json.dumps(result, allow_nan=False)

    assert FakeInitializer.calls == [
        ("init", {
            "latitude": 31.7683,
            "longitude": 35.2137,
            "expected_geoid_range": (19.0, 20.5),
            "repo_dir": Path("/yael"),
        }),
        ("run", False),
    ]


def test_failed_stage_is_data_not_an_adapter_exception():
    class Failed(FakeInitializer):
        def run(self, fail_fast=True):
            return Report([
                Stage("geoid", Status.FAILED, [], error=RuntimeError("grid missing"))
            ], ok=False)

    result = MapInitializationAPI(Failed).initialize(MapInitRequest(31.7, 35.2))
    assert result["ok"] is False
    assert result["geoid"] is None
    assert result["stages"][0]["error"] == {
        "type": "RuntimeError",
        "message": "grid missing",
    }


@pytest.mark.parametrize("lat, lon", [(91, 0), (-91, 0), (0, 181), (0, -181), (math.nan, 0)])
def test_request_refuses_invalid_coordinates(lat, lon):
    with pytest.raises(ValueError):
        MapInitRequest(lat, lon)


def test_request_refuses_reversed_expected_range():
    with pytest.raises(ValueError, match="low <= high"):
        MapInitRequest(31.7, 35.2, (21.0, 19.0))


def test_contract_drift_fails_loudly():
    class BrokenInitializer:
        def __init__(self, **kwargs):
            pass

        def run(self, fail_fast=True):
            return object()  # missing .ok, .results and .summary

    with pytest.raises(MapAPIContractError, match="InitReport must expose .ok"):
        MapInitializationAPI(BrokenInitializer).initialize(MapInitRequest(31.7, 35.2))


def test_missing_yael_runtime_dependency_is_actionable():
    class MissingDependency(FakeInitializer):
        def run(self, fail_fast=True):
            error = ModuleNotFoundError("No module named 'rasterio'")
            error.name = "rasterio"
            raise error

    with pytest.raises(MapAPIUnavailable, match="rasterio"):
        MapInitializationAPI(MissingDependency).initialize(MapInitRequest(31.7, 35.2))


# --- finding Yael's package. It is never merged, so the adapter has to locate
#     the checkout; a wrong pointer must fail with the path it tried.

def _fake_mapinit(root: Path) -> Path:
    (root / "mapinit").mkdir(parents=True)
    (root / "mapinit" / "__init__.py").write_text("")
    return root


def test_locate_prefers_explicit_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(map_api, "MAPINIT_CANDIDATES", ())
    root = _fake_mapinit(tmp_path / "yael")
    assert map_api.locate_mapinit(root) == root
    assert str(root) in map_api.sys.path


def test_locate_reads_env_var(tmp_path, monkeypatch):
    monkeypatch.setattr(map_api, "MAPINIT_CANDIDATES", ())
    root = _fake_mapinit(tmp_path / "elsewhere")
    monkeypatch.setenv(map_api.MAPINIT_ENV, str(root))
    assert map_api.locate_mapinit() == root


def test_locate_falls_back_to_known_layouts(tmp_path, monkeypatch):
    monkeypatch.delenv(map_api.MAPINIT_ENV, raising=False)
    root = _fake_mapinit(tmp_path / "beside")
    monkeypatch.setattr(map_api, "MAPINIT_CANDIDATES", (tmp_path / "missing", root))
    assert map_api.locate_mapinit() == root


def test_locate_names_a_bad_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(map_api, "MAPINIT_CANDIDATES", ())
    monkeypatch.setenv(map_api.MAPINIT_ENV, str(tmp_path / "nope"))
    with pytest.raises(MapAPIUnavailable) as err:
        map_api.locate_mapinit()
    assert "nope" in str(err.value) and map_api.MAPINIT_ENV in str(err.value)


def test_locate_returns_none_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.delenv(map_api.MAPINIT_ENV, raising=False)
    monkeypatch.setattr(map_api, "MAPINIT_CANDIDATES", (tmp_path / "a", tmp_path / "b"))
    assert map_api.locate_mapinit() is None


# --- the pose block: observations in, a pose with its trust flags out.

@dataclass
class Pose:
    latitude: float = 31.76831
    longitude: float = 35.21372
    height_m: float = 779.7
    yaw_deg: float = 38.5
    sigma_horizontal_m: float = 0.6
    sigma_vertical_m: float = 5.4
    sigma_yaw_deg: float = 2.3


@dataclass
class Fix:
    accepted: bool = True
    ambiguous: bool = False
    ambiguity_axis: str | None = None
    on_boundary: bool = False
    dx_m: float = 1.0
    dy_m: float = -2.0
    dyaw_deg: float = -1.5
    score: float = 0.9
    inlier_fraction: float = 0.88
    n_returns: int = 9
    n_edges: int = 65


class FakeInitializerWithPose(FakeInitializer):
    def run(self, fail_fast=True):
        report = super().run(fail_fast)
        obs = self.kwargs.get("pose_observations")
        if obs is not None:
            report.results.append(Stage("pose_init", Status.OK, [
                Check("pose_init.accepted", True, "fixed")], {"pose": Pose(), "fix": Fix()}))
        else:
            report.results.append(Stage("pose_init", Status.SKIPPED, [
                Check("pose_init.configured", True, "skipped: no observations")]))
        return report


def test_request_without_walls_has_no_pose_observations():
    req = MapInitRequest(latitude=31.7683, longitude=35.2137, heading_prior_deg=40.0)
    assert not req.has_pose_observations
    res = MapInitializationAPI(initializer_factory=FakeInitializerWithPose).initialize(req)
    assert res["pose"] is None
    assert [s["status"] for s in res["stages"] if s["name"] == "pose_init"] == ["skipped"]


def test_request_with_walls_runs_the_pose_stage_and_returns_primitives():
    req = MapInitRequest(latitude=31.7683, longitude=35.2137, heading_prior_deg=40.0,
                         sigma_position_m=5.0, sigma_heading_deg=5.0,
                         wall_returns=((5.0, 10.0), (7.5, -20.0), (12.0, 33.0)))
    assert req.has_pose_observations
    res = MapInitializationAPI(initializer_factory=FakeInitializerWithPose).initialize(req)
    pose = res["pose"]
    assert pose["accepted"] is True and pose["ambiguous"] is False
    assert pose["heading_deg"] == 38.5 and pose["sigma_heading_deg"] == 2.3
    assert pose["dx_m"] == 1.0 and pose["n_returns"] == 9
    json.dumps(res, allow_nan=False)


def test_bad_wall_returns_are_rejected_at_the_boundary():
    with pytest.raises(ValueError):
        MapInitRequest(latitude=31.7683, longitude=35.2137, heading_prior_deg=40.0,
                       wall_returns=((5.0,),))
    with pytest.raises(ValueError):
        MapInitRequest(latitude=31.7683, longitude=35.2137, heading_prior_deg=40.0,
                       sigma_position_m=0.0, wall_returns=((5.0, 1.0),))
