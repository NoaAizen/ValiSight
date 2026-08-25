"""Consumer-contract tests for the Yael map initialization boundary."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pytest

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
