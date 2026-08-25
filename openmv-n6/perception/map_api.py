#!/usr/bin/env python3
"""Stable consumer API between the fusion branch and Yael's ``mapinit``.

Yael's branch owns map initialization and its internal dataclasses.  This
module is the boundary owned by the fusion branch: callers submit a small,
versioned request and receive JSON-safe primitives.  Consequently changes to
``mapinit`` internals do not leak into the live/fusion pipeline.

Typical use, after Yael's ``mapinit/`` package is present at repository root::

    from perception.map_api import MapInitRequest, MapInitializationAPI

    result = MapInitializationAPI().initialize(
        MapInitRequest(latitude=31.7683, longitude=35.2137)
    )
    if not result["ok"]:
        raise RuntimeError(result["summary"])

The module is also a JSON CLI::

    python3 -m perception.map_api --lat 31.7683 --lon 35.2137
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


SCHEMA_VERSION = "1.0"
YAEL_API_COMMIT = "a54b7df309122528067d7c5ea45e1bfc4d134346"


class MapAPIError(RuntimeError):
    """Base error for dependency and API-contract failures."""


class MapAPIUnavailable(MapAPIError):
    """Raised when Yael's package has not been merged/installed."""


class MapAPIContractError(MapAPIError):
    """Raised when the installed package no longer satisfies the contract."""


@dataclass(frozen=True)
class MapInitRequest:
    """Inputs the fusion branch is allowed to send to map initialization."""

    latitude: float
    longitude: float
    expected_geoid_range_m: Optional[Tuple[float, float]] = None
    repo_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        latitude = float(self.latitude)
        longitude = float(self.longitude)
        if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
            raise ValueError("latitude must be finite and in [-90, 90] degrees")
        if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
            raise ValueError("longitude must be finite and in [-180, 180] degrees")

        expected = self.expected_geoid_range_m
        if expected is not None:
            if len(expected) != 2:
                raise ValueError("expected_geoid_range_m must contain (low, high)")
            low, high = (float(expected[0]), float(expected[1]))
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError("expected_geoid_range_m must be finite with low <= high")


def _public_map_initializer() -> Callable[..., Any]:
    """Load only the public object promised by Yael's branch."""
    try:
        module = importlib.import_module("mapinit")
    except ModuleNotFoundError as exc:
        if exc.name != "mapinit":
            raise
        raise MapAPIUnavailable(
            "mapinit is unavailable; merge Yael's branch (minimum commit "
            f"{YAEL_API_COMMIT[:7]}) or install that package on PYTHONPATH"
        ) from exc

    initializer = getattr(module, "MapInitializer", None)
    if initializer is None or not callable(initializer):
        raise MapAPIContractError("mapinit must export callable MapInitializer")
    return initializer


def _require(value: Any, attribute: str, owner: str) -> Any:
    if not hasattr(value, attribute):
        raise MapAPIContractError(f"{owner} must expose .{attribute}")
    return getattr(value, attribute)


def _finite(value: Any) -> Any:
    """Return a JSON number, mapping open numeric bounds to JSON null."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _check_json(check: Any) -> Dict[str, Any]:
    name = _require(check, "name", "Check")
    passed = _require(check, "passed", "Check")
    detail = _require(check, "detail", "Check")
    expected = getattr(check, "expected", None)
    return {
        "name": str(name),
        "passed": bool(passed),
        "detail": str(detail),
        "measured": _finite(getattr(check, "measured", None)),
        "expected": (
            [_finite(expected[0]), _finite(expected[1])]
            if expected is not None
            else None
        ),
        "unit": str(getattr(check, "unit", "")),
    }


def _status_text(status: Any) -> str:
    value = getattr(status, "value", status)
    text = str(value)
    if text not in {"ok", "failed", "skipped"}:
        raise MapAPIContractError(
            f"StageResult.status must be ok, failed, or skipped; got {text!r}"
        )
    return text


def _stage_json(stage: Any) -> Dict[str, Any]:
    name = str(_require(stage, "stage", "StageResult"))
    status = _status_text(_require(stage, "status", "StageResult"))
    checks = _require(stage, "checks", "StageResult")
    error = getattr(stage, "error", None)
    return {
        "name": name,
        "status": status,
        "checks": [_check_json(check) for check in checks],
        "error": (
            {"type": type(error).__name__, "message": str(error)}
            if error is not None
            else None
        ),
    }


def _stage_data(results: Any, name: str) -> Optional[Dict[str, Any]]:
    for result in results:
        if str(getattr(result, "stage", "")) == name:
            data = getattr(result, "data", None)
            if data is None:
                return {}
            if not isinstance(data, dict):
                raise MapAPIContractError(f"StageResult.data for {name!r} must be a dict")
            return data
    return None


def _altitude_json(prior: Any) -> Optional[Dict[str, float]]:
    if prior is None:
        return None
    required = ("orthometric_m", "ellipsoidal_m", "geoid_undulation_m", "sigma_m")
    values = {name: _finite(_require(prior, name, "EgoAltitudePrior")) for name in required}
    return values


class MapInitializationAPI:
    """Adapter from Yael's object API to the fusion branch's JSON contract.

    ``initializer_factory`` is injectable so this boundary is testable without
    a geoid grid, DEM files, or a checkout of another branch.
    """

    def __init__(self, initializer_factory: Optional[Callable[..., Any]] = None) -> None:
        self._initializer_factory = initializer_factory

    def initialize(self, request: MapInitRequest, fail_fast: bool = True) -> Dict[str, Any]:
        factory = self._initializer_factory or _public_map_initializer()
        kwargs: Dict[str, Any] = {
            "latitude": float(request.latitude),
            "longitude": float(request.longitude),
        }
        if request.expected_geoid_range_m is not None:
            kwargs["expected_geoid_range"] = tuple(request.expected_geoid_range_m)
        if request.repo_dir is not None:
            kwargs["repo_dir"] = Path(request.repo_dir)

        initializer = factory(**kwargs)
        run = _require(initializer, "run", "MapInitializer")
        if not callable(run):
            raise MapAPIContractError("MapInitializer.run must be callable")
        try:
            report = run(fail_fast=fail_fast)
        except ModuleNotFoundError as exc:
            # MapInitializer is intentionally cheap to import, so an optional
            # geo dependency can first be imported here when the stages are
            # assembled.  Name the missing deployment dependency explicitly.
            raise MapAPIUnavailable(
                f"mapinit runtime dependency {exc.name!r} is unavailable; "
                "install the dependencies declared by Yael's branch"
            ) from exc

        ok = bool(_require(report, "ok", "InitReport"))
        results = list(_require(report, "results", "InitReport"))
        summary_method = _require(report, "summary", "InitReport")
        if not callable(summary_method):
            raise MapAPIContractError("InitReport.summary must be callable")

        geoid_data = _stage_data(results, "geoid")
        priors_data = _stage_data(results, "priors")
        response: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "ok": ok,
            "location": {
                "latitude_deg": float(request.latitude),
                "longitude_deg": float(request.longitude),
            },
            "geoid": (
                {
                    "undulation_m": _finite(geoid_data.get("undulation_m")),
                    "transform": str(geoid_data.get("transform", "")),
                }
                if geoid_data and geoid_data.get("undulation_m") is not None
                else None
            ),
            "priors": (
                {
                    "glo30_path": str(priors_data.get("glo30_path", "")),
                    "overture_path": str(priors_data.get("overture_path", "")),
                    "provider": str(priors_data.get("provider", "")),
                    "ground_elevation_m": _finite(priors_data.get("ground_elevation_m")),
                    "surface_elevation_m": _finite(priors_data.get("surface_elevation_m")),
                    "dem_posting_m": _finite(priors_data.get("dem_posting_m")),
                }
                if priors_data and priors_data.get("glo30_path") is not None
                else None
            ),
            "ego_altitude_prior": _altitude_json(
                priors_data.get("ego_altitude_prior") if priors_data else None
            ),
            "stages": [_stage_json(stage) for stage in results],
            "summary": str(summary_method()),
        }

        # Prove this is strict JSON now, at the boundary, instead of letting an
        # Infinity from an open check bound break some later HTTP response.
        json.dumps(response, allow_nan=False)
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Yael map initialization and emit fusion API JSON")
    parser.add_argument("--lat", type=float, required=True, help="latitude in decimal degrees")
    parser.add_argument("--lon", type=float, required=True, help="longitude in decimal degrees")
    parser.add_argument("--expect-geoid", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--repo-dir", type=Path, default=None, help="Yael package data root")
    parser.add_argument("--diagnose", action="store_true", help="run all stages after a failure")
    args = parser.parse_args()

    request = MapInitRequest(
        latitude=args.lat,
        longitude=args.lon,
        expected_geoid_range_m=tuple(args.expect_geoid) if args.expect_geoid else None,
        repo_dir=args.repo_dir,
    )
    result = MapInitializationAPI().initialize(request, fail_fast=not args.diagnose)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
