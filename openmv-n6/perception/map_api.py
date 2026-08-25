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
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


SCHEMA_VERSION = "1.0"
# Verified against origin/yael at this commit (2026-08-25): all stages the
# adapter reads still expose the keys below, and a full run on the Jetson with
# the EGM2008 grid and the Jerusalem priors cache returned ok=True.
YAEL_API_COMMIT = "fce2918"

# Where Yael's package is looked for when it is not already importable, in
# order.  Her branch shares no history with this one and is never merged, so
# the package lives in its own checkout and has to be found rather than
# assumed.  The env var is the deployment knob; the two directories are the
# layouts actually used on the Jetson (a worktree beside this repo) and the
# one a future merge would produce (mapinit/ at the repo root).
MAPINIT_ENV = "VALISIGHT_MAPINIT"
_REPO_ROOT = Path(__file__).resolve().parents[2]
MAPINIT_CANDIDATES: Tuple[Path, ...] = (
    _REPO_ROOT,                              # merged: <repo>/mapinit/
    _REPO_ROOT.parent / "ValiSight_yael",    # worktree of origin/yael beside the repo
    Path.home() / "ValiSight_yael",
)


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
    #: Pose observations, optional. When both are given the pose_init stage
    #: runs: a heading prior (compass degrees) with the prior's sigmas, and
    #: the static radar returns as (range_m, azimuth_deg) pairs, azimuth
    #: positive to the right of boresight, as radar_detections_all() reports.
    heading_prior_deg: Optional[float] = None
    sigma_position_m: float = 5.0
    sigma_heading_deg: float = 5.0
    wall_returns: Optional[Tuple[Tuple[float, float], ...]] = None
    prior_source: str = "manual_pin"

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
        if self.heading_prior_deg is not None and not math.isfinite(float(self.heading_prior_deg)):
            raise ValueError("heading_prior_deg must be finite")
        if not (self.sigma_position_m > 0 and self.sigma_heading_deg > 0):
            raise ValueError("prior sigmas must be positive")
        if self.wall_returns is not None:
            for pair in self.wall_returns:
                if len(pair) != 2 or not all(math.isfinite(float(v)) for v in pair):
                    raise ValueError("wall_returns must be finite (range_m, azimuth_deg) pairs")

    @property
    def has_pose_observations(self) -> bool:
        return self.heading_prior_deg is not None and bool(self.wall_returns)


def locate_mapinit(explicit: Optional[Path] = None) -> Optional[Path]:
    """Return the directory that holds ``mapinit/`` and put it on sys.path.

    ``explicit`` (the CLI flag) wins, then ``$VALISIGHT_MAPINIT``, then the
    known layouts.  A directory only counts when ``mapinit/__init__.py`` is
    actually in it, so a stale env var fails here with the path it tried
    rather than later with a bare ImportError.  Returns None when nothing was
    found; an already-importable ``mapinit`` (pip install -e) needs no path.
    """
    tried: List[Path] = []
    for raw in ((explicit,) if explicit is not None else ()) + \
               ((Path(os.environ[MAPINIT_ENV]),) if os.environ.get(MAPINIT_ENV) else ()):
        root = Path(raw).expanduser()
        if not (root / "mapinit" / "__init__.py").is_file():
            raise MapAPIUnavailable(
                f"no mapinit package under {root} (from "
                f"{'--mapinit-dir' if explicit is not None else MAPINIT_ENV}); "
                "expected <dir>/mapinit/__init__.py")
        _prepend_path(root)
        return root
    for root in MAPINIT_CANDIDATES:
        tried.append(root)
        if (root / "mapinit" / "__init__.py").is_file():
            _prepend_path(root)
            return root
    return None


def _prepend_path(root: Path) -> None:
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)


def _public_map_initializer(mapinit_dir: Optional[Path] = None) -> Callable[..., Any]:
    """Load only the public object promised by the map package."""
    return _public_name("MapInitializer", mapinit_dir)


def _public_name(name: str, mapinit_dir: Optional[Path] = None) -> Callable[..., Any]:
    located = locate_mapinit(mapinit_dir)
    try:
        module = importlib.import_module("mapinit")
    except ModuleNotFoundError as exc:
        if exc.name != "mapinit":
            raise
        raise MapAPIUnavailable(
            "mapinit is unavailable; check out origin/yael (minimum commit "
            f"{YAEL_API_COMMIT}) beside this repo, point {MAPINIT_ENV} at it, "
            "or pip install it. Looked in: "
            + ", ".join(str(c) for c in MAPINIT_CANDIDATES)
            + ("" if located is None else f" (found {located} but import failed)")
        ) from exc

    obj = getattr(module, name, None)
    if obj is None or not callable(obj):
        raise MapAPIContractError(f"mapinit must export callable {name}")
    return obj


def pose_observations(request: MapInitRequest, mapinit_dir: Optional[Path] = None) -> Any:
    """Build mapinit's PoseObservations from a request, or None without observations."""
    if not request.has_pose_observations:
        return None
    PoseObservations = _public_name("PoseObservations", mapinit_dir)
    PosePrior = _public_name("PosePrior", mapinit_dir)
    WallReturn = _public_name("WallReturn", mapinit_dir)
    prior = PosePrior(
        latitude=float(request.latitude), longitude=float(request.longitude),
        heading_deg=float(request.heading_prior_deg) % 360.0,
        sigma_position_m=float(request.sigma_position_m),
        sigma_heading_deg=float(request.sigma_heading_deg),
        source=str(request.prior_source),
    )
    returns = tuple(WallReturn(float(r), float(a)) for r, a in request.wall_returns)
    return PoseObservations(prior=prior, wall_returns=returns)


def _pose_json(pose_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The pose stage's output as primitives: the pose, and whether to trust it."""
    if not pose_data or pose_data.get("pose") is None:
        return None
    pose, fix = pose_data["pose"], pose_data.get("fix")
    out: Dict[str, Any] = {
        "latitude_deg": _finite(_require(pose, "latitude", "InitialPose")),
        "longitude_deg": _finite(_require(pose, "longitude", "InitialPose")),
        "height_m": _finite(getattr(pose, "height_m", None)),
        "heading_deg": _finite(_require(pose, "yaw_deg", "InitialPose")),
        "sigma_horizontal_m": _finite(getattr(pose, "sigma_horizontal_m", None)),
        "sigma_vertical_m": _finite(getattr(pose, "sigma_vertical_m", None)),
        "sigma_heading_deg": _finite(getattr(pose, "sigma_yaw_deg", None)),
    }
    if fix is not None:
        out.update({
            "accepted": bool(_require(fix, "accepted", "WallFix")),
            "ambiguous": bool(getattr(fix, "ambiguous", False)),
            "ambiguity_axis": getattr(fix, "ambiguity_axis", None),
            "on_boundary": bool(getattr(fix, "on_boundary", False)),
            "dx_m": _finite(getattr(fix, "dx_m", None)),
            "dy_m": _finite(getattr(fix, "dy_m", None)),
            "dyaw_deg": _finite(getattr(fix, "dyaw_deg", None)),
            "score": _finite(getattr(fix, "score", None)),
            "inlier_fraction": _finite(getattr(fix, "inlier_fraction", None)),
            "n_returns": int(getattr(fix, "n_returns", 0)),
            "n_edges": int(getattr(fix, "n_edges", 0)),
        })
    return out


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

    def __init__(self, initializer_factory: Optional[Callable[..., Any]] = None,
                 mapinit_dir: Optional[Path] = None) -> None:
        self._initializer_factory = initializer_factory
        self._mapinit_dir = mapinit_dir

    def initialize(self, request: MapInitRequest, fail_fast: bool = True) -> Dict[str, Any]:
        factory = self._initializer_factory or _public_map_initializer(self._mapinit_dir)
        kwargs: Dict[str, Any] = {
            "latitude": float(request.latitude),
            "longitude": float(request.longitude),
        }
        if request.expected_geoid_range_m is not None:
            kwargs["expected_geoid_range"] = tuple(request.expected_geoid_range_m)
        if request.repo_dir is not None:
            kwargs["repo_dir"] = Path(request.repo_dir)
        if request.has_pose_observations:
            kwargs["pose_observations"] = (
                pose_observations(request, self._mapinit_dir)
                if self._initializer_factory is None else
                {"heading_prior_deg": request.heading_prior_deg,
                 "wall_returns": list(request.wall_returns)})

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
        pose_data = _stage_data(results, "pose_init")
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
            "pose": _pose_json(pose_data),
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
    parser.add_argument("--heading", type=float, default=None, metavar="DEG",
                        help="heading prior (compass); with --walls runs the pose_init stage")
    parser.add_argument("--sigma-pos", type=float, default=5.0, metavar="M")
    parser.add_argument("--sigma-heading", type=float, default=5.0, metavar="DEG")
    parser.add_argument("--walls", type=Path, default=None, metavar="JSONL",
                        help="radar.jsonl from record_radar_all.py; the static returns of "
                             "the frame given by --frame become the wall observations")
    parser.add_argument("--frame", type=int, default=-1, help="which frame of --walls (default last)")
    parser.add_argument("--mapinit-dir", type=Path, default=None,
                        help=f"directory containing Yael's mapinit/ (else ${MAPINIT_ENV} or the known layouts)")
    args = parser.parse_args()

    walls = None
    if args.walls is not None:
        frames = [json.loads(line) for line in open(args.walls) if line.strip()]
        dets = frames[args.frame]["detections"]
        walls = tuple((d["range_m"], d["azimuth_deg"]) for d in dets
                      if d.get("is_static", abs(d.get("velocity_mps", 0.0)) <= 0.25)
                      and 0.5 <= d["range_m"] <= 40.0)
    request = MapInitRequest(
        latitude=args.lat,
        longitude=args.lon,
        expected_geoid_range_m=tuple(args.expect_geoid) if args.expect_geoid else None,
        repo_dir=args.repo_dir,
        heading_prior_deg=args.heading,
        sigma_position_m=args.sigma_pos, sigma_heading_deg=args.sigma_heading,
        wall_returns=walls,
    )
    result = MapInitializationAPI(mapinit_dir=args.mapinit_dir).initialize(
        request, fail_fast=not args.diagnose)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
