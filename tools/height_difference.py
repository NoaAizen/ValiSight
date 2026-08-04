#!/usr/bin/env python3
"""Compute the height difference between ellipsoidal and orthometric heights.

This module uses an EGM2008 geoid model to calculate the geoid undulation N,
which represents the difference between the two height reference surfaces:

    h = H + N

where:
- h is the ellipsoidal height
- H is the orthometric height
- N is the geoid undulation
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapinit.geo.geoid import GeoidGridUnavailable, GeoidModel  # noqa: E402


class HeightDifferenceCalculator:
    """Calculate the geoid-based difference between two height systems."""

    def __init__(
        self,
        grid_dir: Optional[str] = None,
        latitude: float = 31.7683,
        longitude: float = 35.2137,
    ) -> None:
        self.grid_dir = Path(grid_dir).expanduser() if grid_dir else None
        self.latitude = latitude
        self.longitude = longitude

    def resolve_grid_dir(self) -> Optional[Path]:
        """Return an extra directory for PROJ to search for grids, if one is configured.

        PROJ already searches its own user and installation data dirs; this only
        adds a project-local override.
        """
        candidates: list[Path] = []

        if self.grid_dir is not None:
            candidates.append(self.grid_dir)

        for env_name in ("EGM2008_GRID_DIR", "GEOID_GRID_DIR", "PROJ_DATA"):
            value = os.getenv(env_name)
            if value:
                candidates.append(Path(value).expanduser())

        candidates.append(REPO_ROOT / "data")

        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return None

    def get_geoid_undulation(self) -> float:
        """Return the geoid undulation for the configured point in meters."""
        model = GeoidModel(extra_data_dir=self.resolve_grid_dir())
        return model.undulation(self.latitude, self.longitude)

    def compute_height_difference(
        self,
        ellipsoidal_height_m: float,
        orthometric_height_m: float,
    ) -> dict:
        """Compute the height difference and the geoid undulation."""
        undulation_m = self.get_geoid_undulation()
        difference_m = ellipsoidal_height_m - orthometric_height_m


        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "ellipsoidal_height_m": ellipsoidal_height_m,
            "orthometric_height_m": orthometric_height_m,
            "geoid_undulation_m": undulation_m,
            "height_difference_m": difference_m,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the height difference between ellipsoidal and orthometric heights")
    parser.add_argument("--ellipsoidal", type=float, default=100.0, help="Ellipsoidal height in meters")
    parser.add_argument("--orthometric", type=float, default=82.5, help="Orthometric height in meters")
    parser.add_argument("--lat", type=float, default=31.7683, help="Latitude for the dry-run point")
    parser.add_argument("--lon", type=float, default=35.2137, help="Longitude for the dry-run point")
    parser.add_argument("--grid-dir", type=str, default=None, help="Optional extra directory for PROJ to search for grids")
    args = parser.parse_args()

    try:
        calculator = HeightDifferenceCalculator(
            grid_dir=args.grid_dir,
            latitude=args.lat,
            longitude=args.lon,
        )
        result = calculator.compute_height_difference(args.ellipsoidal, args.orthometric)
    except (FileNotFoundError, ImportError, GeoidGridUnavailable) as exc:
        print(f"Height difference calculation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("Height difference result:")
    print(f"  Latitude: {result['latitude']}")
    print(f"  Longitude: {result['longitude']}")
    print(f"  Ellipsoidal height: {result['ellipsoidal_height_m']:.3f} m")
    print(f"  Orthometric height: {result['orthometric_height_m']:.3f} m")
    print(f"  Geoid undulation (N): {result['geoid_undulation_m']:.3f} m")
    print(f"  Difference (h - H): {result['height_difference_m']:.3f} m")


if __name__ == "__main__":
    main()
