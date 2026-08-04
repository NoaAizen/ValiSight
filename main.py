#!/usr/bin/env python3
"""Entry point: initialize position relative to known maps.

    python main.py --lat 31.7683 --lon 35.2137
    python main.py --diagnose            # run every stage, report every check
    python main.py --targets data/targets.json --expect-geoid 19.0 20.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mapinit import MapInitializer
from mapinit.calibration import SurveyedTargetConstraint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lat", type=float, required=True, help="Latitude to initialize at, in decimal degrees")
    parser.add_argument("--lon", type=float, required=True, help="Longitude to initialize at, in decimal degrees")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run every stage instead of stopping at the first failure",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="JSON file of surveyed calibration targets; omit to skip calibration",
    )
    parser.add_argument(
        "--expect-geoid",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Assert the geoid undulation falls in this range, in meters, for the operating area",
    )
    args = parser.parse_args()

    # Extending the calibration means appending another CalibrationConstraint
    # here; neither CalibrationStage nor the runner needs to know about it.
    constraints = []
    if args.targets is not None:
        constraints.append(SurveyedTargetConstraint.from_json(args.targets))

    initializer = MapInitializer(
        latitude=args.lat,
        longitude=args.lon,
        expected_geoid_range=tuple(args.expect_geoid) if args.expect_geoid else None,
        constraints=constraints,
    )

    report = initializer.run(fail_fast=not args.diagnose)
    print(report.summary())
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
