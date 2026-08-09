#!/usr/bin/env python3
"""Pull Overture building footprints, with heights, for one operating area.

The OpenStreetMap stand-in this replaces carries a footprint for nearly every
building but a height for almost none, which is enough for a ground plane and
useless for anything three-dimensional. Overture merges heights from several
sources, including ones derived from imagery, so small buildings that a 30 m
DEM cannot resolve still get a height here.

    python tools/fetch_overture.py --jerusalem

Reads the release parquet directly over HTTPS with DuckDB and writes GeoJSON
into the prior cache, alongside what fetch_priors.py already puts there.

The scan is slow -- the release is partitioned globally and the bounding box
filter still has to touch every file's metadata -- so this is a background job,
not something to sit and wait on. It only needs running when the operating area
changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapinit.geo.tiles import parse_tile_bounds  # noqa: E402

#: Release to read. Pinned rather than resolved at runtime so a cache rebuild
#: reproduces the same data instead of silently picking up a newer release.
OVERTURE_RELEASE = "2026-07-22.0"
OVERTURE_PATH = (
    f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}"
    f"/theme=buildings/type=building/*"
)

#: Jerusalem and its neighbourhoods, matching tools/fetch_priors.py.
JERUSALEM_BBOX = (31.70, 35.14, 31.89, 35.29)

#: Typical storey height in meters, for estimating from a floor count when no
#: explicit height is published. Israeli residential floors run close to this.
METRES_PER_FLOOR = 3.2


def tile_tag(latitude: float, longitude: float) -> str:
    lat_floor, lon_floor = math.floor(latitude), math.floor(longitude)
    ns, ew = ("N" if lat_floor >= 0 else "S"), ("E" if lon_floor >= 0 else "W")
    return f"{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00"


def fetch(bbox, out_path: Path) -> dict:
    """Query the release for one box and write GeoJSON. Returns a summary."""
    import duckdb

    min_lat, min_lon, max_lat, max_lon = bbox
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2';")

    # Filter on the stored bbox struct rather than on the geometry: it is a
    # plain column, so row groups outside the area are skipped without any
    # geometry being decoded.
    query = f"""
        SELECT
            id,
            height,
            num_floors,
            min_height,
            roof_height,
            roof_shape,
            subtype,
            class,
            names.primary AS name,
            ST_AsGeoJSON(geometry) AS geojson,
            bbox.xmin AS xmin, bbox.ymin AS ymin,
            bbox.xmax AS xmax, bbox.ymax AS ymax
        FROM read_parquet('{OVERTURE_PATH}')
        WHERE bbox.xmin BETWEEN {min_lon} AND {max_lon}
          AND bbox.ymin BETWEEN {min_lat} AND {max_lat}
    """
    started = time.time()
    rows = con.execute(query).fetchall()
    elapsed = time.time() - started

    features = []
    counts = {"explicit": 0, "from_floors": 0, "none": 0}
    for (
        identifier, height, num_floors, min_height, roof_height, roof_shape,
        subtype, building_class, name, geojson, xmin, ymin, xmax, ymax,
    ) in rows:
        # Prefer a published height; fall back to floors, and record which was
        # used so a consumer can weight an estimate differently from a measurement
        if height is not None:
            resolved, source = float(height), "published"
            counts["explicit"] += 1
        elif num_floors is not None:
            resolved, source = float(num_floors) * METRES_PER_FLOOR, "from_floors"
            counts["from_floors"] += 1
        else:
            resolved, source = None, "unknown"
            counts["none"] += 1

        features.append({
            "type": "Feature",
            "geometry": json.loads(geojson),
            "properties": {
                "id": identifier,
                "name": name,
                "class": building_class,
                "subtype": subtype,
                # height_m is the number to extrude with. It is measured from
                # ground, so a roof sits at ground + height_m as an assignment,
                # never added on top of a DSM value that already includes it.
                "height_m": resolved,
                "height_source": source,
                "published_height_m": height,
                "num_floors": num_floors,
                "min_height_m": min_height,
                "roof_height_m": roof_height,
                "roof_shape": roof_shape,
            },
        })

    collection = {
        "type": "FeatureCollection",
        "bbox": [min_lon, min_lat, max_lon, max_lat],
        "properties": {
            "source": f"Overture Maps release {OVERTURE_RELEASE}",
            "metres_per_floor": METRES_PER_FLOOR,
            "feature_count": len(features),
            "with_published_height": counts["explicit"],
            "with_height_from_floors": counts["from_floors"],
            "without_any_height": counts["none"],
            "query_seconds": round(elapsed, 1),
        },
        "features": features,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(collection))
    return collection["properties"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", type=float, nargs=4, default=None,
                        metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"))
    parser.add_argument("--jerusalem", action="store_true", help="Shorthand for the Jerusalem box")
    parser.add_argument("--priors-dir", type=Path, default=REPO_ROOT / "data" / "priors")
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else (JERUSALEM_BBOX if args.jerusalem else None)
    if bbox is None:
        parser.error("pass --jerusalem or --bbox MIN_LAT MIN_LON MAX_LAT MAX_LON")

    tag = tile_tag((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    target = args.priors_dir / "overture" / f"overture_{tag}_buildings.geojson"

    print(f"release {OVERTURE_RELEASE}, bbox {bbox} -> tile {tag}")
    print("scanning the global release; this takes a while...", flush=True)

    summary = fetch(bbox, target)
    total = summary["feature_count"]
    print(f"\nwrote {target} ({target.stat().st_size / 1e6:.1f} MB) in {summary['query_seconds']:.0f}s")
    print(f"  buildings              : {total:,}")
    for label, key in [
        ("published height", "with_published_height"),
        ("height from floors", "with_height_from_floors"),
        ("no height at all", "without_any_height"),
    ]:
        value = summary[key]
        share = 100 * value / total if total else 0.0
        print(f"  {label:23s}: {value:,} ({share:.1f}%)")


if __name__ == "__main__":
    main()
