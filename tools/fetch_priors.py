#!/usr/bin/env python3
"""Populate the local prior cache for one operating area.

Task 3 calls for GLO-30 and Overture cached locally so initialization runs
offline. This fetches them once, into the layout LocalFilePriorProvider expects:

    data/priors/
        glo30/    Copernicus_DSM_COG_10_N31_00_E035_00_DEM.tif
        overture/ overture_N31_00_E035_00_buildings.geojson

Both files carry the tile tag of the 1-degree cell they belong to, which is what
the point-in-tile lookup keys on.

    python tools/fetch_priors.py --lat 31.76465 --lon 35.19134

Sources
-------
DEM      Copernicus GLO-30, from the AWS Open Data mirror. Real data, no
         credentials. One full 1-degree tile, 6-40 MB depending on terrain.
Vectors  OpenStreetMap buildings via Overpass, as a stand-in for an Overture
         extract. Overture releases are partitioned parquet and pulling one
         area out of them needs a spatial query engine; OSM gives the same
         geometry for a bounded area over plain HTTP.

The vector layer covers a RADIUS around the point, not the whole tile. The
actual extent is recorded in the file's ``bbox`` so a consumer can tell what it
really has, rather than inferring full tile coverage from the name.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapinit.geo.tiles import parse_tile_bounds  # noqa: E402

GLO30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

#: Jerusalem and its neighbourhoods, generous enough to include Givat Mordechai,
#: Malha and Givat Ram in the south-west through Pisgat Ze'ev in the north.
JERUSALEM_BBOX = (31.70, 35.14, 31.89, 35.29)


def tile_tag(latitude: float, longitude: float) -> str:
    """SW-corner tag of the 1-degree tile containing the point."""
    lat_floor, lon_floor = math.floor(latitude), math.floor(longitude)
    ns, ew = ("N" if lat_floor >= 0 else "S"), ("E" if lon_floor >= 0 else "W")
    return f"{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00"


def fetch_dem(tag: str, out_dir: Path) -> Path:
    """Download the Copernicus GLO-30 tile, skipping if it is already cached."""
    name = f"Copernicus_DSM_COG_10_{tag}_DEM"
    target = out_dir / f"{name}.tif"
    if target.exists():
        print(f"  DEM already cached: {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
        return target

    url = f"{GLO30_BASE}/{name}/{name}.tif"
    print(f"  downloading {url}")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(target, "wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)  # never leave a truncated tile behind
        raise
    print(f"  wrote {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def fetch_buildings(
    latitude: float,
    longitude: float,
    radius_m: float,
    tag: str,
    out_dir: Path,
    bbox: "tuple[float, float, float, float] | None" = None,
) -> Path:
    """Fetch building footprints and write them as GeoJSON.

    Either a radius around the point, or an explicit bbox
    (min_lat, min_lon, max_lat, max_lon) when a whole built-up area is wanted.
    """
    target = out_dir / f"overture_{tag}_buildings.geojson"

    if bbox is not None:
        area = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        selector = f"way[building]({area});relation[building]({area});"
        extent = f"bbox {area}"
    else:
        around = f"around:{radius_m:.0f},{latitude},{longitude}"
        selector = f"way[building]({around});relation[building]({around});"
        extent = f"{radius_m:.0f} m radius"

    query = f"[out:json][timeout:600];({selector});out geom;"
    print(f"  querying Overpass for buildings in {extent}")

    request = urllib.request.Request(
        OVERPASS_URL,
        data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "valisight-prior-cache"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.load(response)

    features = []
    lats, lons = [], []
    for element in payload.get("elements", []):
        geometry = element.get("geometry")
        if not geometry or len(geometry) < 4:
            continue
        ring = [[p["lon"], p["lat"]] for p in geometry]
        if ring[0] != ring[-1]:
            ring.append(ring[0])  # GeoJSON polygons must close
        lons.extend(p[0] for p in ring)
        lats.extend(p[1] for p in ring)

        tags = element.get("tags", {})
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "id": element.get("id"),
                "class": tags.get("building", "yes"),
                # Height, where OSM has it. Note GLO-30 is a DSM and already
                # includes structures, so ground + height is an assignment, not
                # a sum (CLAUDE.md trap #8).
                "height_m": tags.get("height"),
                "levels": tags.get("building:levels"),
                "name": tags.get("name"),
            },
        })

    if not features:
        raise RuntimeError(f"Overpass returned no buildings for {extent} near {latitude}, {longitude}")

    collection = {
        "type": "FeatureCollection",
        # The real extent, so a consumer is not left inferring full tile
        # coverage from the tile tag in the filename
        "bbox": [min(lons), min(lats), max(lons), max(lats)],
        "properties": {
            "source": "OpenStreetMap via Overpass, stand-in for an Overture extract",
            "tile": tag,
            "requested_extent": extent,
            "center": [longitude, latitude],
            "feature_count": len(features),
        },
        "features": features,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(collection))
    print(f"  wrote {target.name}: {len(features)} buildings, {target.stat().st_size / 1e6:.1f} MB")
    return target


#: OSM highway classes worth caching. Tracks and unclassified roads are
#: included deliberately: the platform does not stay on paved roads, so the
#: rough routes are exactly the ones that carry a heading reference where the
#: paved network has nothing.
ROAD_CLASSES = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential"
    "|service|track|path|living_street"
)


def fetch_roads(bbox, tag: str, out_dir: Path) -> Path:
    """Fetch the road and track network for a box, as GeoJSON linestrings.

    A road carries a bearing, and a vehicle travelling along one is pointing
    roughly that way. That makes the network an absolute heading reference in a
    system with no magnetometer, which is the one thing the IMU can never
    supply on its own.

    It is a prior, not a constraint: off-road the vehicle has no bearing to
    borrow, so anything consuming this has to degrade rather than assume.
    """
    target = out_dir / f"roads_{tag}.geojson"
    area = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    query = (
        f"[out:json][timeout:600];"
        f"way[highway~'^({ROAD_CLASSES})$']({area});"
        f"out geom;"
    )
    print(f"  querying Overpass for roads and tracks in bbox {area}")

    request = urllib.request.Request(
        OVERPASS_URL,
        data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "valisight-prior-cache"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.load(response)

    features, lats, lons = [], [], []
    for element in payload.get("elements", []):
        geometry = element.get("geometry")
        if not geometry or len(geometry) < 2:
            continue
        line = [[p["lon"], p["lat"]] for p in geometry]
        lons.extend(p[0] for p in line)
        lats.extend(p[1] for p in line)

        tags = element.get("tags", {})
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": line},
            "properties": {
                "id": element.get("id"),
                "highway": tags.get("highway"),
                "name": tags.get("name"),
                # A one-way road resolves the 180 degree ambiguity a bearing
                # alone leaves open
                "oneway": tags.get("oneway"),
                "surface": tags.get("surface"),
                "lanes": tags.get("lanes"),
            },
        })

    if not features:
        raise RuntimeError(f"Overpass returned no roads for bbox {area}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath(target.name).write_text(json.dumps({
        "type": "FeatureCollection",
        "bbox": [min(lons), min(lats), max(lons), max(lats)],
        "properties": {
            "source": "OpenStreetMap via Overpass",
            "tile": tag,
            "feature_count": len(features),
        },
        "features": features,
    }))
    print(f"  wrote {target.name}: {len(features):,} ways, {target.stat().st_size / 1e6:.1f} MB")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lat", type=float, default=31.76465, help="Operating area latitude")
    parser.add_argument("--lon", type=float, default=35.19134, help="Operating area longitude")
    parser.add_argument("--radius", type=float, default=3000.0, help="Vector fetch radius in meters")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"),
        default=None,
        help="Fetch vectors for this box instead of a radius, e.g. all of a city",
    )
    parser.add_argument(
        "--jerusalem",
        action="store_true",
        help="Shorthand for the bbox covering Jerusalem and its neighbourhoods",
    )
    parser.add_argument(
        "--priors-dir",
        type=Path,
        default=REPO_ROOT / "data" / "priors",
        help="Where the cache is written",
    )
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else (JERUSALEM_BBOX if args.jerusalem else None)

    tag = tile_tag(args.lat, args.lon)
    bounds = parse_tile_bounds(Path(f"x_{tag}_y.tif"))
    print(f"Operating point {args.lat}, {args.lon} -> tile {tag} {bounds}")

    fetch_dem(tag, args.priors_dir / "glo30")
    fetch_buildings(
        args.lat, args.lon, args.radius, tag, args.priors_dir / "overture", bbox=bbox
    )
    print(f"\nPrior cache ready under {args.priors_dir}")


if __name__ == "__main__":
    main()
