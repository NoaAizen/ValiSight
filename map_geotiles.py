"""
Build a georeferenced north-up satellite map of the campus.

Downloads ESRI World Imagery tiles for a lat/lon bounding box, stitches them
into maps/campus_sat.png, and writes maps/map_config.json with the Web-Mercator
origin so map_anchor.py can place GPS coordinates (--latlon) exactly.
Pixel scale is computed from the zoom level -- no manual calibration needed.

Usage:
    python map_geotiles.py                # default: Machon Lev campus, z=19
    python map_geotiles.py --bbox 31.7632 35.1885 31.7662 35.1945 --zoom 19
"""
import argparse
import json
import math
import os
import sys
import urllib.request

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):        # Hebrew paths on cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_IMAGE = os.path.join("maps", "campus_sat.png")
MAP_CONFIG = os.path.join(HERE, "maps", "map_config.json")
TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
TILE = 256


def latlon_to_px(lat, lon, zoom):
    """Lat/lon -> global Web-Mercator pixel coords at the given zoom."""
    n = TILE * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * n
    r = math.radians(lat)
    y = (1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(z, x, y):
    req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y),
                                 headers={"User-Agent": "VailSight-mapper"})
    data = urllib.request.urlopen(req, timeout=30).read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("bad tile %d/%d/%d" % (z, x, y))
    return img


def main():
    ap = argparse.ArgumentParser(description="Fetch georeferenced satellite map")
    ap.add_argument("--bbox", nargs=4, type=float,
                    metavar=("LAT_S", "LON_W", "LAT_N", "LON_E"),
                    default=[31.7632, 35.1885, 31.7662, 35.1945],
                    help="bounding box (default: Machon Lev campus)")
    ap.add_argument("--zoom", type=int, default=19, help="tile zoom (default 19)")
    args = ap.parse_args()

    lat_s, lon_w, lat_n, lon_e = args.bbox
    z = args.zoom
    x0, y0 = latlon_to_px(lat_n, lon_w, z)      # top-left
    x1, y1 = latlon_to_px(lat_s, lon_e, z)      # bottom-right
    tx0, ty0 = int(x0 // TILE), int(y0 // TILE)
    tx1, ty1 = int(x1 // TILE), int(y1 // TILE)
    cols, rows = tx1 - tx0 + 1, ty1 - ty0 + 1
    print("zoom %d: %d x %d tiles (%d total)" % (z, cols, rows, cols * rows))

    mosaic = np.zeros((rows * TILE, cols * TILE, 3), np.uint8)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            tile = fetch_tile(z, tx, ty)
            mosaic[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE,
                   (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = tile
        print("  row %d/%d" % (ty - ty0 + 1, rows))

    # crop mosaic to the exact bbox
    cx0, cy0 = int(x0 - tx0 * TILE), int(y0 - ty0 * TILE)
    cx1, cy1 = int(x1 - tx0 * TILE), int(y1 - ty0 * TILE)
    img = mosaic[cy0:cy1, cx0:cx1]

    out_abs = os.path.join(HERE, OUT_IMAGE)
    ok, buf = cv2.imencode(".png", img)         # Hebrew-path-safe imwrite
    if not ok:
        sys.exit("PNG encode failed")
    buf.tofile(out_abs)

    lat_mid = (lat_s + lat_n) / 2.0
    mpp = 156543.03392 * math.cos(math.radians(lat_mid)) / (2 ** z)
    cfg = {"image": OUT_IMAGE.replace("\\", "/"),
           "meters_per_px": round(mpp, 5), "estimated": False,
           "geo": {"zoom": z, "origin_x": tx0 * TILE + cx0,
                   "origin_y": ty0 * TILE + cy0}}
    with open(MAP_CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)

    print("saved  : %s  (%d x %d px)" % (OUT_IMAGE, img.shape[1], img.shape[0]))
    print("scale  : %.4f m/px (exact, from zoom %d)" % (mpp, z))
    print("config : %s (georeferenced, north-up)" % MAP_CONFIG)
    print("imagery: Esri World Imagery")


if __name__ == "__main__":
    main()
