"""
Draw a session's radar detections on the campus map.

Needs the session anchored first (position + heading on the map):
    python map_anchor.py [session]        # click position + facing point

Then:
    python map_overlay.py                 # latest session
    python map_overlay.py logs/session_20260714_143531
    python map_overlay.py --show          # also open a window

Reads clusters.csv (radar frame: x=forward, y=left, z=up) and tracks.jsonl,
converts to map pixels via the anchor + scale in maps/map_config.json, and
writes into the session dir:
    map_overlay.png        full campus map with detections
    map_overlay_zoom.png   crop around the radar position

Colors: metal=orange, fabric=green, unknown=gray, moving objects=magenta ring.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):        # Hebrew paths on cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_CONFIG = os.path.join(HERE, "maps", "map_config.json")


def imread_u(path):
    """cv2.imread fails on Hebrew paths on Windows -- decode via numpy."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def imwrite_u(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)
    return ok
FALLBACK_MPP = 0.20                # rough guess (m/px) until --calibrate is run

MAT_COLOR = {                      # BGR
    "metal":   (0, 100, 255),      # orange
    "fabric":  (60, 200, 60),      # green
    "unknown": (170, 170, 170),    # gray
}
MOVING_COLOR = (255, 0, 255)       # magenta ring for non-static clusters
SENSOR_COLOR = (255, 120, 0)       # bright blue


def sessions():
    return sorted(glob.glob(os.path.join(HERE, "logs", "session_*")))


def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def to_map_px(x_fwd, y_left, ax, ay, heading_deg, mpp):
    """Radar-local meters (x=forward, y=left) -> map pixel coords."""
    h = math.radians(heading_deg)
    fx, fy = math.sin(h), -math.cos(h)         # forward on image (y down)
    rx, ry = math.cos(h), math.sin(h)          # right on image
    px = ax + (x_fwd * fx - y_left * rx) / mpp
    py = ay + (x_fwd * fy - y_left * ry) / mpp
    return px, py


def main():
    ap = argparse.ArgumentParser(description="Overlay session detections on map")
    ap.add_argument("session", nargs="?", help="session dir (default: latest)")
    ap.add_argument("--grid", type=float, default=0.25,
                    help="aggregation grid in meters (default 0.25)")
    ap.add_argument("--min-count", type=int, default=3,
                    help="min cluster observations per grid cell (default 3)")
    ap.add_argument("--zoom-radius", type=float, default=15.0,
                    help="zoom crop radius around sensor, meters (default 15)")
    ap.add_argument("--max-tracks", type=int, default=12,
                    help="label at most N longest-lived tracks (default 12)")
    ap.add_argument("--show", action="store_true", help="open result window")
    args = ap.parse_args()

    sdir = args.session or (sessions()[-1] if sessions() else None)
    if not sdir or not os.path.isdir(sdir):
        sys.exit("No session found.")
    print("Session:", sdir)

    with open(os.path.join(sdir, "meta.json")) as f:
        meta = json.load(f)
    if "map" not in meta:
        sys.exit("Session not anchored to the map yet.\n"
                 "Run: python map_anchor.py %s" % os.path.relpath(sdir, HERE))
    anc = meta["map"]

    if os.path.isfile(MAP_CONFIG):
        with open(MAP_CONFIG) as f:
            cfg = json.load(f)
        mpp = cfg["meters_per_px"]
        if cfg.get("estimated"):
            print("WARNING: map scale is a rough estimate -- run "
                  "map_anchor.py --calibrate <meters>")
    else:
        mpp = FALLBACK_MPP
        print("WARNING: maps/map_config.json missing, assuming %.2f m/px. "
              "Run map_anchor.py --calibrate <meters>." % mpp)

    img = imread_u(os.path.join(HERE, anc["image"]))
    if img is None:
        sys.exit("Map image not found: %s" % anc["image"])
    ax, ay, heading = anc["x_px"], anc["y_px"], anc["heading_deg"]

    # ---- aggregate clusters on a grid so 1000s of rows become stable dots --
    cpath = os.path.join(sdir, "clusters.csv")
    if not os.path.isfile(cpath):
        sys.exit("No clusters.csv -- run analyze_session.py first.")
    cells = {}                                  # (gx, gy) -> {mat: count}
    n_rows = 0
    with open(cpath) as f:
        for row in csv.DictReader(f):
            try:
                x, y = float(row["x"]), float(row["y"])
            except (ValueError, TypeError):
                continue
            n_rows += 1
            key = (round(x / args.grid), round(y / args.grid))
            mat = row["material"] or "unknown"
            moving = row["label"] != "static"
            d = cells.setdefault(key, {})
            k = (mat, moving)
            d[k] = d.get(k, 0) + 1

    # ---- collect drawables in map-px coords ---------------------------------
    dots = []                                   # (px, py, weight, mat, moving)
    for (gx, gy), counts in cells.items():
        if sum(counts.values()) < args.min_count:
            continue
        (mat, moving), n = max(counts.items(), key=lambda kv: kv[1])
        px, py = to_map_px(gx * args.grid, gy * args.grid,
                           ax, ay, heading, mpp)
        dots.append((px, py, n, mat, moving))

    tracks = load_jsonl(os.path.join(sdir, "tracks.jsonl"))
    tracks = sorted(tracks, key=lambda t: -t["features"]["hits"])
    tracks = tracks[:args.max_tracks]
    tmarks = []                                 # (px, py, label)
    for tr in tracks:
        f = tr["features"]
        az = math.radians(f["az_deg"])
        x = f["range_m"] * math.cos(az)
        y = -f["range_m"] * math.sin(az)        # az>0 is right, y is left
        px, py = to_map_px(x, y, ax, ay, heading, mpp)
        tmarks.append((px, py, "T%d %s" % (tr["track_id"], tr["material"])))

    hfov = meta.get("hfov", 60.0)

    def render(canvas, scale, ox, oy):
        """Draw everything onto canvas; map-px p -> ((p-o)*scale)."""
        def T(px, py):
            return (int(round((px - ox) * scale)), int(round((py - oy) * scale)))

        overlay = canvas.copy()
        for px, py, n, mat, moving in dots:
            r = max(2, int((1.5 + min(3, math.log1p(n))) * scale * 0.5))
            p = T(px, py)
            cv2.circle(overlay, p, r, MAT_COLOR.get(mat, MAT_COLOR["unknown"]), -1)
            if moving:
                cv2.circle(overlay, p, r + max(1, int(scale)), MOVING_COLOR,
                           max(1, int(scale * 0.4)))
        canvas = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)

        # sensor + camera FOV wedge
        sp = T(ax, ay)
        for sign in (-1, 1):
            h = math.radians(heading + sign * hfov / 2.0)
            ex = ax + 8.0 / mpp * math.sin(h)
            ey = ay - 8.0 / mpp * math.cos(h)
            cv2.line(canvas, sp, T(ex, ey), SENSOR_COLOR, max(2, int(scale)))
        r = max(5, int(3 * scale))
        cv2.circle(canvas, sp, r, SENSOR_COLOR, -1)
        cv2.circle(canvas, sp, r, (255, 255, 255), max(1, int(scale * 0.5)))

        # track labels only when zoomed enough to be readable
        if scale >= 2:
            for i, (px, py, label) in enumerate(tmarks):
                p = T(px, py)
                cv2.drawMarker(canvas, p, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 16, 3)
                cv2.drawMarker(canvas, p, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 1)
                tp = (p[0] + 10, p[1] + 5 + (i % 3 - 1) * 18)   # stagger rows
                cv2.putText(canvas, label, tp, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 3)
                cv2.putText(canvas, label, tp, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

        # legend (fixed size, canvas corner)
        items = [("metal", MAT_COLOR["metal"]), ("fabric", MAT_COLOR["fabric"]),
                 ("unknown", MAT_COLOR["unknown"]), ("moving", MOVING_COLOR),
                 ("sensor", SENSOR_COLOR)]
        cv2.rectangle(canvas, (10, 10), (150, 22 + 22 * len(items)),
                      (255, 255, 255), -1)
        cv2.rectangle(canvas, (10, 10), (150, 22 + 22 * len(items)),
                      (0, 0, 0), 1)
        for i, (name, col) in enumerate(items):
            y = 28 + 22 * i
            cv2.circle(canvas, (24, y), 6, col, -1)
            cv2.putText(canvas, name, (38, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1)
        return canvas

    out_full = os.path.join(sdir, "map_overlay.png")
    imwrite_u(out_full, render(img.copy(), 1.0, 0, 0))

    # ---- zoom crop around the sensor: upscale FIRST, then draw crisp -------
    rad_px = args.zoom_radius / mpp
    x0 = max(0, int(ax - rad_px)); y0 = max(0, int(ay - rad_px))
    x1 = min(img.shape[1], int(ax + rad_px)); y1 = min(img.shape[0], int(ay + rad_px))
    crop = img[y0:y1, x0:x1]
    zoom = max(1.0, 900.0 / max(1, max(crop.shape[:2])))
    crop = cv2.resize(crop, None, fx=zoom, fy=zoom,
                      interpolation=cv2.INTER_CUBIC)
    crop = render(crop, zoom, x0, y0)
    out_zoom = os.path.join(sdir, "map_overlay_zoom.png")
    imwrite_u(out_zoom, crop)

    print("clusters: %d rows -> %d grid cells drawn (grid %.2fm, min %d)"
          % (n_rows, len(dots), args.grid, args.min_count))
    print("tracks  : %d labeled (longest-lived, --max-tracks)" % len(tmarks))
    print("saved   : %s" % out_full)
    print("          %s" % out_zoom)

    if args.show:
        cv2.imshow("map_overlay (any key to close)", crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
