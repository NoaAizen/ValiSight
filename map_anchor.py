"""
Anchor a recorded session to the campus map.

Modes:

1. GPS anchoring (needs the georeferenced map from map_geotiles.py):
       python map_anchor.py --latlon 31.76465 35.19134 --heading 350
   Heading is a compass bearing (0 = north, clockwise). If --heading is
   omitted, the map opens for ONE click in the direction the radar faced.

2. Click anchoring on any map image: click the point where the radar
   stood, then a second click in the direction it was facing, press S:
       python map_anchor.py
       python map_anchor.py logs/session_20260714_143531

3. One-time scale calibration for a NON-georeferenced map image
   (maps/campus_sat.png from map_geotiles.py never needs this):
       python map_anchor.py --calibrate 50
   Click TWO points a known real distance apart (here 50 m), press S.

Writes into the session's meta.json:
    "map": {"image": ..., "x_px": ..., "y_px": ..., "heading_deg": ...,
            "lat": ..., "lon": ...}   # heading: 0 = map-up, clockwise

Keys:  S/Enter = save   R = reset clicks   Q/Esc = quit without saving
"""
import argparse
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
MAP_IMAGE = os.path.join("maps", "campus_lev.png")
MAP_CONFIG = os.path.join(HERE, "maps", "map_config.json")
FIT_W, FIT_H = 1400, 850                      # max on-screen window size


def sessions():
    return sorted(glob.glob(os.path.join(HERE, "logs", "session_*")))


def load_config():
    if not os.path.isfile(MAP_CONFIG):
        return None
    with open(MAP_CONFIG) as f:
        return json.load(f)


def load_map(path):
    try:                                       # cv2.imread fails on Hebrew paths
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        img = None
    if img is None:
        sys.exit("Map image not found: %s" % path)
    return img


class ClickCollector:
    """Fit-to-screen viewer that records clicks in full-res map coords."""

    def __init__(self, img, window):
        self.window = window
        h, w = img.shape[:2]
        self.scale = min(1.0, FIT_W / w, FIT_H / h)
        self.view = cv2.resize(img, None, fx=self.scale, fy=self.scale) \
            if self.scale < 1.0 else img.copy()
        self.base = self.view.copy()
        self.clicks = []                       # full-res (x, y)
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicks) < 2:
            self.clicks.append((x / self.scale, y / self.scale))
            self.redraw()

    def redraw(self):
        self.view = self.base.copy()
        pts = [(int(x * self.scale), int(y * self.scale))
               for x, y in self.clicks]
        for p in pts:
            cv2.circle(self.view, p, 6, (0, 0, 255), 2)
        if len(pts) == 2:
            cv2.arrowedLine(self.view, pts[0], pts[1], (0, 0, 255), 2,
                            tipLength=0.15)

    def reset(self):
        self.clicks = []
        self.redraw()

    def run(self, need, banner):
        """Loop until save (returns clicks) or quit (returns None)."""
        while True:
            frame = self.view.copy()
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (40, 40, 40), -1)
            cv2.putText(frame, "%s   [%d/%d clicks]  S=save R=reset Q=quit"
                        % (banner, len(self.clicks), need),
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)
            cv2.imshow(self.window, frame)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                return None
            if k == ord("r"):
                self.reset()
            if k in (ord("s"), 13) and len(self.clicks) == need:
                return self.clicks


def calibrate(map_path, known_m):
    cfg = load_config()
    if cfg and "geo" in cfg and cfg["image"] == map_path.replace("\\", "/"):
        sys.exit("The georeferenced map has an exact scale already -- "
                 "no calibration needed.")
    img = load_map(os.path.join(HERE, map_path))
    cc = ClickCollector(img, "calibrate scale")
    clicks = cc.run(2, "Click 2 points that are %.1f m apart" % known_m)
    cv2.destroyAllWindows()
    if not clicks:
        sys.exit("Cancelled.")
    (x1, y1), (x2, y2) = clicks
    px = math.hypot(x2 - x1, y2 - y1)
    if px < 5:
        sys.exit("Points too close together, try again.")
    mpp = known_m / px
    cfg = {"image": map_path.replace("\\", "/"),
           "meters_per_px": round(mpp, 5), "estimated": False}
    with open(MAP_CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Scale: %.2f px = %.1f m  ->  %.4f m/px" % (px, known_m, mpp))
    print("Saved:", MAP_CONFIG)


def save_anchor(sdir, map_path, x, y, heading, lat=None, lon=None):
    meta_p = os.path.join(sdir, "meta.json")
    if not os.path.isfile(meta_p):
        sys.exit("No meta.json in %s" % sdir)
    with open(meta_p) as f:
        meta = json.load(f)
    meta["map"] = {"image": map_path.replace("\\", "/"),
                   "x_px": round(x, 1), "y_px": round(y, 1),
                   "heading_deg": round(heading % 360.0, 1)}
    if lat is not None:
        meta["map"]["lat"] = lat
        meta["map"]["lon"] = lon
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=2)
    print("Anchored %s at (%.0f, %.0f) heading %.0f deg (0=up, clockwise)"
          % (os.path.basename(sdir), x, y, heading % 360.0))
    print("Now run: python map_overlay.py %s" % os.path.relpath(sdir, HERE))


def anchor_clicks(map_path, sdir):
    img = load_map(os.path.join(HERE, map_path))
    cc = ClickCollector(img, "anchor " + os.path.basename(sdir))
    clicks = cc.run(2, "Click radar position, then a point it was FACING")
    cv2.destroyAllWindows()
    if not clicks:
        sys.exit("Cancelled.")
    (x1, y1), (x2, y2) = clicks
    heading = math.degrees(math.atan2(x2 - x1, -(y2 - y1)))
    save_anchor(sdir, map_path, x1, y1, heading)


def anchor_latlon(sdir, lat, lon, heading):
    cfg = load_config()
    if not cfg or "geo" not in cfg:
        sys.exit("No georeferenced map. Run: python map_geotiles.py")
    from map_geotiles import latlon_to_px
    geo = cfg["geo"]
    gx, gy = latlon_to_px(lat, lon, geo["zoom"])
    x, y = gx - geo["origin_x"], gy - geo["origin_y"]
    img = load_map(os.path.join(HERE, cfg["image"]))
    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        sys.exit("(%.6f, %.6f) is outside the map -- refetch with a larger "
                 "--bbox in map_geotiles.py" % (lat, lon))
    if heading is None:                        # one click to set facing
        cc = ClickCollector(img, "anchor " + os.path.basename(sdir))
        cc.clicks = [(x, y)]
        cc.redraw()
        clicks = cc.run(2, "GPS position marked -- click a point it was FACING")
        cv2.destroyAllWindows()
        if not clicks:
            sys.exit("Cancelled.")
        (_, _), (x2, y2) = clicks
        heading = math.degrees(math.atan2(x2 - x, -(y2 - y)))
    save_anchor(sdir, cfg["image"], x, y, heading, lat=lat, lon=lon)


def main():
    ap = argparse.ArgumentParser(description="Anchor a session to the campus map")
    ap.add_argument("session", nargs="?", help="session dir (default: latest)")
    ap.add_argument("--calibrate", type=float, metavar="METERS",
                    help="calibrate map scale with a known distance")
    ap.add_argument("--latlon", nargs=2, type=float, metavar=("LAT", "LON"),
                    help="anchor at GPS coords (georeferenced map required)")
    ap.add_argument("--heading", type=float,
                    help="compass bearing the radar faced (0=north, clockwise)")
    ap.add_argument("--map", default=None,
                    help="map image (default: configured map, else %s)" % MAP_IMAGE)
    ap.add_argument("--list", action="store_true", help="list sessions")
    args = ap.parse_args()

    if args.list:
        for s in sessions():
            print(os.path.basename(s))
        return

    cfg = load_config()
    map_path = args.map or (cfg["image"] if cfg else MAP_IMAGE)
    if args.calibrate:
        calibrate(map_path, args.calibrate)
        return

    sdir = args.session or (sessions()[-1] if sessions() else None)
    if not sdir or not os.path.isdir(sdir):
        sys.exit("No session found. Run live_radar_camera.py first.")
    if args.latlon:
        anchor_latlon(sdir, args.latlon[0], args.latlon[1], args.heading)
    else:
        if not cfg:
            print("NOTE: no map scale yet -- map_overlay.py will use a rough "
                  "estimate. Run map_geotiles.py (georeferenced) or "
                  "map_anchor.py --calibrate <meters>.")
        anchor_clicks(map_path, sdir)


if __name__ == "__main__":
    main()
