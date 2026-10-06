#!/usr/bin/env python3
"""Live thermal/visible fusion: board streams, this host fuses, browser watches.

    ./live.py                       then open http://localhost:8088
    ./live.py --warp calib/warp.lut --gain 220
    ./live.py --radar               live + IWR1843 overlay (USB DATA port)
    ./live.py --radar --channel thermal_radar
    ./live.py --radar --channel ai  final detections/tracks without raw radar clutter
    ./live.py --record              live + recording to captures/live-<stamp>/
    ./live.py --radar --students    live + the three AI channels (see below)
    ./live.py --radar --record DIR  a full calibration session: session.mp4,
                                    frames.jsonl, thermal.bin, radar.bin+jsonl

The board sends a hardware-JPEG of the visible frame (~11KB for 640x400, 4% of
raw) plus the thermal frame *uncompressed* - the thermal data is the measurement,
and lossy compression on radiometry is not a trade worth making for 17KB. That
puts a frame at ~30KB, which the FS-speed link carries faster than the Lepton
produces frames, so the stream is paced by the sensor rather than the pipe.

Fusion runs through libfusion.so - the same fusion.c that compiles into the
firmware - so what you see here is what the board will do once flashed.

Controls are live, no restart: http://localhost:8088/set?gain=250&eps=120&radius=5

During a calibration session, name each station before holding it - from the
page's pose bar, or /pose?id=N04 - and every frame written from then on carries
that pose_id. /pose?clear=1 between stations, /pose to read back what has been
stamped. Frames written with no station are labelled UNLABELLED on the page
rather than passing quietly; recovering the split afterwards from timestamps is
what the V3 plan section 5d exists to stop.

The channel selector is the product surface: thermal, visible+radar,
thermal+radar, three-sensor fusion, and AI. The separate view selector is there
to judge registration inside the fusion channel - see the VIEWS comment below
for why, and use blink/edges during a calibration session rather than trusting
how sharp the picture looks. The visible source is currently PAG7936 luma; the
rgb_radar API id is reserved for true colour once RGB565 passes its link test.

--students brings up three AI channels, switched on and off live from the page
(keys t, r, c) or over /set?ai_thermal=0&ai_radar=1&ai_fusion=1, and read back
on /ai. They feed fusion and tracking but their intermediate boxes are hidden
by default; final lock boxes are drawn on the thermal-bearing
products and the AI product. Key e (or /set?evidence=1) reveals intermediate
evidence for model debugging:

    thermal   orange - the thermal student, drawn on the visible plane through
              the warp LUT. Without a LUT its boxes are still reported, on the
              thermal plane, but cannot be placed on the picture.
    radar     cyan   - the radar student, straight onto the visible plane.
    fusion    white  - the two of them agreeing: one person, seen by both. The
              candidate pair is made on u only, inside 180 px, because live
              motion can put the radar box behind the thermal box while radar elevation
              comes off a two-element aperture and says almost nothing about
              which box a return belongs to. A fused box carries the range.

`lock` (key l, on by default) is what makes a person survive a frame nobody
found them in. A person seen twice becomes a track with an id: associated
frame to frame by IoU with a centre-distance fallback, smoothed alpha-beta,
carried on their own velocity through the gaps, and dropped 1.5 s after the
last measurement. All three sensors feed one tracker and what describes one
person is merged before association, so a lock the detector loses in glare can
be held by the thermal student - and the letters on the label (D T R F) say
which sensors are holding it right now. A coasting track is amber and dashed
and says so: it is where somebody probably is, not where anybody saw them.

A candidate the DETECTOR has never confirmed must move before it is drawn as a
person. Measured in the lobby this rig sits in: a lit glass door reads 31.7 C
mean / 34.1 C p90 and a person reads 30.9 / 34.1 - the same temperature to
within the sensor's noise, so no radiometric test can separate them, and the
students and the radar will agree with each other about a door all day. What a
door has never done is move. Those candidates are counted as `static` on the
card and in /health rather than dropped quietly, because something warm and
motionless is not nobody. A person the detector sees is drawn at once, standing
still or not.

With `thermal outline` on (key s, opt-in), a detection is drawn as the warm
shape the thermal frame holds inside it rather than as a rectangle - the
detector's green person boxes included, which is the pairing worth having: the
COCO detector is the locator this rig trusts, and the thermal frame is the only
sensor here that knows the shape. Fusion draws its ring around that shape. The
split inside a box is Otsu, not a fixed body-heat threshold, and where no shape
can be found (no LUT, no thermal coverage, nothing warm in the box) the box
comes back.

Each channel has a confidence slider on the same card. It hides boxes rather
than restarting the engines, so the card counts `shown of found` and a quiet
scene never looks like a slider parked too high; below 0.75 a box is drawn as
four corner ticks rather than a rectangle, because a candidate and a detection
should not look alike. Overlapping and nested duplicates - the students emit
eight slots with no NMS of their own - are dropped before any of that.

They are three switches rather than a mode selector because agreement is only
readable beside its components: two channels that fire on everything agree on
everything too. Switching a channel off stops its engine - except when fusion
needs it, which still runs it and simply does not draw it.
"""
import argparse
import glob
import math
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect   # noqa: E402
from detector_models import MODEL_FILES   # noqa: E402
import radar_overlay
import tracker as tracking  # noqa: E402
import recorder  # noqa: E402
import soc as hostsoc  # noqa: E402
import live_channels  # noqa: E402
from web.assets import read_bytes

from viewer.constants import LIB, OUT_W, OUT_H, TH_W, TH_H
from viewer.native import Cfg, Fusion, Temp, Region, PALETTES, Pipeline
from viewer.display import (
    VIEWS, CHANNELS, thermal_edges, coverage_outline, operator_fusion, agc8,
    compose, display_sharpen, display_visible, compose_channel,
)

from viewer.constants import HEAP_FLOOR, LEPTON_SAFE_GAP_MS, DETECT_STALE_S
from viewer.rendering import (
    LIGHT_FLOOR, LIGHT_BLIND_FRAC, LIGHT_LIT_FRAC, scene_light, Latest, Renderer,
)
from viewer.streaming import (
    SETUP_CODE, BATCH_CODE, HEAP_EVERY, _board_error, _PlannedRestart,
    _DroppedBatch, _BATCH_FOOTERS, Streamer,
)
from viewer.telemetry import STALE_S, LEPTON_FPS, SKEW_WARN, health, timing

PORT = "/dev/ttyACM0"

# ---------------------------------------------------------------- http


def worst_level(checks):
    return ("fail" if any(c["level"] == "fail" for c in checks) else
            "warn" if any(c["level"] == "warn" for c in checks) else "ok")


def ui_payload(pipe, state, now):
    """Everything the page redraws each second, in one response.

    The page used to poll five endpoints a second - /health, /timing, /stats,
    /detections, /stat. That is five handler threads a second competing with the
    MJPEG writer on a ThreadingHTTPServer for numbers that all come out of the
    same state dict, and it read them at five *different* instants: a health row
    taken before an FFC could sit on screen beside a temperature band taken after
    it, with nothing on the page to say so. One payload is one moment.

    The five endpoints stay. They are the debugging surface - `curl /timing` is
    worth having - and nothing else in the tree consumes them, so there is no
    compatibility argument either way; this is about what the browser does 86400
    times an hour.

    `cfg` is the part that is new rather than merely moved. The old page hardcoded
    its slider positions in the HTML, so `--gain 220` drew a slider sitting at 200
    over a pipeline running at 220, and the first touch of that slider silently
    moved the pipeline to wherever the handle happened to be.
    """
    checks = health(pipe, state, now)
    stats = pipe.frame_stats()
    lo, hi = state.get("range", (0, 0))
    rp = state.get("radar_proj")
    c = pipe.f.cfg
    stream_status = None
    if state.get("restarting_since") is not None:
        stream_status = {
            "kind": "restart",
            "title": "RECONNECTING SENSORS",
            "detail": "%s · %.0fs" % (
                state.get("last_restart", "stream fault"),
                now - state["restarting_since"]),
        }
    elif state.get("link_recovering_since") is not None:
        stream_status = {
            "kind": "drop",
            "title": "RECOVERING LINK",
            "detail": state.get("link_recovering_reason", "incomplete USB batch"),
        }
    elif state.get("last_frame_t") is not None and now - state["last_frame_t"] > 1.0:
        # The MJPEG connection deliberately keeps the last good frame visible.
        # Without this overlay that honest last frame looks like a live one.
        stream_status = {
            "kind": "stale",
            "title": "WAITING FOR A NEW FRAME",
            "detail": "last frame %.1fs ago" % (now - state["last_frame_t"]),
        }
    return {
        "worst": worst_level(checks),
        "checks": checks,
        "stream_status": stream_status,
        "frames": {"received": state.get("frames", 0),
                   "age_s": round(now - state["last_frame_t"], 2)
                   if state.get("last_frame_t") else None,
                   "batch_drops": state.get("batch_drops", 0)},
        "ai_diagnostics": state.get("student_diagnostics"),
        "server_time": now,
        "timing": timing(state, now),
        "stats": dict(stats, valid=True) if stats else {"valid": False},
        # Only with --raw16, hence None rather than a default: the page hides
        # the row instead of showing an invented one. The two decimal places it
        # carries are real, which is the entire distinction from "stats" above -
        # that is this same scene after the sensor window has quantised it.
        "raw_stats": state.get("raw_stats"),
        "detections": {
            "warped": pipe.warped,
            "age_s": round(now - state["detect_t"], 2) if state.get("detect_t") else None,
            "ms": state.get("detect_ms"),
            "list": state.get("detections") or [],
            "rejected": state.get("detections_rejected") or [],
        },
        # Quantities that drift rather than jump, and which the page draws as a
        # 60s trace: a heap reading is not interesting, a heap reading that is
        # 0.6MB lower than a minute ago is the whole early-warning system.
        # The locks. Separate from `detections`: a detection is what a sensor
        # said about one frame, a track is what the viewer believes about a
        # person across frames, and conflating them is how a coasted box ends
        # up quoted as a measurement.
        "tracks": state.get("tracks") or [],
        "tracks_static": state.get("tracks_static") or [],
        # What the rig did between frames, and whether it could be read at all.
        # The lock silently assumes a still rig whenever this is refused, and
        # an assumption nobody can see is one nobody thinks to doubt.
        "ego": state.get("ego"),
        # Whether the visible camera had light. Beside the tracks rather than
        # buried in health, because it is the number that says how to read
        # every other channel on this frame.
        "light": state.get("light"),
        "heap_free": state.get("heap_free"),
        "restarts": state.get("restarts", 0),
        "batch_drops": state.get("batch_drops", 0),
        "coverage": round(float(pipe.cover_grid().mean()), 4) if pipe.f.have_frame else None,
        "recording": os.path.basename(state["recording"]) if state.get("recording") else None,
        # Which station the frames going to disk right now are labelled with.
        # The operator is at the target, metres from the host, holding a phone;
        # this is the only way to see that the stamp took before walking back.
        # `null` while recording means frames are being written with no pose_id,
        # which is the state a calibration session must never sit in unnoticed.
        "pose": None if state.get("video") is None else {
            "id": state["video"].pose_id,
            "stamped": len(state["video"].meta_poses()),
            "frames": state["video"].frames,
        },
        # Cached inside Soc for 0.4s, so health() above and this share one sample
        # rather than each taking a delta over a near-zero interval.
        "soc": state["soc"].read() if state.get("soc") else None,
        # Yael's map layer (perception.map_api contract). None without --map;
        # the page hides the card. Small and already JSON-safe, so sent whole.
        "map": state.get("map"),
        "map_fix": state.get("map_fix"),
        "map_prior": state.get("map_prior"),
        "nav": nav_horizon(state) if state.get("map_prior") is not None else None,
        "cfg": {
            "visible_detail": pipe.visible_detail, "thermal_detail": pipe.thermal_detail,
            "gain": c.detail_gain, "eps": c.gf_eps, "radius": c.gf_radius,
            "agc": c.agc_permille, "palette": pipe.palette_name,
            "channel": pipe.channel, "channels": live_channels.public_specs(),
            "view": pipe.view,
            "mix": pipe.mix, "outline": pipe.outline, "boxes": pipe.show_detections,
            "lock": pipe.ai_lock,
            "emissivity": round(pipe.eps, 3), "reflected": pipe.refl,
            "warped": pipe.warped, "range": [lo, hi],
            # None, not a zeroed dict: "no radar attached" and "radar attached
            # and pointing straight ahead" are different states, and the page
            # hides the whole card on the first rather than offering knobs that
            # move nothing.
            "radar": None if rp is None else {
                "on": pipe.show_radar, "whisker": pipe.radar_whisker,
                "yaw": round(rp.yaw, 2), "pitch": round(rp.pitch, 2),
                "roll": round(rp.roll, 2),
                "tx": round(rp.t[0] * 1000), "ty": round(rp.t[1] * 1000),
                "tz": round(rp.t[2] * 1000)},
            # None when --students was not given or its engines did not come
            # up: same argument as the radar block above, and the page hides
            # the card rather than offering switches that move nothing.
            # `ready` is separate from `on` because a channel can be switched on
            # and still be unable to draw - a thermal box with no warp LUT has
            # nowhere to go, and fusion needs both channels to exist at all.
            "ai": None if state.get("students") is None else {
                # `found` is what the engine produced this tick and `n` is what
                # survived the confidence floor and the dedup. Both, always: a
                # card that only showed `n` cannot tell a quiet scene from a
                # slider parked too high.
                "floor": round(pipe.ai_conf_floor, 2),
                "evidence": pipe.ai_evidence,
                "silhouette": pipe.ai_silhouette,
                "thermal": {"on": pipe.ai_thermal,
                            "n": len(state.get("student_thermal") or []),
                            "found": (state.get("student_seen") or {}).get("thermal", 0),
                            "conf": round(pipe.ai_conf_thermal, 2),
                            "ready": state["students"].get("th2vis") is not None},
                "radar": {"on": pipe.ai_radar,
                          "n": len(state.get("student_radar") or []),
                          "found": (state.get("student_seen") or {}).get("radar", 0),
                          "conf": round(pipe.ai_conf_radar, 2),
                          "ready": state["students"].get("radar") is not None},
                "fusion": {"on": pipe.ai_fusion,
                           "n": len(state.get("student_fused") or []),
                           "rejected": len(state.get("student_fused_rejected") or []),
                           "ready": (state["students"].get("th2vis") is not None
                                     and state["students"].get("radar") is not None)},
                "age_s": (round(now - state["student_t"], 2)
                          if state.get("student_t") else None),
                "error": state.get("student_error"),
            },
        },
    }


PAGE = read_bytes("live.html")


def start_map_init(state, latlon, geoid_range=None, mapinit_dir=None):
    """Run Yael's map initialisation off the main thread, into state["map"].

    The result is Moshe's JSON contract (perception.map_api, schema 1.0), so
    what /map serves is exactly what the CLI prints.  Three shapes land in
    state["map"]:  {"pending": True} while it runs; the contract dict once it
    returns (ok may be false - that is a MAP finding, e.g. no EGM grid); and
    {"ok": False, "error": ...} when the boundary itself failed (package or a
    dependency missing) - a DEPLOYMENT finding, named so in the text.
    """
    try:
        lat, lon = (float(v) for v in latlon.split(","))
    except ValueError:
        raise SystemExit("--map wants LAT,LON in decimal degrees, got %r" % latlon)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from perception import map_api
    request = map_api.MapInitRequest(
        latitude=lat, longitude=lon,
        expected_geoid_range_m=tuple(geoid_range) if geoid_range else None)
    state["map"] = {"pending": True, "location": {"latitude_deg": lat,
                                                   "longitude_deg": lon}}

    def run():
        try:
            api = map_api.MapInitializationAPI(mapinit_dir=mapinit_dir)
            # fail_fast=False: the health line should name every failed stage,
            # not only the first, because the fixes differ (grid vs priors).
            result = api.initialize(request, fail_fast=False)
        except map_api.MapAPIError as e:
            result = {"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                      "deployment": True, "location": state["map"]["location"]}
        except Exception as e:  # never take the viewer down for the map
            result = {"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                      "location": state["map"]["location"]}
        state["map"] = result
        if result.get("ok"):
            alt = result.get("ego_altitude_prior") or {}
            print("map: initialised at %.5f,%.5f - geoid %.2f m, ground %.1f m "
                  "orthometric (sigma %.1f m), %s"
                  % (lat, lon, (result.get("geoid") or {}).get("undulation_m", float("nan")),
                     alt.get("orthometric_m", float("nan")), alt.get("sigma_m", float("nan")),
                     os.path.basename((result.get("priors") or {}).get("overture_path", "?"))),
                  file=sys.stderr)
        else:
            print("map: FAILED - %s" % (result.get("error") or result.get("summary", "")),
                  file=sys.stderr)

    threading.Thread(target=run, name="map-init", daemon=True).start()


def static_wall_returns(points, max_speed=0.25, min_range=0.5, max_range=40.0):
    """live.py's radar points (x fwd, y left, z, v, snr) -> (range, azimuth) walls.

    Azimuth positive to the RIGHT, as the rig reports it to the map layer:
    y is left in the radar frame, so azimuth = atan2(-y, x).
    """
    out = []
    for p in points or []:
        x, y = float(p["x"]), float(p["y"])
        rng = math.hypot(x, y)
        if not (min_range <= rng <= max_range):
            continue
        if abs(float(p.get("v") or 0.0)) > max_speed:
            continue
        out.append((rng, math.degrees(math.atan2(-y, x))))
    return out


def run_map_fix(state):
    """Solve position+heading from the current static returns, into state["map_fix"].

    Synchronous and a few seconds long (a 3-sigma grid at 5 m / 5 deg): the
    HTTP handler runs it on its own thread, so the stream never waits. The
    result is the map_api contract's "pose" block, or an error the operator
    can act on. It never touches state["map"], the startup layer.
    """
    prior = state.get("map_prior")
    if prior is None:
        return {"ok": False, "error": "no map layer; start live.py with --map LAT,LON"}
    if prior.get("heading_deg") is None:
        return {"ok": False, "error": "no heading prior; give --map-heading DEG or /set?heading=DEG"}
    pts = state.get("radar_points")
    age = time.time() - state.get("radar_points_t", 0) if pts is not None else None
    if pts is None or age > 2.0:
        return {"ok": False, "error": "no fresh radar frame (age %s s)" % (
            None if age is None else round(age, 1))}
    walls = static_wall_returns(pts)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from perception import map_api
    t0 = time.time()
    try:
        req = map_api.MapInitRequest(
            latitude=prior["latitude_deg"], longitude=prior["longitude_deg"],
            expected_geoid_range_m=state.get("map_geoid_range"),
            heading_prior_deg=prior["heading_deg"],
            sigma_position_m=prior["sigma_position_m"],
            sigma_heading_deg=prior["sigma_heading_deg"],
            wall_returns=tuple(walls) or None)
        if not req.has_pose_observations:
            return {"ok": False, "error": "no static returns in range 0.5-40 m (%d points)" % len(pts),
                    "walls": walls}
        res = map_api.MapInitializationAPI(mapinit_dir=state.get("mapinit_dir")).initialize(
            req, fail_fast=False)
    except map_api.MapAPIError as e:
        return {"ok": False, "error": "deployment: %s: %s" % (type(e).__name__, e)}
    except Exception as e:                        # noqa: BLE001
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    pose = res.get("pose")
    stage = next((s for s in res.get("stages", []) if s["name"] == "pose_init"), None)
    out = {"ok": bool(pose and pose.get("accepted")), "pose": pose,
           "prior": prior, "walls": walls, "n_walls": len(walls),
           "checks": (stage or {}).get("checks", []),
           "stage_status": (stage or {}).get("status"),
           "error": ((stage or {}).get("error") or {}).get("message"),
           "solve_s": round(time.time() - t0, 2), "t": time.time()}
    return out


def topdown_view(state, radius_m=60.0):
    """Footprint edges and radar returns in metres east/north of the prior.

    What the page draws so an operator can SEE whether the walls the radar
    reports sit on the map's walls - the check that no number replaces.
    Returns None without a map layer or before its priors loaded.
    """
    prior = state.get("map_prior")
    m = state.get("map") or {}
    pr = m.get("priors") or {}
    if prior is None or not pr.get("overture_path"):
        return None
    cache = state.get("_topdown_cache")
    if cache is None or cache["path"] != pr["overture_path"]:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from perception import map_api
        map_api.locate_mapinit(state.get("mapinit_dir"))
        from mapinit import BuildingLayer
        layer = BuildingLayer.from_geojson(pr["overture_path"])
        # The deployed mapinit checkout exports BuildingLayer but has no
        # mapinit.nav.walls module.  Top-down rendering only needs the public
        # footprint rings, so convert them locally instead of depending on a
        # private/nonexistent navigation helper. x is east, y is north.
        lat0, lon0 = prior["latitude_deg"], prior["longitude_deg"]
        lat_scale = 111_320.0
        lon_scale = lat_scale * math.cos(math.radians(lat0))
        segs = []
        for building in layer.near(lat0, lon0, radius_m * 2.0):
            ring = list(building.ring)
            for (lon_a, lat_a), (lon_b, lat_b) in zip(ring, ring[1:]):
                edge = [(lon_a - lon0) * lon_scale, (lat_a - lat0) * lat_scale,
                        (lon_b - lon0) * lon_scale, (lat_b - lat0) * lat_scale]
                if min(math.hypot(edge[0], edge[1]),
                       math.hypot(edge[2], edge[3])) <= radius_m * 1.5:
                    segs.append([round(v, 2) for v in edge])
        cache = {"path": pr["overture_path"], "edges": segs,
                 "scales": (lon_scale, lat_scale)}
        state["_topdown_cache"] = cache
    heading = prior.get("heading_deg")
    walls = static_wall_returns(state.get("radar_points"))
    fix = (state.get("map_fix") or {}).get("pose") or {}

    def place(h, dx, dy):
        if h is None:
            return []
        return [[round(dx + r * math.sin(math.radians(h + a)), 2),
                 round(dy + r * math.cos(math.radians(h + a)), 2)] for r, a in walls]

    lon_scale, lat_scale = cache["scales"]
    fx = fy = None
    if fix.get("latitude_deg") is not None:
        fx = (fix["longitude_deg"] - prior["longitude_deg"]) * lon_scale
        fy = (fix["latitude_deg"] - prior["latitude_deg"]) * lat_scale
    return {"radius_m": radius_m, "edges": cache["edges"],
            "prior": {"x": 0.0, "y": 0.0, "heading_deg": heading,
                      "sigma_m": prior["sigma_position_m"]},
            "returns_at_prior": place(heading, 0.0, 0.0),
            "fix": None if fx is None else {
                "x": round(fx, 2), "y": round(fy, 2), "heading_deg": fix.get("heading_deg"),
                "accepted": fix.get("accepted"), "ambiguous": fix.get("ambiguous"),
                "sigma_m": fix.get("sigma_horizontal_m")},
            "returns_at_fix": [] if fx is None else place(fix.get("heading_deg"), fx, fy),
            "n_walls": len(walls)}


def nav_horizon(state):
    """How long a fix survives, from the map package's DeadReckoner.

    Speed aiding comes from the radar: with the rig still, the static
    returns say speed 0 and the regime is speed-aided at zero speed, which
    is the honest reading for a parked rig. The IMU constants are the
    package's RIG_ERROR_MODEL (some measured, some assumed - the budget says
    which). Cheap: pure arithmetic, no I/O.
    """
    if state.get("map_prior") is None:
        return None
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from perception import map_api
        map_api.locate_mapinit(state.get("mapinit_dir"))
        from mapinit import DeadReckoner, SpeedAiding
    except Exception as e:                        # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, e)}
    pts = state.get("radar_points") or []
    statics = [p for p in pts if abs(float(p.get("v") or 0.0)) <= 0.25 and math.hypot(p["x"], p["y"]) >= 0.5]
    # Ego speed from the static returns' radial velocities (a still rig: 0).
    speed = 0.0
    if statics:
        speed = float(sum(abs(float(p.get("v") or 0.0)) for p in statics) / len(statics))
    fix = (state.get("map_fix") or {}).get("pose") or {}
    heading_sigma = fix.get("sigma_heading_deg") if fix.get("accepted") else         (state.get("map_prior") or {}).get("sigma_heading_deg") or 5.0
    aided = DeadReckoner(aiding=SpeedAiding(speed_mps=speed, sigma_speed_mps=0.05),
                         initial_heading_sigma_deg=float(heading_sigma))
    inertial = DeadReckoner(initial_heading_sigma_deg=float(heading_sigma))
    b = aided.budget_at(60.0)
    dom = b.dominant
    dom = dom() if callable(dom) else dom
    return {"regime": aided.regime, "speed_mps": round(speed, 2),
            "heading_sigma_deg": round(float(heading_sigma), 2),
            "horizon_1m_s": round(aided.horizon_for(1.0), 1),
            "horizon_1m_inertial_s": round(inertial.horizon_for(1.0), 1),
            "error_60s_m": round(b.total_m, 1),
            "dominant": None if dom is None else str(dom),
            "rests_on_assumption": bool(b.rests_on_assumption() if callable(b.rests_on_assumption)
                                        else b.rests_on_assumption)}


def make_handler(state, pipe):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(PAGE)
            elif u.path == "/set":
                q = {k: v[0] for k, v in parse_qs(u.query).items()}

                if "palette" in q:
                    pipe.set_palette(q.pop("palette"))
                if q.get("channel") in CHANNELS:
                    pipe.channel = q.pop("channel")
                if q.get("view") in VIEWS:
                    pipe.view = q.pop("view")
                if "mix" in q:
                    pipe.mix = max(0, min(100, int(q.pop("mix"))))
                if "outline" in q:
                    pipe.outline = q.pop("outline") not in ("0", "false", "")
                if "boxes" in q:
                    pipe.show_detections = q.pop("boxes") not in ("0", "false", "")
                if "radar" in q:
                    pipe.show_radar = q.pop("radar") not in ("0", "false", "")
                if "whisker" in q:
                    pipe.radar_whisker = q.pop("whisker") not in ("0", "false", "")
                # The AI channels. Accepted even when --students was not given:
                # the flags are display state like every other switch here, and
                # a 404 on a live control is harder to read than a switch that
                # holds a value nothing is currently drawing.
                if "lock" in q:
                    pipe.ai_lock = q.pop("lock") not in ("0", "false", "")
                if "silhouette" in q:
                    pipe.ai_silhouette = q.pop("silhouette") not in (
                        "0", "false", "")
                if "evidence" in q:
                    pipe.ai_evidence = q.pop("evidence") not in (
                        "0", "false", "")
                for k in ("ai_thermal", "ai_radar", "ai_fusion"):
                    if k in q:
                        setattr(pipe, k, q.pop(k) not in ("0", "false", ""))
                # Clamped at the bottom by the floor the engines were built
                # with: a slider below it would look like it was letting more
                # through while changing nothing at all.
                for k, attr in (("conf_thermal", "ai_conf_thermal"),
                                ("conf_radar", "ai_conf_radar")):
                    if k in q:
                        setattr(pipe, attr,
                                min(0.99, max(pipe.ai_conf_floor,
                                              float(q.pop(k)))))
                rp = state.get("radar_proj")
                if rp is not None:
                    for k in ("yaw", "pitch", "roll"):
                        if k in q:
                            setattr(rp, k, float(q.pop(k)))
                    for i, k in enumerate(("tx", "ty", "tz")):
                        if k in q:      # millimetres in the URL, metres inside
                            rp.t[i] = float(q.pop(k)) / 1000.0
                if "heading" in q and state.get("map_prior") is not None:
                    state["map_prior"]["heading_deg"] = float(q.pop("heading")) % 360.0
                if "emissivity" in q or "reflected" in q:
                    eps = float(q.pop("emissivity", pipe.eps))
                    refl = float(q.pop("reflected", pipe.refl))
                    pipe.set_emissivity(min(1.0, max(0.05, eps)), refl)

                for key in ("visible_detail", "thermal_detail"):
                    if key in q:
                        setattr(pipe, key, min(100, max(0, int(float(q.pop(key))))))
                pipe.tune(**{("detail_gain" if k == "gain" else
                              "gf_eps" if k == "eps" else
                              "gf_radius" if k == "radius" else
                              "agc_permille" if k == "agc" else k): v for k, v in q.items()})
                self.send_response(204)
                self.end_headers()
            elif u.path == "/pose":
                # Name the station the frames from here on belong to. The
                # recorder has carried pose_id since it was written, but nothing
                # could ever set it, so every session so far went to disk
                # unlabelled and "which rows are pose N04" was reconstructed
                # afterwards from timestamps and memory. V3 plan section 5d
                # requires the id on the row itself.
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                video = state.get("video")
                if video is None:
                    # Not a 404: the request is well-formed and the operator
                    # believes a session is being labelled. Failing loudly here
                    # is the difference between noticing now and discovering at
                    # solve time that 36 stations share one nameless heap.
                    self.send_response(409)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": "not recording; start live.py with --record",
                    }).encode())
                elif "id" in q or "clear" in q:
                    pid = None if "clear" in q else q["id"]
                    video.set_pose(pid)
                    body = json.dumps({"pose": video.pose_id,
                                       "first_frame": video.frames,
                                       "stamped": len(video.meta_poses())})
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body.encode())
                else:
                    body = json.dumps({"pose": video.pose_id,
                                       "frames": video.frames,
                                       "poses": video.meta_poses()})
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body.encode())
            elif u.path == "/ui":
                # The page's one poll. Everything else here is for curl.
                body = json.dumps(ui_payload(pipe, state, time.time()))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            elif u.path == "/mapfix":
                # Solve now, on this handler thread, and remember the answer.
                res = run_map_fix(state)
                state["map_fix"] = res
                self.send_response(200 if res.get("ok") else 409)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res, allow_nan=False, default=str).encode())
            elif u.path == "/topdown":
                try:
                    td = topdown_view(state)
                except Exception as e:            # noqa: BLE001
                    td = {"error": "%s: %s" % (type(e).__name__, e)}
                self.send_response(200 if td else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(json.dumps(
                        td or {"error": "no map layer or priors not loaded"}).encode())
                except (BrokenPipeError, ConnectionResetError):
                    # The first top-down build may outlive a browser navigation
                    # or readiness probe. The computed cache is still valid;
                    # a client going away is not a live-system exception.
                    pass
            elif u.path == "/map":
                # Yael's map layer as the schema-1.0 contract, or the reason
                # there is none. 404 only when live.py was started without
                # --map: then "no map" is configuration, not a failure.
                m = state.get("map")
                if m is None:
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": "no map layer; start live.py with --map LAT,LON",
                    }).encode())
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(m).encode())
            elif u.path == "/health":
                checks = health(pipe, state, time.time())
                body = json.dumps({"worst": worst_level(checks), "checks": checks})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            elif u.path in ("/temp", "/stats"):
                q = parse_qs(u.query)
                if u.path == "/temp":
                    d = pipe.temp_at(int(q.get("x", ["0"])[0]), int(q.get("y", ["0"])[0]))
                else:
                    d = pipe.frame_stats()
                body = json.dumps({"valid": False} if d is None else dict(d, valid=True))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            elif u.path == "/timing":
                body = json.dumps(timing(state, time.time()))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            elif u.path == "/detections":
                # `warped` rides along with every response. A consumer that logs
                # these numbers has no other way to know whether the box and the
                # temperature refer to the same place in the world.
                body = json.dumps({
                    "warped": pipe.warped,
                    "age_s": round(time.time() - state.get("detect_t", 0), 2)
                             if state.get("detect_t") else None,
                    "ms": state.get("detect_ms"),
                    "detections": state.get("detections") or []})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            elif u.path in ("/ai", "/students"):
                # The curl-side view of the three channels, with the same
                # numbers the page draws. `vis` on a thermal box is what it maps
                # to on the visible plane - null means it could not be placed,
                # and a consumer that logs a thermal box without it is logging a
                # coordinate in the wrong plane.
                st = state.get("students")
                body = json.dumps({
                    "available": st is not None,
                    "age_s": (round(time.time() - state["student_t"], 2)
                              if state.get("student_t") else None),
                    "gate_du_px": Renderer.FUSION_DU_PX,
                    "dedup_iou": Renderer.DEDUP_IOU,
                    "conf": {"thermal": round(pipe.ai_conf_thermal, 2),
                             "radar": round(pipe.ai_conf_radar, 2),
                             "engine_floor": round(pipe.ai_conf_floor, 2)},
                    "found": state.get("student_seen") or {},
                    "lock": {"on": pipe.ai_lock,
                             "coast_s": tracking.MAX_COAST_S,
                             "min_hits": tracking.MIN_HITS,
                             "static_px": tracking.STATIC_PX,
                             "tracks": state.get("tracks") or [],
                             "suppressed": state.get("tracks_static") or []},
                    "error": state.get("student_error"),
                    "channels": {
                        "thermal": {"on": pipe.ai_thermal,
                                    "list": state.get("student_thermal") or []},
                        "radar": {"on": pipe.ai_radar,
                                  "list": state.get("student_radar") or []},
                        "fusion": {"on": pipe.ai_fusion,
                                   "list": state.get("student_fused") or [],
                                   "rejected": state.get("student_fused_rejected") or []},
                    }})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            elif u.path == "/stat":
                lo, hi = state.get("range", (0, 0))
                msg = "%.1f fps | range %d..%dC (%.3f C/code) | %s | eps %.2f, refl %.0fC%s" % (
                    state.get("fps", 0.0), lo, hi, (hi - lo) / 255.0,
                    "calibrated warp" if pipe.warped else "PLACEHOLDER warp - not registered",
                    pipe.eps, pipe.refl,
                    "  | ERROR: " + state["error"] if state.get("error") else "")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(msg.encode())
            elif u.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    last = None
                    while True:
                        f = state.get("frame")
                        if f is not None and f is not last:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             b"Content-Length: %d\r\n\r\n" % len(f))
                            self.wfile.write(f)
                            self.wfile.write(b"\r\n")
                            last = f
                        else:
                            time.sleep(0.02)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(404)
                self.end_headers()
    return H


def session_meta(args):
    """Everything about a recording that cannot be recovered from the data.

    The radar cannot be asked which config it is running and the config is
    sent by a separate tool, so the one claim that matters - which chirp
    profile and which RX phase table produced these points - is copied from
    the stamp send_radar_cfg.py leaves behind. It is carried with its own
    timestamp beside the recording's, because the check a reader has to make
    is not "is there a stamp" but "was it stamped BEFORE this session, with no
    power cycle in between". Changing the phase table moves boresight and
    voids any extrinsic solved on older data (frame_conventions.txt), and that
    is invisible in the points themselves.
    """
    stamp, stamp_note = None, 'no stamp: send_radar_cfg.py has not run since ' \
                              'this checkout, so the radar config is UNKNOWN'
    try:
        import send_radar_cfg
        if os.path.exists(send_radar_cfg.STAMP_PATH):
            with open(send_radar_cfg.STAMP_PATH) as f:
                stamp = json.load(f)
            age = time.monotonic() - stamp.get('sent_at_mono', 0)
            stamp_note = ('sent %.0f s before this session started; valid only '
                          'if the radar was not power-cycled since' % age)
            if age < 0:
                stamp_note = ('STAMP IS FROM A PREVIOUS BOOT (negative age): '
                              'the monotonic clock restarted, so this cannot '
                              'be the config now on the sensor')
    except Exception as e:                     # provenance never breaks a run
        stamp_note = 'stamp unreadable: %s' % e

    # The thermal window is what turns the recorded uint8 back into degrees:
    # c_per_lsb = (tmax - tmin) / 255. Every session before 2026-08-19 omitted
    # it, which is why perception/out/gexport v1 carries c_per_lsb: null and
    # its thermal plane is intensity, not temperature. The default matches the
    # bring-up default in Streamer._setup (fixed_range or (-10, 140)); if that
    # pair ever moves, this one must move with it or the meta lies.
    if args.range:
        tmin_c, tmax_c = (int(v) for v in args.range.split(':'))
    else:
        tmin_c, tmax_c = -10, 140

    # --raw16 changes the unit and not merely the width, so everything that
    # describes how a count becomes a temperature has to move with it. Leaving
    # the uint8 block in place would declare 0.588C steps over a plane whose
    # steps are 0.01C, and a reader has no way to notice: both are plausible
    # numbers. This is the same failure that shipped gexport v1 with c_per_lsb
    # null and turned its thermal plane into intensity.
    if args.raw16:
        thermal_scale = {
            'thermal_dtype': 'uint16_le',
            # Absolute, straight off the sensor's TLinear output. Nothing to
            # undo: C = count / 100 - 273.15, whatever tmin/tmax happen to be.
            'thermal_encoding': 'radiometric_centikelvin',
            'c_per_lsb': 0.01,
            'thermal_counts_max': 65535,
        }
    else:
        thermal_scale = {
            'thermal_dtype': 'uint8',
            'thermal_encoding': 'linear_set_range',
            'c_per_lsb': (tmax_c - tmin_c) / 255.0,
            'thermal_counts_max': 255,
        }

    def _sha(path):
        # Calibration files are small (a LUT is ~1 MB); hashing at session
        # start is the only moment the file on disk is KNOWN to be the file
        # the session ran with.
        if not path or not os.path.exists(path):
            return None
        import hashlib
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    detector_engine = args.detect_engine
    if (args.detect != 'off' and args.detect_backend != 'cpu' and
            detector_engine is None):
        try:
            import trt_detect
            detector_engine = trt_detect.model_paths(args.detect_model)[1]
        except Exception:
            detector_engine = None

    # Which student engines this session ran, hashed for the same reason the
    # detector engine is: an engine is rebuilt in place whenever TensorRT or the
    # GPU moves under it, and "the v2 students" names a file, not a model.
    student_engines = {}
    if args.students:
        try:
            import trt_students
            for name in ('thermal_student.engine', 'radar_student.engine'):
                path = os.path.join(trt_students.ENGINE_DIR, name)
                student_engines[name] = {'path': os.path.abspath(path),
                                         'sha256': _sha(path)}
        except Exception:                      # provenance never breaks a run
            student_engines = {}

    return {
        'tool': 'live.py',
        'argv': sys.argv[1:],
        'board_port': args.port,
        'radar_port': args.radar,
        'channel': args.channel,
        'view': args.view,
        'warp_lut': args.warp,
        'warp_lut_sha256': _sha(args.warp),
        'radar_calib': args.radar_calib,
        'radar_calib_sha256': _sha(args.radar_calib),
        'detector': args.detect,
        'detector_model': (args.detect_model if args.detect != 'off' else None),
        'detector_engine': detector_engine,
        'detector_engine_sha256': _sha(detector_engine),
        'students': bool(args.students),
        'student_conf': args.student_conf if args.students else None,
        # What the session STARTED with. The channels are switchable live, so a
        # reader wanting to know what was on screen at frame N has to read the
        # picture, not this - but the engines and the threshold cannot change
        # under a running session, and those are what a label depends on.
        'ai_channels_at_start': args.ai_channels if args.students else None,
        'student_engines': student_engines or None,
        'radar_cfg_stamp': stamp,
        'radar_cfg_stamp_note': stamp_note,
        # Still recorded under --raw16, and still true: the window goes on
        # deciding the board's own 8-bit frame, which live.py reproduces for
        # the pipeline. It just stops being what turns a count into a degree.
        'tmin': tmin_c,
        'tmax': tmax_c,
        **thermal_scale,
        # The bring-up has run the Lepton in HIGH gain since 2026-08-09
        # (capture._BRINGUP, SET_MODE(True, False)); recorded so a future
        # low-gain session cannot be silently mixed in as the same scale.
        'lepton_gain': 'high',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--port", default=PORT)
    ap.add_argument("--http", type=int, default=8088)
    ap.add_argument("--warp", help="calibrated warp LUT; without it the thermal layer "
                                   "is merely stretched and is NOT registered")
    ap.add_argument("--gain", type=int, default=200)
    ap.add_argument("--eps", type=int, default=200)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--palette", default="ironbow",
                    choices=["ironbow", "white", "black", "gray"],
                    help="white/black are the mono ramps; black also flips the detail sign")
    ap.add_argument("--emissivity", type=float, default=1.0, metavar="E",
                    help="surface emissivity 0.05..1.0. 1.0 is what the sensor assumes; "
                         "bright metal is ~0.1 and reads tens of degrees cold without this")
    ap.add_argument("--reflected", type=float, default=20.0, metavar="C",
                    help="temperature the surface reflects, usually ambient (default 20)")
    ap.add_argument("--agc", type=int, default=0, metavar="PERMILLE",
                    help="scene AGC, per-mille clipped each end (20 = 2%%). "
                         "Makes tone scene-relative rather than absolute")
    ap.add_argument("--thermal-noise-mc", type=int, default=148, metavar="MC",
                    help="thermal temporal-filter noise estimate in milli-Celsius; "
                         "0 disables it (default 148, measured on this Lepton)")
    ap.add_argument("--thermal-frames", type=int, default=8, metavar="N",
                    help="equivalent frames averaged by the motion-adaptive thermal "
                         "filter; 1 disables smoothing (default 8)")
    # The visible temporal filter. fusion.c leaves it off because it costs 768KB
    # (fusion.c:993-994) that only pays back in the dark - but
    # dark is the case this project exists for. In an unlit cabinet the detail
    # layer amplifies read noise exactly as hard as it amplifies edges, so a
    # noisy high-pass makes the fused image worse than none at all. Motion-adaptive
    # IIR: 8 frames buys ~3.9x for one buffer, where an 8-frame boxcar buys 2.8x
    # for eight - and a host-side burst average of 8 measured only 1.86x on this
    # board (2026-08-09), because every blend step re-quantises to 8 bits.
    ap.add_argument("--y-knee", type=int, default=0, metavar="CODES",
                    help="visible temporal filter knee in 8-bit luma codes; 0 = off. "
                         "Try 3-6 for a dark scene. Costs 768KB when enabled "
                         "(512KB of Q8 state + a 256KB output plane)")
    ap.add_argument("--y-frames", type=int, default=8, metavar="N",
                    help="equivalent frames averaged by the visible filter (default 8)")
    # Raised from 50 on 2026-08-10, measured on captures/handwave3. The visible
    # frame is the guided filter's *guide*, so the codec's artefacts are not
    # cosmetic here - they are fed into the one layer whose job is to be
    # amplified. Energy sitting exactly on the JPEG 8x8 grid, against the
    # off-grid energy of the same frame (scene detail has no reason to prefer a
    # period of 8, so anything above 1.0 is the codec):
    #
    #     q30  2.57x   q50  1.97x   q65  1.74x
    #     q80  1.52x   q90  1.32x   q95  1.16x
    #
    # At q30 the fused frame carries 116% of the reference's high-frequency
    # energy - more detail than the uncompressed original, which is the pipeline
    # sharpening blocking artefacts into scene texture. That is the failure mode
    # to avoid, and q50 was closer to it than it looked.
    #
    # The cost is link time, and this link has a hard edge: the board's
    # out.write() discards the tail of a frame after 500ms of no progress. Frame
    # rate at 8.772 fps including the uncompressed 18.75KB thermal plane:
    # q50 256KB/s, q80 325KB/s, q90 420KB/s. FS CDC measures out around
    # 700-900KB/s, so q80 sits near 40% duty and q90 near 55%. q80 buys most of
    # the improvement for the smaller share of the pipe.
    ap.add_argument("--quality", type=int, default=70,
                    help="board-side JPEG quality (default 70 leaves CDC headroom; "
                         "80 was measured dropping 512-byte USB tails)")
    # Costs 18.75KB/frame more on the wire - 164KB/s at 8.772fps, taking a q80
    # session from ~325 to ~490KB/s against a link measured at 700-900. Sending
    # it is a swap, not an addition: the 8-bit plane is rebuilt on this side
    # from the same words, so nothing downstream sees a difference. Needs the
    # raw-passthrough firmware; an older build fails the bring-up with a clear
    # message rather than quietly streaming 8-bit.
    ap.add_argument("--raw16", action="store_true",
                    help="stream the Lepton's 16-bit radiometric frame (0.01C "
                         "per code) instead of the board's 8-bit plane, and "
                         "record it as the session's thermal.bin")
    ap.add_argument("--range", metavar="TMIN:TMAX",
                    help="pin the sensor range instead of auto-ranging, e.g. "
                         "--range 10:45. Auto-range picks off ONE frame at "
                         "bring-up and the sensor drifts for minutes after; pin "
                         "it when the session must outlast that (calibration, "
                         "long recordings) or to measure the drift itself")
    ap.add_argument("--detect", default="off", metavar="WHAT",
                    help="object detection: 'off', 'all' for COCO-80, or a comma-separated "
                         "class list such as 'person,cat'. Attaches a temperature to "
                         "every box")
    ap.add_argument("--detect-backend", default="auto", choices=["auto", "gpu", "cpu"],
                    help="'gpu' runs the selected TensorRT model; 'cpu' runs "
                         "yolov4-tiny with cv2.dnn. 'auto' prefers GPU")
    ap.add_argument("--detect-model", default="yolov10n",
                    choices=sorted(MODEL_FILES),
                    help="TensorRT detector model (GPU only; default yolov10n)")
    ap.add_argument("--detect-engine", metavar="PATH",
                    help="override the registered TensorRT engine path")
    ap.add_argument("--detect-size", type=int, default=416, choices=[320, 416],
                    help="detector input size, CPU backend only. 416 is 68ms and 320 is "
                         "46ms on this host, against a 114ms frame period. The GPU "
                         "backend is fixed at 640 by its engine (default 416)")
    ap.add_argument("--detect-conf", type=float, default=0.35,
                    help="detection confidence threshold (default 0.35)")
    # Radar overlay. Off unless a port is given, because live.py must keep
    # working on a rig that has no radar attached.
    ap.add_argument("--radar", nargs="?", const=radar_overlay.DATA_PORT, default=None,
                    metavar="PORT",
                    help="overlay IWR1843 detections from this DATA port "
                         "(default %s when the flag is given bare)" % radar_overlay.DATA_PORT)
    ap.add_argument("--record", nargs="?", const="", default=None, metavar="DIR",
                    help="record the session: session.mp4 (clean of overlays), "
                         "frames.jsonl and thermal.bin - plus radar.bin and "
                         "radar.jsonl when --radar is on, which is what an offline "
                         "calibration is solved against. A bare --record picks "
                         "captures/live-<timestamp>")
    ap.add_argument("--radar-record", metavar="DIR",
                    help="deprecated alias for --record DIR")
    ap.add_argument("--radar-hfov", type=float, default=70.0, metavar="DEG",
                    help="assumed horizontal FOV used to guess the focal length when no "
                         "solved intrinsics exist (default 70). DESIGN.md:239 records this "
                         "as UNVERIFIED and the measured triple implies 63.8")
    ap.add_argument("--radar-ai", action="store_true",
                    help="draw the radar person/clutter classifier (perception/out/radar_ai/"
                         "cluster_model_v0.pkl) on the picture: green ring = PERSON")
    ap.add_argument("--radar-calib", metavar="JSON",
                    help="solved intrinsics/extrinsics to project with, instead of the guess")
    ap.add_argument("--channel", default="fusion", choices=list(CHANNELS),
                    help="operator display channel: thermal, rgb_radar (currently "
                         "visible luma), thermal_radar, fusion, or ai")
    ap.add_argument("--view", default="fused",
                    choices=list(VIEWS),
                    help="view to start in, and therefore what gets recorded. "
                         "For picking a pixel off the recording, see the note in "
                         "radar_correspond.py: the thermal layer is NOT registered "
                         "until a warp LUT exists, so a pixel taken from the thermal "
                         "content carries that unknown offset into the extrinsic")
    ap.add_argument("--students", action="store_true",
                    help="run the trained thermal+radar person students "
                         "(TensorRT engines from perception/out/gexport/v6/"
                         "models/) as extra detection channels: orange boxes "
                         "= thermal student, cyan = radar student, white = the "
                         "two of them agreeing. Pick which channels are on with "
                         "--ai-channels, or live from the page")
    ap.add_argument("--student-conf", type=float, default=0.5, metavar="C",
                    help="confidence threshold for both students")
    ap.add_argument("--students-dir", metavar="DIR",
                    help="load the student engines from this directory "
                         "instead of perception/out/gexport/v6/models. The "
                         "engines belong to the export that trained them; the "
                         "session meta records which ones actually loaded")
    ap.add_argument("--no-lock", action="store_true",
                    help="start with the person lock off, so every frame's "
                         "boxes stand on that frame alone. The lock is on by "
                         "default and switchable live (key l, /set?lock=0)")
    ap.add_argument("--ai-channels", default="fusion", metavar="LIST",
                    help="which AI channels start switched on: any of "
                         "thermal,radar,fusion (or 'none'). All three are "
                         "switchable live from the page and over "
                         "/set?ai_thermal=0&ai_radar=1&ai_fusion=1, so this only "
                         "picks what the first frame shows")
    ap.add_argument("--map", metavar="LAT,LON", default=None,
                    help="initialise Yael's map layer (mapinit) at this WGS84 "
                         "position: geoid undulation, ground/surface height, "
                         "building priors. Served on /map, graded in /health. "
                         "Runs in the background; the viewer never waits on it")
    ap.add_argument("--map-geoid", type=float, nargs=2, metavar=("LOW", "HIGH"),
                    default=None, help="assert the geoid undulation is in this "
                                       "range (m) - Jerusalem is 19..20.5")
    ap.add_argument("--map-heading", type=float, default=None, metavar="DEG",
                    help="compass heading prior for the map fix (0 = north); "
                         "changeable live with /set?heading=. Without it /mapfix "
                         "cannot run: heading has no other source on this rig")
    ap.add_argument("--map-sigma", type=float, nargs=2, default=(5.0, 5.0),
                    metavar=("M", "DEG"), help="how good the --map position and "
                    "--map-heading are believed to be (1 sigma). Be honest: the "
                    "fix searches 3 sigma and no further")
    ap.add_argument("--mapinit-dir", default=None, metavar="DIR",
                    help="directory holding Yael's mapinit/ package (default: "
                         "$VALISIGHT_MAPINIT, then a ValiSight_yael checkout "
                         "beside this repo)")
    ap.add_argument("--seconds", type=int, default=0, help="exit after N seconds (for tests)")
    args = ap.parse_args()

    if not os.path.exists(args.port):
        # The board does not always come back on the node it left. Anything that
        # re-enumerates it - a replug, or the USB reset that follows a wedge -
        # can hand it ttyACM1 while the stale ttyACM0 is still being released,
        # and then the default looks exactly like a board that is not there.
        # Measured 2026-08-09: a healthy board sat on ttyACM1 while this printed
        # "replug the board", which is advice that would not have helped.
        found = sorted(glob.glob("/dev/ttyACM*"))
        if args.port == PORT and found:
            print("%s is gone; using %s instead" % (args.port, found[0]), file=sys.stderr)
            args.port = found[0]
        else:
            raise SystemExit("%s is not present%s - replug the board"
                             % (args.port, "" if not found else
                                " (found %s, pass --port)" % ", ".join(found)))
    if not os.path.exists(LIB):
        raise SystemExit("%s missing - run 'make libfusion.so' in host/" % LIB)

    # Set before anything imports trt_students: the module reads it at import
    # time, and a later assignment would silently load the default engines.
    if args.students_dir:
        os.environ["STUDENT_ENGINES"] = os.path.abspath(args.students_dir)

    pipe = Pipeline(args)
    pipe.channel = args.channel
    pipe.view = args.view

    # Validated here rather than shrugged off later: a typo in --ai-channels
    # would otherwise silently switch a channel off for a whole session, and a
    # channel that is off is indistinguishable on screen from a channel that is
    # on and finding nobody.
    want_ai = {c.strip() for c in args.ai_channels.split(",") if c.strip()} - {"none"}
    unknown_ai = want_ai - {"thermal", "radar", "fusion"}
    if unknown_ai:
        raise SystemExit("--ai-channels: not a channel: %s (thermal, radar, "
                         "fusion, none)" % ", ".join(sorted(unknown_ai)))
    pipe.ai_thermal = "thermal" in want_ai
    pipe.ai_radar = "radar" in want_ai
    pipe.ai_fusion = "fusion" in want_ai
    # The engines are built with --student-conf and cannot go below it later,
    # so that is where both sliders start and where they stop going down.
    pipe.ai_lock = not args.no_lock
    pipe.ai_conf_floor = args.student_conf
    pipe.ai_conf_thermal = pipe.ai_conf_radar = args.student_conf
    # One sampler for the process; it primes its own counters, so the first
    # /ui already carries a CPU figure rather than a null.
    state = {"soc": hostsoc.Soc(), "detector_available": False}

    detector = None
    if args.detect != "off":
        classes = None if args.detect == "all" else [
            c.strip() for c in args.detect.split(",") if c.strip()]
        if classes:
            unknown = [c for c in classes if c not in detect.COCO]
            if unknown:
                raise SystemExit("not COCO classes: %s\navailable: %s"
                                 % (", ".join(unknown), " ".join(detect.COCO)))
        try:
            detector = detect.make_detector(
                args.detect_backend, size=args.detect_size,
                conf=args.detect_conf, classes=classes,
                model=args.detect_model, engine=args.detect_engine)
        except (FileNotFoundError, RuntimeError) as e:
            # The stream, the radar and the thermal measurement do not need the
            # detector; losing all of them to a missing weights file or a GPU
            # that would not come up turned every detector fault into a dead
            # system (2026-08-23). Degrade loudly instead.
            print("detector unavailable, RUNNING WITHOUT DETECTION: %s" % e,
                  file=sys.stderr)
            detector = None
            args.detect = "off"
        if detector:
            # Session provenance must describe the backend that actually won.
            # In auto mode this may be the CPU fallback, not the requested engine.
            args.detect_backend = detector.backend
            args.detect_model = detector.model_name
            args.detect_engine = getattr(detector, "engine_path", None)
    state["detector_available"] = detector is not None

    work = Latest()
    radar = radar_proj = video = None

    record_dir = args.record if args.record is not None else args.radar_record
    if args.radar_record and args.record is None:
        print("--radar-record is now --record (recording no longer needs the "
              "radar); kept as an alias", file=sys.stderr)
    if record_dir == "":                            # bare --record
        record_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "captures",
            time.strftime("live-%Y%m%d-%H%M%S"))
    if record_dir:
        video = recorder.SessionRecorder(record_dir, meta=session_meta(args))
        state["recording"] = record_dir
        # Also in state, beside radar_proj: the recorder is owned by the
        # Renderer thread, but /pose is served on the HTTP thread and has no
        # other way to reach it. set_pose only appends to a list and rewrites
        # meta.json, so the cross-thread call is a write the capture path never
        # races on - it reads pose_id, and a str assignment is atomic.
        state["video"] = video
        print("recording to %s/" % record_dir, file=sys.stderr)

    if args.radar:
        radar = radar_overlay.RadarReader(args.radar, record_dir=record_dir)
        radar.start()
        radar_proj = radar_overlay.Bootstrap(OUT_W, OUT_H, hfov_deg=args.radar_hfov,
                                             calib_path=args.radar_calib)
        state["radar_proj"] = radar_proj
        print("radar overlay: %s, intrinsics from %s, f=%.1f px"
              % (args.radar, radar_proj.source, radar_proj.f), file=sys.stderr)
        if not args.radar_calib:
            print("  the projection is a GUESS until radar_extrinsics is solved - "
                  "nudge it with /set?yaw=..&pitch=..&tz=..", file=sys.stderr)

    # The thermal->visible mapper, loaded once for everything that needs it:
    # the person outline on the detector's boxes, the student channels, and
    # the fusion ring. Independent of --students on purpose - the outline is
    # worth having on a run with no students at all - and never fatal: a bad
    # LUT costs the outline, not the viewer.
    th2vis = None
    if args.warp:
        try:
            import trt_students
            th2vis = trt_students.ThermalToVisible(args.warp)
        except Exception as e:
            print("person outline unavailable (%s: %s) - boxes only"
                  % (type(e).__name__, e), file=sys.stderr)

    students = None
    if args.students:
        # The students are an OPT-IN extra: any failure here (missing engine,
        # GPU not up, bad LUT) must leave the plain pipeline running.
        try:
            import trt_students
            th_model = trt_students.ThermalStudentTrt(conf=args.student_conf)
            rd_model = (trt_students.RadarStudentTrt(conf=args.student_conf)
                        if args.radar else None)
            if not args.range:
                print("students: --range is not pinned, so the thermal input "
                      "is scene-relative instead of the Celsius the model was "
                      "trained on - expect degraded detections. Use the "
                      "training range (e.g. --range 0:60).", file=sys.stderr)
            if th2vis is None:
                print("students: no --warp LUT, thermal-student boxes cannot "
                      "be drawn on the visible frame (still served on "
                      "/state)", file=sys.stderr)
            # Same range->degrees mapping session_meta() documents: the
            # recorded uint8 is (tmax - tmin) / 255 per count above tmin.
            if args.range:
                s_tmin, s_tmax = (int(v) for v in args.range.split(':'))
            else:
                s_tmin, s_tmax = -10, 140
            students = {"thermal": th_model, "radar": rd_model,
                        "th2vis": th2vis,
                        "c_per_lsb": (s_tmax - s_tmin) / 255.0,
                        "tmin": float(s_tmin)}
            print("students: thermal%s engine(s) up, conf>=%.2f, channels on: %s"
                  % ("+radar" if rd_model else "", args.student_conf,
                     "+".join(sorted(want_ai)) or "none"),
                  file=sys.stderr)
            if pipe.ai_fusion and (rd_model is None or th2vis is None):
                print("students: the fusion channel cannot pair without %s - it "
                      "will draw nothing" % ("a radar student engine"
                                             if rd_model is None else "a warp LUT"),
                      file=sys.stderr)
        except Exception as e:
            print("students: DISABLED (%s: %s)" % (type(e).__name__, e),
                  file=sys.stderr)

    # The HTTP threads reach the students only through state: /ui hides the
    # card when this is None, and health() asks it whether a channel that is
    # switched on can actually draw.
    state["students"] = students

    if args.map:
        start_map_init(state, args.map, args.map_geoid, args.mapinit_dir)
        lat, lon = (float(v) for v in args.map.split(","))
        state["map_prior"] = {"latitude_deg": lat, "longitude_deg": lon,
                              "heading_deg": args.map_heading,
                              "sigma_position_m": float(args.map_sigma[0]),
                              "sigma_heading_deg": float(args.map_sigma[1])}
        state["map_geoid_range"] = tuple(args.map_geoid) if args.map_geoid else None
        state["mapinit_dir"] = args.mapinit_dir

    render = Renderer(pipe, state, work, detector, radar=radar,
                      radar_proj=radar_proj, video=video, students=students,
                      th2vis=th2vis)
    render.radar_ai = bool(args.radar_ai)
    render.start()
    stream = Streamer(args.port, pipe, args.quality, state, work, raw16=args.raw16)
    if args.range:
        lo, hi = (int(v) for v in args.range.split(":"))
        stream.fixed_range = (lo, hi)
        # Told, not inferred: the board reports its own range on #READY, but the
        # host must know NOW that the tone is absolute rather than scene-relative,
        # because temp_at() converts codes with it.
        pipe.set_range(lo, hi)
        print("range pinned to %d..%d C (auto-range off)" % (lo, hi), file=sys.stderr)
    stream.start()

    srv = ThreadingHTTPServer(("0.0.0.0", args.http), make_handler(state, pipe))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("live fusion on http://localhost:%d  (ctrl-c to stop)" % args.http, file=sys.stderr)
    if not args.warp:
        print("NOTE: no --warp, thermal layer is stretched not registered", file=sys.stderr)
    if detector:
        gpu = getattr(detector, "backend", "cpu") == "gpu"
        print("detecting %s at %d px (%s)"
              % (args.detect, detector.net_w if gpu else args.detect_size,
                 detector.model_name + "/TensorRT" if gpu else
                 detector.model_name + "/CPU"), file=sys.stderr)

    # How long the stream may stay down before this gives up on it. The
    # supervisor in Streamer.run() exists to survive a fault - a wedged Lepton, a
    # resync storm, a board that fell off USB - and a restart costs ~3s of
    # draining plus ~10s of bring-up.
    #
    # This loop used to `return 1` the instant state["error"] appeared, which
    # meant the supervisor was never once allowed to finish: the process was gone
    # a fraction of a second into a recovery designed to take thirteen. Every
    # fault therefore read as fatal, including the ones the code already knew how
    # to repair. The error is only cleared on #READY, so polling for it is
    # polling for "a restart is in progress".
    DEAD_S = 90.0

    t0 = time.time()
    down_since, last_err = None, None
    warned_no_radar = False
    # `run_live.sh` starts this process under nohup/setsid.  A background shell
    # may leave SIGINT ignored in that child, and Python deliberately preserves
    # an inherited SIG_IGN disposition.  The launcher used to wait, conclude
    # that live.py ignored it, and SIGKILL the process; SessionRecorder.close()
    # then never ran and the MP4 had no moov/index atom.  Explicit handlers make
    # both the launcher's SIGINT and an ordinary SIGTERM request the same clean
    # exit through the finally block below.
    stop_requested = threading.Event()

    def request_stop(_signum, _frame):
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stop_requested.wait(0.2):
            # A calibration recording with zero radar frames is a wasted
            # session that looks fine on screen (the camera side records
            # happily). The usual cause on this bench: a power cycle wiped the
            # IWR1843's config and nobody re-sent it. Say so on the console,
            # where the person who just typed the record command is looking.
            # Judged on the READER's own counter, not state["radar_frames"]:
            # that one is written by the renderer, which sits idle through the
            # ~15 s camera bring-up, and the first version of this check fired
            # a false alarm through exactly that window.
            if (radar is not None and record_dir and not warned_no_radar
                    and time.time() - t0 > 15 and radar.frames == 0):
                print("WARNING: recording with --radar but 0 radar frames "
                      "after 10s - the radar is probably unconfigured (a "
                      "power cycle wipes it). Run ./send_radar_cfg.py, then "
                      "restart this recording.", file=sys.stderr)
                warned_no_radar = True
            err = state.get("error")
            if err and err != last_err:
                print("stream fault (recovering): %s" % err, file=sys.stderr)
                last_err = err
            if err or state.get("restarting_since"):
                down_since = down_since or time.time()
                if time.time() - down_since > DEAD_S:
                    print("stream down for %.0fs across %d restart(s) - giving up: %s"
                          % (time.time() - down_since, state.get("restarts", 0), err),
                          file=sys.stderr)
                    return 1
            elif down_since is not None:
                print("stream recovered after %.0fs (%d restart(s))"
                      % (time.time() - down_since, state.get("restarts", 0)),
                      file=sys.stderr)
                down_since, last_err = None, None
            if args.seconds and time.time() - t0 > args.seconds:
                tm = timing(state, time.time())
                print("fps %.1f, frames %d, rendered %d, dropped %d, bad jpeg %d, "
                      "batches %d, batch drops %d, resyncs %d, restarts %d, "
                      "frames flowing: %s" % (
                          state.get("fps", 0.0), state.get("frames", 0),
                          state.get("rendered", 0), state.get("dropped", 0),
                          state.get("bad_jpeg", 0), state.get("batches", 0),
                          state.get("batch_drops", 0),
                          state.get("resyncs", 0), state.get("restarts", 0),
                          state.get("frame") is not None), file=sys.stderr)
                # Board clock separately: the fps above is arrival times and says
                # whether the link kept up, not what the sensors did.
                print("board clock: thermal %s, visible +%s ms, %d FFC(s)" % (
                    "%d ms (%.2f fps)" % (tm["thermal_ms"]["median"], tm["thermal_fps"])
                    if tm["thermal_ms"] else "not reported",
                    tm["skew_ms"]["median"] if tm["skew_ms"] else "?",
                    tm["ffcs"]), file=sys.stderr)
                return 0
    except KeyboardInterrupt:
        return 0
    finally:
        stream.stop.set()
        render.stop.set()
        if radar is not None:
            radar.stop.set()
        # The render thread owns the recorder, and the mp4 is only finalized by
        # its close() - so wait for the thread rather than letting the daemon
        # flag kill it mid-write.
        render.join(timeout=3.0)
        # Streamer.release() drains and interrupts the raw REPL in its own
        # finally block.  Give that cleanup a bounded chance to finish so a
        # replacement viewer does not inherit a board still writing payload.
        stream.join(timeout=5.0)
        if radar is not None:
            radar.join(timeout=2.0)
            radar.close()
        if video is not None and video.frames:
            print("recorded %d frames -> %s/" % (video.frames, record_dir),
                  file=sys.stderr)
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    sys.exit(main())
