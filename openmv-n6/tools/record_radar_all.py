#!/usr/bin/env python3
"""Record what Yael's ``radar_detections_all()`` really returns in the field.

    python3 record_radar_all.py --seconds 60 --out DIR [--note "corridor, rig static"]

Her heading/position-from-walls work is blocked on this call "reaching us in
the field" (mapinit/nav/heading.py): the call exists on origin/IMU and returns
every return including static ones (clutterRemoval 0 in the cfg), but nothing
had been captured from the real IWR1843 for her to develop against. This
writes exactly her DRISHOT.md record - range_m, azimuth_deg (+ = right),
elevation_deg (+ = up), velocity_mps (+ = receding), snr_db, timestamp_ms on
the shared clock - plus is_static from RadarFeed and the per-frame ego-velocity
estimate, one JSON line per radar frame, 10 Hz.

Uses yael_api.LiveRadar as-is, so what lands on disk is the API, not a
re-implementation of it. Needs a checkout that holds yael_api/ and board/
(origin/IMU); found via --yael-dir, $VALISIGHT_YAEL_API, or ~/ValiSight_yael.
"""
import argparse
import json
import os
import sys
import time


def find_yael_api(explicit=None):
    for root in ([explicit] if explicit else []) + \
               ([os.environ["VALISIGHT_YAEL_API"]] if os.environ.get("VALISIGHT_YAEL_API") else []) + \
               [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
                os.path.expanduser("~/ValiSight_yael")]:
        root = os.path.abspath(os.path.expanduser(root))
        if os.path.isfile(os.path.join(root, "yael_api", "radar.py")) and \
           os.path.isfile(os.path.join(root, "board", "mmwave", "radar_feed.py")):
            return root
    raise SystemExit("no checkout with yael_api/ and board/mmwave/ (origin/IMU); "
                     "use --yael-dir or: git checkout origin/IMU -- yael_api board")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", required=True, help="directory; writes radar.jsonl + meta.json")
    ap.add_argument("--note", default="", help="what the scene was - walls, distances, motion")
    ap.add_argument("--yael-dir", default=None)
    ap.add_argument("--data-port", default=None)
    ap.add_argument("--cli-port", default=None)
    ap.add_argument("--cfg", default=None, help="radar cfg to push if the radar is silent")
    a = ap.parse_args()

    root = find_yael_api(a.yael_dir)
    sys.path.insert(0, root)
    from yael_api.radar import LiveRadar, DEFAULT_CFG
    kw = dict(data_port=a.data_port, cli_port=a.cli_port)
    if a.cfg:
        kw["cfg"] = a.cfg
    radar = LiveRadar(**kw)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "radar.jsonl")
    n_frames = n_dets = n_static = 0
    last_ts = None
    t0 = time.time()
    with open(path, "w") as f:
        while time.time() - t0 < a.seconds:
            time.sleep(0.02)                       # 10 Hz frames, poll at 50 Hz
            dets = radar.radar_detections_all()
            h = radar.sensor_health_flags()
            ts = dets[0]["timestamp_ms"] if dets else None
            if ts is None or ts == last_ts:
                continue                           # nothing new on the wire
            last_ts = ts
            ego = radar.radar_ego_velocity()
            rec = {"timestamp_ms": ts, "n": len(dets),
                   "n_static": sum(1 for d in dets if d.get("is_static")),
                   "frame_gap": h.get("frame_gap"), "ego": ego,
                   "detections": dets}
            f.write(json.dumps(rec) + "\n")
            n_frames += 1; n_dets += len(dets); n_static += rec["n_static"]
            if n_frames % 50 == 0:
                print("  %d frames, %d returns (%d static), link=%s" % (
                    n_frames, n_dets, n_static, h.get("link")), file=sys.stderr)
    radar.stop()
    h = radar.sensor_health_flags()
    meta = {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": a.seconds,
            "frames": n_frames, "returns": n_dets, "static_returns": n_static,
            "frames_lost_on_wire": h.get("frames_lost_total"), "uart_resyncs": h.get("uart_resyncs"),
            "link": h.get("link"), "cfg": a.cfg or DEFAULT_CFG,
            "data_port": radar.data_port, "note": a.note,
            "fields": {"range_m": "m", "azimuth_deg": "+ = right of boresight",
                       "elevation_deg": "+ = up", "velocity_mps": "+ = receding (radial)",
                       "snr_db": "dB", "timestamp_ms": "shared clock (yael_api.clock)",
                       "is_static": "stationary in the world after ego-motion compensation"},
            "frame_convention": "rig: x forward, y left, z up (per Yael's README)"}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print("wrote %s: %d frames, %d returns, %d static (%.0f%%), link=%s" % (
        path, n_frames, n_dets, n_static, 100.0 * n_static / max(1, n_dets), h.get("link")))
    return 0 if n_frames else 1


if __name__ == "__main__":
    sys.exit(main())
