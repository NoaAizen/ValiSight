#!/usr/bin/env python3
"""Record everything Yael's Rig serves - radar, IMU, thermal - on the one clock.

    python3 record_rig_all.py --seconds 60 --out DIR --bridge-out ~/ValiSight_yael/bridge/out

Companion to record_radar_all.py, for when the N6 bridge is running: bridge_rx
writes imu.csv/frames.csv into --bridge-out and yael_api.Rig tails them, so
this is the field version of `python3 -m yael_api.demo` written to disk.

Output, one JSON line per tick (50 Hz poll):
  {"t_ms": shared clock, "radar": <new radar frame or null>,
   "imu": imu_attitude(), "imu_raw": imu_raw(), "thermal": thermal_frame() header
   (no pixels), "health": sensor_health_flags()}
Radar appears only on ticks where a new frame arrived (10 Hz); IMU is the latest
200 Hz sample at the tick. Nothing is interpolated - a consumer that needs a
radar-IMU pair takes the nearest timestamps and knows the gap.
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
        if os.path.isfile(os.path.join(root, "yael_api", "rig.py")) and \
           os.path.isfile(os.path.join(root, "board", "mmwave", "radar_feed.py")):
            return root
    raise SystemExit("no checkout with yael_api/ and board/mmwave/ (origin/IMU)")


def _json_default(o):
    """Raw buffers (a thermal frame's bytes) are not the record; say how big they were."""
    if isinstance(o, (bytes, bytearray, memoryview)):
        return {"bytes": len(o)}
    return str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bridge-out", default=None, help="dir bridge_rx writes into (imu.csv, frames.csv)")
    ap.add_argument("--stats", default=None, help="bridge_rx stats file, if any")
    ap.add_argument("--note", default="")
    ap.add_argument("--yael-dir", default=None)
    ap.add_argument("--no-radar", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, find_yael_api(a.yael_dir))
    from yael_api import Rig
    rig = Rig(out_dir=a.bridge_out, radar=not a.no_radar, stats_path=a.stats)
    geom = rig.camera_geometry()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "rig.jsonl")
    n = {"ticks": 0, "radar_frames": 0, "imu": 0, "thermal": 0}
    last_radar_ts = last_imu_ts = last_th = None
    t0 = time.time()
    with open(path, "w") as f:
        while time.time() - t0 < a.seconds:
            time.sleep(0.02)
            now = rig.shared_clock_ms()
            dets = rig.radar_detections_all()
            rts = dets[0]["timestamp_ms"] if dets else None
            radar = None
            if rts is not None and rts != last_radar_ts:
                last_radar_ts = rts
                radar = {"timestamp_ms": rts, "n": len(dets),
                         "n_static": sum(1 for d in dets if d.get("is_static")),
                         "ego": rig.radar_ego_velocity(), "detections": dets}
                n["radar_frames"] += 1
            att = rig.imu_attitude()
            raw = rig.imu_raw()
            its = (att or {}).get("timestamp_ms")
            if its is not None and its != last_imu_ts:
                last_imu_ts = its; n["imu"] += 1
            fr = rig.thermal_frame(celsius=False)
            th = None
            if fr:
                th = {k: fr[k] for k in fr if k != "pixels"}
                if th.get("seq") != last_th:
                    last_th = th.get("seq"); n["thermal"] += 1
            h = rig.sensor_health_flags()
            f.write(json.dumps({"t_ms": now, "radar": radar, "imu": att, "imu_raw": raw,
                                "thermal": th, "health": h}, default=_json_default) + "\n")
            n["ticks"] += 1
            if n["ticks"] % 250 == 0:
                print("  %ds: radar %d frames, imu %d new, thermal %d, imu_ok=%s thermal_ok=%s radar_ok=%s"
                      % (time.time() - t0, n["radar_frames"], n["imu"], n["thermal"],
                         h.get("imu_ok"), h.get("thermal_ok"), h.get("radar_ok")), file=sys.stderr)
    if rig.radar:
        rig.radar.stop()
    h = rig.sensor_health_flags()
    meta = {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": a.seconds,
            "counts": n, "final_health": h, "camera_geometry": geom,
            "bridge_out": rig.out_dir, "note": a.note,
            "drop_policy": Rig.DROP_POLICY, "rates": Rig.RATES}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2, default=str)
    print("wrote %s: %s | imu_ok=%s thermal_ok=%s radar_ok=%s" % (
        path, n, h.get("imu_ok"), h.get("thermal_ok"), h.get("radar_ok")))
    return 0 if n["ticks"] else 1


if __name__ == "__main__":
    sys.exit(main())
