#!/usr/bin/env python3
"""הדגמה חיה: python3 -m yael_api.demo [--seconds 10] [--no-radar]
מדפיס פעם בשנייה את כל מה שיעל מקבלת. דורש שהגשר רץ (bridge_rx -o bridge/out)."""
import argparse
import sys
import time

from . import Rig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--no-radar", action="store_true")
    a = ap.parse_args()
    rig = Rig(out_dir=a.out, radar=not a.no_radar, stats_path=a.stats)
    print("camera_geometry:", rig.camera_geometry())
    print("gnss:", rig.gnss_status_and_security())
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        time.sleep(1.0)
        now = rig.shared_clock_ms()
        dets = rig.radar_detections_all()
        ego = rig.radar_ego_velocity()
        att = rig.imu_attitude()
        raw = rig.imu_raw()
        fr = rig.thermal_frame(celsius=False)
        h = rig.sensor_health_flags()
        print("\n[shared_clock %d ms]" % now)
        print("  radar: %d dets (static %d)  ego=%s  health=%s" % (
            len(dets), sum(d["is_static"] for d in dets),
            None if ego is None else {k: (round(v, 2) if isinstance(v, float) else v) for k, v in ego.items()},
            {k: h["radar"][k] for k in ("radar_ok", "frame_gap", "stale_ms", "link")} if h.get("radar") else None))
        for d in dets[:3]:
            print("     ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()})
        print("  imu:   att=%s" % att)
        print("         raw=%s" % raw)
        if fr:
            print("  thermal: seq %d  ts %s  mean %.1f C  std %.1f  ffc=%s  (lag %s ms)" % (
                fr["seq"], fr["timestamp_ms"], fr["mean_c"], fr["std_gray"], fr["ffc_in_progress"],
                None if fr["timestamp_ms"] is None else now - fr["timestamp_ms"]))
        else:
            print("  thermal: none yet")
        print("  health: imu_ok=%s (%s ms) thermal_ok=%s (%s ms) radar_ok=%s clock_map=%s bridge=%s" % (
            h["imu_ok"], h["imu_stale_ms"], h["thermal_ok"], h["thermal_stale_ms"], h["radar_ok"],
            h["clock_map_ready"], h["bridge"]))
    if rig.radar:
        rig.radar.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
