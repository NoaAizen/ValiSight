#!/usr/bin/env python3
"""Live radar scan: people and vehicles, as a terminal table.

    ./radar_scan.py                 scan until ctrl-c
    ./radar_scan.py --seconds 30    scan for 30s and exit

Same pipeline the live view draws (gate -> cluster -> classify), printed once a
second: one line per detected object with its class, position and motion.

The port is exclusive: stop live.py first (or run with --no-radar there) - two
readers on one serial port corrupt both.

Classes and what earns them (rule-based, radar_classify_n6):
  PERSON   moving, human-sized, wide limb velocity spread (micro-Doppler)
  VEHICLE  moving, extent > 1.5 m with 6+ returns
  static   |velocity| under 0.25 m/s - walls, furniture, parked anything
  ?        moving but matches neither shape

The vehicle rule is literature-based and was tuned on walking people indoors;
it has NOT yet been validated against a real vehicle. First parking-lot session
should check it before anyone quotes it.
"""
import argparse
import math
import sys
import time

from radar import RadarFeed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=0, help="0 = until ctrl-c")
    args = ap.parse_args()

    feed = RadarFeed()
    t0 = time.time()
    print("scanning... (ctrl-c to stop)", file=sys.stderr)
    try:
        while not args.seconds or time.time() - t0 < args.seconds:
            time.sleep(1.0)
            stamp = time.strftime("%H:%M:%S")
            objs = [c for c in feed.clusters if c["label"] != "static"]
            statics = len(feed.clusters) - len(objs)
            if feed.status != "ok":
                print("%s  radar: %s" % (stamp, feed.status))
                continue
            if not objs:
                print("%s  clear  (%d static returns)" % (stamp, statics))
                continue
            for c in objs:
                az = math.degrees(math.atan2(-c["centroid"][1], c["centroid"][0]))
                print("%s  %-8s  %5.1f m  az %+5.1f deg  %+5.2f m/s  "
                      "spread %.2f  pts %d" % (
                          stamp, c["label"].upper(), c["range_m"], az,
                          c["doppler_mps"], c["v_spread"], c["n_points"]))
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
