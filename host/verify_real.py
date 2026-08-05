#!/usr/bin/env python3
"""Check the dead-row repair against every recorded frame, not against synthetics.

    ./verify_real.py ../captures/handwave3 ../captures/handwave2

The synthetic tests in verify.py prove the repair does what it says on a frame
built to have a known answer. They cannot prove the *detector* fires on the real
defect, because the synthetic damage is whatever the test author imagined it to
be - and the first version of this detector was wrong for exactly that reason.
It thresholded on brightness, ported from handreg.py, and the real frames say a
dead row is only saturated when the frame's own level happens to be high.

So this runs the built binary over real captures and asserts, per sequence:

  * the count of condemned rows is the same in every frame of that sequence -
    the defect is fixed while it is present, so a count that moves frame to
    frame means the detector is tracking the scene instead
  * a brightness threshold would have missed some of those frames, which is the
    regression this exists to prevent
  * where rows were condemned, the repair pulls the frame's reported maximum off
    the top of the range, where the dead rows otherwise pin it

Per sequence, deliberately. Whether the sensor is in the failed state at all is a
session-level property: on this unit every capture taken up to 15:23 is clean and
every capture from 21:51 onward has all 14 rows, in 100% of frames. So the defect
is a condition the sensor enters and then stays in, not something it was born
with - which is the argument against hardcoding the row numbers, however stable
they look within one session.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

FUSE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fuse")


def run(rgb, thermal, tmin, tmax, *extra):
    out = subprocess.run([FUSE, rgb, thermal, "/tmp/_vr.ppm",
                          "--range", "%d,%d" % (tmin, tmax), "--stats", *extra],
                         check=True, capture_output=True, text=True).stdout
    rows = int(re.search(r"deadrows: (\d+)", out).group(1))
    m = re.search(r"max (-?[\d.]+) C", out)
    return rows, (float(m.group(1)) if m else None)


def frames(d):
    for jp in sorted(glob.glob(os.path.join(d, "*.json"))):
        st = jp[:-5]
        rgb = st + "_rgb0.raw"
        if not os.path.exists(rgb):
            rgb = st + "_rgb.raw"
        th = st + "_thermal.raw"
        if os.path.exists(rgb) and os.path.exists(th):
            yield rgb, th, json.load(open(jp)), os.path.basename(st)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--sat-level", type=int, default=250,
                    help="the brightness threshold this replaced, for comparison")
    args = ap.parse_args()

    fails, total, affected, missed_total = [], 0, 0, 0

    for d in args.dirs:
        counts, max_on, max_off, missed = [], [], [], 0
        print("%s" % d)

        for rgb, th, meta, name in frames(d):
            tmin, tmax = meta["tmin"], meta["tmax"]
            rows, mx = run(rgb, th, tmin, tmax)
            _, mx_off = run(rgb, th, tmin, tmax, "--no-deadrow")

            # what the old brightness test would have condemned on this frame
            t = np.fromfile(th, np.uint8).reshape(meta["th_h"], meta["th_w"])
            old = int(((t >= args.sat_level).mean(1) > 0.6).sum())
            missed += (old < rows)

            counts.append(rows)
            max_on.append(mx)
            max_off.append(mx_off)
            print("  %-10s %2d rows rebuilt (brightness test: %2d) | max %6.2f -> %6.2f C"
                  % (name, rows, old, mx_off, mx))

        if not counts:
            print("  no frame pairs here")
            continue

        counts = np.array(counts)
        total += len(counts)
        missed_total += missed

        if counts.min() != counts.max():
            fails.append("%s: the condemned-row count varies frame to frame (%d..%d). While "
                         "the defect is present it is fixed, so a varying count means the "
                         "detector is following the scene." % (d, counts.min(), counts.max()))

        if counts.max() == 0:
            # A clean session is a legitimate outcome, not a failure: this sensor
            # spends whole sessions out of the failed state.
            print("  -> no dead rows in this sequence; sensor was healthy here")
            continue

        affected += len(counts)
        lift = float(np.mean(np.array(max_off) - np.array(max_on)))
        print("  -> %d rows every frame | brightness test would under-repair %d/%d frames"
              " | repair moves the reported max by %.2f C" % (counts.max(), missed,
                                                              len(counts), lift))
        if lift < 1.0:
            fails.append("%s: the repair barely changes the reported maximum (%.2f C) - the "
                         "dead rows should be pinning it well above the scene" % (d, lift))

    if not total:
        raise SystemExit("no frame pairs found - point this at a capture directory")

    print("\n%d frames, %d of them from sequences showing the defect" % (total, affected))
    print("frames the old brightness threshold would have under-repaired: %d" % missed_total)
    if affected and not missed_total:
        print("NOTE: no frame here distinguishes the two detectors, so this run does not"
              " actually test the change. Include a sequence with a lower frame level.")

    print()
    for f in fails:
        print("FAIL: %s" % f)
    if not fails:
        print("all real-frame checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
