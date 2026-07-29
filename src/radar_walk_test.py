"""End-to-end radar test with a real moving target: does anything survive?

The static-room capture showed 3 strong but isolated points per frame, all of
which the isolation gate deletes. That says nothing about whether the chain
works on a person, so this walks the same data through TWO gates at once:

  strict  -- the shipping gate, isolation rejection on (min_neighbors=1)
  relaxed -- identical but isolation rejection OFF (min_neighbors=0)

If clusters appear only in the relaxed column, the isolation gate is the
blocker. If neither column produces clusters, CFAR is not returning enough
points on a real target and the config is the problem.

    python3 radar_walk_test.py [--seconds 40]
"""
import argparse, collections, glob, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serial
from iwr1843_uart import RadarReader
import radar_gate
import radar_classify_n6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    args = ap.parse_args()

    port = os.path.realpath(sorted(glob.glob("/dev/serial/by-id/*XDS110*if03"))[0])
    print("DATA %s  --  %.0f s\n" % (port, args.seconds), flush=True)
    print("  %-6s %-7s %-7s %-8s %-8s %s" %
          ("t", "pts/s", "maxpts", "strict", "relaxed", "labels (relaxed)"), flush=True)

    reader = RadarReader()
    tot = collections.Counter()
    labels = collections.Counter()
    best = {"n": 0, "t": 0.0}
    bucket = collections.Counter()
    bucket_labels = collections.Counter()
    t0 = t_bucket = time.time()

    with serial.Serial(port, 921600, timeout=0.1) as ser:
        ser.reset_input_buffer()
        while time.time() - t0 < args.seconds:
            for fr in reader.feed(ser.read(8192)):
                pts, snr, noise = fr["points"], fr["snr"], fr["noise"]
                n = len(pts)
                tot["frames"] += 1
                tot["raw"] += n
                bucket["frames"] += 1
                bucket["raw"] += n
                if n > best["n"]:
                    best["n"], best["t"] = n, time.time() - t0

                strict, _ = radar_gate.gate_points(pts, snr, noise)
                relaxed, _ = radar_gate.gate_points(pts, snr, noise, min_neighbors=0)
                tot["strict"] += len(strict)
                tot["relaxed"] += len(relaxed)
                bucket["strict"] += len(strict)
                bucket["relaxed"] += len(relaxed)

                cs = radar_classify_n6.classify_frame(strict)
                cr = radar_classify_n6.classify_frame(relaxed)
                tot["cl_strict"] += len(cs)
                tot["cl_relaxed"] += len(cr)
                bucket["cl_strict"] += len(cs)
                bucket["cl_relaxed"] += len(cr)
                for c in cr:
                    labels[c["label"]] += 1
                    bucket_labels[c["label"]] += 1

            now = time.time()
            if now - t_bucket >= 2.0:
                print("  %-6.0f %-7.1f %-7d %-8s %-8s %s"
                      % (now - t0,
                         bucket["raw"] / (now - t_bucket),
                         best["n"],
                         "%d pt/%d cl" % (bucket["strict"], bucket["cl_strict"]),
                         "%d pt/%d cl" % (bucket["relaxed"], bucket["cl_relaxed"]),
                         dict(bucket_labels) or ""), flush=True)
                bucket.clear()
                bucket_labels.clear()
                t_bucket = now

    print("\n== summary over %d frames ==" % tot["frames"], flush=True)
    print("  raw points            : %d  (%.2f per frame)"
          % (tot["raw"], tot["raw"] / max(1, tot["frames"])))
    print("  busiest single frame  : %d points at t=%.1fs" % (best["n"], best["t"]))
    print("  STRICT  gate (shipping): %d points kept, %d clusters"
          % (tot["strict"], tot["cl_strict"]))
    print("  RELAXED (no isolation) : %d points kept, %d clusters"
          % (tot["relaxed"], tot["cl_relaxed"]))
    print("  labels (relaxed)      : %s" % (dict(labels) or "none"))

    print("\n== reading ==")
    if tot["cl_strict"] > 0:
        print("  The shipping chain produced clusters on a real target. It works.")
    elif tot["cl_relaxed"] > 0:
        print("  Clusters appear ONLY without isolation rejection -> that gate is")
        print("  what is emptying the panel, not the radar and not CFAR.")
    elif best["n"] <= 4:
        print("  A moving person never lifted the point count above %d. CFAR is not"
              % best["n"])
        print("  returning enough points on a real target -- the .cfg is the problem,")
        print("  not the gate. Lower the cfarCfg threshold in the active .cfg and")
        print("  retest -- but predict the cost first: the false-alarm rate goes as")
        print("  (1 + alpha/N)^-N, so 15 -> 10 dB is not a small step.")
    else:
        print("  Points rose on the target but never got within %.1f m of each other."
              % radar_gate.NEIGHBOR_EPS_M)
    return 0


if __name__ == "__main__":
    sys.exit(main())
