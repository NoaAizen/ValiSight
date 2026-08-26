#!/usr/bin/env python3
"""Check the Lepton's 16-bit radiometric channel on the board, and price it.

    ./verify_raw16.py            # needs the raw-passthrough firmware

Two halves, and they answer different questions.

The check half asserts one thing: that the board's OWN 8-bit frame can be
rebuilt from the words, exactly, pixel for pixel. That single comparison covers
the firmware's copy, thermal_io's conversion and the session's window at once,
and it cannot pass by accident - the board computed its side independently, in C,
from the same frame. If it passes, --raw16 is additive and nothing downstream
of the 8-bit plane can tell the difference.

The measurement half exists because the value of the extra bits is not a
constant and this project has already been misled by quoting it as one. The
words never change; the window only decides how the 8-bit plane is derived from
them. So one capture prices every window, and what it keeps showing is that the
answer depends entirely on which window the session ran:

    wide open (-10..140)   the quantiser is several times coarser than the
                           sensor's own temporal noise, and the 8-bit plane
                           throws away real signal
    auto-ranged            the step lands BELOW the noise, and 16 bits buys
                           nothing for single-frame precision

What does not depend on the window, and is the honest reason to keep the
channel: the words are absolute temperature with no meta.json lookup, they
cannot be pinned at a window edge, and they do not inherit a per-session
auto-range decision - which this rig has already had go wrong, with 100% of the
frame pinned at its floor and the health panel reporting "ok" (capture.py).
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import capture      # noqa: E402
import thermal_io   # noqa: E402

TH_W, TH_H = 160, 120


def board_port():
    for d in sorted(glob.glob("/dev/ttyACM*")):
        p = subprocess.run(["udevadm", "info", "-q", "property", "-n", d],
                           capture_output=True, text=True).stdout
        if "ID_VENDOR=MicroPython" in p:
            return d
    raise SystemExit("no MicroPython port - the board is not enumerated")


def capture_frames(port, n, tmin, tmax, autorange):
    code = (capture._BRINGUP
            .replace("__TMIN__", str(tmin)).replace("__TMAX__", str(tmax))
            .replace("__AUTORANGE__", str(bool(autorange)))
            .replace("__RAW__", "True")
            + '''
out = sys.stdout.buffer
sys.stdout.write("#RANGE %d %d\\n" % (TMIN, TMAX))
for _ in range(''' + str(n) + '''):
    t = lep.snapshot()
    lep.ioctl(csi.IOCTL_LEPTON_GET_RAW, _raw)
    tb = t.bytearray()
    sys.stdout.write("#PAIR %d %d\\n" % (len(tb), len(_raw)))
    for buf in (memoryview(tb), memoryview(_raw)):
        off = 0
        while off < len(buf):
            w = out.write(buf[off:off + 4096])
            if w:
                off += w
sys.stdout.write("#DONE\\n")
''')

    # Board.run() ends with read_until(b"OK"), and skipping that is a trap: the
    # raw REPL acknowledges a submission with a bare "OK" and it lands glued to
    # the front of whatever the script prints first. With auto-ranging on that
    # was absorbed by a line this parser ignores anyway; with it off the first
    # line became "OK#RANGE ..." and the range was silently never read.
    b = capture.Board(port)
    b.run(code)

    pairs, rng = [], None
    while True:
        line = b.read_line(timeout=90).decode("utf-8", "replace").strip()
        if line.startswith("\x04") or "Traceback" in line:
            raise SystemExit("BOARD ERROR:\n" + line.lstrip("\x04")
                             + b.drain().decode("utf-8", "replace"))
        if line.startswith("#RANGE"):
            rng = tuple(int(v) for v in line.split()[1:3])
        elif line.startswith("#PAIR"):
            n8, n16 = (int(v) for v in line.split()[1:3])
            pairs.append((b.read_exact(n8), b.read_exact(n16)))
        elif line.startswith("#DONE"):
            return pairs, rng


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=12, help="frames to stack")
    ap.add_argument("-p", "--port", default=None)
    ap.add_argument("--tmin", type=int, default=-10)
    ap.add_argument("--tmax", type=int, default=140)
    ap.add_argument("--no-autorange", action="store_true")
    args = ap.parse_args()

    port = args.port or board_port()
    print("board on %s, capturing %d frames" % (port, args.n))
    pairs, rng = capture_frames(port, args.n, args.tmin, args.tmax,
                                not args.no_autorange)

    W = np.stack([np.frombuffer(r, "<u2").reshape(TH_H, TH_W) for _, r in pairs])
    C = np.stack([np.frombuffer(c, np.uint8).reshape(TH_H, TH_W) for c, _ in pairs])
    degc = W.astype(np.float64) * 0.01 - 273.15
    window = thermal_io.window_ck({"tmin": rng[0], "tmax": rng[1]})

    # ---- the check -------------------------------------------------------
    fails = 0
    if not all(len(r) == 2 * TH_W * TH_H for _, r in pairs):
        print("FAIL  the raw plane is not two bytes per pixel")
        fails += 1

    diff = int((thermal_io.codes8(W, window) != C).sum())
    if diff:
        print("FAIL  %d of %d pixels differ between the rebuilt and the board's "
              "own 8-bit plane" % (diff, C.size))
        fails += 1
    else:
        print("PASS  the board's own 8-bit plane is rebuilt exactly from the "
              "words (%d pixels over %d frames)" % (C.size, len(pairs)))

    # ---- the measurement -------------------------------------------------
    vals = np.unique(W)
    gaps = np.diff(vals.astype(np.int64))
    quantum = int(np.gcd.reduce(gaps)) / 100.0 if len(gaps) else 0.0
    noise = float(np.median(degc.std(axis=0)))

    print("\nscene            %.2f .. %.2f C, %d distinct words"
          % (degc.min(), degc.max(), len(vals)))
    print("sensor quantum   %.2f C" % quantum)
    print("temporal noise   %.4f C  (median per-pixel std over the stack)" % noise)
    if noise > 0.5:
        # The auto-range path settles the sensor before it measures; --no-autorange
        # does not, so a stack taken right after bring-up is still riding the
        # post-FFC drift and reads an order of magnitude high. Say so rather than
        # let the number be quoted: this part's settled figure is 126-148 mK.
        print("                 ^ far above this part's settled 126-148 mK - the "
              "sensor is still drifting after bring-up, so treat the window "
              "comparison below as indicative only")

    print("\n8-bit plane per window - the step it can express, against that noise")
    print("  %-24s %-9s %-8s %-7s %s"
          % ("window", "step C", "vs noise", "pinned", "what limits it"))
    windows = [(args.tmin, args.tmax, "as asked"), (rng[0], rng[1], "auto-ranged"),
               (0, 60, "0:60, recorded sessions")]
    for lo, hi, label in windows:
        step = (hi - lo) / 255.0
        w = thermal_io.window_ck({"tmin": lo, "tmax": hi})
        pinned = float(((W <= w[0]) | (W >= w[1])).mean() * 100)
        print("  %-24s %-9.4f %-8s %-7s %s"
              % ("%d..%d (%s)" % (lo, hi, label), step, "%.1fx" % (step / noise),
                 "%.2f%%" % pinned,
                 "the quantiser" if step > noise else "the sensor's noise"))

    print("\nagainst the words, this session's 8-bit step is %.1fx coarser"
          % (((rng[1] - rng[0]) / 255.0) / max(quantum, 1e-9)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
