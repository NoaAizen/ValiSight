"""Run warm-blob detection over recorded raw Lepton frames.

    python3 thermal_detect.py                          # newest recording with thermal/
    python3 thermal_detect.py data/recordings/<run>    # that recording
    python3 thermal_detect.py path/to/frame.bin        # one raw 19200 B frame
    python3 thermal_detect.py --limit 50               # first N frames only

This is the DECISION-PATH pipeline, end to end, offline:

    decode -> destripe -> repair -> [FPN subtract if cfg/lepton_fpn.npy] ->
    core.detect.thermal.detect -> core.fusion.uncertainty.thermal_estimate

Regain / palettes / JPEG are display and deliberately absent. The dead rows
go in as invalid_rows so their interpolated pixels bridge blobs but never
count as evidence.

Prints one line per frame (best box + thermal sigma) and a summary. Like the
other src/ tools it resolves data/ relative to this file, so it runs from
anywhere.
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import lepton_fix
from core.detect import thermal as td
from core.fusion import uncertainty as unc

RECORDINGS = os.path.join(_ROOT, "data", "recordings")
FPN_PATH = os.path.join(_HERE, "cfg", "lepton_fpn.npy")


def _frame_files(target):
    """Resolve the CLI target to a list of raw .bin frame paths."""
    if target and target.endswith(".bin"):
        return [target]
    if target:
        runs = [target]
    else:
        runs = sorted(glob.glob(os.path.join(RECORDINGS, "*")), reverse=True)
    for run in runs:
        files = sorted(glob.glob(os.path.join(run, "thermal", "*.bin")))
        if files:
            return files
    return []


def _load_fpn():
    if not os.path.exists(FPN_PATH):
        return None
    m = np.load(FPN_PATH, allow_pickle=True)
    return m.item().get("map") if m.dtype == object else m


def process(raw, fpn=None, dead=lepton_fix.DEAD_ROWS,
            offset=lepton_fix.OFFSET_ROWS):
    """Raw sensor bytes -> (boxes, thermal SensorEstimate), or None."""
    frame = lepton_fix.decode(raw)
    if frame is None:
        return None
    frame = lepton_fix.repair(lepton_fix.destripe(frame, dead, offset), dead)
    if fpn is not None:
        frame = lepton_fix.apply_pixel_offsets(frame, fpn)
    boxes = td.detect(frame, dead)
    return boxes, unc.thermal_estimate(boxes[0] if boxes else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", nargs="?", help="recording dir or one .bin frame")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames")
    args = ap.parse_args(argv)

    files = _frame_files(args.target)
    if not files:
        print("no thermal .bin frames under %s" % (args.target or RECORDINGS))
        return 1
    if args.limit:
        files = files[:args.limit]
    fpn = _load_fpn()
    print("%d frames from %s   (FPN map: %s)"
          % (len(files), os.path.dirname(files[0]), "yes" if fpn is not None else "no"))

    n_det = 0
    confs = []
    for path in files:
        with open(path, "rb") as fh:
            out = process(fh.read(), fpn)
        if out is None:
            print("%s  truncated frame, skipped" % os.path.basename(path))
            continue
        boxes, est = out
        if boxes:
            b = boxes[0]
            n_det += 1
            confs.append(b.confidence)
            print("%s  %d box(es)  best conf %.2f  sigma %.2f  "
                  "at x=%d y=%d %dx%d  contrast %.1f DN  valid %.0f%%"
                  % (os.path.basename(path), len(boxes), b.confidence,
                     est.sigma, b.x, b.y, b.w, b.h, b.contrast_dn,
                     100.0 * b.valid_fraction))
        else:
            print("%s  no target  (thermal blind, sigma %.1f)"
                  % (os.path.basename(path), est.sigma))
    print("detections on %d/%d frames%s"
          % (n_det, len(files),
             "" if not confs else ", median conf %.2f" % float(np.median(confs))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
