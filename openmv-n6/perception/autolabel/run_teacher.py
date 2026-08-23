#!/usr/bin/env python3
"""Teacher stage of the autolabel pipeline: detector + thermal heat score.

Runs a selectable TensorRT YOLO teacher over every video frame of a recorded
session and, for each detection, scores the box's heat through the warp LUT.
No labels are decided here - this stage only measures. The build stage picks
thresholds from the measured distributions and turns detections into labels.

Only 'visible'-view sessions are eligible: walk1-4 recorded the fused view,
where the thermal overlay is baked into the very pixels the detector would
see, and a detector must never be fed its own other modality (that rule is
measured and documented for the Jetson GPU detector).

Output: one JSONL line per video frame; thermal deltas are tagged as
uint8_equivalent after normalization to the declared session scale -
  {"i", "t_mono", "bg", "thermal_counts_max", "thermal_delta_unit", "dets": [{"cls","conf","x","y","w","h",
                                  "th": {"n","max","mean","p90"} | null}]}

Usage:
    python3 perception/autolabel/run_teacher.py captures/holds1 \
        [--out perception/out/autolabel] [--conf 0.30] [--model yolo11n] [--limit N]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from perception.dataset import LiveSession          # noqa: E402
from perception.autolabel.thermal_check import ThermalBoxCheck  # noqa: E402
from detect import COCO                             # noqa: E402
import detect as _detect                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--out', default=os.path.join(ROOT, 'perception', 'out',
                                                  'autolabel'))
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--model", default="yolov10n",
                    choices=["yolov10n", "yolov8n", "yolo11n"],
                    help="TensorRT teacher model (default yolov10n)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sess = LiveSession(a.session)
    view = sess.frames[0].get('view') if sess.frames else None
    if view != 'visible':
        sys.exit(f'refusing {a.session}: view={view!r} - the teacher only '
                 f'eats visible-view sessions (fused pixels contain the '
                 f'thermal overlay)')

    try:
        det = _detect.make_detector("auto", conf=a.conf,
                                    model=a.model)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(f"teacher detector unavailable: {e}")
    print(f"teacher: {det.model_name}/{det.backend}", file=sys.stderr)
    thermal_rows = [m for m in sess.frames if m.get("thermal_off") is not None]
    counts_max = sess.meta.get("thermal_counts_max")
    if thermal_rows:
        thermal_dtype = sess.thermal_dtype(thermal_rows[0])
        if counts_max is None and thermal_dtype.itemsize > 1:
            sys.exit(f"refusing {a.session}: uint16 thermal requires "
                     "thermal_counts_max in meta.json before auto-labeling")
    if counts_max is None:
        counts_max = 255                 # legacy uint8 sessions
    check = ThermalBoxCheck(counts_max=counts_max)
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(os.path.normpath(a.session))
    out_path = os.path.join(a.out, f'{name}_teacher.jsonl')

    n_frames = n_dets = 0
    t0 = time.time()
    with open(out_path, 'w') as f:
        for tr in sess.triplets():
            if a.limit and n_frames >= a.limit:
                break
            dets = det(tr.rgb)
            bg = check.background(tr.thermal) if tr.thermal is not None else None
            for d in dets:
                d['th'] = (check.region(tr.thermal, d['x'], d['y'],
                                        d['w'], d['h'])
                           if tr.thermal is not None else None)
            f.write(json.dumps({"i": tr.i, "t_mono": tr.t_mono, "bg": bg,
                                "thermal_counts_max": check.counts_max,
                                "thermal_delta_unit": "uint8_equivalent",
                                "dets": dets}) + "\n")
            n_frames += 1
            n_dets += len(dets)

    if hasattr(det, "close"):
        det.close()
    dt = time.time() - t0
    print(f'{name}: {n_frames} frames, {n_dets} detections, '
          f'{dt:.1f}s ({1e3 * dt / max(n_frames, 1):.1f} ms/frame) -> {out_path}')


if __name__ == '__main__':
    main()
