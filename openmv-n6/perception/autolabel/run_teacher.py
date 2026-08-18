#!/usr/bin/env python3
"""Teacher stage of the autolabel pipeline: detector + thermal heat score.

Runs the TensorRT yolov10n teacher over every video frame of a recorded
session and, for each detection, scores the box's heat through the warp LUT.
No labels are decided here - this stage only measures. The build stage picks
thresholds from the measured distributions and turns detections into labels.

Only 'visible'-view sessions are eligible: walk1-4 recorded the fused view,
where the thermal overlay is baked into the very pixels the detector would
see, and a detector must never be fed its own other modality (that rule is
measured and documented for the Jetson GPU detector).

Output: one JSONL line per video frame -
  {"i", "t_mono", "bg", "dets": [{"cls","conf","x","y","w","h",
                                  "th": {"n","max","mean","p90"} | null}]}

Usage:
    python3 perception/autolabel/run_teacher.py captures/holds1 \
        [--out perception/out/autolabel] [--conf 0.30] [--limit N]
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
from trt_detect import TrtDetector                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--out', default=os.path.join(ROOT, 'perception', 'out',
                                                  'autolabel'))
    ap.add_argument('--conf', type=float, default=0.30)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    sess = LiveSession(a.session)
    view = sess.frames[0].get('view') if sess.frames else None
    if view != 'visible':
        sys.exit(f'refusing {a.session}: view={view!r} - the teacher only '
                 f'eats visible-view sessions (fused pixels contain the '
                 f'thermal overlay)')

    det = TrtDetector(conf=a.conf, names=COCO)
    check = ThermalBoxCheck()
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
            f.write(json.dumps({'i': tr.i, 't_mono': tr.t_mono, 'bg': bg,
                                'dets': dets}) + '\n')
            n_frames += 1
            n_dets += len(dets)

    dt = time.time() - t0
    print(f'{name}: {n_frames} frames, {n_dets} detections, '
          f'{dt:.1f}s ({1e3 * dt / max(n_frames, 1):.1f} ms/frame) -> {out_path}')


if __name__ == '__main__':
    main()
