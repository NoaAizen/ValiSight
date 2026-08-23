#!/usr/bin/env python3
"""Turn manual_boxes.json (label_web.py) into a <sess>_teacher.jsonl.

Manual labels enter the D1 chain as a teacher with conf 1.0 - build_dataset
and export_shards consume the file unchanged, so a hand-labeled session and a
machine-labeled one are interchangeable downstream. The thermal heat score is
still computed per box, exactly as run_teacher.py does: the human asserts
"person", but the thermal grade is what admits the box into the thermal COCO,
and the glass-wall veto (RGB sees, LWIR doesn't) must keep working on human
boxes too.

Only frames present in manual_boxes.json are written: a frame saved with zero
boxes is an explicit "no person" statement and becomes a line with empty dets;
a frame never visited in the labeler stays unknown and is omitted.

An existing machine teacher file is moved aside to <sess>.teacher-machine.jsonl
(a name build_dataset's *_teacher.jsonl glob does not match), never silently
overwritten.

Usage:
    python3 perception/autolabel/manual_to_teacher.py captures/TEST11
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)

from perception.dataset import LiveSession                       # noqa: E402
from perception.autolabel.thermal_check import ThermalBoxCheck   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--out', default=os.path.join(ROOT, 'perception', 'out',
                                                  'autolabel'))
    a = ap.parse_args()

    store = os.path.join(a.session, 'manual_boxes.json')
    if not os.path.exists(store):
        sys.exit(f'{store} not found - label the session with '
                 f'tools/label_web.py first')
    manual = {int(k): v for k, v in json.load(open(store))['boxes'].items()}
    if not manual:
        sys.exit(f'{store} holds no labeled frames')

    sess = LiveSession(a.session)
    view = sess.frames[0].get('view') if sess.frames else None
    if view != 'visible':
        sys.exit(f'refusing {a.session}: view={view!r} - fused pixels carry '
                 f'the thermal overlay; labels there train nothing honest')

    counts_max = sess.meta.get('thermal_counts_max')
    if counts_max is None:
        counts_max = 255                 # legacy uint8 sessions
    check = ThermalBoxCheck(counts_max=counts_max)

    name = os.path.basename(os.path.normpath(a.session))
    out_path = os.path.join(a.out, f'{name}_teacher.jsonl')
    if os.path.exists(out_path):
        first = open(out_path).readline()
        if '"manual"' not in first:
            aside = os.path.join(a.out, f'{name}.teacher-machine.jsonl')
            os.replace(out_path, aside)
            print(f'machine teacher moved aside -> {aside}', file=sys.stderr)

    os.makedirs(a.out, exist_ok=True)
    n_frames = n_boxes = 0
    with open(out_path, 'w') as f:
        for tr in sess.triplets():
            if tr.i not in manual:
                continue
            bg = check.background(tr.thermal) if tr.thermal is not None else None
            dets = []
            for b in manual[tr.i]:
                dets.append({'cls': 'person', 'conf': 1.0,
                             'x': b['x'], 'y': b['y'],
                             'w': b['w'], 'h': b['h'],
                             'th': (check.region(tr.thermal, b['x'], b['y'],
                                                 b['w'], b['h'])
                                    if tr.thermal is not None else None)})
            f.write(json.dumps({'i': tr.i, 't_mono': tr.t_mono, 'bg': bg,
                                'thermal_counts_max': check.counts_max,
                                'thermal_delta_unit': 'uint8_equivalent',
                                'source': 'manual',
                                'dets': dets}) + '\n')
            n_frames += 1
            n_boxes += len(dets)
    print(f'{name}: {n_frames} labeled frames, {n_boxes} person boxes '
          f'-> {out_path}')


if __name__ == '__main__':
    main()
