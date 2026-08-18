#!/usr/bin/env python3
"""Pick the heat threshold from the measured distribution, and show the work.

Reads the teacher JSONLs, splits detections into person / everything-else,
and prints the distribution of (warped thermal max - frame background) for
each. The B-vs-A label boundary is chosen HERE, from data, not assumed -
recorded sessions have no radiometric scale, so the absolute 31-39 C gate
from live detection does not transfer.

Also writes a contact sheet of sampled person boxes - lowest deltas first -
because the labels those borderline boxes get is exactly what a human must
eyeball before the build stage trusts the threshold.

Usage:
    python3 perception/autolabel/analyze_teacher.py [--sheet-session captures/holds1]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import cv2

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)
from perception.dataset import LiveSession   # noqa: E402

OUT = os.path.join(ROOT, 'perception', 'out', 'autolabel')


def load_all():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, '*_teacher.jsonl'))):
        sess = os.path.basename(path).replace('_teacher.jsonl', '')
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                for d in r['dets']:
                    rows.append({'sess': sess, 'i': r['i'], 'bg': r['bg'],
                                 **d})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet-session', default=None)
    ap.add_argument('--sheet-n', type=int, default=24)
    a = ap.parse_args()

    rows = load_all()
    person = [r for r in rows if r['cls'] == 'person' and r['th']]
    other = [r for r in rows if r['cls'] != 'person' and r['th']]
    nocov = [r for r in rows if r['cls'] == 'person' and not r['th']]

    def deltas(rs):
        return np.array([r['th']['max'] - r['bg'] for r in rs])

    dp, do = deltas(person), deltas(other)
    print(f'detections: {len(rows)} total, {len(person)} person w/ thermal, '
          f'{len(nocov)} person w/o coverage, {len(other)} other w/ thermal')
    for name, d in (('person', dp), ('other', do)):
        if not len(d):
            continue
        q = np.percentile(d, [5, 25, 50, 75, 95])
        print(f'  {name:7s} delta(max-bg): p5 {q[0]:6.1f}  p25 {q[1]:6.1f}  '
              f'median {q[2]:6.1f}  p75 {q[3]:6.1f}  p95 {q[4]:6.1f}')

    # Threshold candidate: the valley between the two populations - halfway
    # between other's p75 and person's p25, reported alongside what each side
    # loses so the choice is inspectable.
    if len(dp) and len(do):
        thr = 0.5 * (np.percentile(do, 75) + np.percentile(dp, 25))
        print(f'\ncandidate threshold: delta >= {thr:.0f} counts')
        print(f'  person kept: {(dp >= thr).mean() * 100:.1f}%   '
              f'other passing (false heat): {(do >= thr).mean() * 100:.1f}%')

    per_sess = {}
    for r in person:
        per_sess.setdefault(r['sess'], []).append(r['th']['max'] - r['bg'])
    print('\nperson delta by session (median [n]):')
    for s, v in sorted(per_sess.items()):
        print(f'  {s:24s} {np.median(v):6.1f}  [{len(v)}]')

    if a.sheet_session:
        make_sheet(a.sheet_session, a.sheet_n)


def make_sheet(session_dir, n):
    """Contact sheet of person crops, hardest (lowest delta) first."""
    name = os.path.basename(os.path.normpath(session_dir))
    path = os.path.join(OUT, f'{name}_teacher.jsonl')
    picks = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            for d in r['dets']:
                if d['cls'] == 'person' and d['th']:
                    picks.append((d['th']['max'] - r['bg'], r['i'], d))
    picks.sort(key=lambda p: p[0])
    idx = np.unique(np.linspace(0, len(picks) - 1, n).astype(int))
    chosen = {picks[k][1]: picks[k] for k in idx}

    tiles = []
    for tr in LiveSession(session_dir).triplets():
        if tr.i not in chosen:
            continue
        delta, _, d = chosen[tr.i]
        img = cv2.cvtColor(tr.rgb, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (d['x'], d['y']),
                      (d['x'] + d['w'], d['y'] + d['h']), (0, 255, 0), 2)
        crop = img[max(0, d['y'] - 20):d['y'] + d['h'] + 20,
                   max(0, d['x'] - 20):d['x'] + d['w'] + 20]
        tile = cv2.resize(crop, (160, 200))
        cv2.putText(tile, f'd={delta:.0f} c={d["conf"]:.2f}', (4, 14),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 255), 1)
        tiles.append(tile)
    if tiles:
        cols = 6
        rows_n = int(np.ceil(len(tiles) / cols))
        sheet = np.zeros((rows_n * 200, cols * 160, 3), np.uint8)
        for k, t in enumerate(tiles):
            r_, c_ = divmod(k, cols)
            sheet[r_ * 200:(r_ + 1) * 200, c_ * 160:(c_ + 1) * 160] = t
        out = os.path.join(OUT, f'{name}_person_sheet.png')
        cv2.imwrite(out, sheet)
        print(f'\ncontact sheet ({len(tiles)} crops, lowest delta first): {out}')


if __name__ == '__main__':
    main()
