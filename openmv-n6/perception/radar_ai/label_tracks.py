#!/usr/bin/env python3
"""Gate-5 labels: radar track -> person / clutter, from the camera teacher.

Replays channel B + the u-only association (run_associate) and, per radar
track id, counts frames MATCHED to a person box against frames where the
track was in the camera band with no box (CONTRADICTION). A track is
'person' when >= --pos of its in-band frames were matched, 'clutter' when
<= --neg were, 'ambiguous' otherwise (kept out of training).
Joins the label onto the track features run_channel_b.py wrote.

Usage:
    python3 perception/radar_ai/label_tracks.py captures/session_dyn
        -> perception/out/radar_ai/<sess>_labelled.json
"""
import argparse, json, os, sys, collections
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)
from perception.run_associate import channel_b_snapshots      # noqa: E402
from perception.associate import Associator                   # noqa: E402
OUT = os.path.join(ROOT, 'perception', 'out')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session'); ap.add_argument('--occ', type=float, default=0.35)
    ap.add_argument('--pos', type=float, default=0.5); ap.add_argument('--neg', type=float, default=0.1)
    a = ap.parse_args()
    name = os.path.basename(os.path.normpath(a.session))
    snaps = channel_b_snapshots(a.session, a.occ)
    teacher = {}
    with open(os.path.join(OUT, 'autolabel', f'{name}_teacher.jsonl')) as f:
        for line in f:
            r = json.loads(line); teacher[r['i']] = [d for d in r['dets'] if d['cls'] == 'person']
    with open(os.path.join(a.session, 'frames.jsonl')) as f:
        frames = [json.loads(l) for l in f if l.strip()]
    assoc = Associator()
    cnt = collections.defaultdict(lambda: {'matched': 0, 'contra': 0, 'silent': 0, 'frames': 0})
    for m in frames:
        boxes = teacher.get(m['i'], []); tracks = snaps.get(m.get('radar_frame'), [])
        if not tracks: continue
        xy = np.array([[x, y] for _, x, y in tracks])
        for p in assoc.pair(xy, boxes):
            tid = tracks[p.track_idx][0]; c = cnt[tid]; c['frames'] += 1
            if p.status == 'MATCHED': c['matched'] += 1
            elif p.status == 'CONTRADICTION': c['contra'] += 1
            else: c['silent'] += 1
    feats = json.load(open(os.path.join(OUT, 'radar_ai', f'{name}_radar_tracks.json')))
    out = []; lab = collections.Counter()
    for tr in feats:
        c = cnt.get(tr['tid']); 
        if not c: continue
        inband = c['matched'] + c['contra']
        frac = c['matched'] / inband if inband else None
        label = 'unknown' if frac is None else ('person' if frac >= a.pos else ('clutter' if frac <= a.neg else 'ambiguous'))
        lab[label] += 1
        out.append(dict(tr, session=name, matched=c['matched'], contra=c['contra'], silent=c['silent'], matched_frac=frac, label=label))
    json.dump(out, open(os.path.join(OUT, 'radar_ai', f'{name}_labelled.json'), 'w'), indent=1)
    print(f'{name}: {len(out)} tracks labelled -> {dict(lab)}')

if __name__ == '__main__':
    main()
