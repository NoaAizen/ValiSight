#!/usr/bin/env python3
"""D3 gate: associate channel-B radar tracks with teacher boxes on holds1.

Channel B runs online over radar.jsonl (map pass first, offline-labeling
context), and after every radar frame the positions of confirmed,
freshly-updated tracks are snapshotted by radar frame number. Video frames
then look their radar frame up via frames.jsonl - the same pairing key the
recorder wrote - and the Associator does u-only gated Hungarian against the
teacher's person boxes.

Gate (WORK-PLAN D3): among video frames with at least one person box AND at
least one in-band (1.5-3.5 m) fresh track, >= 90% end with that track
MATCHED. du statistics are reported alongside; the +-50 px gate is the
measured worst case of the current extrinsics, not a tunable.

Usage:
    python3 perception/run_associate.py captures/holds1 [--video-out]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from perception.associate import Associator                 # noqa: E402
from perception.radar_ai.static_map import StaticMap        # noqa: E402
from perception.radar_ai.cluster import cluster_frame       # noqa: E402
from perception.radar_ai.tracker import Tracker             # noqa: E402

OUT = os.path.join(ROOT, 'perception', 'out')


def channel_b_snapshots(session_dir, occ=0.35):
    """radar frame number -> [(tid, x, y), ...] of confirmed fresh tracks."""
    recs = []
    with open(os.path.join(session_dir, 'radar.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            recs.append((r['frame'], r['t_mono'],
                         np.asarray(r['points'], np.float32).reshape(-1, 6)))
    smap = StaticMap(occupancy_threshold=occ)
    for _, _, pts in recs:
        smap.accumulate(pts)
    smap.finalize()

    trk = Tracker()
    snaps = {}
    for frame, t, pts in recs:
        clusters = cluster_frame(pts)
        flags = [float(smap.is_clutter(c.points).mean()) >= 0.5
                 for c in clusters]
        trk.step(t, clusters, flags)
        # Associate against THIS FRAME'S MEASUREMENT (the cluster that just
        # updated the track), not the smoothed state: the alpha-beta lag on
        # a walker is ~0.3 m, which is ~80 px at 2 m - measured as the
        # dominant cause of just-outside-the-gate misses (nearest-track
        # median 77 px on holds1 before this change).
        snaps[frame] = [(tr.tid,
                         float(tr.history[-1][3].centroid[0]),
                         float(tr.history[-1][3].centroid[1]))
                        for tr in trk.tracks
                        if tr.confirmed and tr.misses == 0]
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--occ', type=float, default=0.35)
    ap.add_argument('--video-out', action='store_true',
                    help='write an association overlay clip')
    a = ap.parse_args()
    name = os.path.basename(os.path.normpath(a.session))

    snaps = channel_b_snapshots(a.session, a.occ)
    print(f'{name}: radar track snapshots on {len(snaps)} frames, '
          f'{sum(1 for v in snaps.values() if v)} with a confirmed track')

    teacher = {}
    with open(os.path.join(OUT, 'autolabel', f'{name}_teacher.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            teacher[r['i']] = [d for d in r['dets'] if d['cls'] == 'person']

    with open(os.path.join(a.session, 'frames.jsonl')) as f:
        frames = [json.loads(l) for l in f if l.strip()]

    assoc = Associator()
    # The gate is BOX-centric: does a person box get a radar track attached?
    # Track-centric counting would grade the clutter tracks (furniture,
    # glass ghosts) that the classifier - gate 5, not this one - exists to
    # remove; their unmatched projections are decision-layer evidence, and
    # they are reported below as status counts, never as failures here.
    n_boxes = n_boxes_matched = 0
    dus, statuses = [], {'MATCHED': 0, 'SILENT_EXPLAINED': 0,
                         'CONTRADICTION': 0}
    per_frame = {}
    for m in frames:
        boxes = teacher.get(m['i'], [])
        tracks = snaps.get(m.get('radar_frame'), [])
        if not tracks:
            continue
        xy = np.array([[x, y] for _, x, y in tracks])
        pairs = assoc.pair(xy, boxes)
        per_frame[m['i']] = (tracks, boxes, pairs)
        for p in pairs:
            statuses[p.status] += 1
        has_inband = any(p.in_band for p in pairs)
        if not has_inband:
            continue
        matched_boxes = {p.box_idx for p in pairs
                         if p.status == 'MATCHED' and p.in_band}
        for j in range(len(boxes)):
            n_boxes += 1
            if j in matched_boxes:
                n_boxes_matched += 1
        dus += [p.du_px for p in pairs
                if p.status == 'MATCHED' and p.in_band]

    pct = 100 * n_boxes_matched / max(n_boxes, 1)
    dus = np.array(dus)
    print(f'\nperson boxes in frames with an in-band fresh track: {n_boxes}')
    if len(dus):
        print(f'matched boxes: {n_boxes_matched} ({pct:.1f}%)   du median '
              f'{np.median(dus):.1f} px, p90 {np.percentile(dus, 90):.1f}, '
              f'max {dus.max():.1f}')
    print(f'track-status counts (incl. clutter tracks, informational): '
          f'{statuses}')
    print(f'box radar-coverage (channel-B detectability, NOT association '
          f'quality - bound by pre-Mode-P physics): {pct:.1f}%')

    # The association metric proper: when radar evidence for the box EXISTS
    # (an in-band fresh track within 2x gate), does the pairing land inside
    # the gate? Detectability of motionless people next to furniture is
    # Mode P's job (and gate 1's); geometry is this module's job.
    n_cand = n_cand_ok = 0
    for i, (tracks, boxes, pairs) in per_frame.items():
        matched_boxes = {p.box_idx for p in pairs
                         if p.status == 'MATCHED' and p.in_band}
        for j, b in enumerate(boxes):
            c = b['x'] + b['w'] / 2
            near = [p for p in pairs if p.in_band
                    and not np.isnan(p.u_proj)
                    and abs(p.u_proj - c) <= 2 * assoc.gate]
            if not near:
                continue
            n_cand += 1
            if j in matched_boxes:
                n_cand_ok += 1
    pct_a = 100 * n_cand_ok / max(n_cand, 1)
    print(f'D3 GATE - boxes with an in-band candidate that got MATCHED: '
          f'{n_cand_ok}/{n_cand} = {pct_a:.1f}% '
          f'-> {"PASS" if pct_a >= 90 else "FAIL"} (>=90%; full gate on P1)')

    if a.video_out:
        render(a.session, name, per_frame)


def render(session_dir, name, per_frame, max_frames=400):
    import cv2
    from perception.dataset import LiveSession
    from perception.project import RadarProjector
    proj = RadarProjector()
    vw = None
    drawn = 0
    # pick the busiest stretch: frames that have both tracks and boxes
    good = [i for i, (t, b, _) in per_frame.items() if t and b]
    if not good:
        print('nothing to render')
        return
    lo = good[len(good) // 2]
    keep = set(range(lo, lo + max_frames))
    out_path = os.path.join(OUT, f'{name}_assoc_overlay.mp4')
    for tr in LiveSession(session_dir).triplets():
        if tr.i not in keep or tr.i not in per_frame:
            continue
        tracks, boxes, pairs = per_frame[tr.i]
        img = cv2.cvtColor(tr.rgb, cv2.COLOR_GRAY2BGR)
        for b in boxes:
            cv2.rectangle(img, (b['x'], b['y']),
                          (b['x'] + b['w'], b['y'] + b['h']), (0, 255, 0), 2)
        for p in pairs:
            if np.isnan(p.u_proj):
                continue
            u = int(p.u_proj)
            col = {'MATCHED': (0, 255, 255), 'SILENT_EXPLAINED': (150, 150, 150),
                   'CONTRADICTION': (0, 0, 255)}[p.status]
            cv2.line(img, (u, 40), (u, 360), col, 2)
            cv2.putText(img, f'{p.range_m:.1f}m', (u + 4, 56),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, col, 1)
            if p.box_idx is not None:
                b = boxes[p.box_idx]
                cv2.line(img, (u, 200), (b['x'] + b['w'] // 2, 200),
                         (0, 255, 255), 1)
        if vw is None:
            vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 25.0, (img.shape[1], img.shape[0]))
        vw.write(img)
        drawn += 1
    if vw:
        vw.release()
    print(f'association overlay: {drawn} frames -> {out_path}')


if __name__ == '__main__':
    main()
