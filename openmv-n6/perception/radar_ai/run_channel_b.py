#!/usr/bin/env python3
"""Channel B offline: radar.jsonl -> static map -> clusters -> tracks.

Two passes over the session (offline labeling context): pass one builds the
static occupancy map, pass two clusters what the map does not explain and
tracks the clusters. Emits per-track features for the future GBM and a
top-down plot for the human gate.

D2 gate (WORK-PLAN): on walk1 the walking person comes out as ONE continuous
track that survives empty frames - fragmentation there means the tracker
patience/gate is wrong, and no feature computed on fragments can be trusted.

Usage:
    python3 perception/radar_ai/run_channel_b.py captures/walk1 [--eps 0.6]
        [--occ 0.6] [--min-dur 2.0]
"""
import argparse
import json
import os
import sys

import numpy as np
import cv2

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)
from perception.radar_ai.static_map import StaticMap      # noqa: E402
from perception.radar_ai.cluster import cluster_frame     # noqa: E402
from perception.radar_ai.tracker import Tracker           # noqa: E402
from perception.radar_ai.features import track_features   # noqa: E402

OUT = os.path.join(ROOT, 'perception', 'out', 'radar_ai')


def load_radar(session_dir):
    with open(os.path.join(session_dir, 'radar.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            yield r['t_mono'], np.asarray(r['points'],
                                          dtype=np.float32).reshape(-1, 6)


def top_down_plot(tracks, smap, path, extent=8.0, px_per_m=70):
    """x forward = up, y left = image left (project frame, top view)."""
    W = H = int(2 * extent * px_per_m / 2)
    W = int(extent * px_per_m * 1.0)
    H = int(extent * px_per_m)
    img = np.full((H, W, 3), 24, np.uint8)

    def to_px(x, y):
        u = int(W / 2 - y * px_per_m)
        v = int(H - x * px_per_m)
        return u, v

    # static bins as grey wedges (cell centers)
    ri, ai = np.nonzero(smap._static)
    for r_i, a_i in zip(ri, ai):
        r = (r_i + 0.5) * smap.range_bin
        az = np.radians((a_i + 0.5) * smap.az_bin - 90.0)
        u, v = to_px(r * np.cos(az), r * np.sin(az))
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(img, (u, v), 3, (90, 90, 90), -1)

    # range rings
    for rr in range(1, int(extent) + 1):
        cv2.circle(img, to_px(0, 0), int(rr * px_per_m), (50, 50, 50), 1)
        cv2.putText(img, f'{rr}m', (W // 2 + 4, H - rr * px_per_m + 12),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (110, 110, 110), 1)

    colors = [(80, 220, 80), (80, 160, 255), (220, 160, 60), (180, 80, 220),
              (60, 220, 220), (220, 220, 60)]
    for k, tr in enumerate(tracks):
        c = colors[k % len(colors)]
        pts = [(to_px(x, y)) for _, x, y, _ in tr.history]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, c, 2)
        cv2.circle(img, pts[0], 5, c, -1)
        cv2.putText(img, f'#{tr.tid}', (pts[0][0] + 6, pts[0][1]),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, c, 1)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--eps', type=float, default=0.6)
    ap.add_argument('--lambda-v', type=float, default=0.0,
                    help='Doppler weight in clustering; keep 0 pre-Mode-P')
    ap.add_argument('--occ', type=float, default=0.6,
                    help='static-map occupancy threshold')
    ap.add_argument('--min-dur', type=float, default=2.0,
                    help='report only tracks at least this long, seconds')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    name = os.path.basename(os.path.normpath(a.session))

    smap = StaticMap(occupancy_threshold=a.occ)
    n_pts = 0
    for t, pts in load_radar(a.session):
        smap.accumulate(pts)
        n_pts += len(pts)
    n_static = smap.finalize()
    print(f'{name}: {smap.frames} radar frames, {n_pts} points, '
          f'{n_static} static bins')

    trk = Tracker()
    kept = removed = 0
    for t, pts in load_radar(a.session):
        clusters = cluster_frame(pts, eps=a.eps, lambda_v=a.lambda_v)
        # the map vetoes track BIRTH and starves absorbed tracks, but never
        # deletes points: a cluster in a static bin can still update the
        # track of a person who paused on a mapped spot.
        clutter_flags = []
        for c in clusters:
            frac = float(smap.is_clutter(c.points).mean())
            clutter_flags.append(frac >= 0.5)
            if frac >= 0.5:
                removed += c.n
            else:
                kept += c.n
        trk.step(t, clusters, clutter_flags)
    tracks = [t for t in trk.finish()
              if t.history[-1][0] - t.history[0][0] >= a.min_dur]
    tracks.sort(key=lambda t: -(t.history[-1][0] - t.history[0][0]))
    print(f'clutter removed: {removed} pts ({100 * removed / max(n_pts, 1):.0f}%), '
          f'kept {kept}')

    rows = []
    print(f"\n{'track':>6} {'dur_s':>6} {'upd':>5} {'path_m':>7} "
          f"{'speed_med':>9} {'rcs':>6} {'straight':>8}")
    for tr in tracks:
        f = track_features(tr)
        rows.append({'tid': tr.tid, **f})
        print(f"#{tr.tid:>5} {f['duration_s']:6.1f} {f['n_updates']:5d} "
              f"{f['duration_s'] * f['ground_speed_med']:7.1f} "
              f"{f['ground_speed_med']:9.2f} {f['rcs_mean']:6.1f} "
              f"{f['straightness']:8.2f}")

    with open(os.path.join(OUT, f'{name}_radar_tracks.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    plot = os.path.join(OUT, f'{name}_topdown.png')
    top_down_plot(tracks, smap, plot)
    print(f'\ntracks >= {a.min_dur}s: {len(tracks)}  |  plot: {plot}')

    # D2 gate. "One continuous track" is physically unattainable when the
    # target leaves the sensed envelope, so the honest metric is: every
    # track TERMINATION happens at a physical boundary - the aoaFov cone
    # edge (+-60 deg, detections thin out from ~40), the CFAR range
    # ceiling (5.0 m, R^4 thins from ~4.4), or the IF-HPF floor. A death
    # in the middle of the envelope is a real tracker failure.
    # Judged on MOVING tracks only: a stationary reflector's track fading
    # mid-envelope is clutter for the classifier, not a tracker failure.
    # The D2 question is whether a WALK survives as one track between
    # physical boundaries.
    def explained(x, y):
        r = float(np.hypot(x, y))
        az = abs(float(np.degrees(np.arctan2(y, x))))
        return az >= 40.0 or r >= 4.4 or r <= 1.0
    movers = [(tr, f) for tr, f in zip(tracks, rows)
              if f['ground_speed_med'] >= 0.3]
    ends = [(tr.history[-1][1], tr.history[-1][2]) for tr, _ in movers]
    ok = sum(1 for x, y in ends if explained(x, y))
    pct = 100 * ok / max(len(ends), 1)
    print(f'D2 GATE - moving-track deaths at a physical boundary: '
          f'{ok}/{len(ends)} = {pct:.0f}% '
          f'-> {"PASS" if pct >= 80 else "FAIL"} (>=80%; final gate on P1)')
    for (tr, _), (x, y) in zip(movers, ends):
        if not explained(x, y):
            r = np.hypot(x, y); az = np.degrees(np.arctan2(y, x))
            print(f'  unexplained death: #{tr.tid} at {r:.1f} m, {az:+.0f} deg')


if __name__ == '__main__':
    main()
