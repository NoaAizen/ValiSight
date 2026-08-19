#!/usr/bin/env python3
"""Package recorded sessions into training shards for Colab (Google Drive).

The collection algorithm, stated once:

  1. A session is recorded by live.py exactly as today - session.mp4 (view),
     thermal.bin (measurement), frames.jsonl, radar.jsonl. Nothing here asks
     the recorder to change; this stage only refuses what it cannot use.
  2. Labels come from the D1 teacher chain (run_teacher.py -> build_dataset.py).
     Two label planes are exported per frame, because the student and the
     teacher live in different image planes:
       - rgb_boxes: every person the teacher saw, RGB pixel coords, with conf.
         These are TEACHER OPINIONS, ungraded - the notebook decides how to
         weight them (distillation targets).
       - th_boxes: grade-A boxes only, thermal pixel coords, from the COCO
         build. These are the trusted supervision - the thermal independently
         confirmed them (median delta >= 60).
  3. Frames are stored IN TIME ORDER, contiguous per session, with dt to the
     previous stored frame. Temporal channels (frame differencing) are a
     loader-side derivation, and an FFC gap (1824 ms) or a skipped unclean
     frame must not masquerade as motion - dt is what lets the loader refuse
     a pair rather than difference across a hole.
  4. Derived channels (gradients, |grad|, delta-from-background) are NOT
     stored. They are deterministic functions of the raw frame; storing them
     would double the shard for information the GPU recomputes in microseconds.
     Preserve-information-first applies to what cannot be recomputed: the raw
     thermal counts, the radar points, the timing.
  5. Split is BY SESSION, never by frame. Consecutive frames are near
     duplicates; a frame-level split leaks the train set into val and reports
     a model better than the one you have.

What the thermal sample IS (so the notebook never guesses): 8-bit radiometric,
linear over the board's SET_RANGE window [TMIN, TMAX] - c_per_lsb =
(tmax - tmin) / 255. The Lepton's 14-bit never leaves the board with current
firmware; the range window is the honest resolution statement, and it goes in
the manifest so a future narrower-range session is not silently mixed with a
wide-range one. Sessions recorded before meta.json carried the range have
c_per_lsb: null - intensity, not temperature.

Radar points are exported exactly as radar.jsonl stores them - project frame,
(x fwd, y left, z up, m), v FOLDED at 1.298 m/s (sign and magnitude untrusted),
snr_db, noise_db. Padded to RADAR_K by SNR rank; n_radar says how many are real
and radar_dropped counts what the pad refused, so saturation is visible instead
of silent.

Usage:
    python3 perception/export/export_shards.py --sessions live-20260812-100507 \
        [--out-name v1] [--frames-per-shard 256] [--no-rgb] [--val SESS ...]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)
from perception.dataset import LiveSession, THERMAL_W, THERMAL_H, RGB_W, RGB_H  # noqa: E402

CAPTURES = os.path.join(ROOT, 'captures')
AUTOLABEL = os.path.join(ROOT, 'perception', 'out', 'autolabel')
COCO_PATH = os.path.join(ROOT, 'perception', 'out', 'thermal_coco', 'annotations.json')
OUT_ROOT = os.path.join(ROOT, 'perception', 'out', 'gexport')

RADAR_K = 64        # points kept per frame, strongest SNR first
MAX_BOXES = 8       # persons per frame; indoor sessions have 1-3
V_ALIAS = 1.298     # m/s fold period, restated in the manifest


def load_teacher(sess):
    """Per-video-frame teacher detections, or {} when the session has none.

    A missing teacher file is a WARNING, not an error: a dark session where
    the RGB teacher saw nothing is exactly the hard-negative material the
    student needs, and it must still export - with empty label planes.
    """
    path = os.path.join(AUTOLABEL, f'{sess}_teacher.jsonl')
    if not os.path.exists(path):
        print(f'[export] {sess}: no teacher file - exporting unlabeled')
        return {}
    by_frame = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_frame[r['i']] = [d for d in r['dets'] if d['cls'] == 'person']
    return by_frame


def load_coco_boxes():
    """Grade-A thermal boxes keyed by (session, video frame index).

    The COCO build names images '<sess>_<i:05d>.png' with i the VIDEO frame
    index, which is the same i frames.jsonl carries - the join key exists by
    construction, not by luck.
    """
    if not os.path.exists(COCO_PATH):
        print('[export] no thermal COCO - grade-A plane will be empty')
        return {}
    with open(COCO_PATH) as f:
        coco = json.load(f)
    img_key = {}
    for im in coco['images']:
        stem = os.path.splitext(im['file_name'])[0]
        sess, idx = stem.rsplit('_', 1)
        img_key[im['id']] = (sess, int(idx))
    boxes = {}
    for a in coco['annotations']:
        boxes.setdefault(img_key[a['image_id']], []).append(
            list(a['bbox']) + [a.get('delta', 0.0)])
    return boxes


def pad_boxes(rows, width):
    out = np.zeros((MAX_BOXES, width), np.float32)
    n = min(len(rows), MAX_BOXES)
    for j in range(n):
        out[j] = rows[j][:width]
    return out, n


def export_session(sess, teacher, coco_boxes, keep_rgb):
    """One session -> list of per-frame dicts, in time order."""
    ls = LiveSession(os.path.join(CAPTURES, sess))
    frames = []
    prev_t = None
    n_dropped_radar = 0
    for tr in ls.triplets():
        if tr.thermal is None:      # fused-view sessions (walk1-4): banned
            continue
        if not tr.clean:            # overlay burned in - not training pixels
            continue
        row = {
            'thermal': tr.thermal,
            'i': np.int32(tr.i),
            't_mono': np.float64(tr.t_mono),
            # dt to the PREVIOUS EXPORTED frame - holes from skipped frames
            # and FFC gaps show up here as a large dt, which is the signal
            # the temporal-channel builder keys on.
            'dt_ms': np.float32(-1.0 if prev_t is None
                                else (tr.t_mono - prev_t) * 1e3),
        }
        prev_t = tr.t_mono
        if keep_rgb:
            row['rgb'] = tr.rgb
        pts = np.zeros((RADAR_K, 6), np.float32)
        n_pts = 0
        if tr.radar is not None and len(tr.radar.points):
            p = tr.radar.points
            order = np.argsort(-p[:, 4])            # SNR desc
            n_pts = min(len(p), RADAR_K)
            n_dropped_radar += max(0, len(p) - RADAR_K)
            pts[:n_pts] = p[order[:n_pts]]
        row['radar'] = pts
        row['n_radar'] = np.int16(n_pts)
        row['radar_age_ms'] = np.float32(tr.age_ms if tr.age_ms is not None
                                         else np.nan)

        rgb_rows = [[d['x'], d['y'], d['w'], d['h'], d['conf']]
                    for d in teacher.get(tr.i, [])]
        row['rgb_boxes'], row['n_rgb_boxes'] = pad_boxes(rgb_rows, 5)
        th_rows = coco_boxes.get((sess, tr.i), [])
        row['th_boxes'], row['n_th_boxes'] = pad_boxes(th_rows, 5)
        frames.append(row)
    if n_dropped_radar:
        print(f'[export] {sess}: {n_dropped_radar} radar points beyond '
              f'K={RADAR_K} dropped (weakest SNR first)')
    return frames


def session_c_per_lsb(sess):
    """(tmin, tmax, c_per_lsb) from meta.json, or Nones when unrecorded."""
    path = os.path.join(CAPTURES, sess, 'meta.json')
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        if 'tmin' in m and 'tmax' in m:
            return m['tmin'], m['tmax'], (m['tmax'] - m['tmin']) / 255.0
    return None, None, None


def write_shards(frames, sess, out_dir, per_shard, keep_rgb):
    paths = []
    for s0 in range(0, len(frames), per_shard):
        chunk = frames[s0:s0 + per_shard]
        arrs = {k: np.stack([f[k] for f in chunk])
                for k in chunk[0] if k != 'rgb' or keep_rgb}
        path = os.path.join(out_dir, f'{sess}-{s0 // per_shard:03d}.npz')
        np.savez_compressed(path, **arrs)
        paths.append(os.path.basename(path))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--sessions', nargs='+', required=True,
                    help='session dir names under captures/')
    ap.add_argument('--val', nargs='*', default=[],
                    help='sessions held out for validation (whole sessions - '
                         'a frame split leaks near-duplicates)')
    ap.add_argument('--out-name', default='v1')
    ap.add_argument('--frames-per-shard', type=int, default=256)
    ap.add_argument('--no-rgb', action='store_true',
                    help='omit the RGB plane (student-only export, ~14x smaller)')
    a = ap.parse_args()

    out_dir = os.path.join(OUT_ROOT, a.out_name)
    os.makedirs(out_dir, exist_ok=True)
    coco_boxes = load_coco_boxes()
    keep_rgb = not a.no_rgb

    manifest = {
        'version': a.out_name,
        'thermal': {'w': THERMAL_W, 'h': THERMAL_H, 'dtype': 'uint8',
                    'note': 'linear over [tmin,tmax]; c_per_lsb null = '
                            'intensity only, do not mix as temperature'},
        'rgb': ({'w': RGB_W, 'h': RGB_H, 'dtype': 'uint8',
                 'note': 'mp4-decoded grayscale, lossy - teacher input, '
                         'never a measurement'} if keep_rgb else None),
        'radar': {'k': RADAR_K, 'cols': ['x', 'y', 'z', 'v', 'snr_db',
                                         'noise_db'],
                  'frame': 'project (x fwd, y left, z up, metres)',
                  'v_alias_m_s': V_ALIAS,
                  'note': 'v folded - sign and magnitude untrusted'},
        'labels': {'rgb_boxes': 'teacher persons, RGB px, [x,y,w,h,conf]',
                   'th_boxes': 'grade-A only, thermal px, [x,y,w,h,delta]',
                   'max_boxes': MAX_BOXES},
        'split': {'train': [], 'val': []},
        'sessions': {},
    }

    for sess in a.sessions:
        teacher = load_teacher(sess)
        frames = export_session(sess, teacher, coco_boxes, keep_rgb)
        if not frames:
            print(f'[export] {sess}: nothing exportable (fused view or no '
                  f'thermal) - skipped')
            continue
        shards = write_shards(frames, sess, out_dir, a.frames_per_shard,
                              keep_rgb)
        tmin, tmax, c = session_c_per_lsb(sess)
        n_lab = sum(int(f['n_th_boxes']) > 0 for f in frames)
        manifest['sessions'][sess] = {
            'shards': shards, 'frames': len(frames),
            'frames_with_gradeA': n_lab,
            'tmin': tmin, 'tmax': tmax, 'c_per_lsb': c,
        }
        which = 'val' if sess in a.val else 'train'
        manifest['split'][which].append(sess)
        print(f'[export] {sess}: {len(frames)} frames ({n_lab} with grade-A '
              f'boxes) -> {len(shards)} shard(s) [{which}]')

    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    total = sum(os.path.getsize(os.path.join(out_dir, p))
                for p in os.listdir(out_dir))
    print(f'[export] {out_dir}: {total / 1e6:.1f} MB total. Upload this '
          f'directory to Google Drive and open the Colab notebook.')


if __name__ == '__main__':
    main()
