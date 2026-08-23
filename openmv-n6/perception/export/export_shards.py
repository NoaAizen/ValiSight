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

What the thermal sample IS (so the notebook never guesses): uint8 or
little-endian uint16 raw counts, with dtype and scale declared per session.
Current live capture is uint8 radiometric, linear over the board's SET_RANGE
window [TMIN, TMAX]. Sessions recorded before meta.json carried the range have
c_per_lsb: null - intensity, not temperature. Unknown encodings fail closed.

Radar points are exported exactly as radar.jsonl stores them - project frame,
(x fwd, y left, z up, m), v FOLDED at 1.298 m/s (sign and magnitude untrusted),
snr_db, noise_db. Padded to RADAR_K by SNR rank; n_radar says how many are real
and radar_dropped counts what the pad refused, so saturation is visible instead
of silent.

Usage:
    python3 perception/export/export_shards.py --sessions live-20260812-100507 \
        [--out-name v2] [--frames-per-shard 256] [--no-rgb] [--val SESS ...] \
        [--verified-negative EMPTY_SESS ...]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
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
LABEL_UNKNOWN = -1
LABEL_NEGATIVE = 0
LABEL_POSITIVE = 1


def label_state(has_positive, verified_negative=False):
    """Tri-state target. Missing evidence is never negative evidence."""
    if verified_negative:
        return LABEL_NEGATIVE
    return LABEL_POSITIVE if has_positive else LABEL_UNKNOWN


def load_teacher(sess):
    """Per-video-frame teacher detections, or {} when the session has none.

    A missing teacher file means UNKNOWN supervision, not a negative. A dark
    session where RGB saw nothing is exactly where absence of a box cannot be
    interpreted as absence of a person.
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


def load_heatmaps(sess):
    """Radar dense streams keyed by frame number, or {} for point-only sessions.

    Probes the first rows cheaply: a session recorded before
    radar_people_ra.cfg has no TLV 4/5 anywhere, and scanning its whole
    radar.bin to learn that would double export time for nothing.
    """
    from perception.export import radar_heatmaps
    sess_dir = os.path.join(CAPTURES, sess)
    if not os.path.exists(os.path.join(sess_dir, 'radar.bin')):
        return {}
    probe = {}
    gen = radar_heatmaps.frames(sess_dir)
    for i, fr in enumerate(gen):
        if any(fr[k] is not None for k in ('ra', 'rd', 'range_profile')):
            probe[fr['frame']] = fr
        elif i >= 20 and not probe:
            return {}                      # 20 frames, no heatmap TLVs: old era
    if not probe:
        return {}
    out = {}
    for f, fr in probe.items():
        out[f] = {
            'ra': (radar_heatmaps.ra_image(fr['ra']).astype(np.float16)
                   if fr['ra'] is not None else None),
            'rd': (fr['rd'].astype(np.float16) if fr['rd'] is not None
                   else None),
            'rp': (fr['range_profile'].astype(np.float16)
                   if fr['range_profile'] is not None else None),
        }
    return out


def export_session(sess, teacher, coco_boxes, keep_rgb,
                   teacher_available=True, verified_negative=False):
    """One session -> list of per-frame dicts, in time order."""
    ls = LiveSession(os.path.join(CAPTURES, sess))
    meta_by_i = {m['i']: m for m in ls.frames}
    heatmaps = load_heatmaps(sess)
    if heatmaps:
        print(f'[export] {sess}: radar dense streams present '
              f'({len(heatmaps)} radar frames) - exporting RA/RD/RP')
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
        # RA/RD planes ride along when the session's radar.bin carries them
        # (radar_people_ra.cfg sessions). Keyed by the SAME radar frame the
        # points came from, so points and map describe one measurement.
        # Sessions without heatmaps simply lack these keys - a loader checks
        # for the key instead of trusting zeros that never happened.
        if heatmaps and tr.radar is not None:
            hm = heatmaps.get(tr.radar.frame)
            if hm is not None:
                if hm['ra'] is not None:
                    row['range_angle'] = hm['ra']
                if hm['rd'] is not None:
                    row['range_doppler'] = hm['rd']
                if hm['rp'] is not None:
                    row['range_profile'] = hm['rp']

        rgb_rows = [[d['x'], d['y'], d['w'], d['h'], d['conf']]
                    for d in teacher.get(tr.i, [])]
        row['rgb_boxes'], row['n_rgb_boxes'] = pad_boxes(rgb_rows, 5)
        th_rows = coco_boxes.get((sess, tr.i), [])
        row['th_boxes'], row['n_th_boxes'] = pad_boxes(th_rows, 5)
        # Tri-state supervision: +1 positive, 0 independently verified empty,
        # -1 unknown. In particular, "teacher emitted no box" is UNKNOWN.
        frame_negative = bool(
            verified_negative or
            meta_by_i.get(tr.i, {}).get('verified_negative', False))
        row['radar_label_state'] = np.int8(
            label_state(bool(rgb_rows), frame_negative))
        row['thermal_label_state'] = np.int8(
            label_state(bool(th_rows), frame_negative))
        row['teacher_available'] = np.bool_(teacher_available)
        frames.append(row)
    if n_dropped_radar:
        print(f'[export] {sess}: {n_dropped_radar} radar points beyond '
              f'K={RADAR_K} dropped (weakest SNR first)')
    # Shard rows must be key-uniform (np.stack). Frames that missed a map
    # (session start, stale radar) get zeros plus a validity flag, so the
    # loader can mask them instead of learning from silence.
    for key, flag in (('range_angle', 'ra_valid'),
                      ('range_doppler', 'rd_valid'),
                      ('range_profile', 'rp_valid')):
        shapes = [f[key].shape for f in frames if key in f]
        if not shapes:
            continue
        shape = shapes[0]
        for f in frames:
            f[flag] = np.bool_(key in f)
            if key not in f:
                f[key] = np.zeros(shape, np.float16)
    return frames


def session_thermal_meta(sess, frames):
    """Thermal scale and dtype, refusing mixed/ambiguous frame formats."""
    path = os.path.join(CAPTURES, sess, 'meta.json')
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    dtypes = {str(f['thermal'].dtype) for f in frames}
    if len(dtypes) != 1:
        raise ValueError(f'{sess}: mixed thermal dtypes in one session: {dtypes}')
    dtype = dtypes.pop()
    dtype_name = 'uint16_le' if dtype == 'uint16' else dtype
    counts_max = int(meta.get(
        'thermal_counts_max', np.iinfo(frames[0]['thermal'].dtype).max))
    tmin, tmax = meta.get('tmin'), meta.get('tmax')
    c_per_lsb = meta.get('c_per_lsb')
    if c_per_lsb is None and tmin is not None and tmax is not None:
        c_per_lsb = (tmax - tmin) / counts_max
    return {
        'tmin': tmin, 'tmax': tmax, 'c_per_lsb': c_per_lsb,
        'thermal_dtype': dtype_name,
        'thermal_encoding': meta.get('thermal_encoding', 'unknown'),
        'thermal_counts_max': counts_max,
    }


def session_provenance(sess):
    """Source hashes needed to detect incompatible sessions before training."""
    path = os.path.join(CAPTURES, sess, "meta.json")
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    stamp = meta.get("radar_cfg_stamp") or {}
    return {
        "schema_version": meta.get("schema_version"),
        "lepton_gain": meta.get("lepton_gain"),
        "warp_lut_sha256": meta.get("warp_lut_sha256"),
        "radar_calib_sha256": meta.get("radar_calib_sha256"),
        "radar_cfg_sha256": stamp.get("sha256"),
        "radar_cfg_name": stamp.get("name"),
        "radar_cfg_stamp_note": meta.get("radar_cfg_stamp_note"),
        "detector_model": meta.get("detector_model"),
        "detector_engine_sha256": meta.get("detector_engine_sha256"),
    }


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


TRAINING_CODE_FILES = (
    'perception/__init__.py',
    'perception/student_data.py',
    'perception/students.py',
    'perception/train_students.py',
)


def write_training_bundle(out_dir):
    """Snapshot runnable training code beside the shards for reproducible Colab."""
    code_root = os.path.join(out_dir, 'code')
    os.makedirs(os.path.join(code_root, 'perception'), exist_ok=True)
    hashes = {}
    for rel in TRAINING_CODE_FILES:
        src = os.path.join(ROOT, rel)
        dst = os.path.join(code_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        with open(dst, 'rb') as f:
            hashes[rel] = hashlib.sha256(f.read()).hexdigest()

    notebook_name = 'train_students_colab.ipynb'
    notebook_src = os.path.join(ROOT, 'perception', 'export', notebook_name)
    shutil.copy2(notebook_src, os.path.join(out_dir, notebook_name))
    with open(notebook_src, 'rb') as f:
        hashes[notebook_name] = hashlib.sha256(f.read()).hexdigest()
    requirements = 'numpy\nscipy\ntorch\n'
    requirements_path = os.path.join(code_root, 'requirements.txt')
    with open(requirements_path, 'w') as f:
        f.write(requirements)
    hashes['requirements.txt'] = hashlib.sha256(
        requirements.encode('ascii')).hexdigest()

    return {
        'code_dir': 'code',
        'notebook': notebook_name,
        'sha256': hashes,
        'entrypoint': 'python -m perception.train_students',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--sessions', nargs='+', required=True,
                    help='session dir names under captures/')
    ap.add_argument('--val', nargs='*', default=[],
                    help='sessions held out for validation (whole sessions - '
                         'a frame split leaks near-duplicates)')
    ap.add_argument('--verified-negative', nargs='*', default=[], metavar='SESS',
                    help='sessions manually verified to contain no target; '
                         'only these may supply negative supervision')
    ap.add_argument("--out-name", default="v2")
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
        'thermal': {'w': THERMAL_W, 'h': THERMAL_H,
                    'dtype': 'per-session; uint8 or uint16_le',
                    'note': 'c_per_lsb null = intensity only; training must '
                            'not silently mix calibrated and unit scales'},
        'rgb': ({'w': RGB_W, 'h': RGB_H, 'dtype': 'uint8',
                 'note': 'mp4-decoded grayscale, lossy - teacher input, '
                         'never a measurement'} if keep_rgb else None),
        'radar': {'k': RADAR_K, 'cols': ['x', 'y', 'z', 'v', 'snr_db',
                                         'noise_db'],
                  'frame': 'project (x fwd, y left, z up, metres)',
                  'v_alias_m_s': V_ALIAS,
                  'optional_streams': ['range_angle', 'range_doppler',
                                       'range_profile'],
                  'note': 'v folded - sign and magnitude untrusted'},
        'labels': {'rgb_boxes': 'teacher persons, RGB px, [x,y,w,h,conf]',
                   'th_boxes': 'grade-A only, thermal px, [x,y,w,h,delta]',
                   'radar_label_state': '-1 unknown, 0 verified negative, 1 positive',
                   'thermal_label_state': '-1 unknown, 0 verified negative, 1 positive',
                   'max_boxes': MAX_BOXES},
        'split': {'train': [], 'val': []},
        'sessions': {},
    }

    verified_negative = set(a.verified_negative)
    for sess in a.sessions:
        teacher_path = os.path.join(AUTOLABEL, f'{sess}_teacher.jsonl')
        teacher_available = os.path.exists(teacher_path)
        teacher = load_teacher(sess)
        frames = export_session(
            sess, teacher, coco_boxes, keep_rgb,
            teacher_available=teacher_available,
            verified_negative=sess in verified_negative)
        if not frames:
            print(f'[export] {sess}: nothing exportable (fused view or no '
                  f'thermal) - skipped')
            continue
        shards = write_shards(frames, sess, out_dir, a.frames_per_shard,
                              keep_rgb)
        thermal_meta = session_thermal_meta(sess, frames)
        n_lab = sum(int(f['n_th_boxes']) > 0 for f in frames)
        n_neg = sum(int(f['thermal_label_state']) == 0 for f in frames)
        manifest['sessions'][sess] = {
            'shards': shards, 'frames': len(frames),
            'frames_with_gradeA': n_lab,
            'verified_negative_frames': n_neg,
            'provenance': session_provenance(sess),
            **thermal_meta,
        }
        which = 'val' if sess in a.val else 'train'
        manifest['split'][which].append(sess)
        print(f'[export] {sess}: {len(frames)} frames ({n_lab} with grade-A '
              f'boxes) -> {len(shards)} shard(s) [{which}]')

    manifest['training_bundle'] = write_training_bundle(out_dir)

    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    total = sum(os.path.getsize(os.path.join(out_dir, p))
                for p in os.listdir(out_dir))
    print(f'[export] {out_dir}: {total / 1e6:.1f} MB total. Upload this '
          f'directory to Google Drive and open the Colab notebook.')


if __name__ == '__main__':
    main()
