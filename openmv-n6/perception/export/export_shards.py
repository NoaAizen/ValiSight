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
from perception.student_data import LABEL_UNKNOWN  # noqa: E402
from perception.export.negative_gate import (  # noqa: E402
    NegativeGateError, occupied_frames)

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
                   teacher_available=True, verified_negative=False,
                   radar_unusable=False, occupied=()):
    """One session -> iterator of per-frame dicts, in time order.

    A generator rather than a list on purpose: multi5 alone is ~25k paired
    frames, and holding one session's RGB in memory OOM-killed the export
    twice on the 7.6GB Jetson (2026-08-23). Rows stream straight into
    write_shards, which flushes a file every per_shard rows.
    """
    ls = LiveSession(os.path.join(CAPTURES, sess))
    meta_by_i = {m['i']: m for m in ls.frames}
    heatmaps = load_heatmaps(sess)
    if heatmaps:
        print(f'[export] {sess}: radar dense streams present '
              f'({len(heatmaps)} radar frames) - exporting RA/RD/RP')
    # Shard rows must be key-uniform (np.stack). The map shapes are known
    # from radar.bin before any row is built, so a frame that missed a map
    # (session start, stale radar) gets zeros plus a validity flag right
    # here, instead of in a second pass over a full in-memory session.
    map_shape = {}
    if heatmaps:
        for key, short in (('range_angle', 'ra'), ('range_doppler', 'rd'),
                           ('range_profile', 'rp')):
            for hm in heatmaps.values():
                if hm[short] is not None:
                    map_shape[key] = hm[short].shape
                    break
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
        #
        # `occupied` is negative_gate's per-frame veto on the operator's
        # session-level claim. It only ever moves NEGATIVE -> UNKNOWN, so a
        # frame it is wrong about costs one negative and never teaches that a
        # person is nobody. Measured on radar3-dark-negative1 before this
        # existed: 320 of its 600 frames hold a person and every one of them
        # was exported as "there is nobody here", into the val split.
        frame_negative = bool(
            (verified_negative or
             meta_by_i.get(tr.i, {}).get('verified_negative', False))
            and tr.i not in occupied)
        row['radar_label_state'] = np.int8(
            LABEL_UNKNOWN if radar_unusable
            else label_state(bool(rgb_rows), frame_negative))
        row['thermal_label_state'] = np.int8(
            label_state(bool(th_rows), frame_negative))
        row['teacher_available'] = np.bool_(teacher_available)
        for key, flag in (('range_angle', 'ra_valid'),
                          ('range_doppler', 'rd_valid'),
                          ('range_profile', 'rp_valid')):
            if key in map_shape:
                row[flag] = np.bool_(key in row)
                if key not in row:
                    row[key] = np.zeros(map_shape[key], np.float16)
        yield row
    if n_dropped_radar:
        print(f'[export] {sess}: {n_dropped_radar} radar points beyond '
              f'K={RADAR_K} dropped (weakest SNR first)')


def session_thermal_meta(sess, thermal_dtypes):
    """Thermal scale and dtype, refusing mixed/ambiguous frame formats."""
    path = os.path.join(CAPTURES, sess, 'meta.json')
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    thermal_dtypes = tuple(thermal_dtypes)
    if thermal_dtypes and isinstance(thermal_dtypes[0], dict):
        # Backward-compatible convenience for callers/tests that still pass
        # exported rows instead of write_shards()' compact dtype set.
        dtypes = {
            str(np.asarray(row['thermal']).dtype)
            for row in thermal_dtypes
        }
    else:
        dtypes = set(thermal_dtypes)
    if len(dtypes) != 1:
        raise ValueError(f'{sess}: mixed thermal dtypes in one session: {dtypes}')
    dtype = dtypes.pop()
    dtype_name = 'uint16_le' if dtype == 'uint16' else dtype
    counts_max = int(meta.get(
        'thermal_counts_max', np.iinfo(np.dtype(dtype)).max))
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


def session_provenance(sess, radar_unusable=False):
    """Source hashes needed to detect incompatible sessions before training.

    A session whose radar is not being used contributes no radar hashes. The
    compatibility check treats a missing hash as "says nothing" rather than as
    a conflict, which is exactly right here: this session is not claiming its
    radar is comparable to the others', it is withdrawing it. The config's
    NAME stays, so a reader can still see what it was recorded under.
    """
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
        "radar_calib_sha256": (None if radar_unusable
                               else meta.get("radar_calib_sha256")),
        "radar_cfg_sha256": None if radar_unusable else stamp.get("sha256"),
        "radar_unusable": True if radar_unusable else None,
        "radar_cfg_name": stamp.get("name"),
        "radar_cfg_stamp_note": meta.get("radar_cfg_stamp_note"),
        "detector_model": meta.get("detector_model"),
        "detector_engine_sha256": meta.get("detector_engine_sha256"),
    }


def write_shards(rows, sess, out_dir, per_shard, keep_rgb):
    """Consume the row iterator, flushing a shard file every per_shard rows.

    Returns (paths, stats); stats carries what the manifest needs and was
    previously recomputed from the full in-memory session list.
    """
    paths, buf = [], []
    stats = {'frames': 0, 'n_lab': 0, 'n_neg': 0, 'thermal_dtypes': set()}

    def flush():
        arrs = {k: np.stack([f[k] for f in buf])
                for k in buf[0] if k != 'rgb' or keep_rgb}
        path = os.path.join(out_dir, f'{sess}-{len(paths):03d}.npz')
        np.savez_compressed(path, **arrs)
        paths.append(os.path.basename(path))
        buf.clear()

    for row in rows:
        stats['frames'] += 1
        stats['n_lab'] += int(int(row['n_th_boxes']) > 0)
        stats['n_neg'] += int(int(row['thermal_label_state']) == 0)
        stats['thermal_dtypes'].add(str(row['thermal'].dtype))
        buf.append(row)
        if len(buf) == per_shard:
            flush()
    if buf:
        flush()
    return paths, stats


TRAINING_CODE_FILES = (
    'perception/__init__.py',
    'perception/student_data.py',
    'perception/students.py',
    'perception/train_students.py',
    'perception/export_students_onnx.py',
    # The public-dataset converter travels with the bundle because the
    # pretraining half of a run happens in Colab, where the datasets are, and
    # a converter that lives only in the repo is a converter somebody
    # reimplements from memory at 2am.
    'perception/external/__init__.py',
    'perception/external/public_thermal.py',
)


def write_training_bundle(out_dir, out_name):
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

    # The production Colab flow validates the GPU and
    # class balance, copies shards off the Drive mount, runs a one-epoch
    # preflight, trains the two students independently with resumable
    # checkpoints, and exports ONNX.  Keep the stable exported filename so
    # existing documentation and Drive layouts continue to work.
    notebook_name = 'train_students_colab.ipynb'
    notebook_src = os.path.join(ROOT, 'perception', 'export', notebook_name)
    notebook_dst = os.path.join(out_dir, notebook_name)
    # The notebook's DATA path is REWRITTEN to this export's name, not copied.
    # It used to be copied verbatim, so v3's bundle shipped with
    # DATA=.../gexport/v2: opening it beside the v3 shards and hitting Run all
    # trained on v2 and wrote a checkpoint into v3/models. The two failures
    # that costs - training on the wrong data, and a checkpoint whose name
    # lies about its dataset - are both silent, and the only way anyone would
    # notice is by hashing the manifest afterwards.
    with open(notebook_src, encoding='utf-8') as f:
        nb = json.load(f)
    def _retarget(line):
        # The notebook names its export in one place. Which line that is has
        # moved once already (a hardcoded DATA path became EXPORT_NAME), so
        # both are handled and a notebook carrying neither is refused below
        # rather than shipped pointing somewhere else.
        if line.startswith('EXPORT_NAME = '):
            keep = line.split('#', 1)
            comment = ('  #' + keep[1]) if len(keep) > 1 else '\n'
            return "EXPORT_NAME = '%s'%s" % (out_name, comment)
        if line.startswith('DATA = /'):
            return line
        if line.startswith("DATA = '/content/drive"):
            return ("DATA = '/content/drive/MyDrive/thermal-fusion/gexport/%s'\n"
                    % out_name)
        return line

    retargeted = 0
    for cell in nb.get('cells', []):
        out_lines = []
        for line in cell.get('source', []):
            new_line = _retarget(line)
            retargeted += new_line != line
            out_lines.append(new_line)
        cell['source'] = out_lines
    if not retargeted:
        raise SystemExit(
            'the notebook names no export (no EXPORT_NAME/DATA line to '
            'retarget) - shipping it beside %s shards would point Colab at '
            'whatever it was last edited for' % out_name)
    body = json.dumps(nb, ensure_ascii=False, indent=1)
    with open(notebook_dst, 'w', encoding='utf-8') as f:
        f.write(body)
    # Hash what was WRITTEN, not the template it came from.
    hashes[notebook_name] = hashlib.sha256(body.encode('utf-8')).hexdigest()
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


# Above this share of a "verified empty" session flagged occupied, the claim
# itself is wrong rather than imperfect and the export stops. Below it the
# occupied frames are demoted and the count is printed loudly. radar3-dark-
# negative1 sits at 53%: the operator recorded a person walking about and
# filed it as an empty room, and 20% is already far past a mistake worth
# passing over in a log line.
NEGATIVE_WARN_FRACTION = 0.20
NEGATIVE_REFUSE_FRACTION = 0.80


def audit_negative(sess):
    """Per-frame occupancy for a session claimed empty. -> (set of i, stats)."""
    occupied, stats = occupied_frames(LiveSession(os.path.join(CAPTURES, sess)))
    n, total = stats['frames_occupied'], stats['frames_checked']
    frac = stats['fraction']
    if frac >= NEGATIVE_REFUSE_FRACTION:
        raise SystemExit(
            f'[export] {sess}: {n} of {total} frames ({frac:.0%}) are not '
            f'empty. This is not a negative session - drop it from '
            f'--verified-negative, or trim it and re-record frames.jsonl.')
    if frac >= NEGATIVE_WARN_FRACTION:
        print(f'[export] {sess}: *** {n} of {total} frames ({frac:.0%}) hold '
              f'something warm and were NOT recorded as empty. A session this '
              f'far from empty was mis-verified; the runs are listed below so '
              f'they can be read back against what was recorded.')
    elif n:
        print(f'[export] {sess}: {n} of {total} frames ({frac:.1%}) demoted '
              f'to UNKNOWN - warm, so not evidence of an empty room')
    else:
        print(f'[export] {sess}: {total} frames, none occupied - clean negative')
    for lo, hi in stats['runs']:
        print(f'[export]     occupied i {lo}-{hi} ({hi - lo + 1} frames)')
    return occupied, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--sessions', nargs='+', required=True,
                    help='session dir names under captures/')
    ap.add_argument('--val', nargs='*', default=[],
                    help='sessions held out for validation (whole sessions - '
                         'a frame split leaks near-duplicates)')
    ap.add_argument('--verified-negative', nargs='*', default=[], metavar='SESS',
                    help='sessions manually verified to contain no target; '
                         'only these may supply negative supervision. Every '
                         'frame is still checked by negative_gate, which '
                         'demotes the occupied ones to UNKNOWN')
    ap.add_argument('--negative-audit', action='store_true',
                    help='report per-frame occupancy for the --verified-'
                         'negative sessions and exit, exporting nothing')
    ap.add_argument('--radar-unusable', nargs='*', default=[], metavar='SESS',
                    help='sessions whose RADAR must not be used as evidence, '
                         'while their thermal still is. The case this exists '
                         'for is a session recorded under a different chirp '
                         'config: radar_10hz.cfg aliases Doppler, so its '
                         'velocities are not the quantity the other sessions '
                         'measured, but its Lepton frames are exactly the '
                         'same measurement. Their radar labels go to UNKNOWN '
                         '(masked in training) and their radar provenance '
                         'hashes are omitted, so the split-wide compatibility '
                         'check neither trips on them nor blesses them.')
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
    radar_unusable = set(a.radar_unusable)
    unknown = (radar_unusable | verified_negative | set(a.val)) - set(a.sessions)
    if unknown:
        raise SystemExit('not in --sessions: %s' % ', '.join(sorted(unknown)))

    # Before a byte is written: a session claimed empty is checked frame by
    # frame. Up front rather than inside the loop so a mis-verified session
    # stops the run before shards land on disk half written.
    occupancy, occupancy_stats = {}, {}
    for sess in sorted(verified_negative):
        try:
            occupancy[sess], occupancy_stats[sess] = audit_negative(sess)
        except NegativeGateError as exc:
            raise SystemExit(f'[export] {sess}: cannot be verified empty: {exc}')
    if a.negative_audit:
        raise SystemExit('[export] --negative-audit: nothing exported')

    for sess in a.sessions:
        teacher_path = os.path.join(AUTOLABEL, f'{sess}_teacher.jsonl')
        teacher_available = os.path.exists(teacher_path)
        teacher = load_teacher(sess)
        rows = export_session(
            sess, teacher, coco_boxes, keep_rgb,
            teacher_available=teacher_available,
            verified_negative=sess in verified_negative,
            radar_unusable=sess in radar_unusable,
            occupied=occupancy.get(sess, ()))
        shards, stats = write_shards(rows, sess, out_dir, a.frames_per_shard,
                                     keep_rgb)
        if not stats['frames']:
            print(f'[export] {sess}: nothing exportable (fused view or no '
                  f'thermal) - skipped')
            continue
        thermal_meta = session_thermal_meta(sess, stats['thermal_dtypes'])
        n_lab = stats['n_lab']
        n_neg = stats['n_neg']
        manifest['sessions'][sess] = {
            'shards': shards, 'frames': stats['frames'],
            'frames_with_gradeA': n_lab,
            'verified_negative_frames': n_neg,
            'provenance': session_provenance(sess, sess in radar_unusable),
            **thermal_meta,
        }
        # What the operator claimed minus what the frames showed. Recorded so a
        # checkpoint can be traced to the exact negative set that trained it -
        # the thing that was missing when v3 and v4 were selected on a val
        # split that was half occupied.
        if sess in occupancy_stats:
            manifest['sessions'][sess]['negative_gate'] = occupancy_stats[sess]
        which = 'val' if sess in a.val else 'train'
        manifest['split'][which].append(sess)
        print(f'[export] {sess}: {stats["frames"]} frames ({n_lab} with grade-A '
              f'boxes) -> {len(shards)} shard(s) [{which}]')

    manifest['training_bundle'] = write_training_bundle(out_dir, a.out_name)

    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    total = sum(os.path.getsize(os.path.join(out_dir, p))
                for p in os.listdir(out_dir))
    print(f'[export] {out_dir}: {total / 1e6:.1f} MB total. Upload this '
          f'directory to Google Drive and open the Colab notebook.')


if __name__ == '__main__':
    main()
