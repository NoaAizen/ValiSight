#!/usr/bin/env python3
"""Convert a public thermal person dataset into the shard format we train on.

Why this exists: the students have seen one building. Every verified-empty
frame in v4 comes from three scenes, none of them the lobby the rig runs in,
and the thermal student answers "person" for almost any warm human-sized shape
- a lit glass door included. Public thermal datasets do not fix the domain, but
they carry two things the rig cannot produce quickly: tens of thousands of
people in thermal from other cameras and other scenes, and - the part that
matters most here - thousands of EXHAUSTIVELY ANNOTATED frames with no person
in them at all.

That last distinction is the whole reason these frames are usable as negatives
when our own unlabelled frames are not. "The teacher found nobody" is not
evidence of absence (measured: witness-negatives poison 24.5% of labels). "A
human annotated every person in this image and there were none" is.

    LLVIP     15,488 infrared frames, 1280x1024, night street, pedestrians
              boxed in PASCAL VOC XML.
    FLIR ADAS ~26k thermal frames, 640x512, day and night driving, COCO json
              over 15 classes - so a frame with cars and lamps and no person
              is a labelled negative full of warm structure.

Both are licensed for NON-COMMERCIAL use. That is a decision to take on
purpose, not to discover later; it is recorded in the manifest of every export
this writes.

The output is a normal export directory (shards + manifest) that
perception.student_data can load, so it trains through the existing
entrypoint. It is a SEPARATE export from the rig's, not a merge, for a
mechanical reason: our thermal plane is absolute Celsius and these frames are
8-bit AGC intensity, and load_split refuses to mix the two scales in one split
(rightly - the absolute-intensity channel means different things in each).
Pretrain on this export, fine-tune on the rig's.

    python3 -m perception.external.public_thermal llvip \\
        --root /content/LLVIP --out perception/out/gexport/llvip
    python3 -m perception.external.public_thermal coco \\
        --root /content/FLIR_ADAS_v2/images_thermal_train \\
        --annotations coco.json --out perception/out/gexport/flir

Runs on numpy + cv2 alone - no torch - because it belongs upstream of training
and is normally run in Colab, where the datasets are downloaded: they are tens
of GB and this rig's uplink moves 5 MB a minute.
"""
import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

THERMAL_W, THERMAL_H = 160, 120
MAX_BOXES = 8
FRAMES_PER_SHARD = 256
MAX_RADAR_POINTS = 64

LABEL_UNKNOWN, LABEL_NEGATIVE, LABEL_POSITIVE = -1, 0, 1

# No temporal chain exists between unrelated stills, and build_temporal_indices
# links frame i to i-1 only when dt <= 300 ms. Stamping a full second says
# "these are not consecutive frames" in the one language the loader reads, so
# the model trains on them with valid_prev1/2 = 0 - which is also the state
# every session's first frames are in, and the state its worst answers come
# from today.
STILL_DT_MS = 1000.0


def _boxes_to_thermal_plane(boxes, src_w, src_h):
    """Public-frame boxes -> our 160x120 plane, cropping to 4:3 first.

    A straight resize would squash 5:4 (both datasets) into 4:3 and hand the
    model people 7% wider than they are. Shape is the only thing separating a
    person from a warm rectangle here, so the crop is not a detail: it is the
    difference between teaching shape and teaching a distortion.
    """
    want = THERMAL_W / float(THERMAL_H)
    have = src_w / float(src_h)
    if have > want:                      # too wide: trim the sides
        keep_w = int(round(src_h * want))
        x0, y0, cw, ch = (src_w - keep_w) // 2, 0, keep_w, src_h
    else:                                # too tall: trim top and bottom
        keep_h = int(round(src_w / want))
        x0, y0, cw, ch = 0, (src_h - keep_h) // 2, src_w, keep_h
    sx, sy = THERMAL_W / float(cw), THERMAL_H / float(ch)

    out = []
    for x, y, w, h in boxes:
        nx0, ny0 = (x - x0) * sx, (y - y0) * sy
        nx1, ny1 = (x + w - x0) * sx, (y + h - y0) * sy
        nx0, ny0 = max(0.0, nx0), max(0.0, ny0)
        nx1, ny1 = min(float(THERMAL_W), nx1), min(float(THERMAL_H), ny1)
        # A person the crop cut away is not a person that got smaller: drop the
        # box rather than clip it into a sliver the model has to explain.
        if nx1 - nx0 < 2.0 or ny1 - ny0 < 3.0:
            continue
        out.append((nx0, ny0, nx1 - nx0, ny1 - ny0))
    return (x0, y0, cw, ch), out


def _to_thermal_frame(img, crop):
    x0, y0, cw, ch = crop
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        # 16-bit thermal (some FLIR frames) carries counts, not degrees, and
        # the range differs per capture. Per-frame min/max is the only scale
        # available; it is exactly what "unit" mode in the manifest means.
        lo, hi = float(img.min()), float(img.max())
        img = ((img.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0)
        img = img.astype(np.uint8)
    sub = img[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(sub, (THERMAL_W, THERMAL_H), interpolation=cv2.INTER_AREA)


def read_llvip(root):
    """LLVIP: infrared jpgs + PASCAL VOC xml, one file per image."""
    img_dirs = [os.path.join(root, 'infrared', 'train'),
                os.path.join(root, 'infrared', 'test'),
                os.path.join(root, 'images', 'train'),
                os.path.join(root, 'images', 'val'),
                root]
    ann_dirs = [os.path.join(root, 'Annotations'),
                os.path.join(root, 'annotations'), root]
    images = []
    for d in img_dirs:
        images = sorted(glob.glob(os.path.join(d, '*.jpg')))
        if images:
            break
    if not images:
        raise SystemExit('no *.jpg under %s - is this an LLVIP root?' % root)
    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        xml = None
        for d in ann_dirs:
            cand = os.path.join(d, stem + '.xml')
            if os.path.exists(cand):
                xml = cand
                break
        if xml is None:
            # Silence here would turn an annotated frame into a negative, which
            # is the one mistake this file exists to avoid.
            raise SystemExit('%s has no annotation xml - refusing to guess '
                             'that it is empty' % path)
        boxes = []
        for obj in ET.parse(xml).getroot().iter('object'):
            name = (obj.findtext('name') or '').strip().lower()
            if name not in ('person', 'people', 'pedestrian'):
                continue
            bb = obj.find('bndbox')
            x0, y0 = float(bb.findtext('xmin')), float(bb.findtext('ymin'))
            x1, y1 = float(bb.findtext('xmax')), float(bb.findtext('ymax'))
            boxes.append((x0, y0, x1 - x0, y1 - y0))
        yield path, boxes


def read_coco(root, annotations, person_names=('person', 'people')):
    """FLIR ADAS and anything else that ships a COCO json.

    Exhaustive annotation is the assumption that makes an image with no person
    box a NEGATIVE rather than an unknown. It holds for these datasets and is
    stated in the manifest so a reader can check it rather than trust it.
    """
    path = annotations if os.path.isabs(annotations) else os.path.join(
        root, annotations)
    with open(path) as f:
        coco = json.load(f)
    people = {c['id'] for c in coco['categories']
              if c['name'].strip().lower() in person_names}
    if not people:
        raise SystemExit('no person category in %s (have: %s)'
                         % (path, ', '.join(sorted(
                             c['name'] for c in coco['categories']))))
    by_image = {}
    for a in coco['annotations']:
        if a['category_id'] in people and not a.get('iscrowd'):
            by_image.setdefault(a['image_id'], []).append(a['bbox'])
    for img in sorted(coco['images'], key=lambda im: im['id']):
        file_name = img['file_name']
        p = os.path.join(root, file_name)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(path), file_name)
        yield p, [tuple(float(v) for v in b)
                  for b in by_image.get(img['id'], [])]


def convert(pairs, out_dir, name, limit=0, per_shard=FRAMES_PER_SHARD):
    os.makedirs(out_dir, exist_ok=True)
    shards, rows = [], []
    n_pos = n_neg = n_dropped = 0

    def flush():
        if not rows:
            return
        idx = len(shards)
        path = os.path.join(out_dir, '%s-%03d.npz' % (name, idx))
        stack = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        np.savez_compressed(path, **stack)
        shards.append(os.path.basename(path))
        rows.clear()

    for i, (img_path, boxes) in enumerate(pairs):
        if limit and i >= limit:
            break
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            n_dropped += 1
            continue
        h, w = img.shape[:2]
        crop, th_boxes = _boxes_to_thermal_plane(boxes, w, h)
        frame = _to_thermal_frame(img, crop)
        # A frame whose only people were cropped away is UNKNOWN, never
        # negative: somebody was there, the crop simply cannot see them.
        if boxes and not th_boxes:
            state = LABEL_UNKNOWN
        elif th_boxes:
            state = LABEL_POSITIVE
        else:
            state = LABEL_NEGATIVE
        n_pos += state == LABEL_POSITIVE
        n_neg += state == LABEL_NEGATIVE

        padded = np.zeros((MAX_BOXES, 5), np.float32)
        for j, (x, y, bw, bh) in enumerate(th_boxes[:MAX_BOXES]):
            padded[j] = (x, y, bw, bh, 0.0)   # 5th is thermal delta, unused
        rows.append({
            'thermal': frame,
            'dt_ms': np.float32(STILL_DT_MS),
            'radar': np.zeros((MAX_RADAR_POINTS, 6), np.float32),
            'n_radar': np.int32(0),
            'radar_age_ms': np.float32(0.0),
            'th_boxes': padded,
            'n_th_boxes': np.int32(min(len(th_boxes), MAX_BOXES)),
            'rgb_boxes': np.zeros((MAX_BOXES, 5), np.float32),
            'n_rgb_boxes': np.int32(0),
            'thermal_label_state': np.int8(state),
            # No radar was anywhere near these frames.
            'radar_label_state': np.int8(LABEL_UNKNOWN),
            'teacher_available': np.bool_(True),
            'i': np.int32(i),
            't_mono': np.float64(i * STILL_DT_MS / 1000.0),
        })
        if len(rows) >= per_shard:
            flush()
    flush()
    return {'shards': shards, 'positive': int(n_pos), 'negative': int(n_neg),
            'unreadable': int(n_dropped)}


def write_manifest(out_dir, sessions, val, source, license_note):
    # Merge rather than overwrite, so two datasets can be converted into one
    # pretraining export: `... llvip --out X` then `... coco --out X` leaves a
    # single manifest holding both. Overwriting here would silently drop the
    # first dataset and produce a pretrain that is half the size it claims.
    existing = {}
    path = os.path.join(out_dir, 'manifest.json')
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
        prior = existing.get('sessions', {})
        clash = set(prior) & set(sessions)
        if clash:
            raise SystemExit(
                'session name(s) already in %s: %s - pass a different --name'
                % (path, ', '.join(sorted(clash))))
        val = set(val) | set(existing.get('split', {}).get('val', []))
        sessions = dict(prior, **sessions)
        source = existing.get('source') if isinstance(
            existing.get('source'), list) else [existing.get('source')]
        source = [s for s in source if s] + [_SOURCE]
        license_note = ' | '.join(
            dict.fromkeys(filter(None, [existing.get('license'),
                                        license_note])))
    manifest = {
        'version': os.path.basename(out_dir.rstrip('/')),
        'thermal': {'w': THERMAL_W, 'h': THERMAL_H, 'dtype': 'uint8',
                    'note': 'public 8-bit thermal, unit scale - c_per_lsb is '
                            'null on purpose: these frames are AGC intensity, '
                            'not radiometry, and must not be mixed into a '
                            'Celsius split'},
        'rgb': None,
        'radar': None,
        'labels': {'th_boxes': 'dataset annotation, thermal px, [x,y,w,h,0]',
                   'thermal_label_state': '-1 unknown, 0 verified negative '
                                          '(exhaustively annotated frame with '
                                          'no person), 1 positive',
                   'radar_label_state': 'always -1: no radar exists here',
                   'max_boxes': MAX_BOXES},
        'source': source,
        'license': license_note,
        'split': {'train': [s for s in sessions if s not in val],
                  'val': [s for s in sessions if s in val]},
        'sessions': sessions,
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest


_SOURCE = None      # set by main() so a merge can append this run's source


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('reader', choices=('llvip', 'coco'))
    ap.add_argument('--root', required=True,
                    help='dataset root (LLVIP) or image directory (coco)')
    ap.add_argument('--annotations', default='coco.json',
                    help='coco json, absolute or relative to --root')
    ap.add_argument('--out', required=True, help='export directory to write')
    ap.add_argument('--name', default=None,
                    help='session name inside the export (default: reader)')
    ap.add_argument('--val-fraction', type=float, default=0.1,
                    help='fraction of shards held out, by whole shards - '
                         'never by frame, for the same reason the rig export '
                         'splits by session (default 0.1)')
    ap.add_argument('--limit', type=int, default=0,
                    help='stop after N frames (smoke test)')
    a = ap.parse_args()

    name = a.name or a.reader
    if a.reader == 'llvip':
        pairs = read_llvip(a.root)
        source = {'dataset': 'LLVIP', 'root': os.path.abspath(a.root)}
        lic = ('LLVIP is released for NON-COMMERCIAL use (academic research, '
               'teaching, publication, personal experimentation). Anything '
               'trained on it inherits that question.')
    else:
        pairs = read_coco(a.root, a.annotations)
        source = {'dataset': 'coco-format', 'root': os.path.abspath(a.root),
                  'annotations': a.annotations}
        lic = ('Teledyne FLIR Free ADAS Thermal Dataset is distributed for '
               'NON-COMMERCIAL research and academic use per its terms of '
               'service. Anything trained on it inherits that question.')

    global _SOURCE
    _SOURCE = source
    stats = convert(pairs, a.out, name, limit=a.limit)
    if not stats['shards']:
        raise SystemExit('nothing converted')

    # Held out by whole shards. Frame-level holdout on a dataset shot in
    # sequences inflates every metric it produces, which is the mistake the
    # rig's own export documents at length.
    n_val = max(1, int(round(len(stats['shards']) * a.val_fraction)))
    # Never hold out everything: a split with an empty train side raises deep
    # inside the loader, an hour after the convert that caused it.
    n_val = min(n_val, max(0, len(stats['shards']) - 1))
    sessions = {}
    val = set()
    for i, shard in enumerate(stats['shards']):
        sess = '%s-%03d' % (name, i)
        sessions[sess] = {
            'shards': [shard], 'frames': None,
            'c_per_lsb': None, 'tmin': None, 'tmax': None,
            'thermal_dtype': 'uint8', 'thermal_counts_max': 255,
            'thermal_encoding': 'agc_8bit_unit',
            'provenance': {'lepton_gain': None, 'warp_lut_sha256': None,
                           'radar_calib_sha256': None,
                           'radar_cfg_sha256': None,
                           'detector_model': None,
                           'detector_engine_sha256': None},
        }
        if i >= len(stats['shards']) - n_val:
            val.add(sess)
    write_manifest(a.out, sessions, val, source, lic)
    print('%s: %d shards, %d positive, %d exhaustively-empty, %d unreadable'
          % (name, len(stats['shards']), stats['positive'], stats['negative'],
             stats['unreadable']))
    print('  val shards: %d of %d' % (n_val, len(stats['shards'])))
    print('  LICENSE: %s' % lic)


if __name__ == '__main__':
    main()
