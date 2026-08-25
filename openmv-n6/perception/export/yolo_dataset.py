#!/usr/bin/env python3
"""Build a YOLO fine-tuning dataset from the HAND-labelled sessions.

The stock yolov10n is a COCO model: colour images, mostly large subjects.
This rig gives it none of that - a grayscale 640x400 sensor, people at 2-15 m,
one indoor scene. Fine-tuning on our own hand labels is the honest way to
close that gap, and it is only honest because the labels are HUMAN: training
a teacher on its own output teaches it nothing but its own mistakes, so the
machine-teacher sessions are deliberately excluded.

Frames the operator visited and left empty are exported as empty label files.
That is not a gap - it is the only signal that says "no person here", and it
is what pulls the false-positive rate down.

The split is by whole session, never by frame: consecutive frames are
near-duplicates and a frame-level split would inflate every number.

Usage:
    python3 perception/export/yolo_dataset.py --val occlusion1 --out /tmp/yolo_ds
"""
import argparse
import glob
import json
import os
import shutil
import sys

import cv2

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
W, H = 640, 400


def sessions_with_hand_labels(captures):
    out = []
    for f in sorted(glob.glob(os.path.join(captures, '*', 'manual_boxes.json'))):
        d = os.path.dirname(f)
        meta = os.path.join(d, 'meta.json')
        view = json.load(open(meta)).get('view') if os.path.exists(meta) else None
        if view is not None and view != 'visible':
            print(f'  skip {os.path.basename(d)}: view={view!r} '
                  f'(fused pixels carry the thermal overlay)', file=sys.stderr)
            continue
        out.append(d)
    return out


def frame_image(session, i, cache_ok=True):
    """The exact frame the operator labelled, as a BGR image."""
    if cache_ok:
        p = os.path.join(session, '.label_frames', '%05d.jpg' % i)
        if os.path.exists(p):
            return cv2.imread(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captures', default=os.path.join(ROOT, 'captures'))
    ap.add_argument('--out', default='/tmp/yolo_ds')
    ap.add_argument('--val', nargs='+', default=['occlusion1'],
                    help='whole sessions held out for validation')
    a = ap.parse_args()

    for split in ('train', 'val'):
        for kind in ('images', 'labels'):
            os.makedirs(os.path.join(a.out, kind, split), exist_ok=True)

    stats = {}
    for session in sessions_with_hand_labels(a.captures):
        name = os.path.basename(session)
        split = 'val' if name in a.val else 'train'
        boxes = json.load(open(os.path.join(session,
                                            'manual_boxes.json')))['boxes']
        n_img = n_box = n_empty = n_missing = 0
        for key, bs in sorted(boxes.items(), key=lambda kv: int(kv[0])):
            i = int(key)
            img = frame_image(session, i)
            if img is None:
                n_missing += 1
                continue
            h, w = img.shape[:2]
            stem = f'{name}_{i:05d}'
            cv2.imwrite(os.path.join(a.out, 'images', split, stem + '.jpg'),
                        img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = []
            for b in bs:
                # YOLO wants normalised centre/size, clipped to the frame
                x0 = max(0.0, float(b['x'])); y0 = max(0.0, float(b['y']))
                x1 = min(float(w), x0 + float(b['w']))
                y1 = min(float(h), y0 + float(b['h']))
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                lines.append('0 %.6f %.6f %.6f %.6f' % (
                    (x0 + x1) / 2 / w, (y0 + y1) / 2 / h,
                    (x1 - x0) / w, (y1 - y0) / h))
            with open(os.path.join(a.out, 'labels', split,
                                   stem + '.txt'), 'w') as f:
                f.write('\n'.join(lines))
            n_img += 1
            n_box += len(lines)
            n_empty += (not lines)
        stats[name] = (split, n_img, n_box, n_empty, n_missing)
        print('%-14s %-5s %5d images %6d boxes %5d empty %s' % (
            name, split, n_img, n_box, n_empty,
            f'({n_missing} frames missing from cache)' if n_missing else ''))

    with open(os.path.join(a.out, 'data.yaml'), 'w') as f:
        f.write('path: %s\ntrain: images/train\nval: images/val\n'
                'names:\n  0: person\n' % os.path.abspath(a.out))
    tr = sum(v[1] for v in stats.values() if v[0] == 'train')
    va = sum(v[1] for v in stats.values() if v[0] == 'val')
    print(f'\n{tr} train / {va} val images -> {a.out}')
    if not va:
        sys.exit('validation split is empty - check --val names')


if __name__ == '__main__':
    main()
