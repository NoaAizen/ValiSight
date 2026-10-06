"""Prepare thermal COCO annotations for multiclass YOLO training.

Unlike public_thermal.py, this preserves vehicle/animal labels. It exports
native thermal images; it does not mix RGB, radiometric Celsius or person-only
student shards into an intensity-trained detector.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2

CLASSES = ('person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
           'dog', 'cat', 'horse', 'cow', 'sheep', 'deer')
ALIASES = {'motorbike': 'motorcycle', 'motor': 'motorcycle', 'bike': 'bicycle'}


def collect(root, annotations, classes=CLASSES):
    root = Path(root).resolve()
    data = json.loads(Path(annotations).read_text())
    categories = {c['id']: ALIASES.get(c['name'].lower(), c['name'].lower())
                  for c in data['categories']}
    by_image = {im['id']: [] for im in data['images']}
    for a in data['annotations']:
        if a['image_id'] not in by_image:
            raise ValueError('Annotation references an unknown image')
        name = categories[a['category_id']]
        if name in classes:
            # YOLO text labels cannot express COCO crowd/ignore regions. Do
            # not silently turn them into negatives or ordinary individual boxes.
            if a.get('iscrowd') or a.get('ignore'):
                raise ValueError('Crowd/ignore annotations require explicit handling')
            by_image[a['image_id']].append((name, a['bbox']))
    result = []
    for im in data['images']:
        path = (root / im['file_name']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image path escapes the supplied root')
        pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if pixels is None:
            raise ValueError(f'Missing or unreadable thermal image: {path}')
        if pixels.dtype.name != 'uint8':
            raise ValueError('Expected 8-bit thermal intensity, not raw radiometric words')
        if pixels.ndim == 3:
            if pixels.shape[2] != 3 or not ((pixels[..., 0] == pixels[..., 1]).all()
                                          and (pixels[..., 0] == pixels[..., 2]).all()):
                raise ValueError('Expected grayscale thermal images, not RGB or palettes')
            pixels = pixels[..., 0]
        h, w = pixels.shape
        if (w, h) != (im['width'], im['height']):
            raise ValueError(f'Annotation/image size mismatch: {path}')
        lines = []
        for name, box in by_image[im['id']]:
            x, y, bw, bh = map(float, box)
            if not all(math.isfinite(v) for v in (x, y, bw, bh)):
                raise ValueError('Nonfinite box coordinate')
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + bw), min(h, y + bh)
            if x1 <= x0 or y1 <= y0:
                raise ValueError('Empty or out-of-frame annotation')
            lines.append((classes.index(name), (x0 + x1) / (2*w),
                          (y0 + y1) / (2*h), (x1 - x0) / w, (y1 - y0) / h))
        digest = hashlib.sha256(f'{w}x{h}:'.encode() + pixels.tobytes()).hexdigest()
        result.append((path, digest, lines))
    return result


def export(train_root, train_annotations, val_root, val_annotations, out, classes=CLASSES):
    if isinstance(classes, str):
        classes = tuple(classes.split(','))
    if not classes or len(set(classes)) != len(classes) or set(classes) - set(CLASSES):
        raise ValueError('Classes must be unique supported names')
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('Output already exists; choose a new directory')
    splits = {'train': collect(train_root, train_annotations, classes),
              'val': collect(val_root, val_annotations, classes)}
    if not all(splits.values()):
        raise ValueError('Both training and validation images are required')
    if {r[1] for r in splits['train']} & {r[1] for r in splits['val']}:
        raise ValueError('The same image occurs in training and validation')
    counts = {}
    for split, rows in splits.items():
        (out / 'images' / split).mkdir(parents=True)
        (out / 'labels' / split).mkdir(parents=True)
        counts[split] = dict.fromkeys(classes, 0)
        for i, (path, digest, labels) in enumerate(rows):
            stem = f'{i:06d}_{digest[:12]}'
            pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if not cv2.imwrite(str(out / 'images' / split / (stem + '.png')), pixels):
                raise OSError('Could not write training image')
            text = '\n'.join(str(row[0]) + ' ' + ' '.join(f'{v:.8f}' for v in row[1:])
                             for row in labels)
            (out / 'labels' / split / (stem + '.txt')).write_text(text + ('\n' if text else ''))
            for row in labels:
                counts[split][classes[row[0]]] += 1
    (out / 'data.yaml').write_text('path: ' + json.dumps(str(out)) + '\n'
                                  'train: images/train\nval: images/val\n'
                                  'names: ' + json.dumps(list(classes)) + '\n')
    report = {'classes': list(classes), 'box_counts': counts,
              'image_counts': {split: len(rows) for split, rows in splits.items()},
              'missing_training_classes': [n for n in classes if not counts['train'][n]],
              'missing_validation_classes': [n for n in classes if not counts['val'][n]],
              'train_annotations': str(Path(train_annotations).resolve()),
              'val_annotations': str(Path(val_annotations).resolve()),
              'note': 'A configured class with zero examples is NOT a detection capability. '
                      'Validate on held-out Lepton 160x120 night recordings before deployment.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('train-root', 'train-annotations', 'val-root', 'val-annotations', 'out'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--classes', default=','.join(CLASSES),
                        help='Comma-separated supported class names, in output model order')
    print(json.dumps(export(**vars(parser.parse_args())), indent=2))
