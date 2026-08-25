#!/usr/bin/env python3
"""Score a detector against a YOLO-format hand-labelled set (P/R/mAP@0.5).

The point is comparability: the same script, the same held-out session and
the same matching rule score the stock detector and any fine-tuned successor,
so "it improved" is a number rather than an impression.

Usage:
    python3 tools/eval_detector.py --data /tmp/yolo_ds --split val
    python3 tools/eval_detector.py --data /tmp/yolo_ds --engine my_new.engine
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_gt(label_path, w, h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    for line in open(label_path):
        p = line.split()
        if len(p) != 5:
            continue
        cx, cy, bw, bh = (float(x) for x in p[1:])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def iou_matrix(pred, gt):
    if not len(pred) or not len(gt):
        return np.zeros((len(pred), len(gt)))
    p = np.asarray(pred, float)[:, None, :]
    g = np.asarray(gt, float)[None, :, :]
    iw = np.clip(np.minimum(p[..., 2], g[..., 2]) - np.maximum(p[..., 0], g[..., 0]), 0, None)
    ih = np.clip(np.minimum(p[..., 3], g[..., 3]) - np.maximum(p[..., 1], g[..., 1]), 0, None)
    inter = iw * ih
    ap = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    ag = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    return inter / np.maximum(ap + ag - inter, 1e-9)


def average_precision(scores, matched, n_gt):
    """Standard all-point-interpolated AP over the detection list."""
    if n_gt == 0:
        return float('nan')
    order = np.argsort(-np.asarray(scores))
    m = np.asarray(matched)[order]
    tp = np.cumsum(m)
    fp = np.cumsum(1 - m)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    ap, prev_r = 0.0, 0.0
    for r, pr in zip(recall, precision):
        ap += (r - prev_r) * pr
        prev_r = r
    return float(ap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='YOLO dataset dir')
    ap.add_argument('--split', default='val')
    ap.add_argument('--engine', help='TensorRT engine (default: the live one)')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    import detect
    det = detect.make_detector(backend='gpu', conf=a.conf, classes=['person'],
                               engine=a.engine)
    print(f'detector: {getattr(det, "model_name", "?")}/{det.backend}'
          + (f' engine={a.engine}' if a.engine else ''), file=sys.stderr)

    imgs = sorted(glob.glob(os.path.join(a.data, 'images', a.split, '*.jpg')))
    if a.limit:
        imgs = imgs[:a.limit]
    scores, matched, n_gt = [], [], 0
    empty_frames = fp_on_empty = 0

    for k, ip in enumerate(imgs):
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        gt = load_gt(os.path.join(a.data, 'labels', a.split,
                                  os.path.basename(ip)[:-4] + '.txt'), w, h)
        n_gt += len(gt)
        dets = [d for d in det(img) if d['cls'] == 'person']
        pred = [[d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h']] for d in dets]
        conf = [d['conf'] for d in dets]
        if not gt:
            empty_frames += 1
            fp_on_empty += bool(pred)
        M = iou_matrix(pred, gt)
        taken = set()
        for j in np.argsort(-np.asarray(conf)) if conf else []:
            best, bi = 0.0, -1
            for g in range(len(gt)):
                if g in taken:
                    continue
                if M[j, g] > best:
                    best, bi = M[j, g], g
            hit = best >= a.iou
            if hit:
                taken.add(bi)
            scores.append(conf[j])
            matched.append(1 if hit else 0)
        if (k + 1) % 250 == 0:
            print(f'  {k + 1}/{len(imgs)}', file=sys.stderr)

    tp = int(sum(matched))
    fp = len(matched) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_gt, 1)
    print(f'\nimages {len(imgs)} | gt boxes {n_gt} | predictions {len(matched)}')
    print(f'precision {precision:.3f}')
    print(f'recall    {recall:.3f}')
    print(f'F1        {2 * precision * recall / max(precision + recall, 1e-9):.3f}')
    print(f'mAP@0.5   {average_precision(scores, matched, n_gt):.3f}')
    if empty_frames:
        print(f'false alarms on {fp_on_empty}/{empty_frames} verified-empty '
              f'frames ({100 * fp_on_empty / empty_frames:.0f}%)')


if __name__ == '__main__':
    main()
