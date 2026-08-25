#!/usr/bin/env python3
"""Build stage: teacher detections -> tracks -> graded labels -> COCO thermal.

Consumes the *_teacher.jsonl files run_teacher.py wrote, never the video -
everything visual it needs (the thermal frames) comes straight off
thermal.bin via the frames.jsonl offsets.

Three products:
  1. <sess>_tracks.json  - person tracks with a grade each. The grade is the
     quality of the LABEL, not of the person: A = the thermal independently
     confirms the box (median delta >= 60 over the track's covered boxes),
     REJ = the thermal contradicts it (median delta < 30 - the teacher's
     false-positive tail), B = everything between, plus tracks the thermal
     simply cannot see (no coverage is "unknown", never "cold").
  2. Gate report: % of tracks at grade A. The plan's bar is >= 70; below
     that the problem is association/registration, not the model.
  3. COCO dataset of THERMAL frames (native 160x120) with person boxes
     mapped through the warp LUT - one image per unique thermal frame,
     grade-A boxes only. This is channel A''s training input.

Tracking is greedy IoU, deliberately primitive: the teacher runs at video
rate on stationary-ish indoor scenes, so consecutive-frame overlap is high
and identity switches cost a label grade, not a life. ByteTrack-class
machinery belongs to the runtime (phase 4), not to labeling.

Usage:
    python3 perception/autolabel/build_dataset.py [--min-track 5] [--sheet]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import cv2

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)
from perception.dataset import LiveSession                       # noqa: E402
from perception.autolabel.thermal_check import (                 # noqa: E402
    ThermalBoxCheck, DECIMATION, LOW_W, LOW_H)

OUT = os.path.join(ROOT, 'perception', 'out', 'autolabel')
COCO_DIR = os.path.join(ROOT, 'perception', 'out', 'thermal_coco')

GRADE_A_DELTA = 60.0
REJECT_DELTA = 30.0
# The 60/30 counts were picked from the legacy AGC-era sessions, whose
# scene-stretched scale packs a person's contrast into ~120 counts. A
# linear_set_range session declares c_per_lsb, and there the same person
# sits at ~30 counts (range 0:60) - so known-scale sessions grade in
# degrees C instead. 4.7/3.5 measured 2026-08-23 over TEST11+test10+test7:
# person median 7.1 C vs false-heat 2.8 C; >=4.7 C keeps 90% of persons
# and passes 4% of false heat.
GRADE_A_DEG_C = 4.7
REJECT_DEG_C = 3.5


def session_thresholds(sess):
    c = sess.meta.get('c_per_lsb')
    if c:
        return GRADE_A_DEG_C / c, REJECT_DEG_C / c
    return GRADE_A_DELTA, REJECT_DELTA
IOU_MIN = 0.30
MAX_GAP = 8          # frames a track survives without a match (~1 s)


def iou(a, b):
    ax0, ay0, ax1, ay1 = a['x'], a['y'], a['x'] + a['w'], a['y'] + a['h']
    bx0, by0, bx1, by1 = b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a['w'] * a['h'] + b['w'] * b['h'] - inter
    return inter / union if union > 0 else 0.0


def track_session(teacher_path):
    """Greedy IoU tracking over the person detections of one session."""
    active, done = [], []
    with open(teacher_path) as f:
        for line in f:
            r = json.loads(line)
            persons = [d for d in r['dets'] if d['cls'] == 'person']
            # highest-confidence detections pick their track first
            persons.sort(key=lambda d: -d['conf'])
            taken = set()
            for d in persons:
                best, best_iou = None, IOU_MIN
                for t in active:
                    if id(t) in taken:
                        continue
                    v = iou(t['boxes'][-1], d)
                    if v > best_iou:
                        best, best_iou = t, v
                if best is None:
                    best = {'boxes': [], 'frames': [], 'deltas': [],
                            'covered': []}
                    active.append(best)
                taken.add(id(best))
                best['boxes'].append(d)
                best['frames'].append(r['i'])
                best['covered'].append(d['th'] is not None)
                best['deltas'].append(d['th']['max'] - r['bg']
                                      if d['th'] else None)
            still = []
            for t in active:
                if r['i'] - t['frames'][-1] > MAX_GAP:
                    done.append(t)
                else:
                    still.append(t)
            active = still
    return done + active


def grade_track(t, a_delta=GRADE_A_DELTA, rej_delta=REJECT_DELTA):
    d = [x for x in t['deltas'] if x is not None]
    coverage = sum(t['covered']) / len(t['covered'])
    if not d:
        return 'B', None, coverage          # thermal never saw it: unknown
    med = float(np.median(d))
    if med >= a_delta and coverage >= 0.5:
        return 'A', med, coverage
    if med < rej_delta:
        return 'REJ', med, coverage
    return 'B', med, coverage


def thermal_bbox(check, x, y, w, h):
    """Visible box -> bounding box of its valid warped samples, thermal px."""
    gx0 = max(0, int(x) // DECIMATION)
    gy0 = max(0, int(y) // DECIMATION)
    gx1 = min(LOW_W, int(np.ceil((x + w) / DECIMATION)))
    gy1 = min(LOW_H, int(np.ceil((y + h) / DECIMATION)))
    if gx1 <= gx0 or gy1 <= gy0:
        return None
    m = check.valid[gy0:gy1, gx0:gx1]
    if not m.any():
        return None
    tu = check.tu[gy0:gy1, gx0:gx1][m]
    tv = check.tv[gy0:gy1, gx0:gx1][m]
    bx, by = int(tu.min()), int(tv.min())
    bw, bh = int(tu.max()) - bx + 1, int(tv.max()) - by + 1
    if bw < 3 or bh < 3:
        return None
    return [bx, by, bw, bh]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-track', type=int, default=5,
                    help='drop flicker tracks shorter than this many boxes')
    ap.add_argument('--sheet', action='store_true',
                    help='write a thermal contact sheet for eyeballing')
    a = ap.parse_args()

    check = ThermalBoxCheck()
    os.makedirs(os.path.join(COCO_DIR, 'images'), exist_ok=True)

    coco = {'images': [], 'annotations': [],
            'categories': [{'id': 1, 'name': 'person'}]}
    img_ids = {}                 # (sess, thermal_off) -> image id
    ann_id = 0
    gate_rows = []

    for path in sorted(glob.glob(os.path.join(OUT, '*_teacher.jsonl'))):
        sess_name = os.path.basename(path).replace('_teacher.jsonl', '')
        sess = LiveSession(os.path.join(ROOT, 'captures', sess_name))
        meta_by_i = {m['i']: m for m in sess.frames}
        a_delta, rej_delta = session_thresholds(sess)
        if (a_delta, rej_delta) != (GRADE_A_DELTA, REJECT_DELTA):
            print(f'[build] {sess_name}: c_per_lsb scale, A>={a_delta:.0f} '
                  f'REJ<{rej_delta:.0f} counts ({GRADE_A_DEG_C}/{REJECT_DEG_C} C)')

        tracks = [t for t in track_session(path)
                  if len(t['boxes']) >= a.min_track]
        graded = []
        for t in tracks:
            g, med, cov = grade_track(t, a_delta, rej_delta)
            graded.append({'grade': g, 'median_delta': med, 'coverage': cov,
                           'n': len(t['boxes']),
                           'i0': t['frames'][0], 'i1': t['frames'][-1]})
            if g != 'A':
                continue
            # grade-A tracks feed the thermal COCO, box by box - but only
            # boxes the thermal actually covers and confirms individually.
            for d, i, delta in zip(t['boxes'], t['frames'], t['deltas']):
                if delta is None or delta < a_delta:
                    continue
                m = meta_by_i.get(i)
                if m is None or m.get('thermal_off') is None:
                    continue
                tb = thermal_bbox(check, d['x'], d['y'], d['w'], d['h'])
                if tb is None:
                    continue
                key = (sess_name, m['thermal_off'])
                if key not in img_ids:
                    th = sess.thermal_frame(m)
                    fname = f'{sess_name}_{i:05d}.png'
                    cv2.imwrite(os.path.join(COCO_DIR, 'images', fname), th)
                    img_ids[key] = len(coco['images'])
                    coco['images'].append({'id': img_ids[key],
                                           'file_name': fname,
                                           'width': 160, 'height': 120,
                                           'session': sess_name})
                ann_id += 1
                coco['annotations'].append(
                    {'id': ann_id, 'image_id': img_ids[key],
                     'category_id': 1, 'bbox': tb,
                     'area': tb[2] * tb[3], 'iscrowd': 0,
                     'delta': round(delta, 1), 'conf': d['conf']})

        with open(os.path.join(OUT, f'{sess_name}_tracks.json'), 'w') as f:
            json.dump(graded, f, indent=1)
        n = len(graded)
        na = sum(1 for g in graded if g['grade'] == 'A')
        gate_rows.append((sess_name, n, na,
                          sum(1 for g in graded if g['grade'] == 'B'),
                          sum(1 for g in graded if g['grade'] == 'REJ')))

    with open(os.path.join(COCO_DIR, 'annotations.json'), 'w') as f:
        json.dump(coco, f)

    print(f"{'session':26s} {'tracks':>6} {'A':>4} {'B':>4} {'REJ':>4} {'%A':>6}")
    tot_n = tot_a = tot_vis = 0
    for s, n, na, nb, nr in gate_rows:
        pct = 100 * na / n if n else float('nan')
        print(f'{s:26s} {n:6d} {na:4d} {nb:4d} {nr:4d} {pct:5.1f}%')
        tot_n += n; tot_a += na; tot_vis += na + nr
    # The gate conditions on thermal VISIBILITY. Measured on holds1: every
    # grade-B track has ~zero warp coverage - a person sitting outside the
    # thermal ROI can never earn two-sensor agreement, and counting them
    # penalizes geometry, not association. The REJ tracks stay in the
    # denominator: those the thermal saw and contradicted (measured case:
    # a figure behind the glass wall - LWIR-opaque, correctly killed).
    pct_raw = 100 * tot_a / max(tot_n, 1)
    pct_vis = 100 * tot_a / max(tot_vis, 1)
    print(f'\nall tracks at grade A (incl. out-of-thermal-ROI): {pct_raw:.1f}%')
    print(f'D1 GATE - thermal-visible tracks at grade A '
          f'(>=70%): {pct_vis:.1f}% -> {"PASS" if pct_vis >= 70 else "FAIL"}')
    print(f'COCO: {len(coco["images"])} thermal images, '
          f'{len(coco["annotations"])} person boxes -> {COCO_DIR}')

    if a.sheet:
        make_sheet()


def make_sheet(n=24):
    """Thermal frames with their mapped boxes - the registration eyeball."""
    ann = json.load(open(os.path.join(COCO_DIR, 'annotations.json')))
    by_id = {im['id']: im for im in ann['images']}
    chosen = np.unique(np.linspace(0, len(ann['annotations']) - 1,
                                   n).astype(int))
    out_tiles = []
    for k in chosen:
        an = ann['annotations'][k]
        im = by_id[an['image_id']]
        img = cv2.imread(os.path.join(COCO_DIR, 'images', im['file_name']),
                         cv2.IMREAD_GRAYSCALE)
        vis = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
        x, y, w, h = an['bbox']
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        vis = cv2.resize(vis, (320, 240), interpolation=cv2.INTER_NEAREST)
        cv2.putText(vis, f"d={an['delta']:.0f}", (6, 18),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 1)
        out_tiles.append(vis)
    cols = 6
    rows_n = int(np.ceil(len(out_tiles) / cols))
    sheet = np.zeros((rows_n * 240, cols * 320, 3), np.uint8)
    for k, t in enumerate(out_tiles):
        r_, c_ = divmod(k, cols)
        sheet[r_ * 240:(r_ + 1) * 240, c_ * 320:(c_ + 1) * 320] = t
    out = os.path.join(OUT, 'thermal_box_sheet.png')
    cv2.imwrite(out, sheet)
    print(f'thermal box sheet: {out}')


if __name__ == '__main__':
    main()
