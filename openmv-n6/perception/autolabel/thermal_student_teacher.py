#!/usr/bin/env python3
"""Auto-label a recorded session with the TRAINED THERMAL STUDENT as teacher.

The yolo teacher reads the visible camera, so it cannot label darkness - and
darkness is exactly where the radar student needs training data. The thermal
student does not care about light: this tool runs its TensorRT engine over a
session's raw thermal stream, maps every detection to the visible plane
through the warp LUT (the same correspondence the fused view is built with),
and writes a normal <sess>_teacher.jsonl that build_dataset and export_shards
consume unchanged.

Video is never touched: the thermal frames come straight off thermal.bin, so
an unfinalized mp4 or a pitch-black recording costs nothing.

Requirements:
  - the session was recorded with a pinned range (c_per_lsb + tmin in
    meta.json) - the student was trained on Celsius and refuses unknown scale
  - thermal_student.engine built on this machine (trt_students.ENGINE_DIR)

Usage:
    python3 perception/autolabel/thermal_student_teacher.py captures/dark1
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from perception.dataset import LiveSession                       # noqa: E402
from perception.autolabel.thermal_check import ThermalBoxCheck   # noqa: E402
import trt_students                                              # noqa: E402

LUT_DEFAULT = os.path.join(ROOT, 'calib-artifacts', 'warp_3.5.lut')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--out', default=os.path.join(ROOT, 'perception', 'out',
                                                  'autolabel'))
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--engine', help='thermal_student.engine override')
    ap.add_argument('--lut', default=LUT_DEFAULT)
    ap.add_argument('--force', action='store_true',
                    help='overwrite an existing teacher file')
    a = ap.parse_args()

    sess = LiveSession(a.session)
    name = os.path.basename(os.path.normpath(a.session))
    c_per_lsb = sess.meta.get('c_per_lsb')
    tmin = sess.meta.get('tmin')
    if c_per_lsb is None or tmin is None:
        sys.exit(f'refusing {a.session}: no c_per_lsb/tmin in meta.json - '
                 'the student was trained on Celsius; record with a pinned '
                 '--range')
    counts_max = sess.meta.get('thermal_counts_max') or 255

    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, f'{name}_teacher.jsonl')
    if os.path.exists(out_path) and not a.force:
        sys.exit(f'{out_path} exists - remove it or pass --force')

    model = trt_students.ThermalStudentTrt(engine=a.engine, conf=a.conf)
    th2vis = trt_students.ThermalToVisible(a.lut)
    check = ThermalBoxCheck(lut_path=a.lut, counts_max=counts_max)

    n_frames = n_dets = n_unmapped = 0
    t0 = time.time()
    with open(out_path, 'w') as f:
        for meta in sess.frames:
            raw = sess.thermal_frame(meta)
            if raw is None:
                continue
            celsius = raw.astype(np.float32) * float(c_per_lsb) + float(tmin)
            tdets = model.push(celsius, meta['t_mono'] * 1e3)

            dets = []
            for d in tdets:
                vis = th2vis.box(d['x'], d['y'], d['w'], d['h'])
                if vis is None:
                    # A person the LUT cannot place on the visible frame is
                    # outside the overlap ROI; the RGB-plane label set cannot
                    # honestly contain it.
                    n_unmapped += 1
                    continue
                x, y, w, h = vis
                det = {'cls': 'person', 'conf': float(d['conf']),
                       'x': x, 'y': y, 'w': w, 'h': h,
                       'th': check.region(raw, x, y, w, h)}
                dets.append(det)
            f.write(json.dumps({
                'i': meta['i'], 't_mono': meta['t_mono'],
                'bg': check.background(raw),
                'thermal_counts_max': counts_max,
                'thermal_delta_unit': 'uint8_equivalent',
                'teacher': 'thermal_student_trt',
                'dets': dets}) + '\n')
            n_frames += 1
            n_dets += len(dets)
            if n_frames % 500 == 0:
                print(f'  {n_frames} frames...', file=sys.stderr)

    dt = time.time() - t0
    print(f'{name}: {n_frames} frames, {n_dets} person boxes '
          f'({n_unmapped} outside the visible overlap), '
          f'{dt:.1f}s ({1e3 * dt / max(n_frames, 1):.1f} ms/frame) '
          f'-> {out_path}')


if __name__ == '__main__':
    main()
