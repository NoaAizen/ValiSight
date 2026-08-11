#!/usr/bin/env python3
"""B2 detect pass for the FOIL board (RGB<->thermal stereo pairs).

Differences from calib.py detect:
  - RGB is median-blurred before detection: the foil squares' wrinkle
    texture defeats both chessboard detectors on the raw image; a 9px
    median suppresses the mottle and leaves the square structure
    (measured on captures/foil_preview2: raw=no, median9=FOUND).
  - Thermal is upscaled 4x and locally normalized (the hot ceiling lights
    otherwise own the 8-bit range and flatten the board's contrast).
  - Corners from the blurred RGB are refined with cornerSubPix on the
    ORIGINAL image so the blur does not shift the geometry.

Output: corners.json in the same schema calib.py solve consumes.

    ./b2_detect.py captures/b2_all --pattern 6x7 --square 0.0383 -o corners_b2.json
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib import detect_corners, RGB_W, RGB_H, TH_W, TH_H  # noqa: E402


def detect_rgb(rgb, pattern):
    c = detect_corners(cv2.medianBlur(rgb, 9), pattern)
    if c is None:
        c = detect_corners(cv2.GaussianBlur(rgb, (0, 0), 5), pattern)
    if c is None:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    return cv2.cornerSubPix(rgb, c.astype(np.float32).reshape(-1, 1, 2),
                            (5, 5), (-1, -1), crit).reshape(-1, 2)


def detect_th(th, pattern):
    big = cv2.resize(th, (TH_W * 4, TH_H * 4), interpolation=cv2.INTER_CUBIC)
    for im in (big, cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX),
               cv2.createCLAHE(3.0, (8, 8)).apply(big)):
        c = detect_corners(im, pattern)
        if c is not None:
            return c / 4.0
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pairs_dir')
    ap.add_argument('--pattern', default='6x7')
    ap.add_argument('--square', type=float, default=0.0383)
    ap.add_argument('-o', '--out', default='corners_b2.json')
    a = ap.parse_args()
    pat = tuple(int(v) for v in a.pattern.split('x'))

    out = {'pattern': list(pat), 'square_m': a.square, 'views': []}
    for jp in sorted(glob.glob(os.path.join(a.pairs_dir, '*.json'))):
        stem = jp[:-5]
        rgb_p = stem + '_rgb0.raw'
        if not os.path.exists(rgb_p):
            rgb_p = stem + '_rgb.raw'
        th_p = stem + '_thermal.raw'
        if not (os.path.exists(rgb_p) and os.path.exists(th_p)):
            continue
        rgb = np.fromfile(rgb_p, np.uint8).reshape(RGB_H, RGB_W)
        th = np.fromfile(th_p, np.uint8).reshape(TH_H, TH_W)
        c_rgb = detect_rgb(rgb, pat)
        c_th = detect_th(th, pat)
        name = os.path.basename(stem)
        if c_rgb is None or c_th is None:
            print('  %-14s skip (rgb=%s thermal=%s)' %
                  (name, c_rgb is not None, c_th is not None), file=sys.stderr)
            continue
        out['views'].append({'name': name, 'rgb': c_rgb.tolist(),
                             'thermal': c_th.tolist()})
        print('  %-14s ok' % name, file=sys.stderr)

    json.dump(out, open(a.out, 'w'), indent=1)
    print('%d stereo view(s) -> %s' % (len(out['views']), a.out))


if __name__ == '__main__':
    main()
