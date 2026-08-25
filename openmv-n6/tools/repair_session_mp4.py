#!/usr/bin/env python3
"""Repair an unfinalized session.mp4 into a decodable session_es.m4v.

live.py's recorder is currently never reaching VideoWriter.release() (every
session recorded 2026-08-23 came out without a moov atom), so the mp4 is a
bare ftyp+free+mdat and nothing will open it. The mdat payload itself is a
healthy MPEG-4 part 2 elementary stream minus its VOS/VOL headers - those
live in the moov's esds box, which was never written.

The repair: encode a 3-frame dummy at the recording resolution with the same
cv2 'mp4v' encoder, lift the VOS..VOL+userdata prefix out of the dummy's esds
(stopping before the 0x06 SLConfig descriptor - including it corrupts the
decode), and prepend it to the mdat payload. perception/dataset.py falls back
to session_es.m4v automatically, so nothing downstream changes.

The header must match the recorded resolution: a prefix from a different
geometry decodes with ac-tex damage everywhere (that is how holds1's prefix
failed on test7). 640x480 is what live.py records today; pass --size if that
ever changes.

Usage:
    ./repair_session_mp4.py captures/aug23-walk-far-01 [more sessions...]
    ./repair_session_mp4.py --size 640x480 captures/foo
"""
import argparse
import os
import sys
import tempfile

import cv2
import numpy as np


def vol_prefix(w, h):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'dummy.mp4')
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (w, h))
        if not vw.isOpened():
            sys.exit('cannot open a cv2 VideoWriter - no mp4v encoder?')
        for _ in range(3):
            vw.write(np.zeros((h, w, 3), np.uint8))
        vw.release()
        d = open(p, 'rb').read()
    i = d.find(b'\x00\x00\x01\xb0')          # visual object sequence start
    j = d.find(b'Lavc', i)                   # encoder userdata ends the config
    if i < 0 or j < 0:
        sys.exit('no VOS/userdata in dummy esds - ffmpeg build changed?')
    j = d.index(b'\x06', j)                  # SLConfig tag: config stops here
    return d[i:j]


def repair(sess, prefix, force):
    mp4 = os.path.join(sess, 'session.mp4')
    es = os.path.join(sess, 'session_es.m4v')
    if not os.path.exists(mp4):
        return '%s: no session.mp4' % sess
    cap = cv2.VideoCapture(mp4, cv2.CAP_FFMPEG)
    ok = cap.isOpened()
    cap.release()
    if ok:
        return '%s: session.mp4 is healthy, nothing to repair' % sess
    if os.path.exists(es) and not force:
        return '%s: session_es.m4v already there (--force to redo)' % sess

    raw = open(mp4, 'rb').read()
    k = raw.find(b'mdat')
    if k < 0 or raw[k + 4:k + 8] != b'\x00\x00\x01\xb3':
        return '%s: mdat does not start at a GOV code - not the known failure, refusing' % sess
    with open(es, 'wb') as f:
        f.write(prefix + raw[k + 4:])

    cap = cv2.VideoCapture(es, cv2.CAP_FFMPEG)
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    idx = os.path.join(sess, 'frames.jsonl')
    ni = sum(1 for _ in open(idx)) if os.path.exists(idx) else -1
    if n == 0:
        os.remove(es)
        return '%s: repair decoded 0 frames, removed - wrong --size?' % sess
    return '%s: repaired, %d video frames (index has %d)' % (sess, n, ni)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sessions', nargs='+')
    ap.add_argument('--size', default='640x480',
                    help='recorded resolution WxH (default matches live.py today)')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split('x'))
    prefix = vol_prefix(w, h)
    for s in a.sessions:
        print(repair(s.rstrip('/'), prefix, a.force))


if __name__ == '__main__':
    main()
