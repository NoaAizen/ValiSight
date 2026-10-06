#!/usr/bin/env python3
"""Browser-based manual person labeling for a recorded session.

The human replaces the yolov10n teacher for one session: draw person boxes on
the RGB video frames, and manual_to_teacher.py turns them into a
<sess>_teacher.jsonl that the normal D1 chain (build_dataset -> export)
consumes unchanged - manual labels ARE a teacher, one with conf 1.0.

Labeling every frame is wasted effort: consecutive frames are near-duplicates
and the tracker links boxes across gaps up to 8 frames (MAX_GAP), so the
default stride of 3 keeps tracks intact at a third of the work. Frames you
navigate to and leave WITHOUT boxes are saved as explicitly empty - that is a
statement ("no person here"), not a skip; frames never visited stay unknown
and are simply absent from the output.

If the machine teacher already ran on this session its boxes are shown dashed
as suggestions - 'a' accepts them onto the frame for correction, which is much
faster than drawing from scratch.

Usage:
    ./label_web.py ../captures/TEST11 [--port 8090] [--stride 3]
then open http://localhost:8090 (or the Jetson's address from another machine).

Keys: ->/Space save+next   <- save+prev   a accept teacher boxes
      c copy from last labeled frame      x clear frame (explicit empty)
      Delete remove selected box          g go to frame
      u jump to next unlabeled frame
"""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from perception.dataset import LiveSession   # noqa: E402
from web.assets import read_text

AUTOLABEL = os.path.join(ROOT, 'perception', 'out', 'autolabel')


def extract_frames(sess, cache_dir):
    """One JPEG per video frame, once - VideoCapture seeking is not trusted
    (the repaired elementary streams have no index), so random access in the
    browser needs the frames on disk."""
    os.makedirs(cache_dir, exist_ok=True)
    have = len([f for f in os.listdir(cache_dir) if f.endswith('.jpg')])
    if have >= min(len(sess.frames), 1):
        # a previous run got at least this far; re-extract only if empty
        if have >= len(sess.frames) - 32:   # tolerate a lost video tail
            return have
    cap = sess._open_video()
    n = 0
    try:
        for meta in sess.frames:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(os.path.join(cache_dir, '%05d.jpg' % meta['i']),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            n += 1
            if n % 200 == 0:
                print('  extracted %d/%d' % (n, len(sess.frames)),
                      file=sys.stderr)
    finally:
        cap.release()
    return n


def load_teacher(sess_name):
    # Suggestions must come from the MACHINE teacher. After manual_to_teacher
    # has run on a session, <sess>_teacher.jsonl IS the manual labels (moved
    # machine file: <sess>.teacher-machine.jsonl) - feeding those back as
    # "suggestions" shows nothing on any frame the human has not labeled yet.
    path = os.path.join(AUTOLABEL, sess_name + '.teacher-machine.jsonl')
    if not os.path.exists(path):
        path = os.path.join(AUTOLABEL, sess_name + '_teacher.jsonl')
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            boxes = [{'x': d['x'], 'y': d['y'], 'w': d['w'], 'h': d['h']}
                     for d in r['dets'] if d['cls'] == 'person']
            if boxes:
                out[r['i']] = boxes
    return out


PAGE = read_text("label.html")


class Handler(BaseHTTPRequestHandler):
    # filled in main()
    ctx = None
    # ThreadingHTTPServer handles saves concurrently; without the lock two
    # rapid navigations both write store+'.tmp' and the loser's os.replace
    # finds it already gone
    save_lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        if ctype == 'image/jpeg':
            # extracted frames never change; without this the browser
            # re-downloads every frame it re-visits
            self.send_header('Cache-Control', 'max-age=86400')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        c = self.ctx
        if self.path == '/':
            page = PAGE % {'name': c['name'], 'w': c['w'], 'h': c['h'],
                           'last': c['last'], 'stride': c['stride'],
                           'start': c['start']}
            return self._send(200, page.encode(), 'text/html; charset=utf-8')
        if self.path.startswith('/frame/'):
            p = os.path.join(c['cache'], '%05d.jpg' % int(self.path[7:]))
            if not os.path.exists(p):
                return self._send(404, b'{}')
            return self._send(200, open(p, 'rb').read(), 'image/jpeg')
        if self.path.startswith('/boxes/'):
            i = int(self.path[7:])
            return self._send(200, json.dumps({
                'manual': c['manual'].get(str(i)),
                'teacher': c['teacher'].get(i, []),
                'n_labeled': len(c['manual'])}).encode())
        if self.path.startswith('/unlabeled/'):
            i = int(self.path[11:])
            # next stride-slot with no manual entry ANYWHERE in [t, t+stride),
            # searching forward from i then wrapping; null when done. Slot
            # coverage (not exact-index) tolerates labels laid on a shifted
            # grid - a label at t+1 covers slot t, re-labeling t is waste.
            s = c['stride']

            def covered(t):
                return any(str(t + o) in c['manual'] for o in range(s))
            targets = range(0, c['last'] + 1, s)
            nxt = next((t for t in targets if t > i and not covered(t)),
                       next((t for t in targets if not covered(t)), None))
            return self._send(200, json.dumps({'i': nxt}).encode())
        if self.path.startswith('/prev/'):
            i = int(self.path[6:])
            prev = [int(k) for k in c['manual'] if int(k) < i]
            if not prev:
                return self._send(200, b'{"boxes": null}')
            return self._send(200, json.dumps(
                {'boxes': c['manual'][str(max(prev))]}).encode())
        self._send(404, b'{}')

    def do_POST(self):
        c = self.ctx
        if not self.path.startswith('/boxes/'):
            return self._send(404, b'{}')
        i = int(self.path[7:])
        n = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(n))
        boxes = [{'x': int(b['x']), 'y': int(b['y']),
                  'w': int(b['w']), 'h': int(b['h'])}
                 for b in body['boxes']]
        with self.save_lock:
            c['manual'][str(i)] = boxes
            tmp = c['store'] + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'stride': c['stride'], 'boxes': c['manual']}, f)
            os.replace(tmp, c['store'])
        self._send(200, b'{"ok": true}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--stride', type=int, default=3,
                    help='label every Nth frame (must stay <= 8, the tracker '
                         'MAX_GAP, or tracks break apart)')
    a = ap.parse_args()
    if a.stride > 8:
        sys.exit('stride > 8 breaks track linking (MAX_GAP) - refusing')

    sess = LiveSession(a.session)
    name = os.path.basename(os.path.normpath(a.session))
    cache = os.path.join(a.session, '.label_frames')
    print('extracting frames (first run only)...', file=sys.stderr)
    n = extract_frames(sess, cache)
    sample = cv2.imread(os.path.join(cache, '%05d.jpg' % sess.frames[0]['i']))
    if sample is None:
        sys.exit('no frames extracted - is the video readable?')

    store = os.path.join(a.session, 'manual_boxes.json')
    manual = {}
    if os.path.exists(store):
        manual = json.load(open(store)).get('boxes', {})
        print('resuming: %d frames already labeled' % len(manual),
              file=sys.stderr)
    labeled = sorted(int(k) for k in manual)
    start = labeled[-1] if labeled else 0

    Handler.ctx = {'name': name, 'cache': cache, 'store': store,
                   'manual': manual, 'teacher': load_teacher(name),
                   'w': sample.shape[1], 'h': sample.shape[0],
                   'last': n - 1, 'stride': a.stride, 'start': start}
    srv = ThreadingHTTPServer(('0.0.0.0', a.port), Handler)
    print('labeling %s (%d frames) on http://localhost:%d' % (name, n, a.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\n%d frames labeled -> %s' % (len(manual), store))


if __name__ == '__main__':
    main()
