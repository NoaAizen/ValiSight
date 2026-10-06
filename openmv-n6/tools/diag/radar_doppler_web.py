#!/usr/bin/env python3
"""Live Doppler view: is the velocity axis actually fixed?

    ./radar_doppler_web.py --config radar_people
    then open http://<host>:8082/

The whole radar-AI plan rests on one config change, and on one question about
it that a static scene cannot answer. `chirp.py` derives v_max +-4.99 m/s and
the recorded Doppler STEP already matches the derivation to 0.0002 m/s -- but a
step is not a span. Under the old config the axis was there too; it was only
+-0.649 m/s wide, so a walker folded into it and read as static.

So this page reports one number and everything else supports it:

    the largest |v| the radar has reported

If it climbs past 0.649, the old config could not have produced this reading
and the fix is real. If it climbs to within a few percent of 4.99 the new
ceiling is in sight too and the target should be slowed down, because at that
point a reading is once again ambiguous.

WHAT THIS PAGE CANNOT TELL YOU. That the reading is CORRECT -- only that it is
unfolded. A velocity is confirmed by an independent measurement of the same
motion, which here means a tape measure and a stopwatch, or the camera channel.
And nothing here says anything about range: the exit gate wants a walker at
12 m, and no amount of waving at 1.5 m substitutes for it.
"""
import argparse
from pathlib import Path
import collections
import json
import os
import sys
import threading
import time

import serial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'radar'))
import mmwave  # noqa: E402

DATA_PORT = '/dev/ttyACM2'
BAUD = 921600
HTTP_PORT = 8082

# The v_max of radar_10hz.cfg. Not a threshold on the target -- a threshold on
# the CONFIG: any |v| above it is a reading the old profile could not have
# produced without folding, so it is the line that proves the change took.
OLD_V_MAX = 0.649

# Below this the point is one of the room's static returns, and including them
# would bury the moving target in a histogram of walls.
MOVING = 0.02


class State:
    def __init__(self, limits):
        self.lock = threading.Lock()
        self.limits = limits
        self.max_abs_v = 0.0
        self.max_v_range = 0.0
        self.frames = 0
        self.frames_with_motion = 0
        self.above_old = 0          # moving points the old config would fold
        self.moving_points = 0
        self.near_ceiling = 0       # within 2% of the NEW v_max: ambiguous again
        self.points = []
        self.fps = 0.0
        self.margin_us = None
        self.dropped = 0
        self.hist = collections.Counter()
        self.err = None

    def snapshot(self):
        with self.lock:
            v = self.limits['v_max_m_s']
            return {
                'v_max': v,
                'old_v_max': OLD_V_MAX,
                'max_abs_v': self.max_abs_v,
                'max_v_range': self.max_v_range,
                'frames': self.frames,
                'frames_with_motion': self.frames_with_motion,
                'moving_points': self.moving_points,
                'above_old': self.above_old,
                'near_ceiling': self.near_ceiling,
                'points': self.points,
                'fps': self.fps,
                'margin_us': self.margin_us,
                'dropped': self.dropped,
                'hist': sorted(self.hist.items()),
                'proven': self.max_abs_v > OLD_V_MAX,
                'err': self.err,
            }


def reader(ser, st):
    sync = mmwave.FrameSync()
    t0 = time.time()
    n0 = 0
    try:
        while True:
            data = ser.read(4096)
            t = time.time()
            if not data:
                time.sleep(0.002)
                continue
            for t_frame, frame in sync.feed(data, t):
                try:
                    fr = mmwave.parse_frame(frame)
                except ValueError:
                    continue
                pts = fr['points']
                moving = [p for p in pts if abs(p['v']) >= MOVING]
                with st.lock:
                    st.frames += 1
                    if moving:
                        st.frames_with_motion += 1
                    st.moving_points += len(moving)
                    for p in moving:
                        a = abs(p['v'])
                        if a > st.max_abs_v:
                            st.max_abs_v = a
                            st.max_v_range = mmwave.range_of(p)
                        if a > OLD_V_MAX:
                            st.above_old += 1
                        if a > 0.98 * st.limits['v_max_m_s']:
                            st.near_ceiling += 1
                        # 0.25 m/s buckets, signed: approaching and receding are
                        # different motions and averaging them hides a walker
                        # who only ever shows up on one side.
                        st.hist[round(p['v'] / 0.25) * 0.25] += 1
                    st.points = [{
                        'r': round(mmwave.range_of(p), 2),
                        'v': round(p['v'], 3),
                        'snr': p['snr'],
                        'az': round(__import__('math').degrees(
                            __import__('math').atan2(p['y'], p['x'])), 1),
                    } for p in sorted(pts, key=lambda q: -abs(q['v']))[:12]]
                    if fr['stats']:
                        st.margin_us = fr['stats']['interframe_margin_us']
                    st.dropped = sync.dropped_bytes
                    if t - t0 >= 1.0:
                        st.fps = (st.frames - n0) / (t - t0)
                        t0, n0 = t, st.frames
    except Exception as e:                      # noqa: BLE001 -- surfaced on the page
        with st.lock:
            st.err = '%s: %s' % (type(e).__name__, e)


PAGE = (
    Path(__file__).resolve().parents[1] / "web" / "radar_doppler.html"
).read_text(encoding="utf-8")


def make_handler(st):
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith('/data'):
                body = json.dumps(st.snapshot()).encode()
                ctype = 'application/json'
            else:
                body = PAGE.encode()
                ctype = 'text/html; charset=utf-8'
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return H


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default=DATA_PORT)
    ap.add_argument('--config', default='radar_people')
    ap.add_argument('--http', type=int, default=HTTP_PORT)
    a = ap.parse_args()

    lim = mmwave.use_config(a.config)
    st = State(lim)

    ser = serial.Serial(a.port, BAUD, timeout=0.05)
    threading.Thread(target=reader, args=(ser, st), daemon=True).start()

    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(('0.0.0.0', a.http), make_handler(st))
    print('%s: v_max +-%.3f m/s (old %.3f)' % (a.config, lim['v_max_m_s'],
                                               OLD_V_MAX))
    print('open http://localhost:%d/   (Ctrl-C to stop)' % a.http)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nmax |v| reported: %.3f m/s at %.2f m' % (st.max_abs_v,
                                                          st.max_v_range))
    return 0


if __name__ == '__main__':
    sys.exit(main())
