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


PAGE = """<!doctype html><meta charset=utf-8>
<title>Doppler gate</title>
<style>
 body{background:#111;color:#ddd;font:14px/1.5 ui-monospace,monospace;margin:0;padding:18px}
 h1{font-size:15px;color:#888;font-weight:400;margin:0 0 14px}
 .big{font-size:64px;line-height:1;margin:6px 0}
 .verdict{padding:12px 16px;border-radius:6px;margin:14px 0;font-size:15px}
 .no{background:#2a1a1a;border-left:4px solid #a44}
 .yes{background:#152a15;border-left:4px solid #4a4}
 .warn{background:#2a2415;border-left:4px solid #aa4}
 table{border-collapse:collapse;margin-top:8px}td,th{padding:2px 14px 2px 0;text-align:right}
 th{color:#777;font-weight:400}
 .bar{background:#2c4a6e;height:11px;display:inline-block;vertical-align:middle}
 .k{color:#777}
 .fold{color:#c66}
</style>
<h1>radar_people.cfg &mdash; live Doppler. Move radially: toward the sensor and away.</h1>
<div id=v></div>
<script>
function pct(x){return (100*x).toFixed(0)}
async function tick(){
 let d = await (await fetch('/data')).json();
 let el = document.getElementById('v');
 if(d.err){el.innerHTML='<div class="verdict no">reader died: '+d.err+'</div>';return}
 let verdict;
 if(d.near_ceiling>0)
   verdict='<div class="verdict warn"><b>near the new ceiling.</b> '+d.near_ceiling+
     ' points within 2% of &plusmn;'+d.v_max.toFixed(2)+' m/s &mdash; that is ambiguous '+
     'again. Slow down, or this reading folds too.</div>';
 else if(d.proven)
   verdict='<div class="verdict yes"><b>the fix is real.</b> '+d.above_old+
     ' points above the old &plusmn;'+d.old_v_max+' m/s ceiling. Under radar_10hz.cfg '+
     'every one of them would have folded and read as near-static.</div>';
 else
   verdict='<div class="verdict no"><b>not yet.</b> nothing above the old '+
     '&plusmn;'+d.old_v_max+' m/s. Either nothing is moving radially, or the profile '+
     'did not take. Walk straight at the sensor &mdash; sideways motion has no '+
     'radial component and reads as zero.</div>';
 let hmax = Math.max(1, ...d.hist.map(h=>h[1]));
 let hist = d.hist.map(h=>'<tr><td'+(Math.abs(h[0])>d.old_v_max?' class=fold':'')+'>'+
   h[0].toFixed(2)+'</td><td style="text-align:left"><span class=bar style="width:'+
   (240*h[1]/hmax)+'px"></span> '+h[1]+'</td></tr>').join('');
 let pts = d.points.map(p=>'<tr><td>'+p.r.toFixed(2)+'</td><td'+
   (Math.abs(p.v)>d.old_v_max?' class=fold':'')+'>'+p.v.toFixed(3)+'</td><td>'+
   p.az.toFixed(0)+'</td><td>'+(p.snr==null?'-':p.snr.toFixed(1))+'</td></tr>').join('');
 el.innerHTML = '<div class=k>largest |v| reported</div>'+
  '<div class="big">'+d.max_abs_v.toFixed(3)+'</div>'+
  '<div class=k>m/s, at '+d.max_v_range.toFixed(2)+' m &nbsp;|&nbsp; old ceiling '+
  d.old_v_max+' &nbsp; new ceiling '+d.v_max.toFixed(2)+'</div>'+
  verdict+
  '<table><tr><th>frames</th><td>'+d.frames+'</td><th>with motion</th><td>'+
  pct(d.frames_with_motion/Math.max(1,d.frames))+'%</td></tr>'+
  '<tr><th>moving points</th><td>'+d.moving_points+'</td><th>above old v_max</th><td>'+
  d.above_old+'</td></tr>'+
  '<tr><th>fps</th><td>'+d.fps.toFixed(1)+'</td><th>margin</th><td>'+
  (d.margin_us==null?'-':(d.margin_us/1000).toFixed(1)+' ms')+'</td></tr>'+
  '<tr><th>dropped bytes</th><td>'+d.dropped+'</td><th></th><td></td></tr></table>'+
  '<h1 style="margin-top:20px">velocity histogram, m/s (red = the old config could not report this)</h1>'+
  '<table>'+hist+'</table>'+
  '<h1 style="margin-top:20px">live points, fastest first</h1>'+
  '<table><tr><th>range m</th><th>v m/s</th><th>az deg</th><th>snr</th></tr>'+pts+'</table>';
}
setInterval(tick,200);tick();
</script>
"""


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
