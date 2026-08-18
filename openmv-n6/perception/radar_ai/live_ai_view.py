#!/usr/bin/env python3
"""Live top-down radar view with the cluster classifier — 'see the radar AI'.

Reads the IWR1843 DATA port, clusters each frame (radar_classify_n6.cluster), computes the
train_clusters.py features (incl. displacement vs previous frame), classifies with
out/radar_ai/cluster_model_v0.pkl (fallback: classify_n6 rules) and serves an MJPEG
top-down picture on http://<jetson>:8091 . Green = person, grey = clutter, text = range/speed.
Usage: python3 perception/radar_ai/live_ai_view.py [--port /dev/ttyACM2] [--http 8091] [--rules]
"""
import argparse, os, sys, math, json, pickle, threading, time, http.server
import numpy as np, cv2, serial
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'radar')); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import mmwave, radar_classify_n6 as rc
FEATS = ['n_pts', 'v_mean_abs', 'v_spread', 'rcs', 'extent', 'range', 'disp', 'disp_minus_v']
W, H, RMAX = 700, 700, 10.0
state = {'jpg': None, 'n': 0, 'persons': 0}

def to_px(x, y):  # radar x fwd (up on screen), y left (left on screen)
    return int(W / 2 - y / RMAX * (W / 2)), int(H - 40 - x / RMAX * (H - 60))

def run(port, model):
    ser = serial.Serial(port, 921600, timeout=0.2); sync = mmwave.FrameSync(); prev = []
    while True:
        data = ser.read(4096)
        if not data: continue
        for _t, frame in sync.feed(data, time.time()):
            try: fr = mmwave.parse_frame(frame)
            except ValueError: continue
            pts = [(p['x'], p['y'], p['z'], p['v'], p['snr'] or 0.0) for p in fr['points'] if math.hypot(p['x'], p['y']) < RMAX]
            groups = rc.cluster([(x, y, z, v) for x, y, z, v, s in pts], eps=0.6, min_pts=1)
            img = np.zeros((H, W, 3), np.uint8)
            for r in (2, 4, 6, 8, 10):
                cv2.circle(img, to_px(0, 0), int(r / RMAX * (H - 60)), (40, 40, 40), 1)
                cv2.putText(img, '%dm' % r, (W // 2 + 4, to_px(r, 0)[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
            for a in (-60, -30, 0, 30, 60):
                cv2.line(img, to_px(0, 0), to_px(RMAX * math.cos(math.radians(a)), -RMAX * math.sin(math.radians(a))), (40, 40, 40), 1)
            cur = []; persons = 0
            for g in groups:
                P = list(g); snr = [s for x, y, z, v, s in pts if any(abs(x - q[0]) < 1e-6 and abs(y - q[1]) < 1e-6 for q in P)]
                cx, cy = float(np.mean([q[0] for q in P])), float(np.mean([q[1] for q in P])); vs = np.array([q[3] for q in P]); rng = math.hypot(cx, cy)
                ext = max((math.hypot(a[0] - b[0], a[1] - b[1]) for a in P for b in P), default=0.0)
                disp = min((math.hypot(cx - px, cy - py) for px, py in prev), default=float('nan'))
                f = [len(P), abs(float(vs.mean())), float(vs.max() - vs.min()), float(max(snr) + 40 * math.log10(max(rng, .1))) if snr else 0.0, ext, rng, disp, (disp - abs(float(vs.mean())) * 0.1) if disp == disp else float('nan')]
                if model is not None:
                    X = np.nan_to_num(np.array([f], float), nan=-1); pr = float(model.predict_proba(X)[0, 1]); is_person = pr >= 0.5
                else:
                    pr = float('nan'); is_person = rc.classify(rc.features([(q[0], q[1], q[2], q[3]) for q in P])) == 'pedestrian'
                col = (60, 220, 60) if is_person else (110, 110, 110); persons += is_person
                for q in P: cv2.circle(img, to_px(q[0], q[1]), 3, col, -1)
                u, v = to_px(cx, cy); cv2.circle(img, (u, v), 12 if is_person else 6, col, 2)
                if is_person or len(P) >= 2:
                    cv2.putText(img, '%.1fm %+.1fm/s%s' % (rng, float(vs.mean()), (' p=%.2f' % pr) if pr == pr else ''), (u + 14, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                cur.append((cx, cy))
            prev = cur
            cv2.putText(img, 'radar AI  frame %d  clusters %d  PERSON %d   (%s)' % (fr['frame_number'], len(groups), persons, 'model v0' if model is not None else 'rules'), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.circle(img, to_px(0, 0), 6, (0, 160, 255), -1)
            ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok: state['jpg'] = enc.tobytes(); state['n'] += 1; state['persons'] = persons

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith('/stream'):
            self.send_response(200); self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f'); self.end_headers(); last = -1
            try:
                while True:
                    if state['jpg'] is not None and state['n'] != last:
                        last = state['n']; j = state['jpg']
                        self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(j) + j + b'\r\n')
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError): return
        else:
            b = b'<html><body style="background:#111;margin:0"><img src="/stream" style="width:100vmin;height:100vmin"></body></html>'
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--port', default='/dev/ttyACM2'); ap.add_argument('--http', type=int, default=8091); ap.add_argument('--rules', action='store_true')
    a = ap.parse_args()
    model = None
    mp = os.path.join(ROOT, 'perception', 'out', 'radar_ai', 'cluster_model_v0.pkl')
    if not a.rules and os.path.exists(mp): model = pickle.load(open(mp, 'rb'))['model']
    threading.Thread(target=run, args=(a.port, model), daemon=True).start()
    print('radar AI live view on http://0.0.0.0:%d  (%s)' % (a.http, 'model v0' if model is not None else 'rules'), flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", a.http), Handler).serve_forever()
