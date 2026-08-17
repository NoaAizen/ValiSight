#!/usr/bin/env python3
"""Live view of the bridge in a browser.

    bridge_view.py out/ [--port 8090]

Runs next to `bridge_rx <port> -o out/`: serves the newest thermal frame
(contrast-stretched, x4), the newest RGB JPEG, the last IMU sample and the
receiver's counters, refreshed by the page a few times per second.
Reads only what bridge_rx wrote — no second reader on the USB port.
"""
import argparse, glob, io, json, os, struct, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
try:
    from PIL import Image
except ImportError:
    sys.exit("pip install pillow")

PAGE = """<!doctype html><html dir="rtl"><head><meta charset="utf-8"><title>Bridge live</title>
<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:1rem}
.row{display:flex;gap:1rem;flex-wrap:wrap}img{image-rendering:pixelated;border:1px solid #444}
pre{background:#1a1a1a;padding:.6rem;border-radius:6px;font-size:.95rem}h3{margin:.4rem 0}</style></head><body>
<h2>גשר N6 → ג'ייסון — חי</h2>
<div class="row"><div><h3>תרמי (Lepton) </h3><img id="t" width="640" height="480"></div>
<div><h3>RGB (PAG7936)</h3><img id="r" width="640" height="400"></div></div>
<h3>IMU</h3><pre id="imu">…</pre><h3>מונים</h3><pre id="st">…</pre>
<script>
let n=0;
async function tick(){
  n++; document.getElementById('t').src='/thermal.png?'+n; document.getElementById('r').src='/rgb.jpg?'+n;
  try{const s=await (await fetch('/stats')).json();
    document.getElementById('imu').textContent=s.imu||'אין עדיין';
    document.getElementById('st').textContent=s.stats||'';}catch(e){}
  setTimeout(tick,200);}
tick();</script></body></html>"""


class State:
    def __init__(self, d): self.d = d
    def newest(self, kind):
        fs = glob.glob(os.path.join(self.d, "*_%s.bin" % kind))
        return max(fs) if fs else None
    def thermal_png(self):
        f = self.newest("thermal")
        if not f: return None
        d = open(f, "rb").read(); w, h, fmt, _ = struct.unpack("<HHBB", d[:6]); px = d[6:6 + w * h]
        if len(px) < w * h: return None
        img = Image.frombytes("L", (w, h), px)
        # contrast stretch 1–99 percentile so a flat room still shows structure
        hist = img.histogram(); tot = w * h; lo = hi = 0; c = 0
        for i, v in enumerate(hist):
            c += v
            if lo == 0 and c >= tot * 0.01: lo = i
            if c >= tot * 0.99: hi = i; break
        hi = max(hi, lo + 1)
        img = img.point(lambda p: max(0, min(255, (p - lo) * 255 // (hi - lo))))
        img = img.resize((w * 4, h * 4), Image.NEAREST)
        b = io.BytesIO(); img.save(b, "PNG"); return b.getvalue()
    def rgb_jpg(self):
        f = self.newest("rgb")
        return open(f, "rb").read()[6:] if f else None
    def imu_line(self):
        p = os.path.join(self.d, "imu.csv")
        if not os.path.exists(p): return None
        with open(p, "rb") as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 4096)); tail = fh.read().decode(errors="replace").strip().splitlines()
        r = None
        for line in reversed(tail):            # the last line may still be half-written by bridge_rx
            f = line.split(",")
            if len(f) == 9 and f[0].isdigit() and f[8].lstrip("-").isdigit(): r = f; break
        if r is None: return None
        ax, ay, az = (int(r[3]), int(r[4]), int(r[5])); gx, gy, gz = (int(r[6]) / 1000, int(r[7]) / 1000, int(r[8]) / 1000)
        mag = (ax * ax + ay * ay + az * az) ** 0.5
        return ("תאוצה  X %+5d  Y %+5d  Z %+5d  mg   |a| = %.0f mg\nסיבוב  X %+6.2f Y %+6.2f Z %+6.2f  °/ש'\nseq %s   ts %s   שורות בקובץ ≈ %s"
                % (ax, ay, az, mag, gx, gy, gz, r[0], r[1], "…"))
    def stats(self):
        p = self.stats_path
        if not p or not os.path.exists(p): return "אין קובץ סטטיסטיקה"
        lines = open(p, errors="replace").read().strip().splitlines()
        return "\n".join(lines[-3:]) if lines else ""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--stats", default=None, help="file where bridge_rx stdout is redirected")
    a = ap.parse_args()
    st = State(a.out_dir); st.stats_path = a.stats

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass
        def _send(self, code, ctype, body):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            try:
                if self.path.startswith("/thermal.png"):
                    b = st.thermal_png(); return self._send(200, "image/png", b) if b else self._send(404, "text/plain", b"no frame yet")
                if self.path.startswith("/rgb.jpg"):
                    b = st.rgb_jpg(); return self._send(200, "image/jpeg", b) if b else self._send(404, "text/plain", b"no frame yet")
                if self.path.startswith("/stats"):
                    return self._send(200, "application/json", json.dumps({"imu": st.imu_line(), "stats": st.stats()}).encode())
                return self._send(200, "text/html; charset=utf-8", PAGE.encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    print("live view: http://<jetson>:%d   (dir %s)" % (a.port, a.out_dir)); srv.serve_forever()


if __name__ == "__main__":
    main()
