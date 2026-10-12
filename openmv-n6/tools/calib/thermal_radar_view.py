#!/usr/bin/env python3
"""Native-thermal (160x120) + radar overlay through T_thermal_radar.json.

Works on a live.py session directory (thermal.bin / frames.jsonl / radar.jsonl),
either a finished recording or one being written right now (--follow tails it,
so run live.py --record ... in one terminal and this in another):

    ./thermal_radar_view.py ../../calib_2026-08-19/session_live1 --follow --port 8089
    ./thermal_radar_view.py ../../openmv-n6/captures/session_dyn --montage out.png
    ./thermal_radar_view.py SESSION --mp4 out.mp4

This is the thermal<->radar check proper: no RGB, no warp LUT - the radar point
goes straight into the Lepton's own pixel grid via the COMPOSED extrinsic
(tools/calib/compose_thermal_radar.py). Red ring = moving (|v|>STATIC_V),
grey ring = static, ring size ~ SNR. Range label in metres.
"""
import argparse, json, os, sys, time, threading
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from perception.project import RadarProjector  # noqa: E402

STATIC_V = 0.25
W, H, SCALE = 160, 120, 4


def load_radar(path, seen):
    out = {}
    with open(path) as f:
        f.seek(seen[0])
        for line in f:
            if not line.endswith("\n"):
                break
            try:
                d = json.loads(line)
            except ValueError:
                break
            out[d["frame"]] = d
            seen[0] += len(line.encode())
    return out


def render(img8, pts, proj, label=""):
    uv, inf = proj.project(pts[:, :3]) if len(pts) else (np.zeros((0, 2)), np.zeros(0, bool))
    big = cv2.cvtColor(cv2.resize(img8, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    n_in = 0
    for k, (u, v) in enumerate(uv):
        if not inf[k] or not np.isfinite(u):
            continue
        x, y = int(u * SCALE), int(v * SCALE)
        moving = abs(pts[k, 3]) > STATIC_V
        col = (0, 0, 255) if moving else (180, 180, 180)
        r = int(np.clip(3 + (pts[k, 4] if pts.shape[1] > 4 else 10) / 5, 3, 12))
        if 0 <= x < W * SCALE and 0 <= y < H * SCALE:
            n_in += 1
        cv2.circle(big, (x, y), r, col, 1 if not moving else 2)
        cv2.putText(big, "%.1f" % np.hypot(pts[k, 0], pts[k, 1]), (x + r + 1, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    cv2.putText(big, "%s radar %d in frame / %d total" % (label, n_in, len(pts)), (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return big


class Session:
    def __init__(self, d):
        self.d = d
        self.proj = RadarProjector(camera="thermal")
        self.radar, self.rseen = {}, [0]
        self.frames, self.fseen = [], [0]
        self.refresh()

    def refresh(self):
        rp, fp = os.path.join(self.d, "radar.jsonl"), os.path.join(self.d, "frames.jsonl")
        if os.path.exists(rp):
            self.radar.update(load_radar(rp, self.rseen))
        if os.path.exists(fp):
            with open(fp) as f:
                f.seek(self.fseen[0])
                for line in f:
                    if not line.endswith("\n"):
                        break
                    self.frames.append(json.loads(line)); self.fseen[0] += len(line.encode())

    def frame(self, i):
        f = self.frames[i]
        th = np.fromfile(os.path.join(self.d, "thermal.bin"), dtype=np.uint8,
                         count=f["thermal_len"], offset=f["thermal_off"])
        if th.size != W * H:
            return None
        r = self.radar.get(f["radar_frame"])
        pts = np.array(r["points"], float) if r and r["points"] else np.zeros((0, 6))
        return render(th.reshape(H, W), pts, self.proj, "f%d" % f["i"])


def serve(sess, port):
    from http.server import ThreadingHTTPServer as HTTPServer, BaseHTTPRequestHandler

    class Hd(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=fr"); self.end_headers()
                last = -1
                while True:
                    sess.refresh()
                    if len(sess.frames) - 1 > last:
                        last = len(sess.frames) - 1
                        img = sess.frame(last)
                        if img is not None:
                            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            try:
                                self.connection.sendall(b"--fr\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
                            except (BrokenPipeError, ConnectionResetError):
                                return
                    time.sleep(0.05)
            else:
                self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
                self.wfile.write(b"<html><body style='background:#111;margin:0'><img src='/stream' style='width:100%;image-rendering:pixelated'></body></html>")
    print("thermal+radar view on http://0.0.0.0:%d" % port, file=sys.stderr)
    HTTPServer(("0.0.0.0", port), Hd).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--follow", action="store_true", help="serve MJPEG of the growing session")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--montage", help="write a PNG grid of frames that have moving radar points")
    ap.add_argument("--mp4", help="write the whole session as a video")
    a = ap.parse_args()
    s = Session(a.session)
    if a.follow:
        serve(s, a.port); return
    if a.mp4:
        vw = cv2.VideoWriter(a.mp4, cv2.VideoWriter_fourcc(*"mp4v"), 8.7, (W * SCALE, H * SCALE))
        for i in range(len(s.frames)):
            img = s.frame(i)
            if img is not None: vw.write(img)
        vw.release(); print("wrote", a.mp4)
    if a.montage:
        tiles = []
        for i in range(0, len(s.frames), 3):
            f = s.frames[i]; r = s.radar.get(f["radar_frame"])
            if not r or not r["points"]: continue
            v = np.array([p[3] for p in r["points"]])
            if (np.abs(v) > STATIC_V).sum() >= 2:
                img = s.frame(i)
                if img is not None: tiles.append(cv2.resize(img, (W * 2, H * 2)))
            if len(tiles) >= 20: break
        if tiles:
            while len(tiles) % 5: tiles.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(tiles[j:j + 5]) for j in range(0, len(tiles), 5)]
            cv2.imwrite(a.montage, np.vstack(rows)); print("wrote", a.montage, len(tiles), "tiles")
        else:
            print("no frames with moving radar points")


if __name__ == "__main__":
    main()
