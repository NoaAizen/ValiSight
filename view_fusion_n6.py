"""PC viewer for the unified N6 fusion node (n6_fusion_stream.py).

Decodes the timestamped thermal + gated-radar records the N6 streams and PAIRS
them by their ON-DEVICE monotonic timestamp using core.timesync.SyncBuffer —
time-aligned fusion, not arrival-order guessing. Shows thermal (black-hot) with
its detections and the azimuth of each time-matched radar cluster, a bird's-eye
of those clusters, and a health line (sync matches/drops + radar reduction).

Honesty: radar and thermal are aligned by AZIMUTH only — the IWR1843 elevation
(sigma_el ~12 deg) is too coarse for pixel-accurate vertical placement, so this
is a coarse cross-sensor cue, not calibrated pixel fusion. The true fusion time
resolution is the radar's 0.5 s aggregation window, not the ~60 ms sync slop.

    python -m mpremote connect COM12 mount . run n6_fusion_stream.py   # on N6
    python view_fusion_n6.py --cfg-port COM8    # PC: config radar + view
    python view_fusion_n6.py --selftest         # no hardware

Keys: q / Esc to quit.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np

from core.timesync import SyncBuffer, Stamped

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE_SCRIPT = "n6_fusion_stream.py"
OPENMV_VID = 0x37C5

PANEL = 480
SLOP_S = 0.06                       # ~half the 100 ms radar period (honest floor)
MAX_RANGE_M = 9.0
THERMAL_HFOV_DEG = 57.0
THERMAL_W = 160
FOCAL_PX = (THERMAL_W / 2.0) / math.tan(math.radians(THERMAL_HFOV_DEG / 2.0))
LABEL_COLORS = {"static": (150, 150, 150), "pedestrian": (0, 255, 0),
                "vehicle": (0, 0, 255), "unknown": (0, 200, 255)}
BAND_COLORS = {"warm": (0, 200, 255), "human": (0, 255, 0), "hot": (0, 0, 255)}


def find_openmv_port():
    from serial.tools import list_ports
    for p in list_ports.comports():
        if p.vid == OPENMV_VID:
            return p.device
    return None


def send_cfg(cfg_port, cfg_path):
    import serial
    import iwr1843_uart
    with serial.Serial(cfg_port, 115200, timeout=1) as s:
        iwr1843_uart.send_config(s, cfg_path)


def start_device(port):
    return subprocess.Popen(
        [sys.executable, "-m", "mpremote", "connect", port,
         "mount", ".", "run", DEVICE_SCRIPT],
        cwd=_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)


def lines(proc):
    buf = b""
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.strip()


def _put(img, text, org, color=(255, 255, 255), scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                cv2.LINE_AA)


def az_to_thermal_x(az_deg):
    """Radar azimuth -> thermal image column (azimuth-only cross-sensor cue)."""
    u = THERMAL_W / 2.0 + FOCAL_PX * math.tan(math.radians(az_deg))
    return int(u / THERMAL_W * PANEL)


def render_thermal(gray, dets, radar_clusters):
    scale = PANEL / gray.shape[0]
    up = cv2.resize(gray, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC)
    view = cv2.cvtColor(255 - up, cv2.COLOR_GRAY2BGR)       # black-hot
    for d in dets:
        x, y, w, h = [int(v * scale) for v in d["rect"]]
        c = BAND_COLORS.get(d["label"], (255, 255, 255))
        cv2.rectangle(view, (x, y), (x + w, y + h), c, 2)
        _put(view, "%s %.1fC" % (d["label"], d["t_mean"]),
             (x, max(y - 6, 12)), c, 0.45)
    for cl in radar_clusters:                               # radar azimuth cues
        cx, cy = cl["centroid"][0], cl["centroid"][1]
        az = math.degrees(math.atan2(-cy, cx))
        px = az_to_thermal_x(az)
        col = LABEL_COLORS.get(cl["label"], (255, 255, 255))
        cv2.line(view, (px, 30), (px, PANEL), col, 1)
        _put(view, "R:%s %.1fm" % (cl["label"], cl["range_m"]), (px + 3, 44),
             col, 0.42)
    _put(view, "THERMAL (black-hot) + radar azimuth", (8, 22))
    return view


def render_birdseye(clusters):
    view = np.zeros((PANEL, PANEL, 3), np.uint8)
    scale = (PANEL - 40) / MAX_RANGE_M
    ox, oy = PANEL // 2, PANEL - 20
    for r in range(2, int(MAX_RANGE_M) + 1, 2):
        cv2.circle(view, (ox, oy), int(r * scale), (40, 40, 40), 1)
    cv2.line(view, (ox, oy), (ox, 20), (40, 40, 40), 1)
    for cl in clusters:
        cx, cy = cl["centroid"][0], cl["centroid"][1]
        px = int(ox - cy * scale)
        py = int(oy - cx * scale)
        col = LABEL_COLORS.get(cl["label"], (255, 255, 255))
        cv2.circle(view, (px, py), 5 + min(cl["n_points"], 18), col, 2)
        _put(view, "%s %.1fm %+.1f" % (cl["label"], cl["range_m"],
             cl["doppler_mps"]), (px + 8, py), col, 0.42)
    _put(view, "RADAR (gated, on-N6)", (8, 22))
    return view


def compose(thermal_img, thermal_dets, radar_clusters, reduction, sync_stats):
    left = (render_thermal(thermal_img, thermal_dets, radar_clusters)
            if thermal_img is not None
            else np.zeros((PANEL, PANEL, 3), np.uint8))
    right = render_birdseye(radar_clusters)
    view = np.hstack([left, right])
    health = "sync matched %d  dropped R %d / T %d" % (
        sync_stats.matched, sync_stats.dropped_a, sync_stats.dropped_b)
    if reduction:
        health += "  |  radar gate: %d->%d (cut %.0f%%)" % (
            reduction["in"], reduction["out"], 100 * reduction["ratio"])
    _put(view, health, (8, view.shape[0] - 12), (0, 255, 255), 0.5)
    return view


class FusionState:
    """Pairs radar (A) and thermal (B) by on-device timestamp."""

    def __init__(self, slop_s=SLOP_S):
        self.sync = SyncBuffer(slop_s=slop_s)
        self._pending_dets = []
        self._pending_stats = None
        self.thermal_img = None
        self.thermal_dets = []
        self.radar_clusters = []
        self.reduction = None

    def on_line(self, tag, t_s, payload):
        if tag == "D":
            self._pending_dets = json.loads(payload)
        elif tag == "S":
            self._pending_stats = json.loads(payload)
        elif tag == "T":
            img = _decode_jpeg(payload)
            if img is not None:
                self.sync.push_b(Stamped(
                    {"img": img, "dets": self._pending_dets}, t_s))
        elif tag == "C":
            rec = json.loads(payload)
            pairs = self.sync.push_a(Stamped(rec, t_s))
            for radar_a, thermal_b in pairs:
                self.thermal_img = thermal_b.value["img"]
                self.thermal_dets = thermal_b.value["dets"]
                self.radar_clusters = radar_a.value["clusters"]
                self.reduction = radar_a.value.get("reduction")


def _decode_jpeg(payload):
    import base64
    try:
        raw = base64.b64decode(payload, validate=True)
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def _parse(line):
    parts = line.split(b":", 2)
    if len(parts) != 3:
        return None
    tag = parts[0].decode(errors="replace")
    try:
        t_s = int(parts[1]) / 1e6
    except ValueError:
        return None
    return tag, t_s, parts[2].decode(errors="replace")


def _selftest():
    st = FusionState(slop_s=0.06)
    # aligned timestamps (us): thermal at 1000000, radar at 1010000 (10 ms apart)
    gray = (np.ones((120, 160), np.uint8) * 60)
    ok, jp = cv2.imencode(".jpg", gray)
    import base64
    b64 = base64.b64encode(jp.tobytes()).decode()
    st.on_line("D", 1.0, json.dumps([{"label": "human", "rect": [70, 40, 20, 40],
                                      "t_mean": 34.0, "t_max": 36.0}]))
    st.on_line("T", 1.0, b64)
    st.on_line("C", 1.01, json.dumps({"frame": 3, "clusters": [{"label":
               "pedestrian", "range_m": 3.2, "doppler_mps": 0.8, "extent_m": 0.6,
               "n_points": 5, "centroid": [3.1, 0.4, 0.1]}],
               "reduction": {"in": 9, "out": 5, "ratio": 0.444, "weak": 2,
                             "fov": 1, "isolated": 1}}))
    assert st.sync.stats().matched == 1, "streams did not pair"
    assert st.radar_clusters and st.thermal_img is not None
    img = compose(st.thermal_img, st.thermal_dets, st.radar_clusters,
                  st.reduction, st.sync.stats())
    os.makedirs(os.path.join(_ROOT, "logs"), exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    with open(os.path.join(_ROOT, "logs", "selftest_fusion_n6.png"), "wb") as f:
        f.write(buf.tobytes())
    print("selftest ok: paired 1 radar+thermal by timestamp, cut 44%% of radar "
          "points -> logs/selftest_fusion_n6.png")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--cfg-port", default=None,
                    help="radar CONFIG COM port — send the .cfg first")
    ap.add_argument("--cfg", default=os.path.join(_ROOT, "configs",
                    "iwr1843_vitals.cfg"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if args.cfg_port:
        send_cfg(args.cfg_port, args.cfg)
    port = args.port or find_openmv_port()
    if not port:
        sys.exit("no OpenMV board found — is the N6 plugged in?")
    proc = start_device(port)
    print("N6 fusion node on %s (thermal + gated radar, paired by timestamp)"
          % port)

    st = FusionState()
    try:
        for line in lines(proc):
            parsed = _parse(line)
            if parsed:
                st.on_line(*parsed)
            elif line:
                print("[device]", line.decode(errors="replace"))
            cv2.imshow("VailSight - N6 fusion (q to quit)",
                       compose(st.thermal_img, st.thermal_dets,
                               st.radar_clusters, st.reduction,
                               st.sync.stats()))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        proc.terminate()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
