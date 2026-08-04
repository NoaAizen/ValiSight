"""Live side-by-side viewer: N6 thermal (Lepton 3.5) + N6 RGB (PAG7936).

Launches n6_dual_stream.py on the OpenMV N6 over raw REPL (direct pyserial —
see N6Stream for why not mpremote), decodes the base64 lines it prints over
USB, and shows both feeds in one OpenCV window —
thermal colorized (black-hot by default) with the on-device warm/human/hot
detections and temperatures drawn on top.

Frame rate: the Lepton 3.5 is a 9 Hz sensor (VoSPI-capped at ~8.7 Hz), so the
THERMAL panel cannot exceed ~8.7 fps — that is the hardware ceiling, not a
software limit. The RGB (PAG7936) is a separate sensor and runs faster; the two
rates are shown independently so the number is honest.

Recording: with --record, both streams are also written through src/recorder.py
into data/recordings/<timestamp>/ — thermal as raw .bin (the sensor's exact
bytes), RGB as the board's JPEG, both stamped on the N6's shared mono_us axis
(ticks carried per-frame by the device protocol, unwrapped host-side). This is
the capture path for the thermal<->RGB registration protocol
(data/recordings/PROTOCOL_thermal_rgb_registration.md).

Usage:
    python view_thermal_rgb.py [--port COM11]
    python view_thermal_rgb.py --palette blackhot   # blackhot|whitehot|inferno
    python view_thermal_rgb.py --selftest        # save a few frames to PNG, no GUI
    python view_thermal_rgb.py --record --note "baseline_mm=25; target=marker; plan=A"

Keys: q / Esc to quit,  m = cycle palette (black-hot / white-hot / inferno).
"""
import argparse
import base64
import json
import math
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE_SCRIPT = os.path.join(_ROOT, "n6_dual_stream.py")

OPENMV_VID = 0x37C5


def find_openmv_port():
    """The N6 re-enumerates on reconnect (COM11 -> COM12...), so detect by VID."""
    from serial.tools import list_ports
    for p in list_ports.comports():
        if p.vid == OPENMV_VID:
            return p.device
    return None

VIEW_H = 480                      # display height of each panel
BOX_COLORS = {"warm": (0, 200, 255), "human": (0, 255, 0), "hot": (0, 0, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
PALETTES = ("blackhot", "whitehot", "inferno")


def put_label(img, text, org, scale=0.55, color=(255, 255, 255)):
    """Text with a black halo so it stays legible on any palette — including
    the white 'cold' regions of a black-hot frame."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def colorize(up, palette):
    """Map an 8-bit thermal frame to BGR. black-hot: hot=black, cold=white."""
    if palette == "blackhot":
        return cv2.cvtColor(255 - up, cv2.COLOR_GRAY2BGR)
    if palette == "whitehot":
        return cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(up, cv2.COLORMAP_INFERNO)


class N6Stream:
    """Runs DEVICE_SCRIPT on the board over raw REPL, on pyserial directly.

    Deliberately NOT mpremote: its stdout pump reads the VCP in small paced
    chunks and caps the link near 60 KB/s. Measured on this rig with the
    26 KB raw-thermal lines: 425 ms/frame (~2.3 Hz) through `mpremote run`,
    114 ms/frame (the Lepton's own 8.7 Hz cadence) over a direct read. Same
    line protocol, same device script — only the transport differs."""

    def __init__(self, port, script=DEVICE_SCRIPT):
        import serial
        self.ser = serial.Serial(port, 115200, timeout=0.2)
        self.ser.write(b"\x03\x03")            # interrupt whatever runs
        time.sleep(0.4)
        self.ser.reset_input_buffer()
        self.ser.write(b"\x01")                # enter raw REPL
        time.sleep(0.3)
        self.ser.reset_input_buffer()
        code = open(script, "rb").read()
        # plain raw REPL has no flow control — pace the paste so the board's
        # input buffer never overruns (verified: 256 B / 10 ms is safe here)
        for i in range(0, len(code), 256):
            self.ser.write(code[i:i + 256])
            time.sleep(0.01)
        self.ser.write(b"\x04")                # execute
        self._preamble = True                  # device answers "OK" first

    def lines(self):
        buf = b""
        while True:
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if chunk:
                buf += chunk
                if self._preamble and buf[:2] == b"OK":
                    buf = buf[2:]
                    self._preamble = False
            else:
                # heartbeat on read timeout, so the caller's stall detection
                # runs even when the device has stopped talking entirely
                yield b""
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.strip()

    def terminate(self):
        try:
            self.ser.write(b"\x03\x03")
            self.ser.close()
        except Exception:
            pass


def start_stream(port):
    return N6Stream(port)


def lines(stream):
    return stream.lines()


def imwrite_unicode(path, img):
    """cv2.imwrite silently fails on non-ASCII paths on Windows."""
    ok, buf = cv2.imencode(".png", img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    return ok


THERMAL_W, THERMAL_H = 160, 120


def split_stamped(payload):
    """'<ticks>:<b64>' -> (int ticks, b64). Un-stamped payloads from an old
    device script decode fine for viewing but return (None, payload) — there
    is nothing honest to pair them on, so recording skips them."""
    head, sep, rest = payload.partition(b":")
    if sep and head.isdigit():
        return int(head), rest
    return None, payload


def decode_b64(payload):
    try:
        return base64.b64decode(payload, validate=True)
    except Exception:
        return None


def decode_thermal(payload):
    """(image_for_view, raw_bytes_for_record). The device sends raw 19200-byte
    frames; a JPEG payload (old device script) still renders but yields no
    recordable raw."""
    raw = decode_b64(payload)
    if raw is None:
        return None, None
    if len(raw) == THERMAL_W * THERMAL_H:
        return np.frombuffer(raw, np.uint8).reshape(THERMAL_H, THERMAL_W), raw
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    return img, None


class DualRecorder:
    """Both N6 camera streams -> src/recorder.py, on the shared mono_us axis.

    One Unwrapper for both sensors, deliberately: they read the same
    time.ticks_us() counter on the board, and the frames arrive in emit order,
    so a single unwrapped axis is the truthful one. epoch is constant 1 — the
    device script initialises both cameras exactly once per run, and the
    counter only restarts with the script (see frame_clock.stamp on why epoch
    exists at all)."""

    def __init__(self, save_frames="both", root=None, note=""):
        sys.path.insert(0, os.path.join(_ROOT, "src"))
        import frame_clock
        from recorder import Recorder
        from lepton_fix import DEAD_ROWS, OFFSET_ROWS
        self._fc = frame_clock
        self._stamp = frame_clock.stamp
        self._unwrap = frame_clock.Unwrapper().unwrap
        self.epoch = 1
        self.seq = {"thermal": 0, "rgb": 0}
        self.dropped_unstamped = 0
        self.rgb_dims = None
        # The RGB claim is stamped from the first recorded frame, not from the
        # framesize request: the board answers a QVGA (320x240) request with
        # 320x200 JPEGs (verified across sessions via the SOF header), so a
        # hard-coded claim poisons the FOV every analyst derives from meta.
        self._meta = {
            "camera": "dual",
            "device_script": "n6_dual_stream.py",
            "thermal": "lepton3.5 raw 160x120, radiometric 15..45C -> 0..255",
            "rgb": "pag7936 jpeg, dims stamped from first recorded frame",
            "dead_rows": list(DEAD_ROWS),
            "offset_rows": list(OFFSET_ROWS),
            # NOT "note": Recorder.write_meta owns that key and overwrites it
            # with the clock-axis explanation.
            "session_note": note,
        }
        self.rec = Recorder(root=root, save_frames=save_frames, meta=self._meta)

    def frame(self, sensor, ticks, payload):
        if ticks is None:
            self.dropped_unstamped += 1
            return
        self.seq[sensor] += 1
        rec = self._stamp(self.seq[sensor], self._unwrap(ticks), time.time(),
                          len(payload), sensor, epoch=self.epoch)
        self.rec.frame(rec, payload)
        if sensor == "rgb" and self.rgb_dims is None:
            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.rgb_dims = (img.shape[1], img.shape[0])
                self._meta["rgb"] = "pag7936 jpeg %dx%d" % self.rgb_dims
                self.rec.write_meta(self._meta)

    def new_epoch(self):
        """Restarting the device script restarts its ticks counter — the exact
        case frame_clock.stamp's epoch exists for. A fresh Unwrapper with a
        bumped epoch keeps (epoch, mono_us) the honest ordering key; feeding
        the old Unwrapper post-restart ticks would fabricate a time axis."""
        self.epoch += 1
        self._unwrap = self._fc.Unwrapper().unwrap

    def summary(self):
        s = "recorded %s: %d thermal / %d rgb frames" % (
            self.rec.dir, self.seq["thermal"], self.seq["rgb"])
        if self.dropped_unstamped:
            s += ("\nWARNING: %d un-stamped frames skipped — the board is "
                  "running an old n6_dual_stream.py without per-frame ticks"
                  % self.dropped_unstamped)
        if self.rec.capped:
            s += "\nWARNING: size cap reached, frame payloads stopped early"
        return s


_LEPTON_FIX = None


def _lepton_fix():
    global _LEPTON_FIX
    if _LEPTON_FIX is None:
        sys.path.insert(0, os.path.join(_ROOT, "src"))
        import lepton_fix
        _LEPTON_FIX = lepton_fix
    return _LEPTON_FIX


THERMAL_MIN_C, THERMAL_MAX_C = 15.0, 45.0     # device radiometric window
HUMAN_BAND_C = (30.0, 38.0)                   # same band the device tags


def detect_human_band(gray):
    """Human-band blobs on the REPAIRED thermal frame, for the fusion panel.

    Band-tagging only — exactly the role CLAUDE.md assigns temperature bands
    ('מתייגות בלבד'); the real detector is core/detect/thermal.py and this
    display aid never replaces it. Runs on the repaired frame, so the
    full-width false blobs the dead rows used to produce are gone; the width
    guard drops any residual stripe shape."""
    span = THERMAL_MAX_C - THERMAL_MIN_C
    lo = int((HUMAN_BAND_C[0] - THERMAL_MIN_C) * 255.0 / span)
    hi = int((HUMAN_BAND_C[1] - THERMAL_MIN_C) * 255.0 / span)
    mask = cv2.inRange(gray, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # clothing breaks a person into fragments; closing glues the fragments
    # back into one blob before labelling
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, cc, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in cc[i]]
        if area < 30 or w >= 0.85 * gray.shape[1]:
            continue
        m = mask[y:y + h, x:x + w] > 0
        t_mean = float(gray[y:y + h, x:x + w][m].mean()) * span / 255.0 + THERMAL_MIN_C
        boxes.append({"x": x, "y": y, "w": w, "h": h, "t": t_mean})
    return boxes


class _Box:
    """Attribute view over a detection dict, for core.fusion.azimuth."""
    def __init__(self, d):
        self.x, self.y, self.w, self.h = d["x"], d["y"], d["w"], d["h"]


class RadarFeed:
    """Live IWR1843 clusters for the fusion view, through the project's real
    pipeline pieces: RadarReader (TLV) -> radar_gate.gate_points ->
    radar_classify_n6.classify_frame. Display only — nothing here feeds
    decisions or recordings. If the radar is silent (fresh power-up), it
    pushes the stock config once via the CLI port, same as live.sh."""

    def __init__(self, cfg="stock_iwr1843.cfg"):
        self.clusters = []
        self.fps = 0.0
        self.status = "starting"
        self.cfg = cfg
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _by_id(tail):
        import glob
        hits = sorted(glob.glob("/dev/serial/by-id/*XDS110*" + tail))
        return os.path.realpath(hits[0]) if hits else None

    def _run(self):
        import serial
        sys.path.insert(0, os.path.join(_ROOT, "src"))
        import iwr1843_uart
        import radar_gate
        import radar_classify_n6

        data_port = self._by_id("if03")
        if not data_port:
            self.status = "no DATA port"
            return
        try:
            ser = serial.Serial(data_port, 921600, timeout=0.005)
        except Exception as e:
            self.status = "open failed: %s" % e
            return
        reader = iwr1843_uart.RadarReader()
        ser.reset_input_buffer()
        n_frames, configured = 0, False
        t0 = win_t = time.time()
        win_n = 0
        while True:
            try:
                n = ser.in_waiting
                chunk = ser.read(n) if n else ser.read(1)
            except Exception as e:
                self.status = "serial died: %s" % e
                return
            if chunk:
                for fr in reader.feed(chunk):
                    keep, _ = radar_gate.gate_points(fr["points"],
                                                     fr["snr"], fr["noise"])
                    self.clusters = radar_classify_n6.classify_frame(keep)[:8]
                    n_frames += 1
                    win_n += 1
            now = time.time()
            if now - win_t >= 1.0:
                self.fps, win_t, win_n = win_n / (now - win_t), now, 0
                if n_frames:
                    self.status = "ok"
            if not n_frames and not configured and now - t0 > 4.0:
                configured = True
                cli = self._by_id("if00")
                cfg_path = os.path.join(_ROOT, "src", self.cfg)
                if cli and os.path.exists(cfg_path):
                    self.status = "pushing config"
                    try:
                        with serial.Serial(cli, 115200, timeout=1) as cs:
                            iwr1843_uart.send_config(cs, cfg_path, verbose=False)
                        self.status = "config sent"
                    except Exception as e:
                        self.status = "config failed: %s" % str(e)[:60]
                else:
                    self.status = "silent, no CLI port"


def repair_for_display(gray):
    """The horizontal stripes are this module's dead/offset rows — hardware,
    'glue for connectivity, zero evidence' (CLAUDE.md). Display repairs them
    (same order live_server uses: level-correct, THEN interpolate); the
    recording keeps the sensor's raw bytes so a future defect map can re-run
    over old data."""
    lf = _lepton_fix()
    return lf.repair(lf.destripe(gray))


def render_thermal(gray, dets, stats, palette="blackhot"):
    """Colorize + upscale the 160x120 radiometric frame, draw detections."""
    gray = repair_for_display(gray)
    scale = VIEW_H / gray.shape[0]
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    view = colorize(up, palette)
    for d in dets:
        x, y, w, h = [int(v * scale) for v in d["rect"]]
        c = BOX_COLORS.get(d["label"], (255, 255, 255))
        cv2.rectangle(view, (x, y), (x + w, y + h), c, 2)
        put_label(view, "%s %.1fC" % (d["label"], d["t_mean"]),
                  (x, max(y - 6, 12)), scale=0.5, color=c)
    if stats:
        put_label(view, "min %.1f  mean %.1f  max %.1f C" % (
            stats["min"], stats["mean"], stats["max"]), (8, VIEW_H - 10))
    return view


def render_rgb(img):
    scale = VIEW_H / img.shape[0]
    return cv2.resize(img, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_LINEAR)


WEB_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>ValiSight - Thermal + RGB live</title><style>
body{background:#0d1117;color:#c9d1d9;font-family:monospace;margin:16px}
img{max-width:100%%;border:1px solid #30363d}
.ctl{margin:10px 0}.ctl label{display:inline-block;width:220px}
input[type=range]{width:340px;vertical-align:middle}
.warn{color:#d3a04a}
</style></head><body>
<h3>ValiSight live</h3>
<div class="ctl">
 <button onclick="mode('overlay')">fusion only</button>
 <button onclick="mode('all')">thermal | fusion | rgb</button>
</div>
<img src="/stream.mjpg">
<p class="warn">the overlay mapping is a MANUAL nominal alignment for viewing —
display only, never a calibration. The real registration comes from the
protocol sessions.</p>
<div class="ctl"><label>scale (rgb px / thermal px)</label>
 <input type="range" min="0.5" max="3.0" step="0.01" value="%(s).2f"
  oninput="set('s',this.value)"><span id="v_s">%(s).2f</span></div>
<div class="ctl"><label>offset x (rgb px)</label>
 <input type="range" min="-160" max="160" step="1" value="%(dx)d"
  oninput="set('dx',this.value)"><span id="v_dx">%(dx)d</span></div>
<div class="ctl"><label>offset y (rgb px)</label>
 <input type="range" min="-120" max="120" step="1" value="%(dy)d"
  oninput="set('dy',this.value)"><span id="v_dy">%(dy)d</span></div>
<div class="ctl"><label>overlay opacity</label>
 <input type="range" min="0" max="1" step="0.05" value="%(alpha).2f"
  oninput="set('alpha',this.value)"><span id="v_alpha">%(alpha).2f</span></div>
<script>
function set(k,v){document.getElementById('v_'+k).textContent=v;
 fetch('/set?'+k+'='+v);}
function mode(m){fetch('/set?mode='+m);}
</script></body></html>"""


class WebView:
    """Live browser view: THERMAL | OVERLAY | RGB as one MJPEG stream.

    Exists because the rig is operated over SSH (no DISPLAY) and the operator
    still needs to aim and to SEE the two sensors together. The overlay is a
    manual nominal alignment (scale + offset sliders) for human viewing only —
    it is not a calibration and nothing downstream may consume it. Watching
    the slider value that aligns a target drift as the target moves in range
    IS the parallax the registration analyst will measure properly."""

    def __init__(self, port, palette):
        self.lock = threading.Lock()
        self.params = {"s": 1.6, "dx": 0.0, "dy": 0.0, "alpha": 0.45}
        self.mode = "overlay"            # "overlay" = fusion panel only
        self.palette = palette
        self.radar = None                # optional RadarFeed
        self._last_t = None              # newest (gray, dets, stats)
        self.rgb = None                  # latest native 320x240 BGR
        self.jpeg = None                 # latest composite, encoded
        self.stamp = 0
        self.port = port
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):   # keep the terminal for status lines
                pass

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/":
                    with view.lock:
                        page = (WEB_PAGE % view.params).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(page)
                elif u.path == "/set":
                    q = parse_qs(u.query)
                    with view.lock:
                        for k in view.params:
                            if k in q:
                                try:
                                    view.params[k] = float(q[k][0])
                                except ValueError:
                                    pass
                        if q.get("mode", [""])[0] in ("overlay", "all"):
                            view.mode = q["mode"][0]
                    self.send_response(204)
                    self.end_headers()
                elif u.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=vf")
                    self.end_headers()
                    seen = -1
                    try:
                        while True:
                            with view.lock:
                                data, cur = view.jpeg, view.stamp
                            if data is None or cur == seen:
                                time.sleep(0.03)
                                continue
                            seen = cur
                            self.wfile.write(b"--vf\r\nContent-Type: image/jpeg"
                                             b"\r\nContent-Length: %d\r\n\r\n"
                                             % len(data))
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()

        self._srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def _overlay(self, rgb_native, gray, params):
        """Blend the colorized thermal into the native RGB frame at the manual
        scale/offset. Pure display math; clipped paste, no resampling of the
        recorded data."""
        s = max(params["s"], 0.1)
        w, h = max(int(THERMAL_W * s), 2), max(int(THERMAL_H * s), 2)
        th = cv2.resize(colorize(gray, self.palette), (w, h),
                        interpolation=cv2.INTER_NEAREST)
        out = rgb_native.copy()
        H, W = out.shape[:2]
        x0 = int(W / 2 + params["dx"] - w / 2)
        y0 = int(H / 2 + params["dy"] - h / 2)
        xa, ya = max(x0, 0), max(y0, 0)
        xb, yb = min(x0 + w, W), min(y0 + h, H)
        if xb > xa and yb > ya:
            roi = out[ya:yb, xa:xb]
            src = th[ya - y0:yb - y0, xa - x0:xb - x0]
            a = params["alpha"]
            out[ya:yb, xa:xb] = cv2.addWeighted(src, a, roi, 1.0 - a, 0)
        return out

    def push(self, gray, dets, stats):
        """Called from the capture loop on every decoded thermal frame, and
        re-run via repush() when a fresher RGB lands — the composite always
        pairs the two newest frames instead of dragging a stale background."""
        self._last_t = (gray, dets, stats)
        gray = repair_for_display(gray)     # overlay blends the repaired frame
        with self.lock:
            rgb_native = None if self.rgb is None else self.rgb.copy()
            params = dict(self.params)
            mode = self.mode
        if rgb_native is None:
            mid = np.zeros((VIEW_H, 640, 3), np.uint8)
        else:
            mid = render_rgb(self._overlay(rgb_native, gray, params))
        put_label(mid, "FUSION  s=%.2f dx=%d dy=%d  (uncalibrated, display only)"
                  % (params["s"], params["dx"], params["dy"]), (8, 22), scale=0.5)
        if stats:
            put_label(mid, "min %.1f  mean %.1f  max %.1f C" % (
                stats["min"], stats["mean"], stats["max"]), (8, VIEW_H - 10))

        # human-band boxes + radar, associated through the project's real
        # calibration-free azimuth mechanism (core/fusion/azimuth.py)
        from core.fusion import azimuth as az_fuse
        humans = detect_human_band(gray)
        clusters = list(self.radar.clusters) if self.radar is not None else []
        matches = [None] * len(humans)
        if humans and clusters:
            try:
                matches = az_fuse.associate([_Box(b) for b in humans],
                                            clusters, THERMAL_W)
            except Exception:
                pass
        k = VIEW_H / 240.0                       # native rgb px -> panel px

        def th2panel(tx, ty):
            return (int(((tx - THERMAL_W / 2.0) * params["s"] + 160 + params["dx"]) * k),
                    int(((ty - THERMAL_H / 2.0) * params["s"] + 120 + params["dy"]) * k))

        for b, m in zip(humans, matches):
            p1, p2 = th2panel(b["x"], b["y"]), th2panel(b["x"] + b["w"], b["y"] + b["h"])
            cv2.rectangle(mid, p1, p2, (0, 255, 0), 2)
            lbl = "HUMAN %.1fC" % b["t"]
            if m is not None:
                lbl += "  %.2fm  v=%.1f" % (m["range_m"], m["doppler_mps"])
            put_label(mid, lbl, (max(p1[0], 2), max(p1[1] - 6, 14)),
                      scale=0.55, color=(0, 255, 0))

        if self.radar is not None:
            put_label(mid, "radar: %s  %.0f fps" % (self.radar.status, self.radar.fps),
                      (mid.shape[1] - 250, 22), scale=0.5, color=(80, 200, 255))
            fx = az_fuse.focal_px(mid.shape[1])  # nominal 57-deg pinhole, display only
            for i, c in enumerate(clusters):
                azd = az_fuse.cluster_azimuth_deg(c["centroid"])
                u = int(mid.shape[1] / 2.0 + fx * math.tan(math.radians(azd)))
                if 0 <= u < mid.shape[1]:
                    cv2.line(mid, (u, VIEW_H - 46), (u, VIEW_H - 22),
                             (80, 200, 255), 2)
                    put_label(mid, "%s %.1fm" % (c["label"], c["range_m"]),
                              (max(u - 40, 2), VIEW_H - 50), scale=0.45,
                              color=(80, 200, 255))

        if mode == "overlay":
            comp = mid
        else:
            left = render_thermal(gray, dets, stats, self.palette)
            if rgb_native is None:
                right = np.zeros((VIEW_H, 640, 3), np.uint8)
            else:
                right = render_rgb(rgb_native)
            put_label(right, "RGB (PAG7936)", (8, 22), scale=0.5)
            comp = np.hstack([left, mid, right])
        ok, buf = cv2.imencode(".jpg", comp, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()
                self.stamp += 1

    def repush(self):
        """Rebuild the composite against the RGB that just arrived."""
        if self._last_t is not None:
            self.push(*self._last_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None,
                    help="serial port of the N6 (default: auto-detect by VID)")
    ap.add_argument("--selftest", action="store_true",
                    help="decode a few frames, save PNGs, exit (no GUI)")
    ap.add_argument("--palette", default="blackhot", choices=PALETTES,
                    help="thermal palette (default: blackhot = hot is black)")
    ap.add_argument("--record", action="store_true",
                    help="write the session via src/recorder.py under data/recordings/")
    ap.add_argument("--save-frames", default="both",
                    choices=("none", "thermal", "rgb", "both"),
                    help="which frame payloads to store when recording (default: both)")
    ap.add_argument("--note", default="",
                    help='session note for meta.json, e.g. "baseline_mm=25; target=marker; plan=A"')
    ap.add_argument("--record-root", default=None,
                    help="override the recordings root directory")
    ap.add_argument("--no-gui", action="store_true",
                    help="no OpenCV window; print status lines instead (Ctrl-C to stop)")
    ap.add_argument("--web", action="store_true",
                    help="serve a live browser view: thermal | overlay | rgb (works over SSH)")
    ap.add_argument("--web-port", type=int, default=8081)
    args = ap.parse_args()
    palette = args.palette

    # Over SSH there is no DISPLAY and cv2.imshow dies in GTK — fall back to
    # headless instead of crashing one frame into a recording session.
    headless = (not args.selftest) and (
        args.no_gui or args.web or not os.environ.get("DISPLAY"))
    if headless and not (args.no_gui or args.web):
        print("no DISPLAY — running headless: status lines instead of a window, "
              "Ctrl-C to stop. (Live view in a browser: --web. "
              "Aim check without it: --selftest saves PNGs.)")

    web = None
    if args.web:
        web = WebView(args.web_port, palette)
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            ip = probe.getsockname()[0]
            probe.close()
        except OSError:
            ip = "127.0.0.1"
        print("web view   : http://%s:%d/" % (ip, args.web_port))
        web.radar = RadarFeed()          # best-effort; panel shows its status

    rec = None
    if args.record:
        rec = DualRecorder(save_frames=args.save_frames,
                           root=args.record_root, note=args.note)
        print("recording  : %s  (frames: %s)" % (rec.rec.dir, args.save_frames))

    port = args.port or find_openmv_port()
    if not port:
        sys.exit("no OpenMV board found — is the N6 plugged in?")

    proc = start_stream(port)
    args.port = port
    print("streaming from N6 on %s (Lepton settle ~5s)..." % args.port)

    thermal = rgb = None
    dets, stats = [], None
    got = {"T": 0, "R": 0}
    # independent fps for each sensor — thermal is Lepton-capped, RGB is not
    t_win, t_cnt, t_fps = time.time(), 0, 0.0
    r_win, r_cnt, r_fps = time.time(), 0, 0.0
    last_status = time.time()
    last_seen = {"T": None, "R": None}
    restarts = 0
    line_iter = lines(proc)

    try:
        while True:
            try:
                line = next(line_iter)
            except Exception:              # serial link died mid-read
                line = None

            # Per-sensor stall watch. The measured failure mode is ONE sensor
            # dying after a CSI overflow while the other keeps flowing, so an
            # "any frame recently" check sleeps right through it — a session
            # that looks alive with no thermal in it is the worst outcome.
            now = time.time()
            dead = [s for s in ("T", "R")
                    if last_seen[s] is not None and now - last_seen[s] > 8.0]
            if dead or line is None:
                restarts += 1
                if restarts > 4:
                    print("ERROR: stream keeps stalling (%d restarts) — giving "
                          "up; check the board." % (restarts - 1), flush=True)
                    break
                print("WARN: %s stalled — restarting device stream (epoch %d)"
                      % ("+".join(dead) if dead else "link", restarts + 1),
                      flush=True)
                proc.terminate()
                time.sleep(1.0)
                port = find_openmv_port() or port   # replug renumbers ttyACM*
                proc = start_stream(port)
                line_iter = lines(proc)
                if rec is not None:
                    rec.new_epoch()       # device restart resets its ticks
                last_seen = {"T": None, "R": None}
                continue

            if headless and now - last_status >= 2.0:
                last_status = now
                status = "thermal %4.1f fps [cap ~8.7] | rgb %4.1f fps" % (t_fps, r_fps)
                if rec is not None:
                    status += " | recorded %d T / %d R" % (
                        rec.seq["thermal"], rec.seq["rgb"])
                print(status, flush=True)
            if len(line) < 2 or line[1:2] != b":":
                if line:                       # device tracebacks / notes
                    print("[device]", line.decode(errors="replace"))
                continue
            tag, payload = line[:1], line[2:]

            if tag == b"S":
                try:
                    stats = json.loads(payload)
                except ValueError:
                    pass
            elif tag == b"D":
                try:
                    dets = json.loads(payload)
                except ValueError:
                    pass
            elif tag == b"T":
                ticks, payload = split_stamped(payload)
                img, raw = decode_thermal(payload)
                if img is not None:
                    if rec is not None and raw is not None:
                        rec.frame("thermal", ticks, raw)
                    if not headless:
                        thermal = render_thermal(img, dets, stats, palette)
                    got["T"] += 1
                    last_seen["T"] = time.time()
                    if web is not None:
                        web.push(img, dets, stats)
                    t_cnt += 1
                    if time.time() - t_win >= 1.0:
                        t_fps = t_cnt / (time.time() - t_win)
                        t_win, t_cnt = time.time(), 0
            elif tag == b"R":
                ticks, payload = split_stamped(payload)
                jpeg = decode_b64(payload)
                img = None if jpeg is None else cv2.imdecode(
                    np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    if rec is not None:
                        rec.frame("rgb", ticks, jpeg)
                    if not headless:
                        rgb = render_rgb(img)
                    got["R"] += 1
                    last_seen["R"] = time.time()
                    if web is not None:
                        with web.lock:
                            web.rgb = img
                        web.repush()   # pair the fresh RGB with newest thermal
                    r_cnt += 1
                    if time.time() - r_win >= 1.0:
                        r_fps = r_cnt / (time.time() - r_win)
                        r_win, r_cnt = time.time(), 0

            if headless:
                continue                       # no window to draw

            if thermal is None and rgb is None:
                continue

            if args.selftest:
                if got["T"] >= 3 and got["R"] >= 3:
                    out = os.path.join(_ROOT, "logs")
                    os.makedirs(out, exist_ok=True)
                    imwrite_unicode(os.path.join(out, "selftest_thermal.png"), thermal)
                    imwrite_unicode(os.path.join(out, "selftest_rgb.png"), rgb)
                    print("selftest ok: %d thermal / %d rgb frames, "
                          "saved logs/selftest_*.png" % (got["T"], got["R"]))
                    return
                continue

            blank = np.zeros((VIEW_H, 640, 3), np.uint8)
            left = thermal if thermal is not None else blank
            right = rgb if rgb is not None else blank
            view = np.hstack([left, right])
            put_label(view, "THERMAL (Lepton 3.5)  %.1f fps [cap ~8.7]  %s"
                      % (t_fps, palette), (8, 22), scale=0.6)
            put_label(view, "RGB (PAG7936)  %.1f fps" % r_fps,
                      (left.shape[1] + 8, 22), scale=0.6)
            cv2.imshow("VailSight - Thermal + RGB (q to quit)", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                palette = PALETTES[(PALETTES.index(palette) + 1) % len(PALETTES)]
    except KeyboardInterrupt:
        pass                                   # Ctrl-C is the headless quit key
    finally:
        proc.terminate()
        cv2.destroyAllWindows()
        if rec is not None:
            print(rec.summary())


if __name__ == "__main__":
    main()
