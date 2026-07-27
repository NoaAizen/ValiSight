"""Live side-by-side viewer: N6 thermal (Lepton 3.5) + N6 RGB (PAG7936).

Launches n6_dual_stream.py on the OpenMV N6 via mpremote, decodes the base64
JPEG lines it prints over USB, and shows both feeds in one OpenCV window —
thermal colorized (black-hot by default) with the on-device warm/human/hot
detections and temperatures drawn on top.

Frame rate: the Lepton 3.5 is a 9 Hz sensor (VoSPI-capped at ~8.7 Hz), so the
THERMAL panel cannot exceed ~8.7 fps — that is the hardware ceiling, not a
software limit. The RGB (PAG7936) is a separate sensor and runs faster; the two
rates are shown independently so the number is honest.

Usage:
    python view_thermal_rgb.py [--port COM11]
    python view_thermal_rgb.py --palette blackhot   # blackhot|whitehot|inferno
    python view_thermal_rgb.py --selftest        # save a few frames to PNG, no GUI

Keys: q / Esc to quit,  m = cycle palette (black-hot / white-hot / inferno).
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time

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


def start_stream(port):
    return subprocess.Popen(
        [sys.executable, "-m", "mpremote", "connect", port, "run", DEVICE_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)


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


def imwrite_unicode(path, img):
    """cv2.imwrite silently fails on non-ASCII paths on Windows."""
    ok, buf = cv2.imencode(".png", img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    return ok


def decode_jpeg(payload, flags):
    try:
        raw = base64.b64decode(payload, validate=True)
        return cv2.imdecode(np.frombuffer(raw, np.uint8), flags)
    except Exception:
        return None


def render_thermal(gray, dets, stats, palette="blackhot"):
    """Colorize + upscale the 160x120 radiometric frame, draw detections."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None,
                    help="serial port of the N6 (default: auto-detect by VID)")
    ap.add_argument("--selftest", action="store_true",
                    help="decode a few frames, save PNGs, exit (no GUI)")
    ap.add_argument("--palette", default="blackhot", choices=PALETTES,
                    help="thermal palette (default: blackhot = hot is black)")
    args = ap.parse_args()
    palette = args.palette

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

    try:
        for line in lines(proc):
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
                img = decode_jpeg(payload, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    thermal = render_thermal(img, dets, stats, palette)
                    got["T"] += 1
                    t_cnt += 1
                    if time.time() - t_win >= 1.0:
                        t_fps = t_cnt / (time.time() - t_win)
                        t_win, t_cnt = time.time(), 0
            elif tag == b"R":
                img = decode_jpeg(payload, cv2.IMREAD_COLOR)
                if img is not None:
                    rgb = render_rgb(img)
                    got["R"] += 1
                    r_cnt += 1
                    if time.time() - r_win >= 1.0:
                        r_fps = r_cnt / (time.time() - r_win)
                        r_win, r_cnt = time.time(), 0

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
    finally:
        proc.terminate()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
