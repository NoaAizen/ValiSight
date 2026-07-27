"""Live viewer for radar processing running ON the OpenMV N6.

Launches n6_radar_classify.py on the N6 (with the repo mounted so its imports
resolve), decodes the cluster records it streams over USB, and draws a top-down
bird's-eye view — the CLUSTERING AND CLASSIFICATION happen on the N6, not here.

Usage:
    python view_radar_n6.py                       # N6 auto-detected by VID
    python view_radar_n6.py --cfg-port COM8        # also send the .cfg first
    python view_radar_n6.py --selftest             # decode a synthetic record, no GUI

Keys: q / Esc to quit.

NOTE: needs the IWR1843 DATA UART wired to the N6 (see n6_radar_classify.py) and
the radar streaming. The N6 runs one program at a time, so this and the thermal
streamer (view_thermal_rgb.py) cannot both use the board at once.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE_SCRIPT = "n6_radar_classify.py"
OPENMV_VID = 0x37C5

VIEW = 720                       # square bird's-eye canvas (px)
MAX_RANGE_M = 9.0               # matches cfarFovCfg gating in the .cfg
LABEL_COLORS = {"static": (150, 150, 150), "pedestrian": (0, 255, 0),
                "vehicle": (0, 0, 255), "unknown": (0, 200, 255)}


def find_openmv_port():
    from serial.tools import list_ports
    for p in list_ports.comports():
        if p.vid == OPENMV_VID:
            return p.device
    return None


def send_cfg(cfg_port, cfg_path):
    import serial
    import iwr1843_uart
    print("sending %s on %s ..." % (os.path.basename(cfg_path), cfg_port))
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


def world_to_px(cx_fwd, cy_left):
    """Bird's-eye: radar at bottom-centre, forward = up, left = left."""
    scale = (VIEW - 40) / MAX_RANGE_M
    px = int(VIEW / 2 - cy_left * scale)
    py = int(VIEW - 20 - cx_fwd * scale)
    return px, py


def render(clusters, frame_no, fps):
    view = np.zeros((VIEW, VIEW, 3), np.uint8)
    scale = (VIEW - 40) / MAX_RANGE_M
    ox, oy = VIEW // 2, VIEW - 20
    for r in range(2, int(MAX_RANGE_M) + 1, 2):       # range rings
        cv2.circle(view, (ox, oy), int(r * scale), (40, 40, 40), 1)
        cv2.putText(view, "%dm" % r, (ox + 4, oy - int(r * scale) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(view, (ox, oy), (ox, 20), (40, 40, 40), 1)  # boresight
    for c in clusters:
        cx, cy = c["centroid"][0], c["centroid"][1]
        px, py = world_to_px(cx, cy)
        col = LABEL_COLORS.get(c["label"], (255, 255, 255))
        rad = 5 + min(c["n_points"], 20)
        cv2.circle(view, (px, py), rad, col, 2)
        cv2.putText(view, "%s %.1fm %+.1f" % (c["label"], c["range_m"],
                    c["doppler_mps"]), (px + rad + 2, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    cv2.putText(view, "N6 radar classify  frame %d  %.1f fps  (%d clusters)"
                % (frame_no, fps, len(clusters)), (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def decode_record(payload):
    try:
        return json.loads(payload)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="N6 serial (default: auto)")
    ap.add_argument("--cfg-port", default=None,
                    help="radar CONFIG COM port — send the .cfg before starting")
    ap.add_argument("--cfg", default=os.path.join(_ROOT, "configs",
                    "iwr1843_live.cfg"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:                                  # no hardware
        rec = decode_record('{"frame": 7, "n": 1, "clusters": [{"label": '
                            '"pedestrian", "range_m": 3.2, "doppler_mps": 0.8, '
                            '"extent_m": 0.6, "n_points": 5, "centroid": '
                            '[3.1, 0.4, 0.1]}]}')
        assert rec and rec["clusters"][0]["label"] == "pedestrian"
        img = render(rec["clusters"], rec["frame"], 0.0)
        ok, buf = cv2.imencode(".png", img)
        with open(os.path.join(_ROOT, "logs", "selftest_radar_n6.png"), "wb") as f:
            f.write(buf.tobytes())
        print("selftest ok: decoded + rendered 1 cluster -> "
              "logs/selftest_radar_n6.png")
        return 0

    if args.cfg_port:
        send_cfg(args.cfg_port, args.cfg)

    port = args.port or find_openmv_port()
    if not port:
        sys.exit("no OpenMV board found — is the N6 plugged in?")
    proc = start_device(port)
    print("N6 processing radar on %s (clusters computed on-device)" % port)

    clusters = []
    frame_no = 0
    t0, n = time.time(), 0
    fps = 0.0
    try:
        for line in lines(proc):
            if line[:2] == b"C:":
                rec = decode_record(line[2:])
                if rec:
                    clusters = rec["clusters"]
                    frame_no = rec["frame"]
                    n += 1
                    if time.time() - t0 >= 1.0:
                        fps = n / (time.time() - t0)
                        t0, n = time.time(), 0
            elif line[:2] == b"L:":
                print("[N6]", line[2:].decode(errors="replace"))
            elif line:
                print("[device]", line.decode(errors="replace"))
            cv2.imshow("VailSight - N6 radar (q to quit)",
                       render(clusters, frame_no, fps))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        proc.terminate()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
