"""Unified sensor node ON the OpenMV N6: thermal + RGB + gated radar, one loop.

This is the "run it all on the N6" build. In one MicroPython program the board:
  * grabs the Lepton 3.5 thermal frame (+ warm/human/hot detections, on-device),
  * grabs the PAG7936 RGB frame,
  * drains the IWR1843 DATA UART, GATES the points (core.radar_gate — drop the
    manufactured/ghost returns before doing any work), then clusters+classifies
    them (radar_classify_n6) — all on the board.

Every record carries a monotonic on-device timestamp (time.ticks_us) so the PC
(view_fusion_n6.py) can pair radar and thermal by TIME, not by arrival order.
Radar sample time is reconstructed from the frame index, not its UART arrival,
so it does not inherit the UART's load-dependent jitter (see core.timesync).

Run with the repo mounted so the pure imports resolve:
    python -m mpremote connect COM12 mount . run n6_fusion_stream.py

Line protocol (one record per line; <t> is on-device microseconds):
    S:<t>:<json>   thermal stats {min,mean,max} C
    D:<t>:<json>   thermal detections [{label,rect,t_mean,t_max}, ...]
    T:<t>:<b64>    thermal JPEG (grayscale, MIN_C..MAX_C -> 0..255)
    R:<t>:<b64>    RGB JPEG
    C:<t>:<json>   gated radar {frame, clusters:[...], reduction:{...}}

WIRING: IWR1843 DATA_UART TX -> N6 UART RX (RADAR_UART), shared GND. The radar
must be configured/streaming (send the .cfg from the PC first). The N6 runs ONE
program, so this replaces the standalone thermal streamer while it runs.
"""
import time
import json
import ubinascii

import csi
from machine import UART

from iwr1843_uart import RadarReader
from radar_classify_n6 import classify_frame
from core.radar_gate import gate_points

# --- thermal bands (same as n6_dual_stream / thermal_heatmap_n6) -------------
MIN_C = 15.0
MAX_C = 45.0
CLASSES = [("warm", 25.0, 30.0), ("human", 30.0, 38.0), ("hot", 38.0, 45.0)]

# --- radar ------------------------------------------------------------------
RADAR_UART = 1
RADAR_BAUD = 921600
RADAR_PERIOD_S = 0.05          # frameCfg period in configs/iwr1843_vitals.cfg (20 Hz)
RGB_EVERY = 2
RGB_QUALITY = 40
THERMAL_QUALITY = 80


def temp_to_g(t):
    t = min(max(t, MIN_C), MAX_C)
    return int((t - MIN_C) * 255.0 / (MAX_C - MIN_C))


def g_to_temp(g):
    return (g * (MAX_C - MIN_C) / 255.0) + MIN_C


BANDS = [(lbl, [(temp_to_g(a),
                 temp_to_g(b) - (0 if i == len(CLASSES) - 1 else 1))])
         for i, (lbl, a, b) in enumerate(CLASSES)]


def _v(x):
    return x() if callable(x) else x


def emit(tag, t_us, body):
    print("%s:%d:%s" % (tag, t_us, body))


def setup():
    lep = csi.CSI(cid=csi.LEPTON)
    lep.reset()
    lep.pixformat(csi.GRAYSCALE)
    lep.framesize(csi.QQVGA)
    lep.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)
    lep.ioctl(csi.IOCTL_LEPTON_SET_RANGE, MIN_C, MAX_C)
    rgb = csi.CSI(cid=csi.PAG7936)
    rgb.reset()
    rgb.pixformat(csi.RGB565)
    rgb.framesize(csi.QVGA)
    time.sleep_ms(5000)            # Lepton settle + first FFC
    uart = UART(RADAR_UART, RADAR_BAUD, bits=8, parity=None, stop=1, timeout=5)
    return lep, rgb, uart


def grab(cam):
    for _ in range(10):
        img = cam.snapshot()
        if img is not None:
            return img
        time.sleep_ms(10)
    return None


def thermal_detections(t):
    dets = []
    for label, thr in BANDS:
        for b in t.find_blobs(thr, pixels_threshold=8, area_threshold=8,
                              merge=True):
            rect = tuple(_v(b.rect))
            s = t.get_statistics(thresholds=thr, roi=rect)
            dets.append({"label": label, "rect": rect,
                         "t_mean": g_to_temp(_v(s.mean)),
                         "t_max": g_to_temp(_v(s.max))})
    return dets


def run():
    lep, rgb, uart = setup()
    radar = RadarReader()
    anchor_us = None
    first_frame = None
    n = 0
    print("L:0:N6 fusion node — thermal + gated radar (UART%d @ %d)"
          % (RADAR_UART, RADAR_BAUD))

    while True:
        # --- thermal (paces the loop at the Lepton's ~8.7 Hz) ---------------
        t = grab(lep)
        if t is not None:
            ts = time.ticks_us()
            st = t.get_statistics()
            emit("S", ts, json.dumps({"min": g_to_temp(_v(st.min)),
                                      "mean": g_to_temp(_v(st.mean)),
                                      "max": g_to_temp(_v(st.max))}))
            emit("D", ts, json.dumps(thermal_detections(t)))
            emit("T", ts, ubinascii.b2a_base64(
                t.compress(quality=THERMAL_QUALITY).bytearray())[:-1].decode())

        # --- radar: drain the UART, gate, classify -------------------------
        if uart.any():
            for fr in radar.feed(uart.read()):
                if anchor_us is None:
                    anchor_us = time.ticks_us()
                    first_frame = fr["frame"]
                # sample time from frame index (jitter-free), in us
                t_us = anchor_us + int(
                    (fr["frame"] - first_frame) * RADAR_PERIOD_S * 1e6)
                kept, rep = gate_points(fr["points"], fr["snr"], fr["noise"])
                clusters = classify_frame(kept)
                emit("C", t_us, json.dumps({
                    "frame": fr["frame"], "clusters": clusters,
                    "reduction": {"in": rep.n_in, "out": rep.n_out,
                                  "ratio": rep.reduction_ratio,
                                  "weak": rep.dropped_weak,
                                  "fov": rep.dropped_fov,
                                  "isolated": rep.dropped_isolated}}))

        n += 1
        if n % RGB_EVERY == 0:
            r = grab(rgb)
            if r is not None:
                emit("R", time.ticks_us(), ubinascii.b2a_base64(
                    r.compress(quality=RGB_QUALITY).bytearray())[:-1].decode())


if __name__ == "__main__":
    run()
