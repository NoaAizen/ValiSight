"""Radar point-cloud processing that runs ON the OpenMV N6 (MicroPython).

Edge version of the PC radar path: instead of the PC parsing the IWR1843 and
clustering/classifying, the N6 does it and streams only the finished clusters
over USB. The heavy lifting (TLV parse + cluster + features + classify) is the
same pure code the PC uses — `iwr1843_uart.RadarReader` and
`radar_classify_n6.classify_frame` — both already MicroPython-clean (pure
struct / math, no numpy).

Run it with the repo mounted so those imports resolve from the PC filesystem:
    python -m mpremote connect COM12 mount . run n6_radar_classify.py
(the PC-side `view_radar_n6.py` does exactly this and draws the result).

Line protocol over USB VCP (one record per line):
    C:<json>   {"frame": int, "n": int, "clusters": [cluster_dict, ...]}
    L:<text>   a log/status line

HARDWARE WIRING (required):
  The IWR1843 DATA/AUX UART (921600 baud, 3V3) must be wired to the N6:
      IWR1843 DATA_UART TX  ->  N6 UART RX pin (RADAR_UART below)
      GND                   ->  GND  (shared ground is mandatory)
  The radar's CONFIG UART still needs the .cfg sent once to start streaming —
  do that from the PC (view_radar_n6.py --cfg-port ...) or a second N6 UART.
"""
import time

try:
    from machine import UART            # on-device only
except ImportError:                     # importable on the PC for syntax checks
    UART = None

from iwr1843_uart import RadarReader
from radar_classify_n6 import classify_frame

# N6 UART wired to the radar DATA line. Pick the UART id whose RX pin you wired;
# on the N6 header UART(1) is the usual choice. 921600 = the SDK demo DATA baud.
RADAR_UART = 1
RADAR_BAUD = 921600
POLL_MS = 5


def _emit(tag, obj):
    import json
    print(tag + ":" + json.dumps(obj))


def run():
    if UART is None:
        raise RuntimeError("machine.UART unavailable — run this on the N6")
    uart = UART(RADAR_UART, RADAR_BAUD, bits=8, parity=None, stop=1, timeout=5)
    radar = RadarReader()
    print("L:N6 radar classify — reading UART%d @ %d" % (RADAR_UART, RADAR_BAUD))
    while True:
        if uart.any():
            for fr in radar.feed(uart.read()):
                clusters = classify_frame(fr["points"])
                _emit("C", {"frame": fr["frame"],
                            "n": len(clusters),
                            "clusters": clusters})
        else:
            time.sleep_ms(POLL_MS)


if __name__ == "__main__":
    run()
