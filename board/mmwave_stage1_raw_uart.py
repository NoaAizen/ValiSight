# mmwave_stage1_raw_uart.py — Stage 1 of the N6<->AWR1843 TLV library bring-up.
#
# Purpose: wiring/baud sanity ONLY. No parsing. Reads raw bytes from the radar
# DATA UART and reports whether the mmWave magic word appears at a stable rate.
#
# Gate (do not proceed to Stage 2 until BOTH hold):
#   1. Magic word 02 01 04 03 06 05 08 07 repeats at a stable rate
#      (should match the frame rate in your radar .cfg, e.g. ~10 Hz).
#   2. No RX overruns: "bytes/s" stays steady and no gaps/garbage between
#      magics beyond a sane frame size.
#
# If NO magic appears: wiring (DATA_TX -> N6 RX), shared ground, baud, or the
# radar isn't streaming (sensorStart not sent). Not a code problem — stop here.
#
# Run: copy to the N6 (or run from OpenMV IDE). Standalone, no camera.

import time
from machine import UART

# --- config ------------------------------------------------------------------
UART_ID = 3          # OpenMV N6: UART 3 is on P4 (TX) / P5 (RX). Adjust to wiring.
BAUD = 921600        # AWR1843 mmWave demo DATA UART, 8N1
RX_BUF = 16 * 1024   # generous: ~1 frame is typically < 4 KB
HEXDUMP_BYTES = 512  # hexdump only the first N bytes, then stats only
STATS_EVERY_MS = 2000

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"

# -----------------------------------------------------------------------------
uart = UART(UART_ID, BAUD, bits=8, parity=None, stop=1, timeout=0, rxbuf=RX_BUF)

def hexdump(data, base=0):
    for row in range(0, len(data), 16):
        chunk = data[row:row + 16]
        print("%06x  %s" % (base + row, " ".join("%02x" % b for b in chunk)))

print("stage1: listening on UART%d @ %d 8N1 (rxbuf=%d)" % (UART_ID, BAUD, RX_BUF))
print("stage1: expecting magic %s" % " ".join("%02x" % b for b in MAGIC))

tail = b""           # last 7 bytes of previous chunk, so a magic split across
                     # chunk boundaries is still counted
dumped = 0
total_bytes = 0
magic_count = 0
last_magic_ms = None
magic_intervals = []  # ms between consecutive magics, for rate stability
window_bytes = 0
window_magics = 0
last_stats_ms = time.ticks_ms()

while True:
    chunk = uart.read()
    now = time.ticks_ms()

    if chunk:
        total_bytes += len(chunk)
        window_bytes += len(chunk)

        if dumped < HEXDUMP_BYTES:
            take = chunk[:HEXDUMP_BYTES - dumped]
            hexdump(take, base=dumped)
            dumped += len(take)
            if dumped >= HEXDUMP_BYTES:
                print("stage1: hexdump limit reached, switching to stats only")

        # count magics, including ones straddling the previous chunk
        buf = tail + chunk
        idx = buf.find(MAGIC)
        while idx >= 0:
            magic_count += 1
            window_magics += 1
            if last_magic_ms is not None:
                magic_intervals.append(time.ticks_diff(now, last_magic_ms))
                if len(magic_intervals) > 50:
                    magic_intervals.pop(0)
            last_magic_ms = now
            idx = buf.find(MAGIC, idx + len(MAGIC))
        tail = buf[-(len(MAGIC) - 1):]

    if time.ticks_diff(now, last_stats_ms) >= STATS_EVERY_MS:
        secs = time.ticks_diff(now, last_stats_ms) / 1000
        if magic_intervals:
            avg = sum(magic_intervals) / len(magic_intervals)
            jit = max(magic_intervals) - min(magic_intervals)
            rate = "avg frame interval %.0f ms (jitter %d ms over last %d)" % (
                avg, jit, len(magic_intervals))
        else:
            rate = "no magic yet"
        print("stage1: %d B/s | magics total %d (+%d) | %s" % (
            window_bytes / secs, magic_count, window_magics, rate))
        window_bytes = 0
        window_magics = 0
        last_stats_ms = now

    if not chunk:
        time.sleep_ms(2)
