# stage1_uart_sanity.py — Stage 1 of the N6 <-> AWR1843 mmWave TLV library.
#
# Clean-room build: written from the staged build plan spec only.
# Wiring/baud sanity ONLY — no frame parsing, no camera.
#
# What it reports every stats window:
#   bytes/s   -> is anything arriving, at a steady rate?
#   frames/s  -> mmWave magic words per second (compare to the radar cfg
#                framePeriodicity, e.g. 100 ms -> ~10 frames/s)
#   byte-gap  -> distance in BYTES between consecutive magic words. This
#                approximates totalPacketLen, so it doubles as a frame-size
#                probe: wild swings = lost/garbled bytes (wiring, baud,
#                noise), not radar behavior. Byte distance is also immune
#                to read batching, unlike wall-clock jitter.
#
# GATE (all must hold for ~60 s before moving to Stage 2):
#   1. hexdump shows the sequence 02 01 04 03 06 05 08 07
#   2. frames/s matches the radar cfg frame rate
#   3. byte-gap min/max sit in a tight, sane band (typically a few KB,
#      varying only with the number of detected objects)
#
# If bytes/s == 0       -> wiring (DATA_TX -> N6 RX), shared ground, baud,
#                          or the radar isn't streaming (no sensorStart).
#                          Hardware problem — stop here, don't touch code.
# If bytes but no magic -> wrong UART tapped (config @115200 instead of
#                          DATA @921600), wrong baud, or noise on the line.
#
# Run: copy to the N6 / run from OpenMV IDE. Standalone.

import time
from machine import UART

# --- config -------------------------------------------------------------
UART_ID = 3            # N6 UART wired to the radar DATA line (check pinout)
BAUD = 921600          # AWR1843 mmWave demo DATA UART, 8N1
RXBUF = 32 * 1024      # generous; a frame is typically a few KB
HEXDUMP_LIMIT = 256    # hexdump this many first bytes, then stats only
STATS_MS = 2000        # stats window length
MAX_GAPS = 128         # cap per-window gap samples (memory safety)

MAGIC = bytes((0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07))

# --- helpers ------------------------------------------------------------

def hexdump(chunk, base):
    for off in range(0, len(chunk), 16):
        row = chunk[off:off + 16]
        print("%06x  %s" % (base + off, " ".join("%02x" % b for b in row)))

# --- main ---------------------------------------------------------------
uart = UART(UART_ID, BAUD, bits=8, parity=None, stop=1, timeout=0,
            rxbuf=RXBUF)

print("stage1: UART%d @ %d 8N1 (rxbuf %d)" % (UART_ID, BAUD, RXBUF))
print("stage1: expecting magic %s" % " ".join("%02x" % b for b in MAGIC))

pos = 0                # absolute byte offset since start of stream
carry = b""            # last 7 bytes of previous read, so a magic word
                       # split across two reads is still found
dumped = 0
magic_total = 0
last_magic_pos = None  # absolute offset of the last magic seen

win_start = time.ticks_ms()
win_bytes = 0
win_magics = 0
win_gaps = []          # byte gaps between consecutive magics, this window

while True:
    data = uart.read()

    if data:
        win_bytes += len(data)

        if dumped < HEXDUMP_LIMIT:
            part = data[:HEXDUMP_LIMIT - dumped]
            hexdump(part, dumped)
            dumped += len(part)
            if dumped >= HEXDUMP_LIMIT:
                print("stage1: hexdump done, stats every %d ms" % STATS_MS)

        # search carry+data so boundary-straddling magics are counted;
        # carry is only 7 bytes so it can never hold a full magic twice
        buf = carry + data
        base = pos - len(carry)
        i = buf.find(MAGIC)
        while i >= 0:
            p = base + i
            if last_magic_pos is not None and len(win_gaps) < MAX_GAPS:
                win_gaps.append(p - last_magic_pos)
            last_magic_pos = p
            magic_total += 1
            win_magics += 1
            i = buf.find(MAGIC, i + len(MAGIC))
        carry = buf[-(len(MAGIC) - 1):]
        pos += len(data)
    else:
        time.sleep_ms(2)

    now = time.ticks_ms()
    if time.ticks_diff(now, win_start) >= STATS_MS:
        secs = time.ticks_diff(now, win_start) / 1000
        line = "stage1: %6.0f B/s | %5.1f frames/s | magics total %d" % (
            win_bytes / secs, win_magics / secs, magic_total)
        if win_gaps:
            line += " | gap min/avg/max %d/%d/%d B" % (
                min(win_gaps),
                sum(win_gaps) // len(win_gaps),
                max(win_gaps))
        print(line)

        if win_bytes == 0:
            print("stage1:   no bytes -- wiring? ground? baud? sensorStart?")
        elif win_magics == 0:
            print("stage1:   bytes but no magic -- DATA port (not config)?"
                  " baud? noise?")

        win_start = now
        win_bytes = 0
        win_magics = 0
        win_gaps = []
