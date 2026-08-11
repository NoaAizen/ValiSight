#!/usr/bin/env python3
"""Stage 1 of putting the radar on the BOARD's UART instead of the host's USB.

    ./radar_stage1_n6.py                    # 60 s, radar only
    ./radar_stage1_n6.py --camera --seconds 180
    ./radar_stage1_n6.py --uart 3           # if the wiring ever moves

Wiring, verified on this board 2026-08-09 -- UART7 is the ONLY conflict-free
port and this is not a preference:

    radar DATA_TX  ->  N6 P13 = PE7 = UART7_RX   (idles with PULL_UP)
    N6 P14 = PE8   ->  UART7_TX (unused; the demo needs no back-channel)
    grounds tied together

    UART2 is the CYW43 Bluetooth. UART3 (P4/P5) shares pins with I2C2, which is
    the FIR/TOF bus. UART4 (P2/P3) shares pins with SPI2, the display. PE7/PE8
    appear nowhere in boards/OPENMV_N6/pins_config.h, and the CSI owns
    PE0/1/2/3/5/6/9/10/13/14/15, so UART7 misses the camera entirely.
    P10 = PD6 = CSI_FSYNC is driven by the camera -- never wire to it.

WHY THIS STAGE EXISTS. MicroPython's stm32 UART has no DMA: uart.c is per-byte
IRQ into a ring buffer, and on a full buffer it DROPS THE BYTE SILENTLY, with
no error and no flag (lib/micropython/ports/stm32/uart.c:1293). At 921600 that
is ~92k interrupts/s. A Lepton snapshot() blocks for 113 ms, during which
~10.4 KB arrives with nobody reading. So the question is not "do bytes arrive"
but "do ALL of them arrive, while the cameras are running" -- which is what
--camera measures and what the host-USB path never has to answer.

The accounting is exact rather than statistical: each magic word declares its
own totalPacketLen, so the byte distance to the next magic must equal it.
"""
import argparse
import sys
import time

import serial

PORT = '/dev/ttyACM0'

BOARD_CODE = r'''
import time
from machine import UART

UART_ID = __UART__
BAUD = 921600          # requested; the N6 divisor actually yields 917431 (-0.45%)
RXBUF = __RXBUF__      # 32 KB minimum: a blocked 113 ms snapshot lets 10.4 KB in
DURATION_MS = __MS__
WINDOW_MS = 2000
HEXDUMP = __HEX__
WITH_CAMERA = __CAM__

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
TOTAL_OFF = 12         # magic(8) + version(4) -> totalPacketLen

lep = rgb = None
if WITH_CAMERA:
    # Visible sensor first, then the Lepton soft-reset. The reset line is shared
    # across CSI devices and a hard reset of one while the other streams locks
    # the bus; this is the same order capture.py uses.
    import csi
    try:
        rgb = csi.CSI(cid=0x7936)
        rgb.reset()
        rgb.pixformat(csi.GRAYSCALE)
        rgb.framesize(csi.VGA)
        lep = csi.CSI(cid=0x5435)
        lep.reset(hard=False)
        lep.pixformat(csi.GRAYSCALE)
        lep.framesize(csi.QQVGA)
        lep.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)   # radiometry, HIGH gain
        time.sleep_ms(2000)                                  # VoSPI sync settle
        lep.snapshot()                                       # first one costs ~1.7 s
        print("stage1: cameras up (PAG + Lepton)")
    except Exception as e:
        print("stage1: camera bring-up failed: %s" % e)
        lep = None

uart = UART(UART_ID, BAUD, bits=8, parity=None, stop=1, timeout=0, rxbuf=RXBUF)
print("stage1: UART%d @%d 8N1 rxbuf=%d camera=%s" % (UART_ID, BAUD, RXBUF,
                                                     lep is not None))
print("stage1: gate is lost=0 for the whole run")

def hexdump(chunk, base):
    for off in range(0, len(chunk), 16):
        row = chunk[off:off + 16]
        print("%06x  %s" % (base + off, " ".join("%02x" % b for b in row)))

carry = b""
pos = 0                # absolute offset of the start of carry
dumped = 0
magics = 0
good = 0
bad = 0
lost = 0
last_pos = None
last_total = None
totals_lo = 0
totals_hi = 0
win_bytes = 0
win_magics = 0
snaps = 0
win_start = time.ticks_ms()
t0 = win_start

# No gc.collect() anywhere in this loop, and none after it while the Lepton is
# up: a collect wedges that CSI object permanently and only a fresh csi.CSI()
# recovers it. The loop is written to allocate as little as possible for the
# same reason -- the automatic collector is the thing being kept away.
while time.ticks_diff(time.ticks_ms(), t0) < DURATION_MS:
    data = uart.read()

    if data:
        win_bytes += len(data)
        if dumped < HEXDUMP:
            take = data[:HEXDUMP - dumped]
            hexdump(take, dumped)
            dumped += len(take)
            if dumped >= HEXDUMP:
                print("stage1: hexdump done, stats every %d ms" % WINDOW_MS)

        buf = carry + data
        base = pos
        i = buf.find(MAGIC)
        while i >= 0:
            if i + TOTAL_OFF + 4 > len(buf):
                break                      # totalPacketLen not here yet
            total = (buf[i + TOTAL_OFF] | (buf[i + TOTAL_OFF + 1] << 8) |
                     (buf[i + TOTAL_OFF + 2] << 16) |
                     (buf[i + TOTAL_OFF + 3] << 24))
            p = base + i
            magics += 1
            win_magics += 1
            if last_pos is not None:
                gap = p - last_pos
                if gap == last_total:
                    good += 1
                else:
                    bad += 1
                    lost += gap - last_total
                    if bad <= 8:
                        print("stage1:   gap %d expected %d (%+d B) after %d frames"
                              % (gap, last_total, gap - last_total, magics))
            if totals_lo == 0 or total < totals_lo:
                totals_lo = total
            if total > totals_hi:
                totals_hi = total
            last_pos = p
            last_total = total
            i = buf.find(MAGIC, i + 8)

        if i >= 0:
            carry = buf[i:]                # replay the incomplete header
        else:
            carry = buf[-7:]               # possible split magic
        pos = base + len(buf) - len(carry)

    if lep is not None:
        # The whole point of --camera: snapshot() parks us in the driver for
        # 113 ms per frame with nobody draining the UART ring.
        try:
            lep.snapshot()
            snaps += 1
        except Exception as e:
            print("stage1: snapshot failed: %s" % e)
            lep = None
    elif not data:
        time.sleep_ms(2)

    now = time.ticks_ms()
    if time.ticks_diff(now, win_start) >= WINDOW_MS:
        secs = time.ticks_diff(now, win_start) / 1000
        print("stage1: %7.0f B/s | %4.1f frames/s | magics %d | frame %d..%d B"
              " | ok %d bad %d lost %+d B | snaps %d" % (
                  win_bytes / secs, win_magics / secs, magics,
                  totals_lo, totals_hi, good, bad, lost, snaps))
        if win_bytes == 0:
            print("stage1:   no bytes -- DATA_TX on P13? ground? sensorStart?")
        elif win_magics == 0:
            print("stage1:   bytes but no magic -- CLI port tapped, or baud")
        win_start = now
        win_bytes = 0
        win_magics = 0

uart.deinit()
print("stage1: %s -- %d frames, %d good gaps, %d bad, %+d bytes, %d snapshots"
      % ("PASS" if (magics > 0 and bad == 0) else "FAIL",
         magics, good, bad, lost, snaps))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default=PORT)
    ap.add_argument('--uart', type=int, default=7)
    ap.add_argument('--rxbuf', type=int, default=32 * 1024)
    ap.add_argument('--seconds', type=float, default=60.0)
    ap.add_argument('--hexdump', type=int, default=256)
    ap.add_argument('--camera', action='store_true',
                    help='run both sensors while listening -- the real test')
    args = ap.parse_args()

    code = (BOARD_CODE
            .replace('__UART__', str(args.uart))
            .replace('__RXBUF__', str(args.rxbuf))
            .replace('__MS__', str(int(args.seconds * 1000)))
            .replace('__HEX__', str(args.hexdump))
            .replace('__CAM__', str(bool(args.camera))))

    s = serial.Serial(args.port, 115200, timeout=0.2, write_timeout=5)
    try:
        s.write(b'\r\x03\x03')                  # interrupt anything running
        time.sleep(0.3)
        s.reset_input_buffer()
        s.write(b'\x01')                        # raw REPL
        time.sleep(0.3)
        s.read_all()
        s.write(code.encode() + b'\x04')

        # Stream the board's lines as they come; a 60 s run with nothing on
        # screen is indistinguishable from a hung board.
        buf = b''
        deadline = time.time() + args.seconds + 60
        while time.time() < deadline:
            chunk = s.read(4096)
            if not chunk:
                continue
            buf += chunk
            if buf.startswith(b'OK'):        # raw-REPL ack, not board output
                buf = buf[2:]
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                text = line.decode('utf8', 'replace').rstrip('\r')
                if text not in ('OK', ''):
                    print(text)
            if buf.count(b'\x04') >= 2:
                break
        tail = buf.replace(b'\x04', b'').decode('utf8', 'replace').strip()
        if tail:
            print(tail)
    except KeyboardInterrupt:
        pass
    finally:
        s.write(b'\x03\x02')                    # interrupt, back to friendly REPL
        time.sleep(0.2)
        s.read_all()
        s.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
