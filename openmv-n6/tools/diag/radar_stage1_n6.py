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
from pathlib import Path
import argparse
import sys
import time

import serial

PORT = '/dev/ttyACM0'

BOARD_CODE = (
    Path(__file__).resolve().parents[1] / "board" / "templates" / "radar_stage1.py.tmpl"
).read_text(encoding="utf-8")


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
