#!/usr/bin/env python3
"""Find the hardware timestamping route for cross-sensor sync. Read-only probe.

    ./radar_sync_probe.py            # inventory + IC routes + dual capture
    ./radar_sync_probe.py --camera   # and again with both sensors streaming

WHAT THIS IS FOR. Thermal runs at 8.772 Hz, the visible sensor at 30, the radar
at 10, and none of the three is disciplined to the others. Pairing them in
software costs a timestamp taken after the fact, in Python, behind a snapshot()
that blocks for 113 ms -- which is a jitter floor of tens of milliseconds on a
sensor whose frame period is 114. A timer input capture latches the edge in
hardware and is read later, so the number is the edge's time and not the
interpreter's. This probe answers whether the N6 has two such channels free.

WHAT IS ALREADY KNOWN from boards/OPENMV_N6/pins_config.h, so the probe does
not have to rediscover it:

    TIM1 is the camera's, driving CSI_CLK on PE9 (AF1). Do not plan around it.
    P10 = PD6 = CSI_FSYNC is an OUTPUT the board drives to the sensor. It is
      the frame-sync signal worth capturing, which means jumpering P10 to one
      of the free pins below rather than configuring P10 itself for capture.
    P13/P14 = PE7/PE8 = UART7 carry the radar (see radar_stage1_n6.py).
    P0-P3, P6, P7, P8 are the SPI display; P4/P5 are I2C2, the FIR/TOF bus.

  So the candidates are the pins that appear in neither file: P9 (PG12),
  P11 (PC13), P15 (PE11), P16 (PE12), P17 (PB6), P18 (PB7), P6_ADC (PA5).

Part C is the question that decides the design. Two captures on ONE timer share
one counter, so the two edges are directly subtractable. Two separate timers
mean two clocks to relate, and their drift becomes another calibration.
"""
from pathlib import Path
import argparse
import sys
import time

import serial

PORT = '/dev/ttyACM0'

BOARD_CODE = (
    Path(__file__).resolve().parents[1] / "board" / "templates" / "radar_sync_probe.py.tmpl"
).read_text(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default=PORT)
    ap.add_argument('--camera', action='store_true',
                    help='bring both sensors up first -- part B and C then '
                         'report what is left over, which is what matters')
    ap.add_argument('--pair', metavar='TIM,PINA,CHA,PINB,CHB',
                    help="part C's pair, e.g. '15,P17,1,P18,2'")
    ap.add_argument('--prescaler', type=int, default=3200,
                    help='timer clock divider. 3200 on the 400 MHz source is '
                         '8 us per tick, which keeps a 16-bit counter from '
                         'wrapping inside a 114 ms thermal frame (default)')
    args = ap.parse_args()

    pair = 'None'
    if args.pair:
        t, pa, ca, pb, cb = [x.strip() for x in args.pair.split(',')]
        pair = "(%d, '%s', %d, '%s', %d)" % (int(t), pa, int(ca), pb, int(cb))

    code = (BOARD_CODE
            .replace('__CAM__', str(bool(args.camera)))
            .replace('__PAIR__', pair)
            .replace('__PRESC__', str(args.prescaler)))

    s = serial.Serial(args.port, 115200, timeout=0.2, write_timeout=5)
    try:
        s.write(b'\r\x03\x03')
        time.sleep(0.3)
        s.reset_input_buffer()
        s.write(b'\x01')
        time.sleep(0.3)
        s.read_all()
        s.write(code.encode() + b'\x04')

        buf = b''
        deadline = time.time() + 180
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
        s.write(b'\x03\x02')
        time.sleep(0.2)
        s.read_all()
        s.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
