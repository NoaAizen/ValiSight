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
import argparse
import sys
import time

import serial

PORT = '/dev/ttyACM0'

BOARD_CODE = r'''
import pyb

WITH_CAMERA = __CAM__

# Header pins that pins_config.h does not claim. P10 is excluded on purpose:
# the board drives it as CSI_FSYNC, so it is a source to jumper FROM, not a pin
# to capture ON.
PINS = ['P9', 'P11', 'P15', 'P16', 'P17', 'P18', 'P6_ADC']

# Sixteen-bit timers wrap every few ms at a useful prescale, which is fine for
# an interval but useless as a shared timebase, so part A asks the width rather
# than assuming it. TIM1 is listed only to see it fail once the camera is up.
TIMERS = [1, 2, 3, 4, 5, 8, 12, 13, 14, 15, 16, 17]

cam = None
if WITH_CAMERA:
    import csi, time as _t
    try:
        rgb = csi.CSI(cid=0x7936)
        rgb.reset()
        rgb.pixformat(csi.GRAYSCALE)
        rgb.framesize(csi.VGA)
        cam = csi.CSI(cid=0x5435)
        cam.reset(hard=False)
        cam.pixformat(csi.GRAYSCALE)
        cam.framesize(csi.QQVGA)
        cam.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)
        _t.sleep_ms(2000)
        cam.snapshot()
        print("cameras up (PAG + Lepton)")
    except Exception as e:
        print("camera bring-up failed: %s -- probing without it" % e)
        cam = None

print("=" * 66)
print("A: timer inventory -- width and source clock  (camera=%s)" % (cam is not None))
print("=" * 66)
# Ask for a wide period and read it back: a 16-bit timer keeps only the low
# half. The request is 0x3FFFFFFF, not 0xFFFFFFFF -- MicroPython's small int is
# 31-bit here and the larger value raises "overflow converting long int to
# machine word" for EVERY timer, which reads as "no timers exist" rather than
# as a bug in the probe.
PROBE_PERIOD = 0x3FFFFFFF
for t in TIMERS:
    try:
        tim = pyb.Timer(t, prescaler=0, period=PROBE_PERIOD)
        real = tim.period()
        print("TIM%-2d  width=%2d-bit  period_readback=0x%X  source=%d Hz"
              % (t, 32 if real > 0xFFFF else 16, real, tim.source_freq()))
        tim.deinit()
    except Exception as e:
        print("TIM%-2d  -- %s" % (t, e))

print()
print("=" * 66)
print("B: which TIMER can capture on each free pin")
print("=" * 66)
# READ THIS BEFORE TRUSTING THE CHANNEL NUMBERS. pyb resolves the pin through
# the alternate-function table, and an AF names a TIMER, not a channel -- so
# channel() accepts ch=1..4 on a timer that physically has two. Measured here:
# TIM15 accepted all four channels on P17. The timer column is evidence; the
# channel column is a candidate list that only a real edge can settle.
# af_list() is printed alongside because it is the board's own answer rather
# than this probe's inference.
for pin_name in PINS:
    found = []
    for t in TIMERS:
        for ch in (1, 2, 3, 4):
            tim = None
            try:
                tim = pyb.Timer(t, prescaler=399, period=0xFFFF)
                tim.channel(ch, pyb.Timer.IC, pin=pyb.Pin(pin_name),
                            polarity=pyb.Timer.RISING)
                found.append("TIM%d_CH%d" % (t, ch))
            except Exception:
                pass
            if tim is not None:
                try:
                    tim.deinit()
                except Exception:
                    pass
    try:
        afs = [str(a).split('.')[-1] for a in pyb.Pin(pin_name).af_list()]
        afs = [a for a in afs if 'TIM' in a]
    except Exception:
        afs = []
    print("%-7s %s" % (pin_name + ":",
                       ", ".join(found) if found else "no IC route"))
    print("        pin AFs naming a timer: %s" % (", ".join(afs) or "none"))

print()
print("=" * 66)
print("C: two capture channels on ONE timer, at the same time")
print("=" * 66)
# Fill these in from part B: the pair must share a timer, or the two edges are
# measured against two different clocks and the drift between them becomes yet
# another thing to calibrate.
PAIR = __PAIR__          # (timer, pin_a, channel_a, pin_b, channel_b)
if PAIR is None:
    print("SKIPPED -- pass --pair 'TIM,PINA,CHA,PINB,CHB' from part B's output")
else:
    t, pin_a, ch_a, pin_b, ch_b = PAIR
    try:
        PRESCALER = __PRESC__
        tim = pyb.Timer(t, prescaler=PRESCALER - 1, period=0x3FFFFFFF)
        a = tim.channel(ch_a, pyb.Timer.IC, pin=pyb.Pin(pin_a),
                        polarity=pyb.Timer.RISING)
        b = tim.channel(ch_b, pyb.Timer.IC, pin=pyb.Pin(pin_b),
                        polarity=pyb.Timer.RISING)
        print("SUCCESS: TIM%d holds both captures (%s ch%d, %s ch%d)"
              % (t, pin_a, ch_a, pin_b, ch_b))
        # Read the period BACK. A 16-bit timer silently keeps the low half of
        # the request, so computing the wrap from what was ASKED FOR overstates
        # it by 16384x -- which is the difference between "wraps every 9
        # minutes, handle it as an exception" and "wraps three times per thermal
        # frame, the whole timestamp is wrong".
        period = tim.period()
        tick_ns = 1e9 * PRESCALER / tim.source_freq()
        print("  requested period 0x3FFFFFFF, hardware kept 0x%X" % period)
        print("  tick = %.0f ns, wrap every %.1f ms"
              % (tick_ns, (period + 1) * tick_ns / 1e6))
        print("  counter now = %d (free-running)" % tim.counter())
        print("  capture A = %d, capture B = %d (both meaningless until an"
              " edge arrives)" % (a.capture(), b.capture()))
        tim.deinit()
    except Exception as e:
        print("FAILED: %s" % e)
        if cam is not None:
            print("  -> with the camera up this is the conflict to expect;"
                  " re-run part B and pick a timer the camera does not use")
print()
print("probe done")
'''


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
