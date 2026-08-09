#!/usr/bin/env python3
"""Read the IWR1843 DATA UART on the host: sanity-check it, parse it, record it.

    ./radar_listen.py --stage1              # link sanity, no parsing (do this first)
    ./radar_listen.py                       # live parsed points
    ./radar_listen.py --record ../captures/radar_calib -n 300

The radar is on USB here (XDS110 -> /dev/ttyACM2 @921600), which is the path
that works today and the one the calibration will be done over. The board path
(radar into the N6's UART7) is a separate question answered by
radar_stage1_n6.py; this file is deliberately independent of it, and both use
the same radar/mmwave.py.

--stage1 is not a warm-up. It is the gate: until the byte accounting is exact,
a missing point is indistinguishable from a dropped byte, and every hour spent
on detections above a lossy link is wasted. It passes when `lost` is 0.
"""
import argparse
import json
import os
import sys
import time

import serial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'radar'))
import mmwave  # noqa: E402

DEFAULT_PORT = '/dev/ttyACM2'
BAUD = 921600


def hexdump(data, base=0):
    for off in range(0, len(data), 16):
        row = data[off:off + 16]
        print('%06x  %s' % (base + off, ' '.join('%02x' % b for b in row)))


def stage1(ser, seconds, window, hexdump_bytes):
    """Raw link sanity: bytes/s, frames/s, and exact byte accounting.

    The magic word alone only proves the baud is right. What decides whether
    the link is usable is whether the byte distance between consecutive magics
    equals the totalPacketLen the earlier one declared -- see
    mmwave.MagicScanner.
    """
    scan = mmwave.MagicScanner()
    dumped = 0
    t_end = time.time() + seconds if seconds else None
    t_win = time.time()
    win_bytes = 0
    win_magics = 0
    total_bytes = 0
    print('stage1: %s @%d, window %.1fs -- gate is lost=0' % (
        ser.port, BAUD, window))

    while t_end is None or time.time() < t_end:
        data = ser.read(4096)
        if data:
            total_bytes += len(data)
            win_bytes += len(data)
            if dumped < hexdump_bytes:
                take = data[:hexdump_bytes - dumped]
                hexdump(take, dumped)
                dumped += len(take)
                if dumped >= hexdump_bytes:
                    print('stage1: hexdump done, stats every %.1fs' % window)
            before = scan.magics
            scan.feed(data)
            win_magics += scan.magics - before
        else:
            time.sleep(0.002)

        now = time.time()
        if now - t_win >= window:
            secs = now - t_win
            line = 'stage1: %7.0f B/s | %4.1f frames/s | magics %d' % (
                win_bytes / secs, win_magics / secs, scan.magics)
            if scan.totals:
                line += ' | frame %d..%d B' % (min(scan.totals),
                                               max(scan.totals))
            line += ' | ok %d bad %d lost %+d B' % (
                scan.good_gaps, scan.bad_gaps, scan.lost_bytes)
            print(line)
            if win_bytes == 0:
                print('stage1:   no bytes -- wrong port, or sensorStart never sent')
            elif win_magics == 0:
                print('stage1:   bytes but no magic -- CLI port instead of DATA,'
                      ' or wrong baud')
            elif scan.bad_gaps:
                bad = [g for g in scan.gaps if g[0] != g[1]][:3]
                print('stage1:   BYTES LOST -- gap/expected %s' % bad)
            t_win = now
            win_bytes = 0
            win_magics = 0
            scan.window_reset()

    ok = scan.magics > 0 and scan.bad_gaps == 0
    print('\nstage1: %s -- %d frames, %d good gaps, %d bad, %+d bytes' % (
        'PASS' if ok else 'FAIL', scan.magics, scan.good_gaps,
        scan.bad_gaps, scan.lost_bytes))
    return 0 if ok else 1


def listen(ser, args):
    """Parse frames and either print them or record them for calibration."""
    sync = mmwave.FrameSync()
    raw_f = json_f = None
    if args.record:
        os.makedirs(args.record, exist_ok=True)
        raw_f = open(os.path.join(args.record, 'radar.bin'), 'wb')
        json_f = open(os.path.join(args.record, 'radar.jsonl'), 'w')
        print('recording to %s/ (radar.bin + radar.jsonl)' % args.record)

    n = 0
    dropped_last = 0
    t0 = time.time()
    prev_frame_no = None
    gaps = 0
    try:
        while args.count == 0 or n < args.count:
            data = ser.read(4096)
            t = time.time()
            if not data:
                time.sleep(0.002)
                continue
            if raw_f:
                raw_f.write(data)
            # FrameSync hands back the time each frame's LAST BYTE arrived,
            # which is not always this read: a frame whose successor has only
            # partly arrived is held until the framing is unambiguous. Using
            # the release time instead would put a frame period of bias into
            # the radar-to-camera offset, in the one direction, every time.
            for t_frame, frame in sync.feed(data, t):
                try:
                    fr = mmwave.parse_frame(frame)
                except ValueError as e:
                    print('parse error: %s' % e)
                    continue
                n += 1
                if prev_frame_no is not None and \
                        fr['frame_number'] != prev_frame_no + 1:
                    gaps += 1
                prev_frame_no = fr['frame_number']

                issues = mmwave.validate_points(
                    fr['points'], min_range=mmwave.CFG_10HZ['hpf_blind_m'])
                if json_f:
                    json_f.write(json.dumps({
                        't': t_frame,
                        'frame': fr['frame_number'],
                        'points': [[p['x'], p['y'], p['z'], p['v'],
                                    p['snr'], p['noise']] for p in fr['points']],
                        'convention': 'x_fwd_y_left_z_up',
                    }) + '\n')
                if args.quiet:
                    continue
                line = 'frame %6d  t=%8.3f  %2d pts  tlv=%s  len=%s' % (
                    fr['frame_number'], t_frame - t0, len(fr['points']),
                    fr['tlv_types'], fr['length_mode'])
                if fr['stats']:
                    # The radar's own margin. At zero the DSP is overrunning
                    # the 100 ms frame period and frames go missing on the
                    # sensor, not on the link -- a different bug entirely.
                    line += '  margin=%d us' % fr['stats']['interframe_margin_us']
                print(line)
                for p in fr['points'][:args.show]:
                    print('    fwd %6.2f  left %6.2f  up %6.2f  v %+6.3f  '
                          'snr %s dB  noise %s dB  r %5.2f m' % (
                              p['x'], p['y'], p['z'], p['v'],
                              '%5.1f' % p['snr'] if p['snr'] is not None else '  n/a',
                              '%5.1f' % p['noise'] if p['noise'] is not None else '  n/a',
                              mmwave.range_of(p)))
                for msg in issues:
                    print('    !! %s' % msg)
            if sync.dropped_bytes != dropped_last:
                print('  (resync: %d bytes dropped, %d resyncs)' % (
                    sync.dropped_bytes, sync.resync_count))
                dropped_last = sync.dropped_bytes
    except KeyboardInterrupt:
        pass
    finally:
        if raw_f:
            raw_f.close()
        if json_f:
            json_f.close()

    dt = time.time() - t0
    print('\n%d frames in %.1fs (%.2f fps), %d frame-number gaps, '
          '%d bytes dropped in %d resyncs' % (
              n, dt, n / dt if dt else 0, gaps,
              sync.dropped_bytes, sync.resync_count))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default=DEFAULT_PORT)
    ap.add_argument('--stage1', action='store_true',
                    help='raw link sanity with exact byte accounting, no parsing')
    ap.add_argument('--seconds', type=float, default=60.0,
                    help='stage1 duration, 0 = until Ctrl-C (default 60)')
    ap.add_argument('--window', type=float, default=2.0)
    ap.add_argument('--hexdump', type=int, default=256)
    ap.add_argument('-n', '--count', type=int, default=0,
                    help='stop after N frames (0 = until Ctrl-C)')
    ap.add_argument('--record', metavar='DIR',
                    help='write radar.bin (raw) + radar.jsonl (parsed)')
    ap.add_argument('--show', type=int, default=4,
                    help='print at most this many points per frame')
    ap.add_argument('-q', '--quiet', action='store_true')
    args = ap.parse_args()

    with serial.Serial(args.port, BAUD, timeout=0.05) as ser:
        if args.stage1:
            return stage1(ser, args.seconds, args.window, args.hexdump)
        return listen(ser, args)


if __name__ == '__main__':
    sys.exit(main())
