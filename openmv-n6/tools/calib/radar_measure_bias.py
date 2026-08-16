#!/usr/bin/env python3
"""Read back the IWR1843's range-bias / RX-phase measurement and reduce it to
one paste-ready line.

    ./radar_measure_bias.py                      # send the measure cfg, take 300, report
    ./radar_measure_bias.py --frames 500
    ./radar_measure_bias.py --no-send            # attach to a measurement already running
    ./radar_measure_bias.py --cfg ../../radar/configs/radar_10hz_measure_bias.cfg

WHERE THE ANSWER COMES OUT, AND WHY THIS FILE EXISTS.

`measureRangeBiasAndRxChanPhase 1 <dist> <win>` does not answer on the data
stream. The demo's MmwDemo_measurementResultOutput() calls CLI_write(), so the
result lands on the CONFIG UART at 115200, once per frame, as

    compRangeBiasAndRxChanPhase <rangeBias> <Re(0,0)> <Im(0,0)> ... <Re(T-1,R-1)> <Im(T-1,R-1)>

For the IWR1843 that is 3 TX x 4 RX = 12 complex pairs, so 24 values after the
bias: 25 numbers, exactly the arity already in the .cfg files. Nothing in this
repo read that port after `sensorStart` - send_radar_cfg.py stops at the config
- so the measurement was being produced and thrown away.

Note the asymmetry this creates and that nothing else in the tree has: the CLI
port is normally write-only to us, and here it is the only source of truth.

WHY THE MEDIAN, AND WHY THE SPREAD IS REPORTED NEXT TO IT.

The reduction is a COMPONENT-WISE MEDIAN, never a mean. The measurement takes
whatever dominates the range gate, and if anything else drifts through it for a
few frames a mean carries that away with it; a median does not.

But a median hides the very thing you need to see. T_camera_radar.json records
station V-C locking onto the EDGE OF A ROLLING TABLE instead of the corner
reflector (el -4.8, snr 18.5, against the true cluster's el 0, snr 25.3), and a
median over contaminated frames is a confident wrong number. So the spread is
printed beside every figure: a measurement of one rigid reflector on a tripod
is tight, and a wide spread means something shared the gate. The spread is the
evidence, the median is only the summary.

The die temperature is logged alongside from the DATA port, because the range
bias this corrects drifts with it (see parse_temperature in radar/mmwave.py):
a table taken cold and used warm is a slowly moving depth offset.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import serial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # openmv-n6/
sys.path.insert(0, os.path.join(ROOT, 'radar'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import mmwave                                           # noqa: E402
import send_radar_cfg                                   # noqa: E402

N_TX, N_RX = 3, 4
N_VALUES = 1 + 2 * N_TX * N_RX                          # 25 for the IWR1843
KEY = 'compRangeBiasAndRxChanPhase'
DEFAULT_CFG = os.path.join(ROOT, 'radar', 'configs',
                           'radar_people_measure_bias.cfg')

# Provisional gates. NO BASELINE EXISTS YET - the first clean run on a tripod
# is what turns these into measurements, exactly as the chirp figures in
# radar_people.cfg were turned from derived into measured. They are here to
# make a bad run loud, not to certify a good one.
#
# CONTAMINATION IS BIMODAL, SO IT IS COUNTED AND NOT MEASURED AS SPREAD. The
# first version of this file gated on p90-p10 and a synthetic run in which one
# frame in ten locked onto a second reflector PASSED: those frames sit at or
# above p90, which is exactly the part a percentile spread discards. Every
# robust width statistic fails here for the same reason - robustness means
# ignoring outliers, and the outliers ARE the signal. What separates one rigid
# target from two is not how wide the distribution is but how many frames sit
# away from the mode, so that is what is counted.
OUTLIER_BIAS_M = 0.020        # a lock on something else lands this far off
OUTLIER_COMP = 0.100
GATE_OUTLIER_FRAC = 0.02      # >2% of frames off-mode = something shared the gate
GATE_BIAS_JITTER_M = 0.010    # p90-p10 of the in-mode frames: still worth seeing
GATE_MAG_DEV = 0.30


def find_port(interface):
    """Resolve an XDS110 interface by USB identity, never by ACM number.

    Same reason as run_live.sh: the numbering on this Jetson has already
    swapped once, and every failure it causes is silent.
    """
    for p in sorted(_glob_acm()):
        try:
            props = subprocess.run(['udevadm', 'info', '-q', 'property', '-n', p],
                                   capture_output=True, text=True, timeout=5).stdout
        except (subprocess.SubprocessError, OSError):
            continue
        if 'ID_VENDOR=Texas_Instruments' in props and \
                ('ID_USB_INTERFACE_NUM=%s' % interface) in props.split('\n'):
            return p
    return None


def _glob_acm():
    import glob
    return glob.glob('/dev/ttyACM*')


def parse_result_line(line):
    """A CLI line -> [25 floats], or None if this is not one / is malformed.

    Truncation is the failure that matters. The demo prints this line every
    frame onto a port we are also draining, so a short read can hand us half a
    line; parsing it as a full one would quietly average a real sample with a
    fragment. Arity is checked, not assumed.
    """
    line = line.strip()
    if not line.startswith(KEY):
        return None
    parts = line[len(KEY):].split()
    if len(parts) != N_VALUES:
        return None
    try:
        return [float(x) for x in parts]
    except ValueError:
        return None


def target_distance_from_cfg(cfg_path):
    """The distance the measurement was told to look at -> provenance."""
    try:
        with open(cfg_path) as f:
            for raw in f:
                line = raw.strip()
                if line.startswith('measureRangeBiasAndRxChanPhase'):
                    parts = line.split()
                    if len(parts) >= 4:
                        return {'enabled': int(parts[1]),
                                'target_m': float(parts[2]),
                                'search_win_m': float(parts[3])}
    except OSError:
        pass
    return None


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def spread(xs):
    """p90-p10. Chosen over stdev for the same reason as the median: one
    contaminated frame must not be able to widen the number that decides
    whether the run is trusted."""
    s = sorted(xs)
    if len(s) < 10:
        return float('nan')
    return s[int(0.9 * (len(s) - 1))] - s[int(0.1 * (len(s) - 1))]


class DieTemp(threading.Thread):
    """Sample die temperature off the DATA port while the CLI is measuring.

    Best-effort by construction: a missing data port must not cost the
    calibration, so every failure here degrades to 'no temperature recorded'.
    """

    def __init__(self, port):
        super().__init__(daemon=True)
        # NOT self._stop: threading.Thread already has a private _stop(),
        # and shadowing it with an Event breaks join() at exit.
        self.port, self.samples, self._halt = port, [], threading.Event()

    def run(self):
        try:
            ser = serial.Serial(self.port, 921600, timeout=0.2)
        except (serial.SerialException, OSError):
            return
        sync = mmwave.FrameSync()
        try:
            while not self._halt.is_set():
                data = ser.read(4096)
                if not data:
                    continue
                for _t, frame in sync.feed(data, time.time()):
                    try:
                        fr = mmwave.parse_frame(frame)
                    except ValueError:
                        continue
                    t = fr.get('temperature')
                    # parse_temperature reports per-sensor: rx_c[4], tx_c[3],
                    # dig_c[2]. There is no single 'die' figure, so carry the
                    # hottest - that is the one a drifting bias tracks.
                    if t and t.get('valid'):
                        dies = (t.get('rx_c') or []) + (t.get('tx_c') or []) \
                            + (t.get('dig_c') or [])
                        if dies:
                            self.samples.append(max(dies))
        finally:
            try:
                ser.close()
            except Exception:
                pass

    def stop(self):
        self._halt.set()


def collect(ser, n_frames, timeout_s):
    """Drain the CLI port until n_frames result lines have been parsed."""
    samples, buf, t0, junk = [], '', time.time(), 0
    while len(samples) < n_frames:
        if time.time() - t0 > timeout_s:
            break
        chunk = ser.read(4096)
        if not chunk:
            time.sleep(0.005)
            continue
        buf += chunk.decode('ascii', 'ignore')
        *lines, buf = buf.split('\n')
        for line in lines:
            vals = parse_result_line(line)
            if vals is not None:
                samples.append(vals)
                if len(samples) % 50 == 0:
                    print('  %d/%d' % (len(samples), n_frames), flush=True)
            elif line.strip().startswith(KEY):
                junk += 1
    return samples, junk


def reduce_and_report(samples, temps, cfg_path, meta):
    cols = list(zip(*samples))
    med = [median(c) for c in cols]
    spr = [spread(c) for c in cols]

    bias_med = med[0]
    comp_med = med[1:]
    mags = [(comp_med[2 * i] ** 2 + comp_med[2 * i + 1] ** 2) ** 0.5
            for i in range(N_TX * N_RX)]
    worst_mag = max(abs(m - 1.0) for m in mags)

    # Count frames that sit away from the mode, and keep the in-mode ones to
    # describe the jitter of the target itself.
    outliers, in_mode = [], []
    for s in samples:
        off = (abs(s[0] - bias_med) > OUTLIER_BIAS_M or
               max(abs(v - c) for v, c in zip(s[1:], comp_med)) > OUTLIER_COMP)
        (outliers if off else in_mode).append(s)
    frac = len(outliers) / float(len(samples))
    bias_jitter = spread([s[0] for s in in_mode])
    biases = [s[0] for s in samples]

    print('\n' + '=' * 68)
    print('samples          %d  (%d in-mode, %d off-mode)'
          % (len(samples), len(in_mode), len(outliers)))
    if temps:
        print('die temperature  %d-%d C  (bias drifts with it; recorded)'
              % (min(temps), max(temps)))
    else:
        print('die temperature  not recorded (data port unavailable)')
    if meta:
        print('target           %.2f m, search window %.2f m'
              % (meta['target_m'], meta['search_win_m']))
    print('rangeBias range  %.5f .. %.5f m  (two clumps here = two targets)'
          % (min(biases), max(biases)))

    print('\n%-22s %-12s %-12s %s' % ('', 'value', 'gate', ''))
    rows = [('off-mode fraction', frac, GATE_OUTLIER_FRAC),
            ('in-mode jitter (m)', bias_jitter, GATE_BIAS_JITTER_M),
            ('worst |mag|-1', worst_mag, GATE_MAG_DEV)]
    # A gate that could not be evaluated is not a gate that failed. Both block
    # the paste, but they send you to different places, so they are named
    # differently: NaN here means too few in-mode frames survived to say
    # anything about jitter, which is a symptom of the contamination above and
    # not of the mount.
    fails, undetermined = [], []
    for name, val, gate in rows:
        if val != val:
            undetermined.append(name)
            status = 'no data'
        elif val > gate:
            fails.append(name)
            status = 'FAIL'
        else:
            status = 'ok'
        print('%-22s %-12s <= %-10.3f %s'
              % (name, '%.5f' % val if val == val else '-', gate, status))

    print('\nper-channel magnitude (want ~1.0):')
    for tx in range(N_TX):
        print('  TX%d  %s' % (tx, '  '.join('%.3f' % mags[tx * N_RX + rx]
                                            for rx in range(N_RX))))

    # One refinement pass: the median over ALL samples is robust enough to
    # locate the mode (it survives up to half the frames being wrong), and the
    # figure actually pasted is then taken over the in-mode frames only.
    final = [median(c) for c in zip(*in_mode)] if in_mode else med

    print('\n' + '-' * 68)
    print('paste into radar_people.cfg, replacing the UNVERIFIED block:\n')
    line = KEY + ' ' + ' '.join(('%.7f' if i == 0 else '%.5f') % v
                                for i, v in enumerate(final))
    print(line)
    print('-' * 68)

    # Each gate fails for its own physical reason, so each says its own thing.
    # "something is wrong" sends you back to the tripod when the fault is a
    # dead RX channel, and costs an afternoon.
    WHY = {
        'off-mode fraction':
            'Frames are landing on TWO different targets. This is the\n'
            '  rolling-table failure recorded in T_camera_radar.json: the\n'
            '  reflector shares its range gate with something else. Put it on\n'
            '  a tripod, clear the gate, re-run.',
        'in-mode jitter (m)':
            'One target, but it is not holding still - a soft mount, a\n'
            '  breathing tripod, or someone moving nearby. The reflector and\n'
            '  everything touching it must be rigid.',
        'worst |mag|-1':
            'A channel is far off unit magnitude. That is not a placement\n'
            '  problem: suspect the RX channel or the antenna connection,\n'
            '  and check which TX/RX cell it is in the table above.',
    }
    if fails or undetermined:
        print('\nDO NOT PASTE.')
        for f in fails:
            print('\n- %s: %s' % (f, WHY.get(f, '')))
        for u in undetermined:
            print('\n- %s: too few in-mode frames to evaluate this. Fix the\n'
                  '  failures above first; this one is downstream of them.' % u)
        print('\nA confident median over a contaminated gate is worse than no')
        print('table at all - it moves boresight silently, and every session')
        print('recorded afterwards carries labels on the wrong tracks.')
    else:
        print('\nOne rigid reflector held the gate for every frame. Paste it.')

    return {
        'line': line,
        'median': final,
        'median_all_samples': med,
        'spread_p90_p10': spr,
        'n_samples': len(samples),
        'n_in_mode': len(in_mode),
        'off_mode_fraction': frac,
        'in_mode_jitter_m': bias_jitter,
        'bias_min_max': [min(biases), max(biases)],
        'die_c': {'min': min(temps), 'max': max(temps)} if temps else None,
        'magnitudes': mags,
        'gates': {'outlier_frac': GATE_OUTLIER_FRAC,
                  'bias_jitter_m': GATE_BIAS_JITTER_M,
                  'mag_dev': GATE_MAG_DEV,
                  'status': 'FAIL' if fails else 'PASS',
                  'failed': fails,
                  'note': 'provisional; no baseline existed when written'},
        'cfg': os.path.relpath(cfg_path, ROOT),
        'measure_cmd': meta,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', default=DEFAULT_CFG)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--timeout', type=float, default=120.0)
    ap.add_argument('--no-send', action='store_true',
                    help='attach to a measurement already running')
    ap.add_argument('--cli-port', default=None)
    ap.add_argument('--data-port', default=None)
    ap.add_argument('--out', default=None, help='write JSON provenance here')
    args = ap.parse_args()

    cli = args.cli_port or find_port('00')
    data = args.data_port or find_port('03')
    if not cli:
        print('no XDS110 CLI port (interface 00). is the radar plugged in?',
              file=sys.stderr)
        return 1
    print('CLI %s   DATA %s' % (cli, data or '(none)'))

    meta = target_distance_from_cfg(args.cfg)
    if meta and not args.no_send and meta['enabled'] != 1:
        print('%s has measureRangeBiasAndRxChanPhase disabled - it will never '
              'print a result.' % os.path.basename(args.cfg), file=sys.stderr)
        return 1

    temp = None
    if data:
        temp = DieTemp(data)
        temp.start()

    try:
        with serial.Serial(cli, send_radar_cfg.BAUD, timeout=0.2) as ser:
            if not args.no_send:
                send_radar_cfg.flush_cli(ser)
                print('sending %s' % os.path.relpath(args.cfg, ROOT))
                send_radar_cfg.send_config(ser, args.cfg)
            print('collecting %d result lines...' % args.frames)
            samples, junk = collect(ser, args.frames, args.timeout)
    finally:
        if temp:
            temp.stop()
            temp.join(timeout=2.0)

    if junk:
        print('  %d malformed result lines skipped (expected %d numbers)'
              % (junk, N_VALUES))
    if not samples:
        print('\nno result lines. Either the measurement is not enabled in the '
              'cfg, or\nthis firmware does not print it. Check that '
              'measureRangeBiasAndRxChanPhase\nis 1 and that sensorStart '
              'succeeded.', file=sys.stderr)
        return 1
    if len(samples) < args.frames:
        print('  timed out with %d/%d' % (len(samples), args.frames))

    out = reduce_and_report(samples, temp.samples if temp else [],
                            args.cfg, meta)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print('\nwrote %s' % args.out)
    return 0 if out['gates']['status'] == 'PASS' else 2


if __name__ == '__main__':
    sys.exit(main())
