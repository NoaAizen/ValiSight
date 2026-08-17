#!/usr/bin/env python3
"""Send a .cfg to the IWR1843 over its CONFIG UART.

    ./send_radar_cfg.py                                   # radar_10hz.cfg on /dev/ttyACM1
    ./send_radar_cfg.py ../radar/configs/radar_10hz.cfg --port /dev/ttyACM1
    ./send_radar_cfg.py --stop                            # sensorStop only

The XDS110 exposes two ports: the lower-numbered one is the CLI at 115200, the
higher one is the data stream at 921600. On this Jetson that is /dev/ttyACM1
and /dev/ttyACM2 (ACM0 is the OpenMV N6).

Each command waits for the radar's OWN response rather than sleeping. A fixed
sleep desynchronizes the log: slow commands like sensorStop answer late, and
from then on every response is attributed to the previous command, so the run
looks clean while reporting a different command's "Done" for a failure.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import serial

DEFAULT_PORT = '/dev/ttyACM1'
DEFAULT_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'radar', 'configs', 'radar_10hz.cfg')
BAUD = 115200


# A command this firmware build does not have answers "'name' is not recognized
# as a CLI command" -- which contains none of Done/Error/Ignored. Without it in
# this list the sender waits out the full timeout and then reports SUCCESS, and
# the radar runs a config quietly different from the one every derived figure
# (range resolution, R_max, v_max) was computed against. Observed on this board:
# `calibData 0 0 0` is rejected exactly this way and was logged as fine.
_TERMINATORS = ('Done', 'Error', 'Ignored', 'not recognized')
_FAILURES = ('Error', 'not recognized')


def send_line(ser, line, timeout=2.0):
    ser.reset_input_buffer()                     # drop anything stale
    ser.write((line + '\n').encode())
    t0 = time.time()
    resp = ''
    while time.time() - t0 < timeout:
        n = ser.in_waiting
        if n:
            resp += ser.read(n).decode('ascii', 'ignore')
            if any(k in resp for k in _TERMINATORS):
                break
        else:
            time.sleep(0.01)
    return resp.strip()


def flush_cli(ser, settle=0.4):
    """Terminate any half-line the radar's CLI is holding, and drain the reply.

    `reset_input_buffer()` clears the HOST's receive buffer; it cannot clear a
    fragment already sitting in the radar's line parser. One gets there easily:
    opening the port toggles DTR and can inject a byte, and anything that wrote
    to this port by mistake leaves a fragment with no newline behind it -
    live.py aimed at the wrong ACM number does precisely that, and the ACM
    numbers on this rig are not stable.

    The next real command is then appended to that fragment, so the radar sees
    `<junk>sensorStop` and answers "'sensorStop' is not recognized as a CLI
    command". That reads like firmware without the command - `help` lists it,
    so it is not - and it costs an hour to believe. One newline fixes it.
    """
    ser.reset_input_buffer()
    ser.write(b'\n')
    time.sleep(settle)
    if ser.in_waiting:
        ser.read(ser.in_waiting)
    ser.reset_input_buffer()


def _rejected(resp):
    return any(k in resp for k in _FAILURES)


def send_config(ser, cfg_path, verbose=True, retries=2):
    """Returns [(command, response)]. Raises RuntimeError on a rejected command.

    Retrying clears the transient CLI errors that a reconfigure throws while
    the sensor is still stopping; a command that fails twice is a real one.
    """
    log = []
    with open(cfg_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('%'):
                continue
            resp = send_line(ser, line)
            attempt = 0
            # Retry only a transient Error. A command the firmware does not
            # have will never start existing, so retrying it just triples the
            # wait before the same failure.
            while 'Error' in resp and attempt < retries:
                attempt += 1
                time.sleep(0.3)
                resp = send_line(ser, line)
            log.append((line, resp))
            if verbose:
                short = resp.replace('\r', '').replace('\n', ' | ')
                note = ' (retry %d)' % attempt if attempt else ''
                print('  cfg> %-58s %s%s' % (line, short[:90], note))
            if _rejected(resp):
                raise RuntimeError('radar rejected: %s -> %s' % (line, resp))
    return log


STAMP_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), '.radar-cfg-stamp.json')


def cfg_digest(cfg_path):
    """sha256 of the config as sent, plus the phase-table line it carries.

    The digest alone is not enough to read later: two configs can differ only
    in compRangeBiasAndRxChanPhase, and that one line is the difference
    between a valid extrinsic and one that was silently voided (the phase
    table moves boresight - frame_conventions.txt). So it is carried in clear
    beside the hash.
    """
    with open(cfg_path, 'rb') as f:
        raw = f.read()
    phase = None
    for line in raw.decode('utf-8', 'replace').splitlines():
        if line.strip().startswith('compRangeBiasAndRxChanPhase'):
            phase = line.strip()
    return {
        'path': os.path.abspath(cfg_path),
        'name': os.path.basename(cfg_path),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'range_bias_m': (float(phase.split()[1]) if phase and
                         len(phase.split()) > 1 else None),
        'phase_line': phase,
    }


def write_stamp(cfg_path, port, log):
    """Record what was actually put on the radar, for the recorder to copy.

    The radar cannot be asked which config it is running, and the config is
    sent by this tool while sessions are recorded by another. Without this
    stamp a session's provenance is the operator's memory, which is exactly
    what PDF 01 sec 11 ("use exactly the same radar configuration during
    validation that you used for calibration") cannot be checked against.

    The stamp is a CLAIM, not proof: it says what this tool sent and when. A
    reader must still check that it predates the recording and that the radar
    was not power-cycled in between - meta.json carries both timestamps so
    that check is possible.
    """
    stamp = dict(cfg_digest(cfg_path),
                 port=port,
                 sent_at_wall=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 sent_at_mono=time.monotonic(),
                 commands=len(log))
    try:
        with open(STAMP_PATH, 'w') as f:
            json.dump(stamp, f, indent=2)
    except OSError as e:                       # never fail a send over this
        print('warning: could not write %s (%s)' % (STAMP_PATH, e),
              file=sys.stderr)
    return stamp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cfg', nargs='?', default=DEFAULT_CFG)
    ap.add_argument('--port', default=DEFAULT_PORT)
    ap.add_argument('--stop', action='store_true',
                    help='send sensorStop and exit, leaving the config loaded')
    args = ap.parse_args()

    with serial.Serial(args.port, BAUD, timeout=0.2) as ser:
        flush_cli(ser)               # before the first command, always
        if args.stop:
            print('  cfg> sensorStop %s' % send_line(ser, 'sensorStop'))
            return 0
        if not os.path.exists(args.cfg):
            print('no such config: %s' % args.cfg, file=sys.stderr)
            return 2
        print('sending %s to %s @%d' % (os.path.basename(args.cfg),
                                        args.port, BAUD))
        log = send_config(ser, args.cfg)
    stamp = write_stamp(args.cfg, args.port, log)
    print('\nstamped %s  sha256 %s  rangeBias %s'
          % (stamp['name'], stamp['sha256'][:12],
             '%.4f m' % stamp['range_bias_m']
             if stamp['range_bias_m'] is not None else 'ABSENT'))
    if not stamp['range_bias_m']:
        print('  NOTE: rangeBias is 0/absent - this config is UNCALIBRATED. '
              'Any (R,t) solved from a session recorded on it is provisional.')
    print('sensor started. next: ./radar_listen.py --stage1')
    return 0


if __name__ == '__main__':
    sys.exit(main())
