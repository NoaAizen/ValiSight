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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cfg', nargs='?', default=DEFAULT_CFG)
    ap.add_argument('--port', default=DEFAULT_PORT)
    ap.add_argument('--stop', action='store_true',
                    help='send sensorStop and exit, leaving the config loaded')
    args = ap.parse_args()

    with serial.Serial(args.port, BAUD, timeout=0.2) as ser:
        if args.stop:
            print('  cfg> sensorStop %s' % send_line(ser, 'sensorStop'))
            return 0
        if not os.path.exists(args.cfg):
            print('no such config: %s' % args.cfg, file=sys.stderr)
            return 2
        print('sending %s to %s @%d' % (os.path.basename(args.cfg),
                                        args.port, BAUD))
        send_config(ser, args.cfg)
    print('\nsensor started. next: ./radar_listen.py --stage1')
    return 0


if __name__ == '__main__':
    sys.exit(main())
