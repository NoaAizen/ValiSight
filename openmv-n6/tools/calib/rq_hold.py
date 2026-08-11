#!/usr/bin/env python3
"""Radar-only qualification hold (plan V2 sec. 4.3, RQ01-RQ06).

Reads the IWR1843 data UART for --seconds, gates detections around the
expected TCR position, and reports the stats the plan asks for: detection
availability, median target location, range/azimuth/elevation sigma, median
SNR, and ambiguity count. Writes one JSON per hold into calib-artifacts/rq/.

    ./rq_hold.py RQ01 --range 1.5
    ./rq_hold.py RQ05 --range 3.0 --az -20
"""
import argparse
import json
import math
import os
import struct
import sys
import time

import serial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'radar'))
import mmwave  # noqa: E402

PORT = '/dev/ttyACM2'
BAUD = 921600
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'calib-artifacts', 'rq')

MAGIC = mmwave.MAGIC
_TOTAL_LEN_OFF = mmwave._TOTAL_LEN_OFF


def frames_from_serial(port, seconds):
    """Yield complete radar frames read from the data UART for `seconds`."""
    s = serial.Serial(port, BAUD, timeout=0.2)
    buf = b''
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            buf += s.read(8192)
            while True:
                i = buf.find(MAGIC)
                if i < 0 or len(buf) < i + _TOTAL_LEN_OFF + 4:
                    break
                total = struct.unpack_from('<I', buf, i + _TOTAL_LEN_OFF)[0]
                if not 40 <= total <= 65536:      # desynced: skip this magic
                    buf = buf[i + len(MAGIC):]
                    continue
                if len(buf) < i + total:
                    break
                yield buf[i:i + total]
                buf = buf[i + total:]
    finally:
        s.close()


def sph(p):
    """Project-frame point -> (range_m, az_deg right-positive, el_deg up)."""
    r = math.sqrt(p['x'] ** 2 + p['y'] ** 2 + p['z'] ** 2)
    az = math.degrees(math.atan2(-p['y'], p['x']))   # empirical: +az tracks +u
    el = math.degrees(math.atan2(p['z'], math.hypot(p['x'], p['y'])))
    return r, az, el


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def std(v):
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('hold_id', help='e.g. RQ01')
    ap.add_argument('--range', type=float, required=True,
                    help='expected TCR range, metres')
    ap.add_argument('--az', type=float, default=0.0,
                    help='expected azimuth, deg (right-positive)')
    ap.add_argument('--rtol', type=float, default=0.5, help='range gate +- m')
    ap.add_argument('--aztol', type=float, default=15.0, help='az gate +- deg')
    ap.add_argument('--eltol', type=float, default=10.0,
                    help='el gate +- deg around 0 (TCR sits at antenna height)')
    ap.add_argument('--seconds', type=float, default=15.0)
    ap.add_argument('--port', default=PORT)
    a = ap.parse_args()

    n_frames = 0
    hits = []          # (r, az, el, snr, x, y, z) of gated points, one list per frame
    ambiguous = 0
    for raw in frames_from_serial(a.port, a.seconds):
        try:
            fr = mmwave.parse_frame(raw)
        except Exception:
            continue
        n_frames += 1
        g = []
        for p in fr['points']:
            r, az, el = sph(p)
            if (abs(r - a.range) <= a.rtol and abs(az - a.az) <= a.aztol
                    and abs(el) <= a.eltol):
                g.append((r, az, el, p['snr'], p['x'], p['y'], p['z']))
        if len(g) > 1:
            ambiguous += 1
            g.sort(key=lambda t: -(t[3] if t[3] is not None else -999))
        if g:
            hits.append(g[0])

    if not n_frames:
        raise SystemExit('no radar frames: is the cfg sent and sensor started?')

    res = {
        'hold_id': a.hold_id,
        'expected': {'range_m': a.range, 'az_deg': a.az},
        'gate': {'rtol_m': a.rtol, 'aztol_deg': a.aztol},
        'frames': n_frames,
        'frames_with_target': len(hits),
        'availability': len(hits) / n_frames,
        'ambiguous_frames': ambiguous,
    }
    if hits:
        rs = [h[0] for h in hits]
        azs = [h[1] for h in hits]
        els = [h[2] for h in hits]
        snrs = [h[3] for h in hits if h[3] is not None]
        res.update({
            'range_median_m': median(rs), 'range_sigma_m': std(rs),
            'az_median_deg': median(azs), 'az_sigma_deg': std(azs),
            'el_median_deg': median(els), 'el_sigma_deg': std(els),
            'snr_median_db': median(snrs) if snrs else None,
            'xyz_median_m': [median([h[4] for h in hits]),
                             median([h[5] for h in hits]),
                             median([h[6] for h in hits])],
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, a.hold_id + '.json')
    json.dump(res, open(out, 'w'), indent=2)
    print(json.dumps(res, indent=2))
    print('-> %s' % out)


if __name__ == '__main__':
    main()
