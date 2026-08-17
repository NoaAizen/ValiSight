#!/usr/bin/env python3
"""Derive a config's physical limits from the .cfg file itself.

Every number that describes what the radar can see -- range resolution, maximum
unambiguous range and velocity, the HPF blind zone, the radar cube size -- is a
consequence of four fields in `profileCfg` and two in `frameCfg`. Until now they
were hand-typed into `mmwave.CFG_10HZ` and into comments, which was fine while
there was exactly one config. SCAN-MODES-PLAN adds more, and a hand-typed limit
table is the kind of thing that goes stale silently: `validate_points` would go
on rejecting good Mode P points using Mode C's v_max, and every rejected point
looks exactly like a point the radar never reported.

So the limits are derived here, and `mmwave` asks for them by config name.

WHAT THIS IS NOT. These are derivations. The three numbers in `radar_10hz.cfg`
that were *measured* on the board -- 0.0436 m range grid over 225k points,
+-0.649 m/s over 16 distinct Doppler values, 11.16 m -- are what anchor the
formulas, and `tools/tests/test_chirp.py` asserts the derivation reproduces
them. A derived number for a config nobody has run is a prediction, and
SCAN-MODES-PLAN gate 5 is where it stops being one.
"""
import os

C = 299_792_458.0

# hpfCornerFreq1 code -> corner in Hz (SDK 3.x CLI, xWR18xx).
_HPF1_HZ = {0: 175e3, 1: 235e3, 2: 350e3, 3: 700e3}

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(_HERE, 'configs')


def parse_cfg(path):
    """.cfg file -> {command: [[args of occurrence 1], [occurrence 2], ...]}.

    ALWAYS a list of occurrences, even for commands that appear once. Several
    of the commands that matter here legitimately repeat and mean different
    things each time -- `cfarFovCfg` is the range gate on one line and the
    Doppler gate on the next, distinguished only by their procId argument, and
    `chirpCfg` appears once per TX. A flat {command: args} dict silently keeps
    the last of each, which reads as "the range gate is -5.0 to 5.0 m" and
    passes every type check on the way past.

    Args stay as floats: every field read here is numeric, and int()-ing at
    parse time would quietly truncate rampEndTime 57.14.
    """
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            parts = line.split()
            try:
                args = [float(a) for a in parts[1:]]
            except ValueError:
                args = parts[1:]
            out.setdefault(parts[0], []).append(args)
    return out


def one(cfg, command):
    """The effective arguments of `command`: the last occurrence, as the CLI
    takes them. Raises if the command is absent, rather than returning a shape
    that fails somewhere further down."""
    return cfg[command][-1]


def fov_gate(cfg, proc_id):
    """cfarFovCfg for procId 0 (range, metres) or 1 (Doppler, m/s) -> (lo, hi).

    None if this config does not gate that axis.
    """
    for args in cfg.get('cfarFovCfg', []):
        if int(args[1]) == proc_id:
            return args[2], args[3]
    return None


def limits(cfg):
    """{command: args} -> the physical limits dict, in the CFG_10HZ shape.

    `cfg` may also be a path.
    """
    if isinstance(cfg, str):
        cfg = parse_cfg(cfg)

    p = one(cfg, 'profileCfg')
    start_ghz, idle_us, ramp_us = p[1], p[2], p[4]
    slope_mhz_us, n_adc, rate_ksps = p[7], p[9], p[10]
    hpf1 = int(p[11])

    f = one(cfg, 'frameCfg')
    chirp_lo, chirp_hi, n_loops = int(f[0]), int(f[1]), int(f[2])
    frame_period_s = f[4] * 1e-3

    # Chirps per loop, i.e. the TDM-MIMO multiplier on Tc. Taken from frameCfg's
    # chirp index span rather than from channelCfg's TX mask: the mask says which
    # transmitters exist, the span says how many are actually stepped through,
    # and only the second one sets the Doppler ambiguity.
    n_tx = chirp_hi - chirp_lo + 1
    n_rx = bin(int(one(cfg, 'channelCfg')[0])).count('1')

    slope_hz_s = slope_mhz_us * 1e12

    # Range resolution comes from the bandwidth actually SAMPLED, not from the
    # full ramp: the ADC opens at adcStartTime and closes n_adc samples later.
    adc_time_s = n_adc / (rate_ksps * 1e3)
    b_valid_hz = slope_hz_s * adc_time_s

    # Velocity comes from the CARRIER, so it needs the full ramp's centre
    # frequency, not startFreq. At 4 GHz of excursion the difference is 2.6%.
    b_total_hz = slope_hz_s * ramp_us * 1e-6
    f_centre_hz = start_ghz * 1e9 + b_total_hz / 2.0
    lam_m = C / f_centre_hz

    tc_s = (idle_us + ramp_us) * 1e-6 * n_tx
    v_max = lam_m / (4.0 * tc_s)

    return {
        'frame_period_s': frame_period_s,
        'range_res_m': C / (2.0 * b_valid_hz),
        'range_max_unambiguous_m': rate_ksps * 1e3 * C / (2.0 * slope_hz_s),
        'v_max_m_s': v_max,
        'v_res_m_s': 2.0 * v_max / n_loops,
        # A function of freqSlopeConst, not a property of the board: it moves
        # with the profile, which is exactly why it cannot stay a constant.
        'hpf_blind_m': _HPF1_HZ[hpf1] * C / (2.0 * slope_hz_s),
        # Not a limit but the thing that decides whether a profile loads at all.
        # Complex int16 per sample per virtual channel; the 1843's L3 is 1 MB and
        # the demo does not get all of it.
        'cube_bytes': int(n_adc * n_loops * n_rx * n_tx * 4),
        'active_s': n_loops * n_tx * (idle_us + ramp_us) * 1e-6,
        'n_tx': n_tx,
        'n_rx': n_rx,
        'n_doppler_bins': n_loops,
        'clutter_removal': bool(one(cfg, 'clutterRemoval')[1])
                           if 'clutterRemoval' in cfg else False,
        'range_gate_m': fov_gate(cfg, 0),
        'doppler_gate_m_s': fov_gate(cfg, 1),
    }


def limits_for(name):
    """'radar_10hz' -> limits, read from configs/radar_10hz.cfg."""
    return limits(os.path.join(CONFIG_DIR, name + '.cfg'))


if __name__ == '__main__':
    import sys
    names = sys.argv[1:] or ['radar_10hz', 'radar_people']
    for n in names:
        path = n if os.path.sep in n else os.path.join(CONFIG_DIR, n + '.cfg')
        d = limits(path)
        print('\n%s' % os.path.basename(path))
        print('  range res      %.4f m' % d['range_res_m'])
        print('  R_max          %.2f m' % d['range_max_unambiguous_m'])
        print('  v_max          +-%.3f m/s   (alias period %.3f)'
              % (d['v_max_m_s'], 2 * d['v_max_m_s']))
        print('  v_res          %.4f m/s  (%d bins)'
              % (d['v_res_m_s'], d['n_doppler_bins']))
        print('  HPF blind      %.3f m' % d['hpf_blind_m'])
        print('  active/period  %.2f / %.0f ms'
              % (d['active_s'] * 1e3, d['frame_period_s'] * 1e3))
        print('  radar cube     %d KB  (%dTX x %dRX)'
              % (d['cube_bytes'] // 1024, d['n_tx'], d['n_rx']))
        print('  walker 1.4 m/s %s'
              % ('k=0, azimuth trustworthy' if d['v_max_m_s'] > 1.4
                 else 'ALIASES -- azimuth is wrong by ~9.6 deg'))
