#!/usr/bin/env python3
"""Check radar/chirp.py against the numbers that were MEASURED on this board.

    ./test_chirp.py

The point of this file is one assertion and everything else is support: the
same code that derives Mode P's limits must reproduce, from radar_10hz.cfg
alone, the three things that were actually measured off that config --

    range grid  0.0436 m   fitted to 225k logged points
    v_max       0.649 m/s  16 distinct Doppler values in the recordings
    R_max       11.16 m

If it does, a derived number for a config nobody has run yet is a prediction
worth acting on. If it does not, radar_people.cfg is arithmetic with a comment
block, and SCAN-MODES-PLAN gate 2 would be run against the wrong expectations
-- which is worse than not running it, because the gate would pass.

What this cannot cover: whether the radar accepts the profile at all (768 KB of
radar cube against a 1 MB L3), and whether the phase table carried over from
radar_10hz.cfg is valid at this profile's band. Both need hardware.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'radar'))
import chirp    # noqa: E402
import mmwave   # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print('  ok   %s' % name)
    else:
        print('  FAIL %s %s' % (name, detail))
        FAILS.append(name)


def close(a, b, tol):
    return abs(a - b) <= tol


print('derivation vs the measured radar_10hz.cfg')
d = chirp.limits_for('radar_10hz')
m = mmwave.CFG_10HZ

# 0.5% rather than exact: the derivation uses the ramp's centre frequency for
# lambda, and the demo's own figure depends on where in the ramp it evaluates.
# A coordinate or units bug would be off by a factor, not by half a percent.
check('range resolution reproduces the fitted 0.0436 m grid',
      close(d['range_res_m'], m['range_res_m'], 0.0005),
      '%.4f vs %.4f' % (d['range_res_m'], m['range_res_m']))
check('R_max reproduces the measured 11.16 m',
      close(d['range_max_unambiguous_m'], m['range_max_unambiguous_m'], 0.05),
      '%.2f vs %.2f' % (d['range_max_unambiguous_m'],
                        m['range_max_unambiguous_m']))
check('v_max reproduces the measured 0.649 m/s within 0.5%',
      close(d['v_max_m_s'], m['v_max_m_s'], 0.005 * m['v_max_m_s']),
      '%.4f vs %.4f' % (d['v_max_m_s'], m['v_max_m_s']))
check('v_res reproduces the 16-bin 0.0812 m/s step',
      close(d['v_res_m_s'], m['v_res_m_s'], 0.0005) and
      d['n_doppler_bins'] == 16,
      '%.4f / %d bins' % (d['v_res_m_s'], d['n_doppler_bins']))
check('the HPF blind zone reproduces 0.375 m',
      close(d['hpf_blind_m'], m['hpf_blind_m'], 0.002),
      '%.3f vs %.3f' % (d['hpf_blind_m'], m['hpf_blind_m']))
check('3 TX x 4 RX read off the config', d['n_tx'] == 3 and d['n_rx'] == 4)

print('\nthe defect Mode P exists to fix')
check('under radar_10hz a 1.4 m/s walker aliases',
      abs(mmwave.fold_velocity(1.4, d['v_max_m_s'])) < 0.11,
      'folds to %.3f' % mmwave.fold_velocity(1.4, d['v_max_m_s']))

p = chirp.limits_for('radar_people')
check('under radar_people a 1.4 m/s walker does not',
      close(mmwave.fold_velocity(1.4, p['v_max_m_s']), 1.4, 1e-9))
check('and neither does a 4.5 m/s runner',
      close(mmwave.fold_velocity(4.5, p['v_max_m_s']), 4.5, 1e-9))

print('\nradar_people.cfg meets what SCAN-MODES-PLAN Mode P claims')
for name, got, want, tol in (
        ('range resolution 7.03 cm', p['range_res_m'], 0.0703, 0.0005),
        ('R_max 18.0 m', p['range_max_unambiguous_m'], 18.0, 0.05),
        ('v_max 4.98 m/s', p['v_max_m_s'], 4.98, 0.02),
        ('v_res 0.156 m/s', p['v_res_m_s'], 0.156, 0.001),
        ('HPF blind zone moved to 0.60 m', p['hpf_blind_m'], 0.604, 0.005),
        ('active time 12.3 ms', p['active_s'] * 1e3, 12.29, 0.05)):
    check(name, close(got, want, tol), '%.4f vs %.4f' % (got, want))
check('radar cube is the 768 KB the plan flags as tight',
      p['cube_bytes'] == 768 * 1024, '%d B' % p['cube_bytes'])
check('the documented 48-loop fallback does fit under 640 KB',
      256 * 48 * 4 * 3 * 4 < 640 * 1024)

print('\nthe cfg lines that are decisions, not arithmetic')
check('clutterRemoval is OFF -- a standing person must not be deleted',
      p['clutter_removal'] is False)
r_lo, r_hi = p['range_gate_m']
check('the range gate floor follows the HPF corner that moved',
      r_lo >= p['hpf_blind_m'] - 0.01,
      'gate %.2f vs blind %.3f' % (r_lo, p['hpf_blind_m']))
check('the range gate ceiling clears the 15 m envelope and stays under R_max',
      15.0 <= r_hi <= p['range_max_unambiguous_m'], '%.1f m' % r_hi)
# The two cfarFovCfg lines differ only in their procId, so a parser that keeps
# one per command reads the Doppler gate as the range gate -- "0.60 to 16 m"
# becomes "-5 to 5 m" and every check above still passes on the wrong numbers.
d_lo, d_hi = p['doppler_gate_m_s']
check('the Doppler gate does not clip a walker at v_max',
      d_hi >= p['v_max_m_s'] and d_lo <= -p['v_max_m_s'],
      '%.1f..%.1f vs v_max %.2f' % (d_lo, d_hi, p['v_max_m_s']))
check('range and Doppler gates were read as different lines',
      (r_lo, r_hi) != (d_lo, d_hi))
check('the measure-bias variant is the same profile',
      chirp.limits_for('radar_people_measure_bias')['v_max_m_s']
      == p['v_max_m_s'])

print('\nuse_config actually redirects the validators')
mmwave.use_config('radar_10hz')
fast = [{'x': 2.0, 'y': 0.0, 'z': 0.0, 'v': 1.30}]
check('1.30 m/s is impossible under radar_10hz',
      len(mmwave.validate_points(fast)) == 1)
mmwave.use_config('radar_people')
check('and possible under radar_people',
      len(mmwave.validate_points(fast)) == 0)
far = [{'x': 14.0, 'y': 0.0, 'z': 0.0, 'v': 0.0}]
check('14 m is inside radar_people R_max', len(mmwave.validate_points(far)) == 0)
mmwave.use_config('radar_10hz')
check('and outside radar_10hz R_max', len(mmwave.validate_points(far)) == 1)
check('use_config restored the measured table, not a derived one',
      mmwave.CFG is mmwave.CFG_10HZ)

print()
if FAILS:
    print('%d FAILED: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('all chirp derivation checks passed')
