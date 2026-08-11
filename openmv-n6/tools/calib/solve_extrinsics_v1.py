#!/usr/bin/env python3
"""Block F of the V2 plan: solve T_C<-R from the picked TCR correspondences.

Formulation per CALIBRATION-PLAN.md stage 5 (reduced DOF, weak elevation):
  - t_y (camera-frame vertical) FIXED at the caliper value: radar sits
    47 mm below the camera -> +0.047 in OpenCV camera frame (y down).
  - free params: rotation (rodrigues, 3) + t_x + t_z, robust soft_l1 loss.
  - fit on the clean holds only (radar az and pixel az agree <= 4 deg);
    every other pick is evaluated but never fitted.

Input:  captures/holds1/corr.json   (recorded xyz verified correct as-is)
        calib-artifacts/calib.json  (K, dist; f tape-fixed at 525)
Output: calib-artifacts/T_camera_radar.json
"""
import json
import math
import os

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')

# project radar frame (x fwd, y left, z up) -> camera (x right, y down, z fwd)
R_CANONICAL = np.array([[0., -1., 0.],
                        [0., 0., -1.],
                        [1., 0., 0.]])
T_Y_MECH = 0.047                 # radar 47 mm below camera, camera y is down
FIT_HOLDS = {'hold_39', 'hold_44', 'hold_50', 'hold_51', 'hold_52',
             'hold_53', 'hold_57'}


def load():
    corr = json.load(open(os.path.join(ROOT, 'captures/holds1/corr.json')))
    cal = json.load(open(os.path.join(ROOT, 'calib-artifacts/calib.json')))
    K = np.array(cal['K_rgb'])
    dist = np.array(cal['dist_rgb'])
    rows = [(x['name'], np.array([x['x'], x['y'], x['z']]),
             np.array([x['u'], x['v']], float))
            for x in corr['correspondences']]
    return rows, K, dist


def project(params, pts3, K, dist):
    rvec = params[:3]
    t = np.array([params[3], T_Y_MECH, params[4]])
    proj, _ = cv2.projectPoints(pts3.reshape(-1, 1, 3), rvec,
                                t.reshape(3, 1), K, dist)
    return proj.reshape(-1, 2)


def fit(pts3, pts2, K, dist):
    rvec0, _ = cv2.Rodrigues(R_CANONICAL)
    x0 = np.concatenate([rvec0.ravel(), [0.0, 0.0]])

    def resid(p):
        return (project(p, pts3, K, dist) - pts2).ravel()

    lo = np.concatenate([rvec0.ravel() - 0.5, [-0.06, -0.06]])
    hi = np.concatenate([rvec0.ravel() + 0.5, [0.06, 0.06]])
    r = least_squares(resid, x0, loss='soft_l1', f_scale=5.0,
                      bounds=(lo, hi))
    return r.x


def ang_deg(R):
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def main():
    rows, K, dist = load()
    fit_rows = [r for r in rows if r[0] in FIT_HOLDS]
    p3f = np.array([r[1] for r in fit_rows])
    p2f = np.array([r[2] for r in fit_rows])

    params = fit(p3f, p2f, K, dist)
    R, _ = cv2.Rodrigues(params[:3])
    t = np.array([params[3], T_Y_MECH, params[4]])

    print('=== fit on %d clean holds, t_y fixed %.3f ===' % (len(fit_rows), T_Y_MECH))
    all3 = np.array([r[1] for r in rows])
    all2 = np.array([r[2] for r in rows])
    res = np.linalg.norm(project(params, all3, K, dist) - all2, axis=1)
    for (nm, r3, _), e in sorted(zip(rows, res), key=lambda z: -z[1]):
        tag = 'FIT ' if nm in FIT_HOLDS else 'eval'
        print('  %s %-9s r=%.2fm  res=%6.1f px' % (tag, nm, np.linalg.norm(r3), e))
    fres = np.array([e for (nm, _, _), e in zip(rows, res) if nm in FIT_HOLDS])
    print('fit-set:  median %.1f  p95 %.1f  max %.1f px' %
          (np.median(fres), np.percentile(fres, 95), fres.max()))
    print('R vs canonical: %.2f deg   t = [%+.1f %+.1f %+.1f] mm' %
          ((ang_deg(R @ R_CANONICAL.T),) + tuple(1000 * t)))

    # leave-one-out on the fit set
    angs, dts = [], []
    for i in range(len(fit_rows)):
        m = np.arange(len(fit_rows)) != i
        pi = fit(p3f[m], p2f[m], K, dist)
        Ri, _ = cv2.Rodrigues(pi[:3])
        angs.append(ang_deg(Ri @ R.T))
        dts.append(1000 * np.linalg.norm([pi[3] - params[3], 0, pi[4] - params[4]]))
    print('LOO: max rotation swing %.2f deg, max |dt| %.1f mm' % (max(angs), max(dts)))

    out = {
        'R': R.tolist(), 't_m': t.tolist(),
        'convention': 'p_C = R @ p_R + t; radar x fwd y left z up; camera '
                      'OpenCV x right y down z fwd; t_y fixed mechanically',
        'K_source': 'calib-artifacts/calib.json (f tape-fixed 525)',
        'fit_holds': sorted(FIT_HOLDS), 'n_fit': len(fit_rows),
        'fit_median_px': float(np.median(fres)),
        'fit_p95_px': float(np.percentile(fres, 95)),
        'fit_max_px': float(fres.max()),
        'R_vs_canonical_deg': ang_deg(R @ R_CANONICAL.T),
        'loo_max_rot_deg': float(max(angs)), 'loo_max_dt_mm': float(max(dts)),
        'per_hold_res_px': {nm: float(e) for (nm, _, _), e in zip(rows, res)},
    }
    path = os.path.join(ROOT, 'calib-artifacts', 'T_camera_radar.json')
    json.dump(out, open(path, 'w'), indent=2)
    print('-> %s' % path)


if __name__ == '__main__':
    main()
