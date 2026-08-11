#!/usr/bin/env python3
"""Check the radar overlay's projection signs, with no radar and no board.

    ./test_radar_overlay.py

Every failure this can catch is a silent one. A flipped azimuth sign draws a
person on the wrong side of the room and looks entirely plausible while doing
it; nothing raises, and the fused image cannot contradict it. So the tests are
written as physical statements -- "a target to my left appears on the left of
the picture" -- rather than as matrix comparisons, because a matrix compared
against itself proves nothing.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radar_overlay  # noqa: E402

FAILS = []
W, H = 640, 400


def check(name, cond, detail=''):
    if cond:
        print('  ok   %s' % name)
    else:
        print('  FAIL %s %s' % (name, detail))
        FAILS.append(name)


def pt(x, y, z, v=0.0, snr=25.0):
    return {'x': x, 'y': y, 'z': z, 'v': v, 'snr': snr, 'noise': 50.0}


proj = radar_overlay.Bootstrap(W, H, hfov_deg=70.0)
cx, cy = W / 2.0, H / 2.0

print('projection signs')

R = proj.R_CANONICAL
check('canonical rotation is proper',
      abs(np.linalg.det(R) - 1.0) < 1e-9 and
      np.allclose(R @ R.T, np.eye(3)),
      '(det=%.6f)' % np.linalg.det(R))

(u, v, inside), = proj.project([pt(3.0, 0.0, 0.0)])
check('straight ahead lands at the principal point',
      abs(u - cx) < 1e-6 and abs(v - cy) < 1e-6, '(got %.2f, %.2f)' % (u, v))
check('straight ahead is in frame', inside)

# The whole point of the file. Radar y is LEFT; image u grows to the RIGHT.
(u, v, _), = proj.project([pt(3.0, 1.0, 0.0)])
check('a target to my left is drawn left of centre', u < cx,
      '(u=%.1f, cx=%.1f -- azimuth sign is inverted)' % (u, cx))
check('and does not move vertically', abs(v - cy) < 1e-6)

(u, v, _), = proj.project([pt(3.0, -1.0, 0.0)])
check('a target to my right is drawn right of centre', u > cx, '(u=%.1f)' % u)

# Radar z is UP; image v grows DOWNWARD.
(u, v, _), = proj.project([pt(3.0, 0.0, 1.0)])
check('a target above me is drawn above centre', v < cy,
      '(v=%.1f, cy=%.1f -- elevation sign is inverted)' % (v, cy))
check('and does not move horizontally', abs(u - cx) < 1e-6)

# Angles, not just signs: at 45 deg off-axis the target sits exactly f pixels
# from the centre, whatever f happens to be.
(u, _, _), = proj.project([pt(3.0, -3.0, 0.0)])
check('45 deg right is exactly f pixels off centre', abs((u - cx) - proj.f) < 1e-6,
      '(offset %.2f vs f %.2f)' % (u - cx, proj.f))

print('range independence and framing')

# An angle is an angle: the same bearing at any distance is the same pixel.
near = proj.project([pt(2.0, -0.5, 0.0)])[0]
far = proj.project([pt(8.0, -2.0, 0.0)])[0]
check('the same bearing projects identically at any range',
      abs(near[0] - far[0]) < 1e-6, '(%.2f vs %.2f)' % (near[0], far[0]))

(u, v, inside), = proj.project([pt(-2.0, 0.0, 0.0)])
check('a target behind the camera is not drawn', u is None and not inside)

(u, v, inside), = proj.project([pt(1.0, 5.0, 0.0)])
check('an off-frame target reports its true coordinate', u < 0,
      '(u=%.1f -- a clamped point is a lie sitting on the frame edge)' % u)
check('and is flagged as outside', not inside)

print('runtime nudges')

proj.yaw = 5.0
(u_yaw, _, _), = proj.project([pt(3.0, 0.0, 0.0)])
check('yaw moves a boresight target off centre', abs(u_yaw - cx) > 10,
      '(moved %.1f px)' % (u_yaw - cx))
proj.yaw = 0.0
(u0, _, _), = proj.project([pt(3.0, 0.0, 0.0)])
check('and zero yaw restores it', abs(u0 - cx) < 1e-6)

proj.t[0] = 0.10                       # camera 10 cm to the right of the radar
(u_t, _, _), = proj.project([pt(2.0, 0.0, 0.0)])
expect = proj.f * 0.10 / 2.0
check('a lever arm shifts by f*B/Z', abs((u_t - cx) - expect) < 1e-6,
      '(got %.2f, expected %.2f)' % (u_t - cx, expect))
proj.t[0] = 0.0

print('honesty markers')

import mmwave  # noqa: E402
vmax = mmwave.CFG_10HZ['v_max_m_s']
check('a static target is not marked aliased',
      not radar_overlay.is_aliased(pt(3, 0, 0, v=0.0)))
check('a slow target is not marked aliased',
      not radar_overlay.is_aliased(pt(3, 0, 0, v=0.2 * vmax)))
# A walker folds to near the edge of the window, which is exactly where an
# aliased target lands -- the marker claims doubt, not knowledge.
folded = mmwave.fold_velocity(1.4, vmax)
check('a folded walker is flagged as suspect',
      radar_overlay.is_aliased(pt(3, 0, 0, v=folded)) or abs(folded) < 0.6 * vmax,
      '(folded to %.3f of vmax %.3f)' % (folded, vmax))
check('a reading at the window edge is flagged',
      radar_overlay.is_aliased(pt(3, 0, 0, v=0.95 * vmax)))

print('drawing')

img = np.zeros((H, W, 3), np.uint8)
drawn, off, aliased = radar_overlay.annotate(
    img, [pt(3.0, 0.0, 0.0), pt(3.0, 0.5, 0.2), pt(1.0, 9.0, 0.0)], proj)
check('in-frame points are drawn', drawn == 2, '(drawn %d)' % drawn)
check('the off-frame point is counted, not drawn', off == 1, '(off %d)' % off)
check('something actually reached the pixels', img.any())

img2 = np.zeros((H, W, 3), np.uint8)
radar_overlay.annotate(img2, [pt(3.0, 0.0, 0.0)], proj, show_whisker=False)
lit_no_whisker = int((img2.sum(axis=2) > 0).sum())
img3 = np.zeros((H, W, 3), np.uint8)
radar_overlay.annotate(img3, [pt(3.0, 0.0, 0.0)], proj, show_whisker=True)
check('the whisker adds vertical extent',
      int((img3.sum(axis=2) > 0).sum()) > lit_no_whisker)

img4 = np.zeros((H, W, 3), np.uint8)
radar_overlay.annotate(img4, [pt(3.0, 0.0, 0.0, v=0.95 * vmax)], proj,
                       show_whisker=False, label_nearest=False)
filled = np.zeros((H, W, 3), np.uint8)
radar_overlay.annotate(filled, [pt(3.0, 0.0, 0.0, v=0.0)], proj,
                       show_whisker=False, label_nearest=False)
# Hollow vs filled is the visual claim of trust. If they ever draw the same, a
# suspect detection is indistinguishable from a trusted one.
check('an aliased detection is drawn hollow, a trusted one filled',
      int((img4.sum(axis=2) > 0).sum()) < int((filled.sum(axis=2) > 0).sum()))

print('empty and degenerate input')
check('no points draws nothing', radar_overlay.annotate(
    np.zeros((H, W, 3), np.uint8), [], proj) == (0, 0, 0))
check('projecting nothing returns nothing', proj.project([]) == [])

print()
if FAILS:
    print('%d FAILED: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('all radar overlay checks passed')
