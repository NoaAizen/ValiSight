#!/usr/bin/env python3
"""Turn a recorded live.py session into radar<->pixel correspondences.

    ./radar_correspond.py holds   ../../captures/walk1
    ./radar_correspond.py pick    ../../captures/walk1 -o corr.json
    ./radar_correspond.py report  corr.json

A session is what `live.py --radar --radar-record DIR` writes: radar.jsonl,
frames.jsonl, session.mp4. Both jsonl files carry `t_mono` from the same
monotonic clock, which is what lets a radar segment be matched to a picture.

THE PROTOCOL THIS TOOL ASSUMES. The subject walks, stops, and holds still for
about three seconds, repeatedly. Holding is not a convenience, it is what makes
the measurement valid, for two independent reasons:

  - A stationary target has zero Doppler, so k = 0 and no aliasing correction is
    applied to its angle. A person walking at 1.4 m/s under radar_10hz.cfg
    aliases (v_max 0.649 m/s) and the demo's TDM-MIMO angle stage then leaves
    ~120 deg of residual phase across the aperture -- roughly 9.6 deg of azimuth
    error, about 88 px. A walking correspondence is a wrong correspondence.
  - Averaging ~30 frames divides the single-frame azimuth noise (measured sigma
    1.10 deg on this rig, p90 2.89 deg) by sqrt(30), which is what turns a 9 px
    wobble into something a solve can use.

WHAT THIS TOOL WILL NOT DO. It will not guess which radar point is the subject.
Picking the nearest point to a projected guess would use the extrinsic that is
being solved for, and would converge on whatever guess it started from. So the
operator clicks the point in a bird's-eye view, in radar coordinates, where no
extrinsic exists yet -- and separately clicks the pixel. Two independent clicks,
no circularity.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:                                   # the core is importable
    cv2 = None                                        # without a GUI stack


# --- session loading ----------------------------------------------------

def load_session(d):
    """Returns (radar_rows, frame_rows, video_path). Raises on a broken set."""
    rp = os.path.join(d, 'radar.jsonl')
    fp = os.path.join(d, 'frames.jsonl')
    vp = os.path.join(d, 'session.mp4')
    if not os.path.exists(rp):
        raise SystemExit('%s has no radar.jsonl -- was it recorded with '
                         '--radar-record?' % d)
    radar = [json.loads(l) for l in open(rp) if l.strip()]
    frames = ([json.loads(l) for l in open(fp) if l.strip()]
              if os.path.exists(fp) else [])
    if not frames:
        # Recorded before frames.jsonl existed. Say so rather than silently
        # falling back to index/fps, which is wrong by design here: the stream
        # is sensor-paced and has 1824 ms FFC gaps in it.
        print('warning: no frames.jsonl -- pictures cannot be matched to radar '
              'segments by time. Re-record to calibrate.', file=sys.stderr)
    return radar, frames, (vp if os.path.exists(vp) else None)


def polar(p):
    """(x_fwd, y_left, z_up) -> (range_m, azimuth_deg, elevation_deg).

    Azimuth is positive to the LEFT, matching the project radar frame. Sign
    conventions get written down here rather than inferred at the call site,
    because this is the number the whole calibration turns on.
    """
    x, y, z = p[0], p[1], p[2]
    r = math.sqrt(x * x + y * y + z * z)
    az = math.degrees(math.atan2(y, x))
    el = math.degrees(math.asin(z / r)) if r > 1e-9 else 0.0
    return r, az, el


# --- hold detection -----------------------------------------------------

# The camera's half-FOV. The radar's aoaFovCfg admits +-60 deg but the PAG7936
# sees about +-35, so a quarter of the radar's cone is space where a subject is
# detected and photographed by nothing. Measured on captures/walk2: 9 of 12
# otherwise-perfect holds were in that gap and could never have become
# correspondences. Marking it is the difference between a wasted session and a
# one-sentence instruction.
CAMERA_HALF_FOV_DEG = 35.0


def find_holds(radar, min_frames=25, tol_m=0.15, tol_deg=4.0, max_gap=2,
               half_fov_deg=CAMERA_HALF_FOV_DEG):
    """Find stretches where one detection stays put -- a subject holding still.

    Association is deliberately crude: nearest point within a gate, frame to
    frame. A real tracker would be better at crossing targets and worse here,
    because what is wanted is not "where did everything go" but "which returns
    sat still long enough to be measured". max_gap tolerates the odd frame where
    CFAR loses the target without ending the hold.
    """
    tracks = []          # each: {'pts': [(row_i, point)], 'last': i, 'ref': (r,az)}
    live = []
    for i, row in enumerate(radar):
        pts = row['points']
        used = set()
        still_live = []
        for tr in live:
            r0, az0 = tr['ref']
            best, best_d = None, None
            for j, p in enumerate(pts):
                if j in used:
                    continue
                r, az, _ = polar(p)
                if abs(r - r0) > tol_m or abs(az - az0) > tol_deg:
                    continue
                d = abs(r - r0) / tol_m + abs(az - az0) / tol_deg
                if best_d is None or d < best_d:
                    best, best_d = j, d
            if best is not None:
                used.add(best)
                tr['pts'].append((i, pts[best]))
                tr['last'] = i
                still_live.append(tr)
            elif i - tr['last'] <= max_gap:
                still_live.append(tr)          # brief dropout, keep the track
            else:
                tracks.append(tr)
        live = still_live
        for j, p in enumerate(pts):
            if j in used:
                continue
            r, az, _ = polar(p)
            live.append({'pts': [(i, p)], 'last': i, 'ref': (r, az)})
    tracks.extend(live)

    holds = []
    for tr in tracks:
        if len(tr['pts']) < min_frames:
            continue
        idx = [i for i, _ in tr['pts']]
        pol = np.array([polar(p) for _, p in tr['pts']])
        # A track that drifts is a walking subject, not a hold. Rejecting it
        # here is the whole point: a walking correspondence is an aliased one.
        if pol[:, 0].std() > tol_m or pol[:, 1].std() > tol_deg:
            continue
        holds.append({
            'i0': idx[0], 'i1': idx[-1], 'n': len(idx),
            't0': radar[idx[0]]['t_mono'], 't1': radar[idx[-1]]['t_mono'],
            'range_m': float(pol[:, 0].mean()),
            'az_deg': float(pol[:, 1].mean()),
            'el_deg': float(pol[:, 2].mean()),
            'range_sd': float(pol[:, 0].std()),
            'az_sd': float(pol[:, 1].std()),
            'el_sd': float(pol[:, 2].std()),
            'xyz': [float(v) for v in
                    np.array([p[:3] for _, p in tr['pts']]).mean(axis=0)],
            'snr_db': float(np.mean([p[4] for _, p in tr['pts']
                                     if p[4] is not None] or [0.0])),
            'v_max_abs': float(max(abs(p[3]) for _, p in tr['pts'])),
        })
    # Scenery, not subjects. A wall or a cabinet is stationary for the whole
    # recording; a person holding a pose is stationary for three seconds. Both
    # look identical in one frame, and duration is the only thing that separates
    # them -- so a return that persists across most of the session is the room,
    # and offering it as a candidate is how a calibration ends up fitted to a
    # radiator. Kept in the list but marked, because a session where EVERY hold
    # is scenery means the subject never actually stopped, and silently
    # returning nothing would hide that.
    #
    # Span alone is not enough, and the first version of this got it wrong on
    # real data. Room clutter sitting near the CFAR threshold BLINKS: it drops
    # below the threshold and comes back, which shatters one permanent return
    # into a dozen short tracks that each pass a span test easily. Measured on
    # captures/walk1: the reported "subjects" were six separate tracks at
    # r=2.97 az=-20.1 and five at r=4.23 az=-46.0, every one with sd(az) of
    # exactly 0.00 -- a person breathes, a table edge lands in the same bin.
    #
    # So the real discriminator is RECURRENCE, not duration. A subject occupies
    # a cell once and leaves; a blinking object returns to the same cell over
    # and over across the whole session.
    if radar:
        span = len(radar)
        for h in holds:
            same = [g for g in holds
                    if abs(g['range_m'] - h['range_m']) < 0.25
                    and abs(g['az_deg'] - h['az_deg']) < 4.0]
            recurrent = len(same) >= 3
            spread = (max(g['i1'] for g in same) -
                      min(g['i0'] for g in same)) > 0.5 * span
            h['revisits'] = len(same) - 1
            h['in_camera'] = abs(h['az_deg']) <= half_fov_deg
            h['scenery'] = ((h['i1'] - h['i0'] + 1) > 0.7 * span
                            or (recurrent and spread)
                            # Zero angular variance over dozens of frames is not
                            # a body. It is a rigid scatterer quantised into one
                            # bin, and it is the cheapest tell in the data.
                            or (h['az_sd'] < 0.02 and h['range_sd'] < 0.002))
    holds.sort(key=lambda h: h['t0'])
    return holds


def frame_at(frames, t):
    """Index of the recorded picture nearest in time to t, or None."""
    if not frames:
        return None, None
    best = min(frames, key=lambda f: abs(f['t_mono'] - t))
    return best['i'], abs(best['t_mono'] - t)


# --- spread, the thing that decides whether a set is worth solving ------

def spread_report(corrs, image_size=(640, 400)):
    """Judge the correspondence SET, not any single correspondence.

    A degenerate set fits beautifully and means nothing -- residual falls
    because the model has more freedom than the data constrains, not because
    the answer is right. So this reports the four spreads that matter and
    refuses to call a set usable without them.
    """
    if not corrs:
        return {'n': 0, 'ok': False, 'issues': ['no correspondences']}
    az = np.array([c['az_deg'] for c in corrs])
    el = np.array([c['el_deg'] for c in corrs])
    rng = np.array([c['range_m'] for c in corrs])
    u = np.array([c['u'] for c in corrs])
    v = np.array([c['v'] for c in corrs])

    issues = []
    if len(corrs) < 12:
        issues.append('only %d correspondences; the solver gate needs 12 and '
                      'the plan calls for ~40' % len(corrs))
    if az.max() - az.min() < 30:
        issues.append('azimuth spans %.1f deg; needs >=30 across BOTH signs, '
                      'or the azimuth sign is unobservable' % (az.max() - az.min()))
    if not (az.min() < -8 and az.max() > 8):
        issues.append('all correspondences are on one side of boresight '
                      '(%.1f..%.1f deg)' % (az.min(), az.max()))
    if el.max() - el.min() < 8:
        issues.append('elevation spans %.1f deg; without vertical spread the '
                      'up/down sign is unobservable and the solver returns its '
                      'initial guess' % (el.max() - el.min()))
    if rng.max() / max(rng.min(), 1e-6) < 1.6:
        issues.append('depth ratio %.2f; needs >=1.6 or rotation and '
                      'translation are indistinguishable'
                      % (rng.max() / max(rng.min(), 1e-6)))
    if len(set(np.round(rng, 1))) < 3:
        issues.append('fewer than 3 distinct depths')
    w, h = image_size
    if (u.max() - u.min()) < 0.4 * w or (v.max() - v.min()) < 0.25 * h:
        issues.append('image coverage %.0fx%.0f px is too clustered'
                      % (u.max() - u.min(), v.max() - v.min()))
    aliased = [c for c in corrs if c.get('v_max_abs', 0) > 0.39]
    if aliased:
        issues.append('%d correspondences were not stationary (|v| up to '
                      '%.2f m/s) -- their azimuth is corrupted by Doppler '
                      'aliasing' % (len(aliased),
                                    max(c['v_max_abs'] for c in aliased)))
    return {
        'n': len(corrs),
        'azimuth_deg': [float(az.min()), float(az.max())],
        'elevation_deg': [float(el.min()), float(el.max())],
        'range_m': [float(rng.min()), float(rng.max())],
        'depth_ratio': float(rng.max() / max(rng.min(), 1e-6)),
        'image_span_px': [float(u.max() - u.min()), float(v.max() - v.min())],
        'mean_frames_averaged': float(np.mean([c['n'] for c in corrs])),
        'ok': not issues,
        'issues': issues,
    }


def save(path, corrs, session, meta=None):
    """Provenance travels with the data or the data is not evidence."""
    with open(path, 'w') as f:
        json.dump({
            'session': os.path.abspath(session),
            'convention': 'radar x_fwd y_left z_up; image u right, v down',
            'protocol': 'stationary hold, frames averaged per correspondence',
            'correspondences': corrs,
            'spread': spread_report(corrs),
            'meta': meta or {},
        }, f, indent=2)


# --- picking ------------------------------------------------------------

def _bird(holds, active, size=400, max_r=None):
    """Top-down radar view. No extrinsic is involved, which is the point."""
    img = np.zeros((size, size, 3), np.uint8)
    max_r = max_r or max([h['range_m'] for h in holds] + [3.0]) * 1.15
    cx, cy = size // 2, size - 20
    scale = (size - 40) / max_r
    for ring in range(1, int(max_r) + 1):
        cv2.circle(img, (cx, cy), int(ring * scale), (40, 40, 46), 1)
        cv2.putText(img, '%dm' % ring, (cx + 4, cy - int(ring * scale) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 100), 1, cv2.LINE_AA)
    for k, h in enumerate(holds):
        a = math.radians(h['az_deg'])
        x = int(cx - math.sin(a) * h['range_m'] * scale)   # +az is LEFT
        y = int(cy - math.cos(a) * h['range_m'] * scale)
        on = (k == active)
        cv2.circle(img, (x, y), 9 if on else 5,
                   (255, 90, 180) if on else (120, 60, 90), -1 if on else 1)
        cv2.putText(img, str(k), (x + 11, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255) if on else (140, 140, 150), 1, cv2.LINE_AA)
    return img


def pick(session_dir, out_path, min_frames=25, image_size=(640, 400)):
    if cv2 is None:
        raise SystemExit('picking needs OpenCV with GUI support')
    radar, frames, video = load_session(session_dir)
    holds = [h for h in find_holds(radar, min_frames=min_frames)
             if not h.get('scenery')]
    if not holds:
        raise SystemExit('no subject holds found -- every stationary return '
                         'lasted most of the session, i.e. it is all room. Did '
                         'the subject stop and hold for ~3 s?')
    if not video:
        raise SystemExit('no session.mp4 in %s' % session_dir)
    cap = cv2.VideoCapture(video)

    corrs, click = [], {}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            click['uv'] = (float(x), float(y))

    cv2.namedWindow('pick')
    cv2.setMouseCallback('pick', on_mouse)
    print('click the subject in the picture, then ENTER to accept. '
          'n=skip  u=undo  q=finish')
    print('if the session was recorded in mix/fused view: the warm blob is the')
    print('easiest way to FIND the subject, but click the VISIBLE body outline,')
    print('not the blob centre -- the thermal layer is unregistered (no warp')
    print('LUT) and its unknown offset must not enter the radar extrinsic.')

    k = 0
    while k < len(holds):
        h = holds[k]
        fi, dt = frame_at(frames, 0.5 * (h['t0'] + h['t1']))
        if fi is None:
            print('hold %d has no picture; skipped' % k)
            k += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            k += 1
            continue
        click.pop('uv', None)
        while True:
            shown = frame.copy()
            hud = ('hold %d/%d  r=%.2fm az=%+.1f el=%+.1f  %d frames  '
                   'sd(az)=%.2f  dt=%.0fms'
                   % (k + 1, len(holds), h['range_m'], h['az_deg'], h['el_deg'],
                      h['n'], h['az_sd'], (dt or 0) * 1000))
            cv2.putText(shown, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if h['v_max_abs'] > 0.39:
                cv2.putText(shown, 'NOT STATIONARY - azimuth suspect',
                            (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (60, 160, 255), 2, cv2.LINE_AA)
            if 'uv' in click:
                u, v = click['uv']
                cv2.drawMarker(shown, (int(u), int(v)), (180, 90, 255),
                               cv2.MARKER_CROSS, 18, 2)
            bird = _bird(holds, k, size=shown.shape[0])
            cv2.imshow('pick', np.hstack([shown, bird]))
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):                      # ENTER
                if 'uv' not in click:
                    print('  click the subject first')
                    continue
                u, v = click['uv']
                corrs.append({'x': h['xyz'][0], 'y': h['xyz'][1], 'z': h['xyz'][2],
                              'u': u, 'v': v, 'range_m': h['range_m'],
                              'az_deg': h['az_deg'], 'el_deg': h['el_deg'],
                              'n': h['n'], 'az_sd': h['az_sd'],
                              'snr_db': h['snr_db'], 'v_max_abs': h['v_max_abs'],
                              'frame_index': fi, 'dt_s': dt})
                print('  %2d: r=%.2f az=%+.1f -> (%.0f, %.0f)'
                      % (len(corrs), h['range_m'], h['az_deg'], u, v))
                k += 1
                break
            if key == ord('n'):
                k += 1
                break
            if key == ord('u') and corrs:
                corrs.pop()
                print('  undone (%d left)' % len(corrs))
            if key == ord('q'):
                k = len(holds)
                break
    cap.release()
    cv2.destroyAllWindows()

    save(out_path, corrs, session_dir)
    print('\nwrote %d correspondences to %s' % (len(corrs), out_path))
    _print_report(spread_report(corrs, image_size))


def _print_report(rep):
    print('\ncorrespondence set: %d' % rep['n'])
    if rep['n']:
        print('  azimuth   %+.1f .. %+.1f deg' % tuple(rep['azimuth_deg']))
        print('  elevation %+.1f .. %+.1f deg' % tuple(rep['elevation_deg']))
        print('  range     %.2f .. %.2f m  (ratio %.2f)'
              % (rep['range_m'][0], rep['range_m'][1], rep['depth_ratio']))
        print('  image     %.0f x %.0f px covered' % tuple(rep['image_span_px']))
        print('  averaged  %.0f radar frames per correspondence'
              % rep['mean_frames_averaged'])
    print('  verdict:  %s' % ('USABLE' if rep['ok'] else 'NOT USABLE'))
    for i in rep['issues']:
        print('    - %s' % i)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    h = sub.add_parser('holds', help='list stationary holds in a session')
    h.add_argument('session')
    h.add_argument('--min-frames', type=int, default=25)
    p = sub.add_parser('pick', help='click correspondences')
    p.add_argument('session')
    p.add_argument('-o', '--out', default='correspondences.json')
    p.add_argument('--min-frames', type=int, default=25)
    r = sub.add_parser('report', help='judge an existing correspondence set')
    r.add_argument('corr')
    a = ap.parse_args()

    if a.cmd == 'holds':
        radar, frames, _ = load_session(a.session)
        holds = find_holds(radar, min_frames=a.min_frames)
        print('%d radar frames, %d pictures, %d holds'
              % (len(radar), len(frames), len(holds)))
        for k, hd in enumerate(holds):
            fi, dt = frame_at(frames, 0.5 * (hd['t0'] + hd['t1']))
            print('  %2d  r=%5.2fm az=%+6.1f el=%+6.1f  %3d frames  '
                  'sd(r)=%.3f sd(az)=%.2f  snr=%.1fdB  frame=%s%s%s'
                  % (k, hd['range_m'], hd['az_deg'], hd['el_deg'], hd['n'],
                     hd['range_sd'], hd['az_sd'], hd['snr_db'], fi,
                     '  MOVING' if hd['v_max_abs'] > 0.39 else '',
                     '  SCENERY' if hd.get('scenery') else
                     ('' if hd.get('in_camera', True) else '  OUTSIDE PICTURE')))
        subj = [h for h in holds if not h.get('scenery')]
        usable = [h for h in subj if h.get('in_camera', True)]
        print('\n%d of %d holds are candidate subjects; %d of those are inside '
              'the camera cone and can become correspondences'
              % (len(subj), len(holds), len(usable)))
        if usable:
            import numpy as _np
            rs = _np.array([h['range_m'] for h in usable])
            az = _np.array([h['az_deg'] for h in usable])
            el = _np.array([h['el_deg'] for h in usable])
            print('  depth ratio %.2f (need >=1.60) | left %d / right %d '
                  '(need both) | elevation span %.1f deg (need >=8)'
                  % (rs.max() / rs.min(), int((az > 0).sum()), int((az < 0).sum()),
                     el.max() - el.min()))
        return 0
    if a.cmd == 'pick':
        pick(a.session, a.out, min_frames=a.min_frames)
        return 0
    with open(a.corr) as f:
        _print_report(spread_report(json.load(f)['correspondences']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
