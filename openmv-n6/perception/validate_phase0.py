#!/usr/bin/env python3
"""Phase 0 gate: does the loader + projection stack put radar on the person?

Two checks, one quantitative and one for eyes:

1. Project the holds1/corr.json correspondences through the frozen
   calibration and compare to the residuals T_camera_radar.json itself
   recorded. This is a regression test of the CODE PATH (loader, frames,
   signs, distortion), not a fresh validation of the calibration - that is
   field gate 2 at 10/15 m. The gate therefore runs on the artifact's own
   fit_holds (clean TCR holds): max |du| <= 10 px, matching the recorded
   du_max_px of 7.9. Holds the calibration round excluded (clutter picks,
   moving target, elevation anomaly) are printed for context, never gated -
   their large du IS the documented contamination story. A mirror or
   convention error would show as u_pick and u_proj moving in opposite
   directions across the FOV, so that correlation is gated too.

2. Render an overlay clip around each hold frame: radar points projected
   onto the video (green = inside the validated range band, gray = outside),
   thermal picture-in-picture, age and point count in the HUD. A human
   confirms the dots sit on the person before anything builds on this.

Usage:
    python3 perception/validate_phase0.py captures/holds1 \
        [--out perception/out] [--window-s 3.0]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from perception.dataset import LiveSession           # noqa: E402
from perception.project import RadarProjector        # noqa: E402

GATE_MAX_ABS_DU_FIT_PX = 10.0


def check_correspondences(session_dir, proj):
    path = os.path.join(session_dir, 'corr.json')
    if not os.path.exists(path):
        print('no corr.json in this session - skipping quantitative check')
        return None, []
    with open(path) as f:
        corr = json.load(f)
    ext_path = os.path.join(os.path.dirname(__file__), '..',
                            'calib-artifacts', 'T_camera_radar.json')
    with open(ext_path) as f:
        ext = json.load(f)
    fit_holds = set(ext.get('fit_holds', []))
    excluded = ext.get('excluded', {})

    rows, hold_frames = [], []
    for c in corr['correspondences']:
        uv, in_front = proj.project(np.array([c['radar']]))
        if not in_front[0]:
            continue
        role = ('FIT' if c['name'] in fit_holds else
                'excl' if c['name'] in excluded else 'other')
        rows.append({'name': c['name'], 'range_m': c['range_m'],
                     'du': uv[0, 0] - c['u'], 'dv': uv[0, 1] - c['v'],
                     'role': role, 'u_pick': c['u']})
        hold_frames.append(c['frame_index'])

    print(f"{'hold':>8} {'range':>6} {'role':>6} {'du_px':>8} {'dv_px':>8}")
    for r in rows:
        print(f"{r['name']:>8} {r['range_m']:>6.2f} {r['role']:>6} "
              f"{r['du']:>8.1f} {r['dv']:>8.1f}")

    fit = [r for r in rows if r['role'] == 'FIT']
    max_fit = max(abs(r['du']) for r in fit) if fit else float('nan')
    # Sign check across ALL holds: with a mirror/convention bug, projected
    # and picked u move in opposite directions across the FOV.
    if len(rows) >= 4:
        u_pick = np.array([r['u_pick'] for r in rows])
        u_proj = u_pick + np.array([r['du'] for r in rows])
        sign_r = float(np.corrcoef(u_pick, u_proj)[0, 1])
    else:
        sign_r = float('nan')

    ok = bool(fit) and max_fit <= GATE_MAX_ABS_DU_FIT_PX and sign_r > 0.9
    print(f'\nmax |du| on the {len(fit)} clean fit holds: {max_fit:.1f} px '
          f'(gate <= {GATE_MAX_ABS_DU_FIT_PX:.0f}; artifact says 7.9)')
    print(f'u_pick vs u_proj correlation, all holds: {sign_r:.3f} (gate > 0.9)')
    print(f'PHASE 0 QUANTITATIVE GATE: {"PASS" if ok else "FAIL"}')
    return ok, hold_frames


def render_overlay(session_dir, proj, out_path, hold_frames, window_s,
                   fps=25.0):
    sess = LiveSession(session_dir)
    if hold_frames:
        t0 = {f['i']: f['t_mono'] for f in sess.frames}
        keep = np.zeros(len(sess.frames), dtype=bool)
        times = np.array([f['t_mono'] for f in sess.frames])
        for hf in hold_frames:
            if hf < len(times):
                keep |= np.abs(times - times[hf]) <= window_s
    else:
        keep = np.ones(len(sess.frames), dtype=bool)

    vw = None
    n_drawn = 0
    for tr in sess.triplets():
        if tr.i >= len(keep) or not keep[tr.i]:
            continue
        img = cv2.cvtColor(tr.rgb, cv2.COLOR_GRAY2BGR)
        if tr.radar is not None and len(tr.radar.points):
            uv, in_front = proj.project(tr.radar.xyz)
            in_band = proj.in_calibrated_band(tr.radar.xyz)
            for k in range(len(uv)):
                if not in_front[k]:
                    continue
                u, v = uv[k]
                if not (0 <= u < img.shape[1] and 0 <= v < img.shape[0]):
                    continue
                snr = tr.radar.points[k, 4]
                rng = float(np.linalg.norm(tr.radar.xyz[k]))
                color = (0, 255, 0) if in_band[k] else (160, 160, 160)
                rad = int(np.clip(2 + snr / 6, 2, 9))
                cv2.circle(img, (int(u), int(v)), rad, color, 2)
                cv2.putText(img, f'{rng:.1f}', (int(u) + 6, int(v) - 6),
                            cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)
        # thermal PIP, top-right; nearest-neighbour so rows stay honest
        pip = cv2.applyColorMap(tr.thermal, cv2.COLORMAP_INFERNO)
        img[4:124, img.shape[1] - 164:img.shape[1] - 4] = pip
        age = f'{tr.age_ms:+.0f} ms' if tr.age_ms is not None else 'no radar'
        npts = len(tr.radar.points) if tr.radar is not None else 0
        cv2.putText(img, f'i={tr.i}  radar age {age}  pts {npts}',
                    (8, img.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1.2,
                    (0, 255, 255), 1)
        if vw is None:
            vw = cv2.VideoWriter(out_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                 (img.shape[1], img.shape[0]))
        vw.write(img)
        n_drawn += 1
    if vw is not None:
        vw.release()
    print(f'overlay: {n_drawn} frames -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'out'))
    ap.add_argument('--window-s', type=float, default=3.0)
    ap.add_argument('--full', action='store_true',
                    help='render the whole session, not just hold windows')
    a = ap.parse_args()

    proj = RadarProjector()
    ok, hold_frames = check_correspondences(a.session, proj)
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(os.path.normpath(a.session))
    out_mp4 = os.path.join(a.out, f'{name}_phase0_overlay.mp4')
    render_overlay(a.session, proj, out_mp4,
                   [] if a.full else hold_frames, a.window_s)


if __name__ == '__main__':
    main()
