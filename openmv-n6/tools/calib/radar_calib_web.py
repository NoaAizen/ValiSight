#!/usr/bin/env python3
"""Radar<->camera calibration, end to end, in a browser.

    ./radar_calib_web.py ../../captures/walk3            # then open :8081
    ./radar_calib_web.py ../../captures/walk3 -i calib.json
    ./radar_calib_web.py ../../captures/walk3 --port 8090

Wraps the two tools that already exist into the one workflow that was missing:
radar_correspond.py finds the stationary holds in a recorded session but its
picker needs a local OpenCV window, and the rig is driven from a browser;
radar_extrinsics.py solves and gates a correspondence set but needs a set to
exist. This serves the picking UI over HTTP - click the subject in the picture
for each hold, then press solve - and runs the solver in place, writing both
corr.json and radar_calib.json next to the session.

The input is a session recorded by:

    ./live.py --radar /dev/ttyACM2 --record DIR --view visible

(visible, not fused: the thermal layer is unregistered and its offset must not
leak into the radar extrinsic through a click on a warm blob.)

EVIDENCE PURITY. Two per-frame flags in frames.jsonl are enforced, not assumed:
'clean' says the frame was recorded BEFORE the guessed radar overlay and the
detection boxes were drawn on it (live.py writes it since this tool was added -
a pixel clicked next to a burned-in radar marker is a correspondence derived
from the projection being solved for), and 'view' must be 'visible'. Holds
whose picture fails either check are excluded from picking; --allow-tainted
admits them, and the taint travels into corr.json where the solver's audit
trail can see it. Holds whose radar track moved (|v| past the aliasing band)
are picked but excluded from the solve unless --allow-moving: their azimuth
carries the TDM-MIMO Doppler-folding error, up to ~9.6 deg.

INTRINSICS. radar_extrinsics.py refuses to run without K, and calib.py has not
been run on this rig (DESIGN.md 2.1). Rather than block the whole workflow,
-i/--intrinsics is optional here: without it K is built from the MEASURED
horizontal FOV of 63.8 deg (DESIGN.md's consistent D72.5/H63.8/V42.3 triple,
f ~= 514 px), zero distortion, and the output file says so in its provenance.
That produces a real, usable overlay alignment - but t absorbs the K error, so
the file must be re-solved (same corr.json, one command) once calib.py has run.
The nominal-K result is written to radar_calib_nominalK.json, not
radar_calib.json, so a consumer can never confuse the two.

The correspondence file this writes carries BOTH shapes: the {radar, pixel}
pairs radar_extrinsics.load_correspondences() reads and the flat {x,y,z,u,v}
rows radar_correspond.py's report reads, so either tool can audit it later.
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radar_correspond as rc      # noqa: E402
import radar_extrinsics as rx      # noqa: E402

# Measured 2026-08-10 with a caliper-measured 38.3 mm checkerboard at
# tape-measured 1.00 m (19.95 px/square -> f=521) and 2.00 m (10.13 px ->
# f=529), subpixel corners: f = 525 +- 5 px -> HFOV = 62.7 deg. This
# OVERTURNS the 2026-08-09 A4-sheet figure of f~=650-700 (49.8 deg), which
# was itself a mismeasurement, and lands next to DESIGN.md's 63.8 triple.
# Full K+distortion: calib-artifacts/calib.json (prefer -i over this nominal).
MEASURED_HFOV_DEG = 62.7

# Glass-wall specular ghosts are persistent, high-SNR, and sign-flipped in
# azimuth - measured on this rig's lobby, they out-voted the direct return in
# every naive picker. With the sensors measured co-aligned and f measured, a
# pick whose radar azimuth disagrees with its pixel azimuth by more than this
# is a ghost pairing, not evidence. (This veto leans on the ~0 deg measured
# mount yaw; re-measure it if the bracket ever changes.)
GHOST_TOL_DEG = 12.0


def nominal_K(width, height, hfov_deg=MEASURED_HFOV_DEG):
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]]), np.zeros(5)


class CalibSession:
    """All mutable state behind one lock: the HTTP handlers stay dumb."""

    def __init__(self, session_dir, min_frames=25, intrinsics=None,
                 hfov_deg=MEASURED_HFOV_DEG, sigma_az=rx.RADAR_SIGMA_AZ_DEG,
                 sigma_el=rx.RADAR_SIGMA_EL_DEG, rig_id=None, mount_token=None,
                 allow_tainted=False, allow_moving=False, unmirror=False,
                 allow_ghosts=False, ghost_tol_deg=GHOST_TOL_DEG):
        # The video is NOT mirrored: the walk-in pixel track matches radar
        # azimuth sign directly (measured 2026-08-09 late; the earlier
        # raised-hand test that said otherwise used the wrong hand). The flip
        # is kept as an option only in case a future sensor config mirrors
        # the readout.
        self.unmirror = unmirror
        self.dir = os.path.abspath(session_dir)
        self.lock = threading.Lock()
        self.sigma_az, self.sigma_el = sigma_az, sigma_el
        self.rig_id, self.mount_token = rig_id, mount_token
        self.allow_tainted, self.allow_moving = allow_tainted, allow_moving
        self.allow_ghosts = allow_ghosts
        self.ghost_tol = float(ghost_tol_deg)   # <= 0 disables the veto

        radar, frames, video = rc.load_session(session_dir)
        if not frames:
            raise SystemExit('%s has no frames.jsonl - pictures cannot be '
                             'matched to radar holds. Re-record with a current '
                             'live.py.' % session_dir)
        if not video:
            raise SystemExit('%s has no session.mp4' % session_dir)
        self.frames_meta = frames

        holds = rc.find_holds(radar, min_frames=min_frames)
        self.n_scenery = sum(1 for h in holds if h.get('scenery'))
        subj = [h for h in holds if not h.get('scenery')]
        # in_camera is NOT used to exclude, only displayed. It assumes the
        # radar and camera boresights roughly agree, which is exactly what is
        # being calibrated - measured on calib4, a hold the flag called
        # "outside" at az -51 deg had the subject standing plainly in frame.
        # The operator sees the picture; whether the subject is in it is their
        # call, made by clicking or skipping.
        self.n_outside = sum(1 for h in subj if not h.get('in_camera', True))

        # Evidence purity, per hold: the picture it will be clicked on must be
        # the visible view with no overlay burned in. A frame that fails is
        # excluded rather than warned about - tainted evidence looks exactly
        # like good evidence once it is a row in corr.json - unless
        # --allow-tainted turns the exclusion into a recorded taint.
        self.n_tainted = 0
        picked_rows = []
        for h in subj:
            row = min(frames, key=lambda f: abs(f['t_mono']
                                                - 0.5 * (h['t0'] + h['t1'])))
            reasons = []
            view = row.get('view')
            if view not in (None, 'visible'):
                reasons.append('recorded in view=%s, not visible' % view)
            if not row.get('clean'):
                reasons.append('no clean flag: the guessed radar overlay may '
                               'be burned into this picture (older live.py)')
            if reasons and not allow_tainted:
                self.n_tainted += 1
                continue
            picked_rows.append((h, row, reasons))
        self.holds = [h for h, _, _ in picked_rows]
        self.taints = [r for _, _, r in picked_rows]
        if not self.holds:
            raise SystemExit(
                'no usable holds: %d scenery, %d outside the camera cone, %d '
                'with tainted pictures (overlay burned in / wrong view; '
                '--allow-tainted admits them, recorded as such). Did the '
                'subject stop and hold INSIDE the picture, recorded with the '
                'current live.py in --view visible?'
                % (self.n_scenery, self.n_outside, self.n_tainted))

        # Decode every hold's picture once, up front. Dozens of 640x400 frames
        # is a few tens of MB; seeking the mp4 per HTTP request is not
        # thread-safe and not worth making so.
        cap = cv2.VideoCapture(video)
        self.images, self.dt_ms, self.frame_idx = [], [], []
        for h, row, _ in picked_rows:
            fi = row['i']
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read()
            if ok and self.unmirror:
                img = np.ascontiguousarray(img[:, ::-1])
            self.images.append(img if ok else None)
            self.dt_ms.append(abs(row['t_mono']
                                  - 0.5 * (h['t0'] + h['t1'])) * 1000.0)
            self.frame_idx.append(fi)
        cap.release()
        shapes = [im.shape for im in self.images if im is not None]
        if not shapes:
            raise SystemExit('session.mp4 yielded no decodable frames')
        self.h, self.w = shapes[0][:2]

        self.intrinsics_path = intrinsics
        if intrinsics:
            self.K, self.dist = rx.intrinsics_from_calib(intrinsics, 'rgb')
            self.k_source = os.path.basename(intrinsics)
            self.k_nominal = False
        else:
            self.K, self.dist = nominal_K(self.w, self.h, hfov_deg)
            self.k_source = ('NOMINAL: f=%.0f px from measured %.1f deg HFOV, '
                             'zero distortion' % (self.K[0, 0], hfov_deg))
            self.k_nominal = True

        self.corr_path = os.path.join(self.dir, 'corr.json')
        self.calib_path = os.path.join(
            self.dir, 'radar_calib_nominalK.json' if self.k_nominal
            else 'radar_calib.json')

        self.picks = {}          # hold index -> (u, v)
        self.result = None       # dict from _solve, or None
        self._load_existing()

    # -- persistence ------------------------------------------------------

    def _load_existing(self):
        """A crashed or reopened session resumes from its own corr.json."""
        if not os.path.exists(self.corr_path):
            return
        try:
            with open(self.corr_path) as f:
                d = json.load(f)
            for c in d.get('correspondences', []):
                # The hold index is a position in a list that find_holds
                # rebuilds every run; a different --min-frames or a code change
                # shifts every index. The saved radar xyz is the durable
                # identity, so picks are re-associated by it, not by index.
                want = np.asarray(c.get('radar', ()), float)
                if want.shape != (3,):
                    continue
                dist = [float(np.linalg.norm(want - np.asarray(h['xyz'])))
                        for h in self.holds]
                k = int(np.argmin(dist)) if dist else -1
                if k < 0 or dist[k] > 0.10:
                    print('warning: saved pick at r=%.2f m no longer matches '
                          'any hold - dropped' % float(np.linalg.norm(want)),
                          file=sys.stderr)
                    continue
                self.picks[k] = (float(c['u']), float(c['v']))
        except (ValueError, KeyError, TypeError):
            print('warning: %s unreadable, starting empty' % self.corr_path,
                  file=sys.stderr)

    def _az_pixel_deg(self, u):
        """Pixel column -> azimuth in the radar's empirical sign convention.

        Measured on the walk-in track of the boresight session: the radar's
        reported azimuth follows atan((u - cx)/fx) directly (left of frame =
        negative az as reported), so that is the convention used for the
        ghost veto. The full solve does not depend on this - R absorbs signs.
        """
        return float(np.degrees(np.arctan2(u - self.K[0, 2], self.K[0, 0])))

    def _is_ghost(self, k, u):
        if self.ghost_tol <= 0:
            return False
        return abs(self._az_pixel_deg(u)
                   - self.holds[k]['az_deg']) > self.ghost_tol

    def _corrs(self):
        out = []
        for k in sorted(self.picks):
            h = self.holds[k]
            u, v = self.picks[k]
            out.append({
                'name': 'hold_%02d' % k, 'hold': k,
                'radar': [h['xyz'][0], h['xyz'][1], h['xyz'][2]],
                'pixel': [u, v],
                'x': h['xyz'][0], 'y': h['xyz'][1], 'z': h['xyz'][2],
                'u': u, 'v': v,
                'range_m': h['range_m'], 'az_deg': h['az_deg'],
                'el_deg': h['el_deg'], 'n': h['n'], 'az_sd': h['az_sd'],
                'snr_db': h['snr_db'], 'v_max_abs': h['v_max_abs'],
                'frame_index': self.frame_idx[k], 'dt_s': self.dt_ms[k] / 1000.0,
                'taint': self.taints[k],
                'az_pixel_deg': self._az_pixel_deg(u),
                'ghost_suspect': self._is_ghost(k, u),
            })
        return out

    def _save_corrs(self):
        """Write corr.json; returns the exact dict written, so solve() can hand
        the same object to provenance and the digest matches the file."""
        corrs = self._corrs()
        doc = {
            'session': self.dir,
            'camera': 'rgb',
            'image_size': [self.w, self.h],
            'convention': 'radar x_fwd y_left z_up; image u right, v down',
            'pixel_frame': ('unmirrored: video flipped horizontally on load '
                            '(live.py stream is mirrored, measured 2026-08-09)'
                            if self.unmirror else
                            'raw video orientation (mirrored stream!)'),
            'protocol': 'stationary hold on a HUMAN subject, frames averaged '
                        'per correspondence. Bootstrap-grade: an extended '
                        'target\'s radar centroid is offset from its visual '
                        'outline by a constant that t absorbs invisibly '
                        '(CALIBRATION-PLAN.md step 4); the corner-reflector '
                        'round supersedes this.',
            'correspondences': corrs,
            'spread': rc.spread_report(corrs, (self.w, self.h)),
        }
        with open(self.corr_path, 'w') as f:
            json.dump(doc, f, indent=2)
        return doc

    # -- actions ----------------------------------------------------------

    def accept(self, k, u, v):
        with self.lock:
            if not (0 <= k < len(self.holds)):
                return {'error': 'no hold %d' % k}
            self.picks[k] = (float(u), float(v))
            self.result = None          # stale the moment the set changes
            self._save_corrs()
            return self.state()

    def remove(self, k):
        with self.lock:
            self.picks.pop(int(k), None)
            self.result = None
            self._save_corrs()
            return self.state()

    def solve(self):
        with self.lock:
            all_corrs = self._corrs()
            # A moving hold's azimuth carries the TDM-MIMO Doppler-folding
            # error (up to ~9.6 deg under radar_10hz.cfg). RANSAC is a net for
            # the odd blunder, not a licence to feed it known-corrupt angles.
            corrs = [c for c in all_corrs
                     if self.allow_moving or c['v_max_abs'] <= 0.39]
            n_moving = len(all_corrs) - len(corrs)
            n_ghost = sum(1 for c in corrs if c['ghost_suspect'])
            if not self.allow_ghosts:
                corrs = [c for c in corrs if not c['ghost_suspect']]
            if len(corrs) < 6:
                dropped = []
                if n_moving:
                    dropped.append('%d moving' % n_moving)
                if n_ghost and not self.allow_ghosts:
                    dropped.append('%d ghost-suspect (radar az far from '
                                   'pixel az - glass specular)' % n_ghost)
                return {'error': 'need at least 6 stationary correspondences '
                                 'to solve, have %d%s (the gate wants 12+)'
                                 % (len(corrs),
                                    ' after excluding ' + ' and '.join(dropped)
                                    if dropped else '')}
            radar = np.array([c['radar'] for c in corrs], float)
            pix = np.array([c['pixel'] for c in corrs], float)
            size = (self.w, self.h)
            try:
                R, t, stats = rx.solve(radar, pix, self.K, self.dist,
                                       sigma_az_deg=self.sigma_az,
                                       sigma_el_deg=self.sigma_el,
                                       image_size=size)
            except (ValueError, RuntimeError, cv2.error) as e:
                return {'error': 'solve failed: %s' % e}
            g = rx.gate(radar, pix, stats, size)
            try:
                hold = rx.holdout_by_depth(radar, pix, self.K, self.dist, size,
                                           sigma_az_deg=self.sigma_az,
                                           sigma_el_deg=self.sigma_el)
            except (ValueError, RuntimeError, cv2.error):
                hold = None

            notes = []
            if self.k_nominal:
                notes.append('intrinsics are NOMINAL (%s) - t has absorbed '
                             'the K error; re-solve the same corr.json '
                             'against calib.py output before trusting '
                             'close-range parallax' % self.k_source)
            notes.append('human-subject bootstrap protocol: the target is an '
                         'extended body, not a corner reflector, so t carries '
                         'an unobservable radar-centroid-vs-outline offset. '
                         'Supersede with the reflector round of '
                         'CALIBRATION-PLAN.md step 4.')
            if n_moving:
                notes.append('%d moving hold(s) excluded from the fit'
                             % n_moving)
            if n_ghost:
                notes.append('%d ghost-suspect pick(s) %s (|az_radar - '
                             'az_pixel| > %.0f deg)'
                             % (n_ghost, 'INCLUDED by --allow-ghost-picks'
                                if self.allow_ghosts else 'excluded',
                                GHOST_TOL_DEG))

            # A failed fit never lands on the path a good one lives at: the
            # CLI refuses to write without --force for the same reason, and an
            # interactive re-solve must not eat yesterday's passing file.
            out_path = (self.calib_path if g['passed']
                        else self.calib_path.replace('.json', '.FAILED.json'))
            saved = rx.save_calib(
                out_path, R, t, self.K, self.dist, size,
                stats, g, camera='rgb', holdout=hold,
                meta={'correspondence_path': self.corr_path,
                      'correspondences': self._save_corrs(),
                      'intrinsics_path': self.intrinsics_path,
                      'rig_id': self.rig_id,
                      'mount_token': self.mount_token,
                      'notes': ' | '.join(notes)})
            # Machine-readable, not just a filename and a prose note: a
            # consumer that would trust nominal-K parallax must be able to
            # branch on a field.
            saved['k_provisional'] = self.k_nominal
            saved['protocol'] = 'human-subject bootstrap'
            with open(out_path, 'w') as f:
                json.dump(saved, f, indent=2)

            uv, _, status = rx.project(radar, R, t, self.K, self.dist, size)
            proj = {}
            for c, (pu, pv), st in zip(corrs, uv, status):
                if np.isfinite(pu):
                    proj[c['hold']] = (float(pu), float(pv), int(st))
            worst = None
            if hold:
                rows = [r for r in hold if 'held_median_px' in r]
                if rows:
                    worst = max(rows, key=lambda r: r['held_median_px'])
            self.result = {
                'passed': g['passed'],
                'summary': rx.summarise(stats, g),
                'holdout_worst': (
                    'worst held-out distance %.2f-%.2f m: median %.2f px '
                    '(in-sample there %.2f px)'
                    % (worst['range_m'][0], worst['range_m'][1],
                       worst['held_median_px'], worst['fit_median_px'])
                    if worst else 'holdout not possible (need 6+ points '
                                  'outside every depth bin)'),
                'proj': proj,
                'wrote': out_path,
                'nominal_k': self.k_nominal,
                'n_moving_excluded': n_moving,
            }
            return self.state()

    # -- views ------------------------------------------------------------

    def state(self):
        """Everything the page renders, in one JSON blob."""
        corrs = self._corrs()
        rep = rc.spread_report(corrs, (self.w, self.h)) if corrs else \
            {'n': 0, 'ok': False, 'issues': ['no correspondences yet']}
        res = None
        if self.result:
            res = {k: v for k, v in self.result.items() if k != 'proj'}
        return {
            'session': self.dir,
            'image_size': [self.w, self.h],
            'k_source': self.k_source, 'k_nominal': self.k_nominal,
            'n_scenery': self.n_scenery, 'n_outside': self.n_outside,
            'n_tainted': self.n_tainted,
            'corr_path': self.corr_path, 'calib_path': self.calib_path,
            'holds': [{
                'k': k, 'range_m': h['range_m'], 'az_deg': h['az_deg'],
                'el_deg': h['el_deg'], 'n': h['n'], 'az_sd': h['az_sd'],
                'snr_db': h['snr_db'], 'moving': h['v_max_abs'] > 0.39,
                'taint': self.taints[k], 'dt_ms': self.dt_ms[k],
                'picked': (list(self.picks[k]) if k in self.picks else None),
                'ghost': (self._is_ghost(k, self.picks[k][0])
                          if k in self.picks else False),
            } for k, h in enumerate(self.holds)],
            'spread': rep,
            'result': res,
        }

    def frame_jpeg(self, k):
        with self.lock:
            if not (0 <= k < len(self.holds)) or self.images[k] is None:
                return None
            img = self.images[k].copy()
            h = self.holds[k]
            hud = ('hold %d/%d  r=%.2fm az=%+.1f el=%+.1f  %d frames  '
                   'sd(az)=%.2f  dt=%.0fms'
                   % (k + 1, len(self.holds), h['range_m'], h['az_deg'],
                      h['el_deg'], h['n'], h['az_sd'], self.dt_ms[k]))
            cv2.putText(img, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if h['v_max_abs'] > 0.39:
                cv2.putText(img, 'NOT STATIONARY - azimuth suspect', (8, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 160, 255), 2,
                            cv2.LINE_AA)
            if k in self.picks:
                u, v = self.picks[k]
                cv2.drawMarker(img, (int(u), int(v)), (180, 90, 255),
                               cv2.MARKER_CROSS, 18, 2)
            if self.result and k in self.result['proj']:
                pu, pv, _ = self.result['proj'][k]
                p = (int(round(pu)), int(round(pv)))
                cv2.circle(img, p, 7, (90, 220, 90), 2)
                if k in self.picks:
                    u, v = self.picks[k]
                    cv2.line(img, (int(u), int(v)), p, (90, 220, 90), 1,
                             cv2.LINE_AA)
                    err = math.hypot(pu - u, pv - v)
                    cv2.putText(img, '%.1f px' % err, (p[0] + 10, p[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 220, 90),
                                1, cv2.LINE_AA)
            ok, buf = cv2.imencode('.jpg', img,
                                   [cv2.IMWRITE_JPEG_QUALITY, 90])
            return buf.tobytes() if ok else None

    def bird_jpeg(self, k):
        with self.lock:
            img = rc._bird(self.holds, k, size=self.h)
            # _bird already highlights the active hold; add a ring on picks
            max_r = max([h['range_m'] for h in self.holds] + [3.0]) * 1.15
            cx, cy = img.shape[1] // 2, img.shape[0] - 20
            scale = (img.shape[0] - 40) / max_r
            for j, h in enumerate(self.holds):
                if j not in self.picks:
                    continue
                a = math.radians(h['az_deg'])
                x = int(cx - math.sin(a) * h['range_m'] * scale)
                y = int(cy - math.cos(a) * h['range_m'] * scale)
                cv2.circle(img, (x, y), 12, (90, 220, 90), 1)
            ok, buf = cv2.imencode('.jpg', img,
                                   [cv2.IMWRITE_JPEG_QUALITY, 90])
            return buf.tobytes() if ok else None


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>radar &harr; camera calibration</title><style>
 body{background:#111;color:#ddd;font:14px system-ui,sans-serif;margin:16px}
 h1{font-size:18px} a{color:#8ac} .row{display:flex;gap:12px;flex-wrap:wrap}
 img{max-width:100%;display:block} #framewrap{position:relative;flex:0 1 640px}
 #frame{cursor:crosshair;width:100%}
 #hint{color:#aaa;margin:6px 0} button{background:#333;color:#ddd;border:1px
 solid #555;padding:6px 14px;margin-right:6px;border-radius:4px;cursor:pointer}
 button:hover{background:#444} #solve{background:#274;border-color:#3a6}
 table{border-collapse:collapse;margin-top:10px;font-size:13px}
 td,th{padding:2px 10px;text-align:right;border-bottom:1px solid #2a2a2a}
 tr.cur{background:#232b36} tr{cursor:pointer} .picked{color:#7d7}
 .mov{color:#fa6} pre{background:#181818;padding:10px;border:1px solid #333;
 white-space:pre-wrap;max-width:900px}
 .ok{color:#7d7}.bad{color:#f77}.warn{color:#fc6}
 #issues li{color:#fc6}
</style></head><body>
<h1>radar &harr; camera calibration</h1>
<div id="hint">click the subject's <b>visible body outline</b> (not a thermal
blob) &middot; <b>enter</b>/click again = accept &middot; <b>n</b>/<b>p</b> =
next/prev hold &middot; <b>u</b> = unpick &middot; <b>s</b> = solve</div>
<div class="row">
 <div id="framewrap"><img id="frame"></div>
 <div><img id="bird"></div>
</div>
<div style="margin-top:10px">
 <button onclick="nav(-1)">&larr; prev</button>
 <button onclick="nav(1)">next &rarr;</button>
 <button onclick="unpick()">unpick</button>
 <button id="solve" onclick="solve()">solve</button>
 <span id="prog"></span>
</div>
<ul id="issues"></ul>
<pre id="result" style="display:none"></pre>
<table id="holds"></table>
<div id="meta" style="color:#888;margin-top:8px"></div>
<script>
let S=null,k=0,pending=null;
const $=id=>document.getElementById(id);
async function post(p,body){const r=await fetch(p,{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
 return r.json();}
async function refresh(){S=await(await fetch('/state')).json();render();}
function bust(u){return u+'?t='+Date.now();}
function render(){
 if(!S)return;
 if(S.error){alert(S.error);return;}
 $('frame').src=bust('/frame/'+k); $('bird').src=bust('/bird/'+k);
 const n=S.holds.filter(h=>h.picked).length;
 $('prog').textContent=' '+n+'/'+S.holds.length+' holds picked';
 const ul=$('issues');ul.innerHTML='';
 (S.spread.issues||[]).forEach(t=>{const li=document.createElement('li');
  li.textContent=t;ul.appendChild(li);});
 const tb=$('holds');tb.innerHTML=
  '<tr><th>#</th><th>range m</th><th>az &deg;</th><th>el &deg;</th>'+
  '<th>frames</th><th>sd(az)</th><th>snr dB</th><th></th></tr>';
 S.holds.forEach(h=>{const tr=document.createElement('tr');
  if(h.k===k)tr.className='cur';
  tr.innerHTML='<td>'+h.k+'</td><td>'+h.range_m.toFixed(2)+'</td><td>'+
   h.az_deg.toFixed(1)+'</td><td>'+h.el_deg.toFixed(1)+'</td><td>'+h.n+
   '</td><td>'+h.az_sd.toFixed(2)+'</td><td>'+h.snr_db.toFixed(1)+'</td><td>'+
   (h.picked?'<span class=picked>&#10003; ('+h.picked[0].toFixed(0)+','+
    h.picked[1].toFixed(0)+')</span>':'')+
   (h.moving?' <span class=mov>moving</span>':'')+
   (h.ghost?' <span class=mov>GHOST? az mismatch</span>':'')+'</td>';
  tr.onclick=()=>{k=h.k;pending=null;render();};
  tb.appendChild(tr);});
 const r=$('result');
 if(S.result){r.style.display='block';
  r.innerHTML='<span class="'+(S.result.passed?'ok':'bad')+'">'+
   (S.result.passed?'GATE PASSED':'GATE FAILED')+'</span>'+
   (S.result.nominal_k?'  <span class=warn>(nominal K - provisional)</span>':'')+
   '\\n'+esc(S.result.summary)+'\\n'+esc(S.result.holdout_worst)+
   '\\nwrote '+esc(S.result.wrote);}
 else r.style.display='none';
 $('meta').textContent='session '+S.session+' | intrinsics: '+S.k_source+
  ' | excluded: '+S.n_scenery+' scenery, '+S.n_outside+' outside camera, '+
  S.n_tainted+' tainted pictures | writes '+S.corr_path;
}
function esc(s){const d=document.createElement('div');
 d.appendChild(document.createTextNode(s||''));return d.innerHTML;}
let lastAccept=0;
$('frame').addEventListener('click',async e=>{
 /* a fast double-click on the accept would otherwise land its echo on the
    NEXT hold and stamp it with the same pixel - seen in real data as two
    different stations sharing one pixel exactly */
 if(Date.now()-lastAccept<600)return;
 const im=$('frame'),r=im.getBoundingClientRect();
 const u=(e.clientX-r.left)*im.naturalWidth/r.width;
 const v=(e.clientY-r.top)*im.naturalHeight/r.height;
 if(pending&&Math.hypot(pending[0]-u,pending[1]-v)<12){await accept();return;}
 pending=[u,v];drawPending();
});
function drawPending(){ /* re-fetch keeps it simple: server draws accepted
 marks; the pending one is shown by a floating div */
 let d=$('pend');if(!d){d=document.createElement('div');d.id='pend';
  d.style.cssText='position:absolute;width:14px;height:14px;margin:-7px 0 0 '+
  '-7px;pointer-events:none;border:2px solid #f6a;border-radius:50%';
  $('framewrap').appendChild(d);}
 const im=$('frame'),r=im.getBoundingClientRect(),
  wr=$('framewrap').getBoundingClientRect();
 d.style.left=(pending[0]*r.width/im.naturalWidth+r.left-wr.left)+'px';
 d.style.top=(pending[1]*r.height/im.naturalHeight+r.top-wr.top)+'px';
 d.style.display='block';}
async function accept(){if(!pending)return;
 S=await post('/accept',{k:k,u:pending[0],v:pending[1]});pending=null;
 lastAccept=Date.now();
 const d=$('pend');if(d)d.style.display='none';
 nav(1);}
async function unpick(){S=await post('/remove',{k:k});render();}
async function solve(){$('solve').textContent='solving...';
 S=await post('/solve',{});$('solve').textContent='solve';
 if(S.error){alert(S.error);await refresh();}else render();}
function nav(d){k=Math.max(0,Math.min(S.holds.length-1,k+d));pending=null;
 const p=$('pend');if(p)p.style.display='none';render();}
document.addEventListener('keydown',e=>{
 if(e.key==='Enter')accept();
 else if(e.key==='n')nav(1);else if(e.key==='p')nav(-1);
 else if(e.key==='u')unpick();else if(e.key==='s')solve();});
refresh();
</script></body></html>"""


def make_handler(cal):
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), 'application/json')

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/':
                self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
            elif path == '/state':
                with cal.lock:
                    self._json(cal.state())
            elif path.startswith('/frame/') or path.startswith('/bird/'):
                try:
                    k = int(path.rsplit('/', 1)[1])
                except ValueError:
                    self._json({'error': 'bad index'}, 404)
                    return
                jpg = (cal.frame_jpeg(k) if path.startswith('/frame/')
                       else cal.bird_jpeg(k))
                if jpg is None:
                    self._json({'error': 'no frame %d' % k}, 404)
                else:
                    self._send(200, jpg, 'image/jpeg')
            else:
                self._json({'error': 'not found'}, 404)

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            try:
                body = json.loads(self.rfile.read(n) or b'{}')
            except ValueError:
                self._json({'error': 'bad json'}, 400)
                return
            if self.path == '/accept':
                self._json(cal.accept(int(body['k']),
                                      body['u'], body['v']))
            elif self.path == '/remove':
                self._json(cal.remove(int(body['k'])))
            elif self.path == '/solve':
                self._json(cal.solve())
            else:
                self._json({'error': 'not found'}, 404)

    return H


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('session', help='directory written by live.py --record')
    ap.add_argument('-i', '--intrinsics', default=None,
                    help='calib.json from calib.py; omit to use the nominal '
                         'measured-FOV K (output is then marked provisional)')
    ap.add_argument('--hfov', type=float, default=MEASURED_HFOV_DEG,
                    help='HFOV in deg for the nominal K (default: measured %.1f)'
                         % MEASURED_HFOV_DEG)
    ap.add_argument('--min-frames', type=int, default=25)
    ap.add_argument('--port', type=int, default=8081)
    ap.add_argument('--sigma-az', type=float, default=rx.RADAR_SIGMA_AZ_DEG)
    ap.add_argument('--sigma-el', type=float, default=rx.RADAR_SIGMA_EL_DEG)
    ap.add_argument('--rig-id', default=None)
    ap.add_argument('--mount-token', default=None,
                    help='a string that changes whenever anything is unbolted')
    ap.add_argument('--allow-tainted', action='store_true',
                    help='admit holds whose picture failed the clean/visible '
                         'checks; the taint is recorded per correspondence')
    ap.add_argument('--allow-moving', action='store_true',
                    help='let non-stationary holds into the fit despite their '
                         'Doppler-folded azimuth')
    ap.add_argument('--ghost-tol', type=float, default=GHOST_TOL_DEG,
                    help='|az_radar - az_pixel| beyond this flags a pick as a '
                         'glass ghost (default %.0f; 0 disables). Valid only '
                         'while the sensors are co-aligned as measured'
                         % GHOST_TOL_DEG)
    ap.add_argument('--allow-ghost-picks', action='store_true',
                    help='let picks whose radar azimuth disagrees with their '
                         'pixel azimuth by more than %.0f deg into the fit; '
                         'they are glass-specular ghosts on this rig'
                         % GHOST_TOL_DEG)
    ap.add_argument('--unmirror', action='store_true',
                    help='flip the video horizontally on load (only if a '
                         'future sensor config mirrors the readout; measured '
                         '2026-08-09: the current stream is NOT mirrored)')
    a = ap.parse_args()

    cal = CalibSession(a.session, min_frames=a.min_frames,
                       intrinsics=a.intrinsics, hfov_deg=a.hfov,
                       sigma_az=a.sigma_az, sigma_el=a.sigma_el,
                       rig_id=a.rig_id, mount_token=a.mount_token,
                       allow_tainted=a.allow_tainted,
                       allow_moving=a.allow_moving,
                       unmirror=a.unmirror,
                       allow_ghosts=a.allow_ghost_picks,
                       ghost_tol_deg=a.ghost_tol)
    print('%d holds (%d scenery, %d outside-camera, %d tainted excluded), '
          'intrinsics: %s'
          % (len(cal.holds), cal.n_scenery, cal.n_outside, cal.n_tainted,
             cal.k_source),
          file=sys.stderr)
    if cal.picks:
        print('resumed %d picks from %s' % (len(cal.picks), cal.corr_path),
              file=sys.stderr)

    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(('0.0.0.0', a.port), make_handler(cal))
    print('open http://localhost:%d  (ctrl-c to stop)' % a.port,
          file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
