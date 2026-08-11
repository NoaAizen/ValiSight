#!/usr/bin/env python3
"""Draw live IWR1843 detections on the visible frame.

Imported by live.py; runs nothing on its own. Two jobs:

  RadarReader   a thread that owns the radar DATA port and keeps the newest
                parsed frame available, plus optional recording of the radar
                stream (radar.bin, radar.jsonl) into a session directory
  Bootstrap     radar point -> pixel, with a hand-adjustable extrinsic

Session video/thermal recording lives in recorder.py - it is a mode of the
live viewer, not a radar feature.

WHAT THIS OVERLAY IS AND IS NOT. Until tools/calib/radar_extrinsics.py has been
solved against measured correspondences, the projection here is a GUESS: the
canonical axis permutation, a lever arm of zero, and a focal length derived from
an assumed field of view. It is for looking, for eyeballing a rough alignment,
and for recording a session you can calibrate from later. Nothing it draws is a
measurement, and a number must never be read off it.

Two specific traps this file cannot fix, only mark:

1. A WALKING PERSON IS DRAWN IN THE WRONG PLACE under radar_10hz.cfg. v_max is
   0.649 m/s, so 1.4 m/s aliases (k=1), and the demo's TDM-MIMO angle stage
   compensates using the *measured* Doppler index -- leaving 120 deg of residual
   phase across the azimuth aperture and roughly 9.6 deg of azimuth error, which
   is ~88 px here. Points whose folded velocity implies aliasing are drawn
   hollow and marked, because the alternative is a confident dot in the wrong
   place. This is a config problem, not a drawing problem.

2. ELEVATION IS NEARLY UNMEASURED. Three TX gives a two-element elevation
   aperture -- an interferometer with no resolution, sigma ~12 deg. The vertical
   position of a dot is therefore the weakest thing on screen, which is why each
   detection also draws a vertical whisker: the honest shape of a radar
   detection in an image is closer to a column than to a point.
"""
import json
import os
import threading
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.join(_HERE, '..', 'radar'))
import mmwave  # noqa: E402

DATA_PORT = '/dev/ttyACM2'
BAUD = 921600

# Distinct from the two colours live.py already uses -- (80,255,255) for thermal
# edges and (255,210,40) for the coverage outline. BGR is applied at encode
# time, so these are RGB.
COL_STATIC = (255, 90, 180)      # magenta: k = 0, azimuth trustworthy
COL_ALIASED = (255, 160, 60)     # amber: Doppler aliased, azimuth suspect
COL_WHISKER = (150, 60, 110)

STALE_S = 0.6                    # older than this and the scene has moved on


class RadarReader(threading.Thread):
    """Owns the radar DATA port. Newest parsed frame in .latest, or None.

    Deliberately does NOT try to pair radar frames to camera frames. Live, the
    newest frame is the right answer and staleness is the only thing worth
    enforcing. Pairing is an offline problem, solved against the recording with
    the radar's own frame counter -- see radar/CALIBRATION-PLAN.md, which
    measures the host arrival stamp as good to only ~+-25 ms while the frame
    counter models the same clock to under a millisecond.
    """

    def __init__(self, port=DATA_PORT, record_dir=None):
        super().__init__(daemon=True)
        self.port = port
        self.stop = threading.Event()
        self.latest = None           # (t_monotonic, parsed_frame)
        self.frames = 0
        self.dropped_bytes = 0
        self.resyncs = 0
        self.parse_errors = 0
        self.error = None
        self._record_dir = record_dir
        self._jsonl = None
        self._raw = None
        self._lock = threading.Lock()

    def _open_recording(self):
        if not self._record_dir:
            return
        os.makedirs(self._record_dir, exist_ok=True)
        self._raw = open(os.path.join(self._record_dir, 'radar.bin'), 'wb')
        self._jsonl = open(os.path.join(self._record_dir, 'radar.jsonl'), 'w')

    def run(self):
        import serial
        try:
            ser = serial.Serial(self.port, BAUD, timeout=0.2)
        except Exception as e:                      # noqa: BLE001
            self.error = 'radar port %s: %s' % (self.port, e)
            return
        self._open_recording()
        sync = mmwave.FrameSync()
        byte_pos = 0
        try:
            while not self.stop.is_set():
                # read(1) blocks until a byte exists, then in_waiting drains the
                # rest without a second wait. monotonic, not time(): a clock step
                # mid-session would corrupt every interval derived from this.
                data = ser.read(1) + ser.read(ser.in_waiting)
                t = time.monotonic()
                if not data:
                    continue
                if self._raw:
                    self._raw.write(data)
                for t_frame, frame in sync.feed(data, t):
                    off = byte_pos                  # offset of THIS frame's magic
                    try:
                        fr = mmwave.parse_frame(frame)
                    except ValueError:
                        self.parse_errors += 1
                        continue
                    finally:
                        byte_pos += len(frame)
                    self.frames += 1
                    with self._lock:
                        self.latest = (t_frame, fr)
                    if self._jsonl:
                        self._write(fr, t_frame, off)
                self.dropped_bytes = sync.dropped_bytes
                self.resyncs = sync.resync_count
        except Exception as e:                      # noqa: BLE001
            self.error = 'radar reader died: %s' % e
        finally:
            try:
                ser.close()
            except Exception:                       # noqa: BLE001
                pass
            self.close()

    def _write(self, fr, t_frame, byte_offset):
        """Everything that distinguishes a good run from a bad one goes in the
        file, not stdout. byte_offset ties this row to radar.bin, without which
        a replay of the raw capture cannot be matched to the parsed rows."""
        t = fr.get('temperature') or {}
        st = fr.get('stats') or {}
        self._jsonl.write(json.dumps({
            't_mono': t_frame,
            'frame': fr['frame_number'],
            'byte_offset': byte_offset,
            'points': [[p['x'], p['y'], p['z'], p['v'], p['snr'], p['noise']]
                       for p in fr['points']],
            'convention': 'x_fwd_y_left_z_up',
            'die_c': t.get('rx_c'),
            'temp_valid': t.get('valid'),
            'interframe_margin_us': st.get('interframe_margin_us'),
            'dropped_bytes': self.dropped_bytes,
            'resyncs': self.resyncs,
        }) + '\n')

    def get(self, now=None):
        """Newest frame if it is fresh, else None."""
        with self._lock:
            item = self.latest
        if item is None:
            return None
        t_frame, fr = item
        if (now or time.monotonic()) - t_frame > STALE_S:
            return None
        return fr

    def close(self):
        for f in (self._jsonl, self._raw):
            try:
                if f:
                    f.close()
            except Exception:                       # noqa: BLE001
                pass
        self._jsonl = self._raw = None


class Bootstrap:
    """Radar point -> pixel under a guessed extrinsic, adjustable at runtime.

    R_CANONICAL is the zero-misalignment permutation from the project radar
    frame (x fwd, y left, z up) to the OpenCV camera frame (x right, y down,
    z fwd), written out so the signs can be read rather than trusted:

        x_cam = -y_radar      left  -> right
        y_cam = -z_radar      up    -> down
        z_cam =  x_radar      fwd   -> fwd

    yaw/pitch/roll are nudges applied on top, in degrees, so a session can be
    walked into rough alignment by eye. They are a starting guess for
    radar_extrinsics.solve(), never a substitute for it.
    """

    R_CANONICAL = np.array([[0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [1.0, 0.0, 0.0]])

    def __init__(self, width, height, hfov_deg=70.0, calib_path=None):
        self.w, self.h = width, height
        self.yaw = self.pitch = self.roll = 0.0
        self.t = np.zeros(3)                         # metres, camera frame
        self.source = 'assumed %.1f deg HFOV' % hfov_deg
        f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        self.K = np.array([[f, 0, width / 2.0],
                           [0, f, height / 2.0],
                           [0, 0, 1.0]])
        self.dist = np.zeros(5)
        if calib_path and os.path.exists(calib_path):
            self._load(calib_path)

    def _load(self, path):
        with open(path) as fh:
            c = json.load(fh)
        if 'K' in c:
            self.K = np.array(c['K'], float)
            self.dist = np.array(c.get('dist', np.zeros(5)), float).ravel()
            self.source = os.path.basename(path)

    @property
    def f(self):
        return float(self.K[0, 0])

    def R(self):
        y, p, r = np.radians([self.yaw, self.pitch, self.roll])
        cy, sy, cp, sp, cr, sr = (np.cos(y), np.sin(y), np.cos(p),
                                  np.sin(p), np.cos(r), np.sin(r))
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        return Rz @ Rx @ Ry @ self.R_CANONICAL

    def project(self, points):
        """[{x,y,z,...}] -> [(u, v, in_frame)]. Never clamps: a detection off to
        the left reads u = -140, because a clamped point is a lie that sits on
        the frame edge looking like a real one."""
        if not points:
            return []
        P = np.array([[p['x'], p['y'], p['z']] for p in points], float)
        C = P @ self.R().T + self.t
        out = []
        for (xc, yc, zc) in C:
            if zc <= 1e-6:                          # behind the camera
                out.append((None, None, False))
                continue
            u = self.K[0, 0] * xc / zc + self.K[0, 2]
            v = self.K[1, 1] * yc / zc + self.K[1, 2]
            out.append((u, v, 0 <= u < self.w and 0 <= v < self.h))
        return out


def is_aliased(p, vmax=None):
    """Is this detection's velocity consistent with Doppler aliasing?

    Cannot be answered from one frame -- a reading near +-vmax is either a
    target at that speed or one folded from any of infinitely many others. What
    it CAN say is that a reading in the outer band of the unambiguous window is
    where an aliased target lands, so the azimuth is suspect. Deliberately
    conservative: it marks doubt, it does not claim knowledge.
    """
    if vmax is None:
        vmax = mmwave.CFG_10HZ['v_max_m_s']
    return abs(p['v']) > 0.6 * vmax


def attach_range(dets, points, proj):
    """Give each detection box the range of the radar return that lands in it.

    Association is horizontal-only: elevation sigma is ~12 deg, so a point's v
    coordinate says almost nothing about which box it belongs to, while u is
    good to a few degrees. Static returns (|v| < 0.05) are excluded outright -
    they are room clutter, and a box standing near a wall must not inherit the
    wall's range. Aliased returns keep their (correct) range but carry ~9.6 deg
    of azimuth error under radar_10hz.cfg, so they associate with a widened
    band and lose to any non-aliased candidate.

    Sets d['radar_m'] (metres) on each matched box; removes it on misses so a
    stale range cannot outlive the return that produced it.
    """
    uv = proj.project(points)
    for d in dets:
        d.pop("radar_m", None)
        best = None
        for p, (u, _v, _in) in zip(points, uv):
            if u is None or abs(p['v']) < 0.05:
                continue
            alias = is_aliased(p)
            pad = 0.3 * d["w"] + (100 if alias else 0)
            if not (d["x"] - pad <= u <= d["x"] + d["w"] + pad):
                continue
            snr = p['snr'] if p['snr'] is not None else 0.0
            key = (not alias, snr)
            if best is None or key > best[0]:
                best = (key, p)
        if best is not None:
            d["radar_m"] = round(mmwave.range_of(best[1]), 2)


def annotate(rgb, points, proj, show_whisker=True, label_nearest=True):
    """Draw detections. Returns (drawn, offscreen, aliased)."""
    uv = proj.project(points)
    drawn = off = aliased = 0
    nearest = None
    for p, (u, v, inside) in zip(points, uv):
        if u is None or not inside:
            off += 1
            continue
        alias = is_aliased(p)
        aliased += alias
        col = COL_ALIASED if alias else COL_STATIC
        x, y = int(round(u)), int(round(v))
        r = mmwave.range_of(p)
        # A zero-Doppler return is room clutter (glass, furniture), not a
        # subject - a person, even one holding still, breathes. Drawn tiny
        # and dim so it cannot masquerade as the person standing near its
        # azimuth, which is exactly the confusion it caused in the field.
        static = abs(p['v']) < 0.05
        # Radius carries SNR, which is the one per-point quality the radar
        # actually reports. Clamped so a 40 dB return does not become a blob.
        snr = p['snr'] if p['snr'] is not None else 15.0
        rad = int(max(3, min(11, 3 + (snr - 11.2) / 3.0)))
        if static:
            cv2.circle(rgb, (x, y), 2, (110, 60, 85), 1)
            drawn += 1
            continue
        if show_whisker:
            # The vertical extent the elevation uncertainty actually spans.
            # sigma_el ~ 12 deg; at range r that is r*tan(12deg) metres, and
            # f/r converts metres at range r to pixels.
            half = int(min(proj.h, proj.f * r * np.tan(np.radians(12.0)) / max(r, 0.3)))
            cv2.line(rgb, (x, max(0, y - half)), (x, min(proj.h - 1, y + half)),
                     COL_WHISKER, 1)
        if alias:
            cv2.circle(rgb, (x, y), rad, col, 1)     # hollow: do not trust me
        else:
            cv2.circle(rgb, (x, y), rad, col, -1)
        drawn += 1
        if nearest is None or r < nearest[0]:
            nearest = (r, x, y, p)
    if label_nearest and nearest is not None:
        r, x, y, p = nearest
        txt = '%.2f m  %+.2f m/s' % (r, p['v'])
        cv2.putText(rgb, txt, (min(x + 12, proj.w - 150), max(14, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_STATIC, 1, cv2.LINE_AA)

    # Coaching HUD. The radar's cone is ~+-60 deg and the camera's is ~+-35, so
    # a subject can be tracked perfectly and photographed by nothing -- measured
    # on captures/walk2, 9 of 12 otherwise-good holds fell in that gap and were
    # unusable. The operator cannot see that gap; standing in it looks exactly
    # like standing anywhere else. So the frame has to say it out loud, while
    # there is still time to take a step.
    if off:
        # Which way to move. Radar azimuth is positive LEFT, so a subject at
        # positive azimuth beyond the cone must move to the operator's right,
        # which is the LEFT of the picture as they face the camera.
        outside = [p for p, (u, _, ins) in zip(points, uv)
                   if u is not None and not ins]
        if outside:
            mean_y = sum(p['y'] for p in outside) / len(outside)
            arrow = '-->' if mean_y > 0 else '<--'
            msg = '%d OUTSIDE THE PICTURE  step %s' % (off, arrow)
            cv2.rectangle(rgb, (0, proj.h - 30), (proj.w, proj.h), (0, 0, 0), -1)
            cv2.putText(rgb, msg, (10, proj.h - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, COL_ALIASED, 2, cv2.LINE_AA)
    cv2.putText(rgb, 'radar %d in / %d out' % (drawn, off),
                (proj.w - 150, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                COL_STATIC if drawn else (150, 150, 160), 1, cv2.LINE_AA)
    return drawn, off, aliased
