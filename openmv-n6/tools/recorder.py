#!/usr/bin/env python3
"""Record a live session to disk: session.mp4, frames.jsonl, thermal.bin.

Imported by live.py; runs nothing on its own. Recording used to live inside
radar_overlay.py because it was born as the radar-calibration recorder, but it
is a mode of the live viewer, not a radar feature: a session is worth keeping
with no radar attached, and the radar files (radar.bin, radar.jsonl) are owned
by RadarReader either way. What lands in a session directory:

  session.mp4     the composed view, copied BEFORE any overlay is drawn. The
                  calibration tools refuse frames without the 'clean' flag,
                  because a pixel clicked next to a burned-in radar marker is a
                  correspondence derived from the very projection being solved.
  frames.jsonl    one row per video frame. Without it the video is unusable
                  for calibration: mp4 carries a nominal frame rate, and this
                  stream is paced by a sensor that delivers at 8.772 fps with
                  1824 ms FFC gaps in it, so frame index times 1/fps is not
                  when anything happened. t_mono is the same monotonic clock
                  radar.jsonl carries.
  thermal.bin     the raw radiometric frames, byte-for-byte as the board sent
                  them, indexed by (thermal_off, thermal_len) in frames.jsonl.
                  The mp4 is for looking; this is the measurement. mp4v is
                  lossy and 8-bit, and a temperature must never be read back
                  off the video.
"""
import json
import os
import time

import cv2


class SessionRecorder:
    """Writes one session directory. Create it, call write() per frame, close().

    The mp4 writer is opened lazily on the first frame because it needs the
    real frame size, and refuses silently-broken output: if the codec is
    unavailable VideoWriter returns an object whose write() does nothing, so
    isOpened() is checked and reported rather than producing a 0-byte file at
    the end. close() matters for the same reason - mp4 is finalized on
    release(), so a recorder that is never closed is a recording that may not
    open.
    """

    def __init__(self, dirpath, fps=8.772, meta=None):
        os.makedirs(dirpath, exist_ok=True)
        self.dir = dirpath
        self.path = os.path.join(dirpath, 'session.mp4')
        self.fps = fps
        self.w = None
        self.frames = 0
        self.error = None
        self._index_path = os.path.join(dirpath, 'frames.jsonl')
        self._thermal_path = os.path.join(dirpath, 'thermal.bin')
        self._index = None
        self._thermal = None
        self._thermal_off = 0
        # Set by the operator between poses; stamped onto every row from then
        # on. A calibration session is a sequence of deliberate placements,
        # and 'which rows belong to pose V07' is otherwise reconstructed from
        # timestamps and memory after the fact.
        self.pose_id = None
        self._write_meta(meta or {})

    def _write_meta(self, meta):
        """Write meta.json FIRST, before a single frame exists.

        Provenance written at close() is provenance a crash destroys, and a
        calibration session that died halfway is still evidence - it just has
        to be readable as what it is. What goes in here is what cannot be
        recovered from the data afterwards: which radar config was on the
        sensor, which ports, when the clock started.
        """
        doc = {
            'created_wall': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'created_mono': time.monotonic(),
            'fps_nominal': self.fps,
            'thermal_frame_bytes': None,      # filled by the first write()
        }
        doc.update(meta)
        try:
            with open(os.path.join(self.dir, 'meta.json'), 'w') as f:
                json.dump(doc, f, indent=2)
        except OSError as e:                  # never fail a recording over it
            self.error = 'meta.json: %s' % e
        self._meta = doc

    def _update_meta(self, **kw):
        self._meta.update(kw)
        try:
            with open(os.path.join(self.dir, 'meta.json'), 'w') as f:
                json.dump(self._meta, f, indent=2)
        except OSError:
            pass

    def write(self, rgb, thermal=None, extra=None):
        if self.w is None:
            h, wd = rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.w = cv2.VideoWriter(self.path, fourcc, self.fps, (wd, h))
            if not self.w.isOpened():
                self.error = 'cannot open %s for writing' % self.path
                self.w = False
            else:
                self._index = open(self._index_path, 'w')
        if self.w is False:
            return
        self.w.write(rgb[:, :, ::-1])
        row = {'i': self.frames, 't_mono': time.monotonic()}
        if thermal is not None:
            if self._thermal is None:
                self._thermal = open(self._thermal_path, 'wb')
            self._thermal.write(thermal)
            row['thermal_off'] = self._thermal_off
            row['thermal_len'] = len(thermal)
            if self._meta.get('thermal_frame_bytes') is None:
                self._update_meta(thermal_frame_bytes=len(thermal))
            self._thermal_off += len(thermal)
        if extra:
            row.update(extra)
        if self.pose_id is not None:
            row['pose_id'] = self.pose_id
        if self._index:
            self._index.write(json.dumps(row) + '\n')
        self.frames += 1

    def set_pose(self, pose_id):
        """Name the placement the next frames belong to. None clears it."""
        self.pose_id = str(pose_id) if pose_id not in (None, '') else None
        poses = self._meta.setdefault('poses', [])
        poses.append({'pose_id': self.pose_id, 'first_frame': self.frames,
                      't_mono': time.monotonic()})
        self._update_meta(poses=poses)
        return self.pose_id

    def meta_poses(self):
        """The pose marks written so far, oldest first.

        A copy: callers are the HTTP thread reporting progress to the operator,
        and handing out the live list would let a poll iterate it while the
        capture thread appends to it.
        """
        return list(self._meta.get('poses', []))

    def close(self):
        self._update_meta(closed_wall=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                          frames=self.frames)
        if self.w not in (None, False):
            self.w.release()
        for f in (self._index, self._thermal):
            if f:
                f.close()
        self.w = self._index = self._thermal = None
