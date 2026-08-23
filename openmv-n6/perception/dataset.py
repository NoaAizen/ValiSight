"""Synced (rgb, thermal, radar) triplets out of a recorded live session.

Everything downstream of this module - autolabel, association, the decision
layer - consumes triplets from here and nowhere else, so the conventions are
settled once, in this file:

  rgb      640x400 uint8 grayscale, frame C orientation. The stream is NOT
           mirrored (measured 2026-08-09, see radar_calib_web.py; the
           "mirrored stream!" note in holds1/corr.json predates that
           measurement and is wrong).
  thermal  120x160 uint8 or little-endian uint16, raw as recorded. The dtype
           is taken from meta.json when present and otherwise inferred from
           thermal_len. Unknown sizes are rejected rather than misaligned.
  radar    points in the PROJECT frame (x fwd, y left, z up, metres),
           exactly as radar.jsonl stores them - never rescaled. v is FOLDED
           (alias period 1.298 m/s); do not trust its sign or magnitude.

Pairing: frames.jsonl carries radar_frame per video frame. When present we
join on it; when null (session start) we fall back to nearest t_mono. Either
way the triplet exposes age_ms = (video t_mono - radar t_mono) so a consumer
can refuse stale radar instead of silently fusing it. The camera/radar frame
ratio is not integer and not stable (walk1: 1074 vs 1351) - no one-to-one
assumption anywhere.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import cv2
import numpy as np

THERMAL_W, THERMAL_H = 160, 120
THERMAL_PIXELS = THERMAL_W * THERMAL_H
THERMAL_BYTES = THERMAL_PIXELS       # legacy uint8 frame size
THERMAL_DTYPES = {
    'uint8': np.dtype('u1'),
    'uint16_le': np.dtype('<u2'),
}
RGB_W, RGB_H = 640, 400

# Beyond this, radar and video are describing different moments: a person at
# 1.4 m/s moves 21 cm in 150 ms, about a body width at 5 m.
DEFAULT_MAX_AGE_MS = 150.0


@dataclass
class RadarObs:
    frame: int
    t_mono: float
    points: np.ndarray          # (N, 6) float32: x, y, z, v, snr_db, noise_db
    dropped_bytes: int
    resyncs: int

    @property
    def xyz(self) -> np.ndarray:
        return self.points[:, :3]


@dataclass
class Triplet:
    i: int                      # video frame index
    t_mono: float
    rgb: np.ndarray             # (400, 640) uint8
    thermal: Optional[np.ndarray]  # (120, 160), uint8 or little-endian uint16
    radar: Optional[RadarObs]   # None when nothing within max_age_ms
    age_ms: Optional[float]     # video t_mono - radar t_mono
    view: str
    clean: bool


@dataclass
class LiveSession:
    """A recorded session directory: session.mp4 + thermal.bin + *.jsonl."""
    path: str
    frames: List[dict] = field(default_factory=list)
    radar_records: List[dict] = field(default_factory=list)

    def __post_init__(self):
        meta_path = os.path.join(self.path, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}
        with open(os.path.join(self.path, 'frames.jsonl')) as f:
            self.frames = [json.loads(l) for l in f if l.strip()]
        radar_path = os.path.join(self.path, 'radar.jsonl')
        if os.path.getsize(radar_path) > 0:
            with open(radar_path) as f:
                self.radar_records = [json.loads(l) for l in f if l.strip()]
        self._by_frame = {r['frame']: r for r in self.radar_records}
        self._radar_t = np.array([r['t_mono'] for r in self.radar_records])
        tb = os.path.join(self.path, 'thermal.bin')
        # Keep the file as bytes. Each row owns its offset/length and decides
        # the dtype explicitly, so a future uint16 session cannot shift every
        # frame after the first by half a frame.
        self._thermal = (np.memmap(tb, dtype=np.uint8, mode='r')
                         if os.path.exists(tb) else None)

    def __len__(self):
        return len(self.frames)

    def thermal_dtype(self, frame_meta: dict):
        """Return the declared/inferred dtype for one thermal frame.

        Legacy sessions have no meta.json but do carry thermal_len, so their
        19,200-byte uint8 frames remain readable. A 38,400-byte frame is only
        interpreted as little-endian uint16. Anything else fails closed.
        """
        nbytes = frame_meta.get('thermal_len')
        if nbytes is None:
            nbytes = self.meta.get('thermal_frame_bytes', THERMAL_BYTES)
        try:
            nbytes = int(nbytes)
        except (TypeError, ValueError):
            raise ValueError(f'{self.path}: invalid thermal_len={nbytes!r}')

        declared = self.meta.get('thermal_dtype')
        if declared is not None:
            if declared not in THERMAL_DTYPES:
                raise ValueError(f'{self.path}: unsupported thermal_dtype '
                                 f'{declared!r}')
            dtype = THERMAL_DTYPES[declared]
            expected = THERMAL_PIXELS * dtype.itemsize
            if nbytes != expected:
                raise ValueError(f'{self.path}: thermal_len={nbytes}, but '
                                 f'thermal_dtype={declared} requires {expected}')
            return dtype

        if nbytes == THERMAL_PIXELS:
            return THERMAL_DTYPES['uint8']
        if nbytes == 2 * THERMAL_PIXELS:
            return THERMAL_DTYPES['uint16_le']
        raise ValueError(f'{self.path}: cannot infer thermal dtype from '
                         f'thermal_len={nbytes}; declare thermal_dtype in meta.json')

    def thermal_frame(self, meta: dict):
        """The raw thermal frame recorded with this frames.jsonl row, without
        touching the video. None when the session (or this row) has none."""
        off = meta.get('thermal_off')
        if off is None or self._thermal is None:
            return None
        dtype = self.thermal_dtype(meta)
        nbytes = THERMAL_PIXELS * dtype.itemsize
        end = int(off) + nbytes
        if int(off) < 0 or end > len(self._thermal):
            raise ValueError(f'{self.path}: thermal frame at offset {off} needs '
                             f'{nbytes} bytes, file has {len(self._thermal)}')
        raw = np.array(self._thermal[int(off):end], copy=True)
        return raw.view(dtype).reshape(THERMAL_H, THERMAL_W)

    def _radar_for(self, meta: dict, max_age_ms: float):
        rec = self._by_frame.get(meta.get('radar_frame'))
        if rec is None and len(self._radar_t):
            # session-start frames have radar_frame null; nearest-t fallback
            k = int(np.searchsorted(self._radar_t, meta['t_mono']))
            best, best_dt = None, None
            for j in (k - 1, k):
                if 0 <= j < len(self.radar_records):
                    dt = abs(meta['t_mono'] - self._radar_t[j])
                    if best_dt is None or dt < best_dt:
                        best, best_dt = self.radar_records[j], dt
            rec = best
        if rec is None:
            return None, None
        age_ms = (meta['t_mono'] - rec['t_mono']) * 1e3
        if abs(age_ms) > max_age_ms:
            return None, age_ms
        pts = np.asarray(rec['points'], dtype=np.float32).reshape(-1, 6)
        return RadarObs(rec['frame'], rec['t_mono'], pts,
                        rec.get('dropped_bytes', 0), rec.get('resyncs', 0)), age_ms

    def _open_video(self):
        """session.mp4, or the repaired raw ES when the recorder died before
        close() and the moov atom never got written (the 2026-08-12 sessions;
        holds1 has the same repair). The ES carries no frame count."""
        mp4 = os.path.join(self.path, 'session.mp4')
        cap = cv2.VideoCapture(mp4, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        cap.release()
        es = os.path.join(self.path, 'session_es.m4v')
        if os.path.exists(es):
            cap = cv2.VideoCapture(es, cv2.CAP_FFMPEG)
            if cap.isOpened():
                print(f'[dataset] {self.path}: session.mp4 unreadable '
                      f'(unfinalized), using repaired session_es.m4v')
                return cap
            cap.release()
        raise RuntimeError(f'{self.path}: no readable video '
                           f'(session.mp4 broken, no session_es.m4v)')

    def triplets(self, max_age_ms: float = DEFAULT_MAX_AGE_MS,
                 clean_only: bool = False) -> Iterator[Triplet]:
        """Sequential iteration - VideoCapture seeking is not trusted."""
        cap = self._open_video()
        try:
            n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n_video > 0 and n_video != len(self.frames):
                # Measured cause (recorder.py write()): the mp4 frame and its
                # jsonl line are written 1:1 in the same call, but the jsonl
                # goes through a buffered file - a crash loses the buffered
                # TAIL. The surplus video frames are at the END, so pairing
                # from the start is sound and the common prefix is correct.
                print(f'[dataset] {self.path}: {n_video} video frames vs '
                      f'{len(self.frames)} frames.jsonl lines - iterating '
                      f'the common prefix (surplus is at the end)')
            for meta in self.frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if clean_only and not meta.get('clean', True):
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # walk1-4 recorded the FUSED view with no thermal_off at all -
                # those sessions have no thermal layer and their video must not
                # feed a detector (the thermal overlay is baked into the pixels).
                off = meta.get('thermal_off')
                if off is None or self._thermal is None:
                    thermal = None
                else:
                    thermal = self.thermal_frame(meta)
                radar, age_ms = self._radar_for(meta, max_age_ms)
                yield Triplet(meta['i'], meta['t_mono'], rgb, thermal,
                              radar, age_ms, meta.get('view', ''),
                              meta.get('clean', True))
        finally:
            cap.release()
