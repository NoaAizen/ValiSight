"""Static background map over (range, azimuth) bins.

The sensor is fixed, so a bin that contains a zero-Doppler return in most
frames of the session is furniture, and no classifier should be asked to
re-discover that the table is a table. This is deliberately NOT the radar's
clutterRemoval: that one runs per-frame in the spectrum and deletes a
standing person too. Persistence over minutes is what separates a couch
from a person holding still for thirty seconds.

This is the OFFLINE, two-pass version for labeling recorded sessions: pass
one measures per-bin occupancy over the whole session, pass two masks. A
runtime version needs a trailing window plus care with long stationary
persons (a 10-minute sleeper will fade into the map - that is a documented
limitation to solve at the decision layer, not here).

Bins follow the sensor's own geometry - range x azimuth, not cartesian -
because the point spread scales with range exactly that way.
"""
import numpy as np

RANGE_BIN_M = 0.25
AZ_BIN_DEG = 3.0
V_STATIC = 0.10          # |v| below this counts as zero-Doppler (folded)


class StaticMap:
    def __init__(self, occupancy_threshold=0.60, min_streak_frames=300,
                 range_bin=RANGE_BIN_M, az_bin=AZ_BIN_DEG, max_range=20.0):
        self.thr = occupancy_threshold
        self.min_streak = min_streak_frames    # ~30 s at 10 Hz
        self.range_bin = range_bin
        self.az_bin = az_bin
        self.n_r = int(np.ceil(max_range / range_bin))
        self.n_a = int(np.ceil(180.0 / az_bin))
        self.counts = np.zeros((self.n_r, self.n_a), np.int32)
        self._streak = np.zeros((self.n_r, self.n_a), np.int32)
        self._max_streak = np.zeros((self.n_r, self.n_a), np.int32)
        self.frames = 0
        self._static = None

    def _bins(self, pts):
        x, y = pts[:, 0], pts[:, 1]
        r = np.hypot(x, y)
        az = np.degrees(np.arctan2(y, x))          # 0 = boresight, +left
        ri = np.clip((r / self.range_bin).astype(int), 0, self.n_r - 1)
        ai = np.clip(((az + 90.0) / self.az_bin).astype(int), 0, self.n_a - 1)
        return ri, ai

    @staticmethod
    def _dilate(m):
        out = m.copy()
        for dr in (-1, 0, 1):
            for da in (-1, 0, 1):
                if dr or da:
                    out |= np.roll(np.roll(m, dr, 0), da, 1)
        return out

    def accumulate(self, pts):
        """Pass one: per-frame occupancy AND its longest continuous streak.

        Occupancy alone cannot tell furniture from a person's favourite
        pausing spot - both rack up frames. What separates them is
        continuity: the couch returns in every frame for the whole session,
        the pausing human in episodes. Occupancy is judged on the dilated
        grid (point jitter smears one reflector over neighbours), and the
        streak is tracked on the same dilated view for the same reason.
        """
        self.frames += 1
        occ = np.zeros((self.n_r, self.n_a), bool)
        if len(pts):
            still = np.abs(pts[:, 3]) < V_STATIC
            if still.any():
                ri, ai = self._bins(pts[still])
                occ[ri, ai] = True
        occ = self._dilate(occ)
        self.counts += occ
        self._streak = np.where(occ, self._streak + 1, 0)
        np.maximum(self._max_streak, self._streak, out=self._max_streak)

    def finalize(self):
        """Static = occupied most of the session AND continuously present
        for at least min_streak frames somewhere in it."""
        self._static = ((self.counts >= self.thr * max(self.frames, 1))
                        & (self._max_streak >= min(self.min_streak,
                                                   self.frames)))
        return int(self._static.sum())

    def is_clutter(self, pts):
        """Pass two: mask of points sitting in a static bin with ~zero v.
        A MOVING point in a static bin is always kept - that is a person
        walking past the couch, not the couch."""
        if not len(pts):
            return np.zeros(0, bool)
        ri, ai = self._bins(pts)
        still = np.abs(pts[:, 3]) < V_STATIC
        return self._static[ri, ai] & still
