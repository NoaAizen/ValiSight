"""Inter-frame tracking of radar clusters: alpha-beta + Hungarian.

Constant-velocity alpha-beta in the ground plane. The Hungarian step is
genuinely needed even indoors: two people crossing is the P3 scenario, and
greedy association swaps identities exactly there. Doppler does not enter
the association cost - it is folded and dead in the recorded data - and z
never does (sigma_el).

A track survives MAX_MISSES consecutive empty frames because a person at
5 m is 1-3 points and CFAR blinks; killing on the first miss is how one
walk becomes eight track fragments. The cost of patience is bounded by the
gate radius.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

GATE_M = 1.0             # association gate around the prediction
MAX_MISSES = 12          # ~1.2 s at 10 Hz
ALPHA, BETA = 0.5, 0.2
MIN_HITS_CONFIRMED = 5


MAX_CLUTTER_STREAK = 30      # updates fed only by static-marked clusters


@dataclass
class Track:
    tid: int
    t: float
    pos: np.ndarray                       # (2,) filtered x, y
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hits: int = 1
    misses: int = 0
    clutter_streak: int = 0
    last_real: int = 0                    # history index of last non-clutter update
    history: list = field(default_factory=list)   # (t, x, y, cluster)

    @property
    def confirmed(self):
        return self.hits >= MIN_HITS_CONFIRMED

    @property
    def absorbed(self):
        """Latched onto furniture: fed only by static-marked clusters for a
        long stretch while going nowhere. The birth veto stops furniture
        from STARTING tracks; this stops it from KEEPING one alive."""
        return self.clutter_streak > MAX_CLUTTER_STREAK

    def predict(self, t):
        dt = max(t - self.t, 0.0)
        return self.pos + self.vel * dt

    def update(self, t, cluster, clutter=False):
        dt = max(t - self.t, 1e-3)
        pred = self.pos + self.vel * dt
        z = cluster.centroid[:2]
        resid = z - pred
        self.pos = pred + ALPHA * resid
        self.vel = self.vel + (BETA / dt) * resid
        self.t = t
        self.hits += 1
        self.misses = 0
        self.history.append((t, float(self.pos[0]), float(self.pos[1]),
                             cluster))
        if clutter:
            self.clutter_streak += 1
        else:
            self.clutter_streak = 0
            self.last_real = len(self.history) - 1

    def truncate_to_real(self):
        """Cut the trailing clutter-fed segment off the history."""
        self.history = self.history[:self.last_real + 1]
        return len(self.history)

    def coast(self, t):
        self.misses += 1


class Tracker:
    def __init__(self, gate=GATE_M, max_misses=MAX_MISSES):
        self.gate = gate
        self.max_misses = max_misses
        self.tracks: List[Track] = []
        self.closed: List[Track] = []
        self._next = 1

    def step(self, t, clusters, clutter=None):
        """clutter[j] True = cluster j sits in the static map. Such a
        cluster may CONTINUE an existing track (the person who paused on a
        mapped spot - deleting its points was measured to be the dominant
        mid-envelope death cause on walk3/walk4) but never START one, and a
        track that lives off such clusters alone is closed as absorbed."""
        if clutter is None:
            clutter = [False] * len(clusters)
        if self.tracks and clusters:
            cost = np.full((len(self.tracks), len(clusters)), 1e6)
            for i, tr in enumerate(self.tracks):
                p = tr.predict(t)
                # the longer a track coasts, the less its prediction is
                # worth - grow the gate with the miss count (a walker at
                # 1.4 m/s opens ~0.14 m per missed radar frame)
                gate = self.gate + 0.20 * tr.misses
                for j, c in enumerate(clusters):
                    d = float(np.hypot(*(c.centroid[:2] - p)))
                    if d <= gate:
                        cost[i, j] = d
            ri, ci = linear_sum_assignment(cost)
            matched_t, matched_c = set(), set()
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e6:
                    self.tracks[i].update(t, clusters[j], clutter[j])
                    matched_t.add(i)
                    matched_c.add(j)
        else:
            matched_t, matched_c = set(), set()

        for i, tr in enumerate(self.tracks):
            if i not in matched_t:
                tr.coast(t)
        for j, c in enumerate(clusters):
            if j not in matched_c:
                if clutter[j]:
                    continue
                tr = Track(self._next, t, c.centroid[:2].copy())
                tr.history.append((t, float(c.centroid[0]),
                                   float(c.centroid[1]), c))
                self.tracks.append(tr)
                self._next += 1

        alive = []
        for tr in self.tracks:
            dead = tr.misses > self.max_misses
            if tr.absorbed:
                tr.truncate_to_real()
                dead = True
            if dead:
                if tr.confirmed and len(tr.history) >= MIN_HITS_CONFIRMED:
                    self.closed.append(tr)
            else:
                alive.append(tr)
        self.tracks = alive

    def finish(self):
        self.closed += [t for t in self.tracks if t.confirmed]
        self.tracks = []
        return self.closed
