"""Approximate-time pairing of two async sensor streams (pure).

``SyncBuffer`` matches a stream A (e.g. radar) and a stream B (e.g. thermal) by
monotonic sample time, within a declared ``slop``. It is a bounded queue with an
explicit drop-oldest policy, and it *counts* every drop — because two sensors at
different rates will leave some frames unpaired every second, and that mismatch
is the STEADY STATE, not an error to hide. Making the drops a reported number is
the honest design.

Rate reality (from the shipped configs/iwr1843_live.cfg, NOT the corpus): the
radar frameCfg period is 100 ms -> ~10 Hz, against the Lepton's ~8.7 Hz, i.e. a
ratio of ~1.15:1 — roughly one radar frame per thermal frame, occasionally two,
so ~13% of radar frames go unpaired in steady state. Honest slop floor at 10 Hz
is ~50-60 ms (half the radar period plus timestamp uncertainty); tighter than
that drops most pairs and cannot be fixed without interpolation.
"""
from collections import deque, namedtuple

Stamped = namedtuple("Stamped", ["value", "t"])

# matched pairs so far, drops per stream, and current queue occupancy.
SyncStats = namedtuple(
    "SyncStats",
    ["matched", "dropped_a", "dropped_b", "pending_a", "pending_b"])

DEFAULT_DEPTH = 8


class SyncBuffer:
    """Pair two monotonic streams within ``slop_s`` seconds.

    push_a/push_b return the list of (a, b) pairs newly matched by that push.
    Items too old to ever match are dropped and counted; a queue longer than
    ``depth`` drops its oldest and counts it. Nothing is silently discarded.
    """

    def __init__(self, slop_s, depth=DEFAULT_DEPTH):
        if slop_s < 0:
            raise ValueError("slop must be >= 0")
        self.slop = slop_s
        self.depth = depth
        self._a = deque()
        self._b = deque()
        self.matched = 0
        self.dropped_a = 0
        self.dropped_b = 0

    def push_a(self, stamped):
        self._a.append(stamped)
        return self._drain()

    def push_b(self, stamped):
        self._b.append(stamped)
        return self._drain()

    def _drain(self):
        pairs = []
        while self._a and self._b:
            a, b = self._a[0], self._b[0]
            dt = a.t - b.t
            if abs(dt) <= self.slop:
                pairs.append((a, b))
                self.matched += 1
                self._a.popleft()
                self._b.popleft()
            elif dt < 0:
                # a is older than b by more than slop -> b is monotonic so no
                # future b can match a; a can never be paired -> drop it.
                self._a.popleft()
                self.dropped_a += 1
            else:
                self._b.popleft()
                self.dropped_b += 1
        # bounded depth: drop oldest beyond the declared queue depth
        while len(self._a) > self.depth:
            self._a.popleft()
            self.dropped_a += 1
        while len(self._b) > self.depth:
            self._b.popleft()
            self.dropped_b += 1
        return pairs

    def stats(self):
        return SyncStats(self.matched, self.dropped_a, self.dropped_b,
                         len(self._a), len(self._b))
