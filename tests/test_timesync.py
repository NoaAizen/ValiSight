"""Clock discipline + async pairing.

These lock in the two things the realtime-soc-auditor calls the top priority:
sample time reconstructed from the frame index (not noisy arrival time), and a
bounded, drop-counting pairing of two streams at different rates.
"""
from core.timesync import (
    MonoClock, sensor_time_from_frame, Stamped, SyncBuffer,
)


class FakeTicks:
    def __init__(self, seq):
        self.seq = list(seq)
        self.i = 0

    def __call__(self):
        v = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return v


# --- monotonic clock ---------------------------------------------------------

def test_monoclock_reports_seconds_since_origin():
    clk = MonoClock(FakeTicks([0, 1000, 3000]), unit=1e-6)  # us ticks
    assert abs(clk.now() - 1e-3) < 1e-12                    # (1000-0)*1e-6
    assert abs(clk.now() - 3e-3) < 1e-12                    # (3000-0)*1e-6


# --- sample time from frame index, robust to dropped frames ------------------

def test_sensor_time_from_frame_index():
    # 20 Hz frames, anchored at t=10.0 s on the monotonic clock
    assert sensor_time_from_frame(0, 0.05, 10.0) == 10.0
    assert sensor_time_from_frame(5, 0.05, 10.0) == 10.25
    # frame 20 arrives after gaps, but the index (not the arrival count) sets
    # the time -> stays correct across drops
    assert abs(sensor_time_from_frame(20, 0.05, 10.0) - 11.0) < 1e-12


# --- pairing within slop -----------------------------------------------------

def test_pairs_within_slop():
    sb = SyncBuffer(slop_s=0.03)
    assert sb.push_b(Stamped("t0", 0.00)) == []            # nothing to pair yet
    pairs = sb.push_a(Stamped("r0", 0.01))                 # |0.01-0.00| <= 0.03
    assert len(pairs) == 1
    (a, b) = pairs[0]
    assert a.value == "r0" and b.value == "t0"
    assert sb.matched == 1


def test_unpairable_old_item_is_dropped_and_counted():
    sb = SyncBuffer(slop_s=0.02)
    sb.push_a(Stamped("r", 0.0))
    pairs = sb.push_b(Stamped("t", 0.5))                   # 0.5 s apart >> slop
    assert pairs == []
    assert sb.dropped_a == 1                               # r can never match


def test_radar_faster_than_thermal_is_reported_not_hidden():
    # radar runs a touch faster than thermal (~10 vs ~8.7 Hz from the config),
    # so some radar frames never find a thermal partner: 3 radar, 1 thermal.
    sb = SyncBuffer(slop_s=0.03)
    sb.push_a(Stamped("r0", 0.00))
    sb.push_a(Stamped("r1", 0.05))
    sb.push_a(Stamped("r2", 0.10))
    pairs = sb.push_b(Stamped("t0", 0.06))
    assert len(pairs) == 1
    assert pairs[0][0].value == "r1"                       # nearest within slop
    s = sb.stats()
    assert s.matched == 1
    assert s.dropped_a == 1                                # r0 unmatchable -> dropped
    assert s.pending_a == 1                                # r2 still waiting


def test_bounded_depth_drops_oldest():
    sb = SyncBuffer(slop_s=0.001, depth=2)
    for i in range(4):                                     # no B ever arrives
        sb.push_a(Stamped("r%d" % i, i * 1.0))
    s = sb.stats()
    assert s.pending_a == 2                                # only depth kept
    assert s.dropped_a == 2                                # oldest two dropped
