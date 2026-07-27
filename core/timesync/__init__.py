"""core.timesync — pure clock discipline for multi-sensor fusion.

Deterministic replay is a precondition for every regression test (CLAUDE.md,
realtime-soc-auditor). The two failures this module exists to prevent:

  * **arrival time used as sample time.** Radar frames cross a UART with
    ms-scale, load-dependent latency; the moment the N6 finishes reading a frame
    is NOT when the scene was measured. ``sensor_time_from_frame`` reconstructs
    the sample time from ``frame_no * frame_period`` anchored to the first
    frame, so it does not inherit UART jitter.
  * **mixing time sources.** Every timestamp here is a monotonic tick from ONE
    injected clock. Wall-clock is never used for a delta.

Then ``SyncBuffer`` pairs two async streams (thermal ~8.7 Hz, radar faster) by
timestamp within a declared slop, with a bounded depth and an explicit
drop-oldest policy — and it *reports* drops instead of hiding them.

Pure: standard library only, no hardware. The clock is injected, so the whole
thing is testable with synthetic ticks.
"""
from .clock import MonoClock, sensor_time_from_frame
from .sync import Stamped, SyncBuffer, SyncStats

__all__ = [
    "MonoClock",
    "sensor_time_from_frame",
    "Stamped",
    "SyncBuffer",
    "SyncStats",
]
