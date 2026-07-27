"""Monotonic timebase and sensor-time reconstruction (pure).

One clock, monotonic, injected. All fusion timestamps are seconds from an
arbitrary but fixed monotonic origin — never wall clock, never mixed sources.
"""


class MonoClock:
    """A monotonic clock wrapper. The tick source is injected so tests can drive
    it deterministically and no wall-clock ever leaks into a delta.

    ``ticks_fn`` returns a monotonically non-decreasing integer of ``unit``
    seconds (e.g. ``time.monotonic_ns`` with unit=1e-9, or the N6's
    ``time.ticks_us`` with unit=1e-6).
    """

    __slots__ = ("_ticks", "_unit", "_origin")

    def __init__(self, ticks_fn, unit=1e-9):
        self._ticks = ticks_fn
        self._unit = unit
        self._origin = ticks_fn()

    def now(self):
        """Seconds since this clock's origin (monotonic)."""
        return (self._ticks() - self._origin) * self._unit

    def reset(self):
        self._origin = self._ticks()


def sensor_time_from_frame(frame_no, frame_period_s, anchor_s, first_frame_no=0):
    """Reconstruct a frame's *sample* time from its index, not its arrival time.

    A sensor emitting frames at a fixed period is its own metronome: the true
    sample time of ``frame_no`` is ``anchor_s + (frame_no - first_frame_no) *
    frame_period_s``, where ``anchor_s`` is the monotonic time captured at the
    first frame. This is drift-free against UART/transport jitter — the arrival
    time of any single frame is noisy, but the frame *index* is exact.

    Failure modes to keep in mind (documented, not hidden):
      * dropped frames still advance ``frame_no``, so the reconstruction stays
        correct across gaps (unlike counting arrivals);
      * the sensor's own oscillator drifts against the host clock over long
        runs — re-anchor periodically for captures longer than tens of seconds.
    """
    return anchor_s + (frame_no - first_frame_no) * frame_period_s
