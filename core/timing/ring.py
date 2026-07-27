"""Pre-allocated ring buffer for latency samples (nanoseconds, int64).

The whole point is that ``push`` never allocates: the backing store is a fixed
``array('q')`` sized once, and a write is an in-place indexed assignment. That
keeps the measurement itself off the allocator, which is a hard requirement on
a real-time hot path.
"""
from array import array


class RingBuffer:
    """Fixed-capacity ring of int64 nanosecond samples. push() is allocation-free."""

    __slots__ = ("_buf", "_cap", "_count")

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1, got %r" % (capacity,))
        self._cap = capacity
        # Zero-initialised, contiguous, allocated exactly once here.
        self._buf = array("q", bytes(8 * capacity))
        self._count = 0

    def push(self, value_ns):
        """Record one sample. No allocation on this path."""
        self._buf[self._count % self._cap] = value_ns
        self._count += 1

    def __len__(self):
        return self._cap if self._count >= self._cap else self._count

    @property
    def total_count(self):
        """Total pushes ever, including those overwritten by wraparound."""
        return self._count

    @property
    def overwritten(self):
        """How many samples wraparound has silently dropped (a reporting flag)."""
        return self._count - len(self)

    def samples(self):
        """Return the live samples as a list. Allocates — for reporting only,
        never call this inside the measured region."""
        n = len(self)
        return list(self._buf[:n])
