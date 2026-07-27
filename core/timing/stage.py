"""Stage timing registry and the ``stage_timer`` context manager.

``stage_timer(name)`` returns a *reusable* context-manager object bound to a
pre-allocated ring buffer. First use of a name allocates its buffer; every use
after that allocates nothing on the measured path — enter reads the clock,
exit writes one int into the ring. That is the whole hot path.
"""
import time
from collections import namedtuple

from .ring import RingBuffer
from .percentiles import summarize

DEFAULT_CAPACITY = 100_000

# Per-stage line in a report. ``status`` is one of:
#   "ok"            measured, within (or no) budget
#   "OVER_BUDGET"   measured, p99 exceeds the declared budget
#   "BUDGET_UNSET"  budget declared as null, awaiting a first measurement
#   "NO_BUDGET"     measured with no declared budget at all -> a finding
StageReport = namedtuple(
    "StageReport", ["stage", "stats", "budget", "status"])


class _Stage:
    """One stage: a name, its ring buffer, and a clock. Reused as a CM."""

    __slots__ = ("name", "_ring", "_clock", "_start")

    def __init__(self, name, ring, clock):
        self.name = name
        self._ring = ring
        self._clock = clock
        self._start = 0

    def __enter__(self):
        self._start = self._clock()
        return self

    def __exit__(self, exc_type, exc, tb):
        # No allocation: one clock read, one subtract, one in-place ring write.
        self._ring.push(self._clock() - self._start)
        return False

    def samples(self):
        return self._ring.samples()


class StageRegistry:
    """Owns the per-stage ring buffers and produces the percentile report."""

    def __init__(self, capacity=DEFAULT_CAPACITY,
                 clock=time.perf_counter_ns, budgets=None):
        self._capacity = capacity
        self._clock = clock
        self._stages = {}
        self._budgets = dict(budgets or {})

    def set_budgets(self, budgets):
        self._budgets = dict(budgets or {})

    def timer(self, name):
        """Return the reusable context manager for ``name`` (creates on first use)."""
        stage = self._stages.get(name)
        if stage is None:
            stage = _Stage(name, RingBuffer(self._capacity), self._clock)
            self._stages[name] = stage
        return stage

    def report(self):
        """Percentile table vs budgets, flagging over-budget / unset / undeclared."""
        rows = []
        for name in sorted(self._stages):
            budget = self._budgets.get(name)
            budget_ms = budget.p99_ms if budget is not None else None
            stats = summarize(self._stages[name].samples(), budget_ms=budget_ms)
            if budget is None:
                status = "NO_BUDGET"
            elif budget.p99_ms is None:
                status = "BUDGET_UNSET"
            elif stats is not None and stats.p99_ms > budget.p99_ms:
                status = "OVER_BUDGET"
            else:
                status = "ok"
            rows.append(StageReport(name, stats, budget, status))
        return rows

    def format_report(self):
        """Human-readable percentile table (values in ms)."""
        lines = ["%-24s %8s %8s %8s %8s %8s %8s  %s"
                 % ("stage", "p50", "p95", "p99", "p99.9", "max",
                    "over", "status")]
        for r in self.report():
            if r.stats is None:
                lines.append("%-24s %8s %8s %8s %8s %8s %8s  %s"
                             % (r.stage, "-", "-", "-", "-", "-", "-",
                                r.status))
                continue
            s = r.stats
            over = "-" if s.over_budget is None else str(s.over_budget)
            lines.append("%-24s %8.3f %8.3f %8.3f %8.3f %8.3f %8s  %s"
                         % (r.stage, s.p50_ms, s.p95_ms, s.p99_ms,
                            s.p999_ms, s.max_ms, over, r.status))
        return "\n".join(lines)


# --- module-level default registry ------------------------------------------

_default = StageRegistry()


def get_default_registry():
    return _default


def reset_default_registry(capacity=DEFAULT_CAPACITY,
                           clock=time.perf_counter_ns, budgets=None):
    """Replace the default registry (used by the harness and by tests)."""
    global _default
    _default = StageRegistry(capacity=capacity, clock=clock, budgets=budgets)
    return _default


def stage_timer(name):
    """Reusable context manager for ``name`` on the default registry."""
    return _default.timer(name)


def report():
    return _default.report()
