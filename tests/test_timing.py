"""Sanity tests for the timing harness itself.

Two things the spec demands:
  * inject a stage with KNOWN latency and prove the reported percentiles are
    right (not an average, and the over-budget count is exact);
  * prove the harness adds < 50 us per reading.

Plus: the mandatory-context gate refuses a context-free run, and an undeclared
budget shows up as a finding rather than a silent default.
"""
import gc
import time

from core.timing import (
    StageRegistry,
    summarize,
    Budget,
    MeasurementContext,
    validate_context,
)
from core.timing.ring import RingBuffer

_MS = 1_000_000  # ns per ms


class FakeClock:
    """Deterministic clock: yields start/end pairs for each injected duration."""

    def __init__(self, durations_ns):
        self._q = []
        t = 0
        for d in durations_ns:
            self._q.append(t)       # __enter__
            t += d
            self._q.append(t)       # __exit__
        self._i = 0

    def __call__(self):
        v = self._q[self._i]
        self._i += 1
        return v


# --- percentiles are exact, and never the mean -------------------------------

def test_summarize_known_distribution():
    # 1..100 ms; mean would be 50.5 ms but we assert the *percentiles*.
    samples = [i * _MS for i in range(1, 101)]
    stats = summarize(samples, budget_ms=90.0)
    assert stats.p50_ms == 50.0
    assert stats.p95_ms == 95.0
    assert stats.p99_ms == 99.0
    assert stats.p999_ms == 100.0
    assert stats.max_ms == 100.0
    assert stats.over_budget == 10          # 91..100 ms are over a 90 ms budget


def test_summarize_empty_is_none_not_zero():
    # No samples -> nothing honest to report. A fabricated 0 would be worse.
    assert summarize([]) is None


# --- inject a stage with known latency, read it back through the registry -----

def test_stage_timer_reports_injected_latency():
    durations = [i * _MS for i in range(1, 101)]     # 1..100 ms
    reg = StageRegistry(capacity=1000, clock=FakeClock(durations),
                        budgets={"inject": Budget("inject", 90.0, None, {})})
    timer = reg.timer("inject")
    for _ in durations:
        with timer:
            pass
    row = reg.report()[0]
    assert row.stage == "inject"
    assert row.stats.p50_ms == 50.0
    assert row.stats.p99_ms == 99.0
    assert row.stats.over_budget == 10
    assert row.status == "OVER_BUDGET"


def test_undeclared_budget_is_a_finding():
    reg = StageRegistry(capacity=16, clock=FakeClock([5 * _MS]))  # no budgets
    with reg.timer("orphan"):
        pass
    row = reg.report()[0]
    assert row.status == "NO_BUDGET"        # not a silent default


def test_declared_null_budget_reports_unset():
    reg = StageRegistry(capacity=16, clock=FakeClock([5 * _MS]),
                        budgets={"s": Budget("s", None, None, {})})
    with reg.timer("s"):
        pass
    assert reg.report()[0].status == "BUDGET_UNSET"


# --- the harness must add < 50 us per reading --------------------------------

def test_stage_timer_overhead_under_50us():
    reg = StageRegistry(capacity=200_000, clock=time.perf_counter_ns)
    timer = reg.timer("overhead")
    for _ in range(2000):                   # warm up the buffer/name
        with timer:
            pass
    n = 50_000
    gc.disable()
    try:
        start = time.perf_counter_ns()
        for _ in range(n):
            with timer:
                pass
        elapsed = time.perf_counter_ns() - start
    finally:
        gc.enable()
    per_call_us = (elapsed / n) / 1000.0
    assert per_call_us < 50.0, "stage_timer cost %.3f us/call" % per_call_us


# --- ring buffer wraparound is accounted for, not silent ---------------------

def test_ring_buffer_wraparound_reports_overwrites():
    rb = RingBuffer(4)
    for i in range(10):
        rb.push(i)
    assert len(rb) == 4
    assert rb.total_count == 10
    assert rb.overwritten == 6


# --- mandatory context gate --------------------------------------------------

def _full_context(**overrides):
    base = dict(nvpmodel_mode="MODE_15W (id 2)", jetson_clocks=True,
                soc_temp_start_c=41.0, soc_temp_end_c=58.0,
                thermal_state="saturated", saturation_minutes=6.0,
                concurrent_load="detector on lepton + rosbag record")
    base.update(overrides)
    return MeasurementContext(**base)


def test_full_context_is_valid():
    ok, missing = validate_context(_full_context())
    assert ok and missing == []


def test_context_missing_nvpmodel_is_invalid():
    ok, missing = validate_context(_full_context(nvpmodel_mode=None))
    assert not ok
    assert any("nvpmodel" in m for m in missing)


def test_saturated_run_needs_five_minutes():
    ok, missing = validate_context(_full_context(saturation_minutes=3.0))
    assert not ok
    assert any("saturation_minutes" in m for m in missing)


def test_cold_run_does_not_need_saturation():
    ok, missing = validate_context(
        _full_context(thermal_state="cold", saturation_minutes=None))
    assert ok, missing
