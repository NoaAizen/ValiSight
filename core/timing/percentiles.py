"""Percentile summary for latency samples. Never the mean.

A real-time budget is a tail statement, so we report p50 / p95 / p99 / p99.9 /
max and the number of samples that blew the budget. Percentiles use the
nearest-rank method on the sorted samples (the standard for latency SLOs — no
interpolation that could invent a value between two measured samples).
"""
import math
from collections import namedtuple

StageStats = namedtuple(
    "StageStats",
    ["count", "p50_ms", "p95_ms", "p99_ms", "p999_ms", "max_ms",
     "over_budget", "budget_ms"])

_NS_PER_MS = 1_000_000.0


def _nearest_rank(sorted_samples, pct):
    n = len(sorted_samples)
    rank = math.ceil(pct / 100.0 * n)
    idx = min(max(rank - 1, 0), n - 1)
    return sorted_samples[idx]


def summarize(samples_ns, budget_ms=None):
    """Summarise nanosecond samples into a ``StageStats`` (values in ms).

    ``over_budget`` counts samples strictly above ``budget_ms`` when a budget is
    given (else ``None``). Returns ``None`` for an empty sample set — there is
    nothing honest to report, and a fabricated zero would be worse.
    """
    if not samples_ns:
        return None
    s = sorted(samples_ns)
    n = len(s)

    def ms(p):
        return _nearest_rank(s, p) / _NS_PER_MS

    if budget_ms is None:
        over = None
    else:
        budget_ns = budget_ms * _NS_PER_MS
        over = sum(1 for x in samples_ns if x > budget_ns)

    return StageStats(
        count=n,
        p50_ms=ms(50.0),
        p95_ms=ms(95.0),
        p99_ms=ms(99.0),
        p999_ms=ms(99.9),
        max_ms=s[-1] / _NS_PER_MS,
        over_budget=over,
        budget_ms=budget_ms,
    )
