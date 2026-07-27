"""core.timing — pure, hardware-free latency measurement primitives.

Standard library only. No hardware, no I/O, no network. The platform side
(reading nvpmodel / SoC temperature / loading the budgets file) lives in
``bench/``; this package only holds the primitives that must be testable with
no rig attached.

Design rules baked in here (see CLAUDE.md and timing_budgets.yaml):

* **Never the mean.** Every stage reports p50 / p95 / p99 / p99.9 / max and the
  count of samples over budget. The average hides exactly the tail that kills a
  real-time system.
* **No allocation inside the measurement.** ``stage_timer(name)`` returns a
  reusable context-manager object bound to a pre-allocated ring buffer. After
  the first use of a name, the hot path allocates nothing.
* **An undeclared budget is a finding, not a default.** ``report()`` marks any
  measured stage that has no declared budget, and any declared budget still
  sitting at ``null`` (awaiting a first measurement).
* **A measurement with no context is meaningless.** ``MeasurementContext`` /
  ``validate_context`` refuse a run that lacks nvpmodel / clocks / SoC temps /
  cold-vs-saturated / concurrent load.
"""
from .ring import RingBuffer
from .percentiles import StageStats, summarize
from .budgets import Budget, budgets_from_dict
from .context import MeasurementContext, validate_context
from .stage import (
    StageRegistry,
    StageReport,
    stage_timer,
    get_default_registry,
    reset_default_registry,
    report,
)

__all__ = [
    "RingBuffer",
    "StageStats",
    "summarize",
    "Budget",
    "budgets_from_dict",
    "MeasurementContext",
    "validate_context",
    "StageRegistry",
    "StageReport",
    "stage_timer",
    "get_default_registry",
    "reset_default_registry",
    "report",
]
