"""Measurement harness: bind budgets + a validated context to a percentile report.

Flow:
  1. Load timing_budgets.yaml into typed budgets.
  2. Run the pipeline, timing stages via the (default) StageRegistry.
  3. Collect the mandatory context (bench.platform_context) and REFUSE to emit
     a report if the context is incomplete — a number without context is not
     recorded.
  4. Fill any still-null budget from its first clean measurement (proposal;
     write-back is explicit, never silent).
"""
import os

from core.timing import (
    budgets_from_dict,
    get_default_registry,
    reset_default_registry,
    validate_context,
)
from . import yaml_min

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BUDGETS_PATH = os.path.join(ROOT, "timing_budgets.yaml")


class ContextIncomplete(Exception):
    """Raised when a run's measurement context is missing mandatory fields."""


def load_budgets(path=DEFAULT_BUDGETS_PATH):
    with open(path, encoding="utf-8") as fh:
        data = yaml_min.load_yaml(fh.read())
    return budgets_from_dict(data)


class BenchHarness:
    def __init__(self, budgets_path=DEFAULT_BUDGETS_PATH, capacity=100_000):
        self.budgets_path = budgets_path
        self.budgets = load_budgets(budgets_path)
        # Point the default registry at these budgets so stage_timer() lines up.
        self.registry = reset_default_registry(capacity=capacity,
                                               budgets=self.budgets)

    def timer(self, name):
        return self.registry.timer(name)

    def emit_report(self, context):
        """Validate context, then return the report. Refuses without context."""
        ok, missing = validate_context(context)
        if not ok:
            raise ContextIncomplete(
                "refusing to record measurements — missing context: %s"
                % ", ".join(missing))
        rows = self.registry.report()
        return {
            "context": context,
            "rows": rows,
            "table": self.registry.format_report(),
        }

    def fill_missing_budgets(self):
        """Propose a p99 budget for every declared-null stage from its first
        measurement. Returns ``{stage: proposed_p99_ms}`` and updates the
        in-memory budgets. Does NOT write the file — call ``write_budgets``.
        """
        proposals = {}
        for row in self.registry.report():
            budget = row.budget
            if budget is None or budget.p99_ms is not None or row.stats is None:
                continue
            proposed = round(row.stats.p99_ms, 3)
            proposals[row.stage] = proposed
            self.budgets[row.stage] = budget._replace(p99_ms=proposed)
        self.registry.set_budgets(self.budgets)
        return proposals

    def write_budgets(self, path=None):
        """Re-emit the (possibly filled) budgets to disk. Explicit only."""
        path = path or self.budgets_path
        stages = {}
        for name, b in self.budgets.items():
            spec = {"p99_ms": b.p99_ms, "target_hz": b.target_hz}
            spec.update(b.meta)
            stages[name] = spec
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(yaml_min.dump_yaml({"stages": stages}))
