"""bench — the measurement harness for core.timing.

Adapter layer: this is where the platform gets touched (nvpmodel, jetson_clocks,
thermal zones, the budgets file). ``core.timing`` stays pure; ``bench`` is what
runs it on a real SoC and refuses to record a number that has no context.
"""
from .harness import BenchHarness, ContextIncomplete, load_budgets

__all__ = ["BenchHarness", "ContextIncomplete", "load_budgets"]
