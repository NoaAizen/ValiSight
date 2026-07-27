# bench — real-time measurement harness

`core/timing/` is pure (stdlib only, no hardware). `bench/` is the adapter that
runs it on a real SoC: it reads the platform context, loads the budgets file,
and refuses to record a number that has no context.

## Usage sketch

```python
from bench.harness import BenchHarness
from bench.platform_context import ContextCollector
from core.timing import stage_timer   # default registry, wired by the harness

harness = BenchHarness()               # loads timing_budgets.yaml
collector = ContextCollector()         # snapshots SoC temp at start

# ... run the pipeline under sustained load (>= 5 min for a 'saturated' run) ...
for frame in stream:
    with stage_timer("radar.cluster"):
        clusters = cluster(frame.points)
    with stage_timer("fusion.associate"):
        tracks = associate(clusters, detections)

ctx = collector.finish(thermal_state="saturated",
                       saturation_minutes=6.0,
                       concurrent_load="detector on lepton + rosbag record")

result = harness.emit_report(ctx)      # raises ContextIncomplete if ctx is thin
print(result["table"])
```

## Rules the harness enforces

- **No mean, ever.** Reports are p50 / p95 / p99 / p99.9 / max + over-budget
  count (`core/timing/percentiles.py`).
- **No context, no record.** `emit_report` calls `validate_context`; a run
  missing nvpmodel / jetson_clocks / SoC temps / cold-vs-saturated /
  concurrent load raises `ContextIncomplete`. A `saturated` run needs >= 5 min
  of prior sustained load.
- **Undeclared budget is a finding.** A stage measured but absent from
  `timing_budgets.yaml` reports `NO_BUDGET`; a declared-but-`null` budget
  reports `BUDGET_UNSET`.
- **Filling budgets is explicit.** `fill_missing_budgets()` proposes a p99 from
  the first clean measurement and updates the in-memory budgets; it does not
  touch the file until you call `write_budgets()`.

## Off-Jetson

`bench.platform_context` returns `None` for fields it cannot read off a Jetson,
so `validate_context` will (correctly) refuse the run. That is the point: a
laptop number with no thermal/power context is not a real-time measurement.
