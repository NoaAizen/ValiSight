"""Budget declarations (pure). The YAML file is loaded in bench/; this module
only turns an already-parsed dict into typed ``Budget`` records.

A budget with ``p99_ms is None`` is *declared but unmeasured* — the harness is
expected to fill it from a first clean measurement. A stage that gets measured
with no ``Budget`` at all is a finding (see ``StageRegistry.report``), never a
silent default.
"""
from collections import namedtuple

Budget = namedtuple("Budget", ["stage", "p99_ms", "target_hz", "meta"])


def budgets_from_dict(data):
    """Build ``{stage_name: Budget}`` from a parsed budgets document.

    Expected shape::

        {"stages": {"osm_extrude.refine": {"p99_ms": 50, "target_hz": None,
                                            ...extra keys become meta...}}}
    """
    stages = (data or {}).get("stages") or {}
    out = {}
    for name, spec in stages.items():
        spec = spec or {}
        p99 = spec.get("p99_ms", None)
        hz = spec.get("target_hz", None)
        meta = {k: v for k, v in spec.items()
                if k not in ("p99_ms", "target_hz")}
        out[name] = Budget(stage=name, p99_ms=p99, target_hz=hz, meta=meta)
    return out
