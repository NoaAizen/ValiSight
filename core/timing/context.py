"""Mandatory measurement context (pure).

A latency number without this context is meaningless and must not be recorded
(CLAUDE.md / the timing spec). This module holds the schema and the validation;
the actual collection off the SoC lives in ``bench/platform_context.py``.
"""
from collections import namedtuple

# Minimum sustained-load duration before a run counts as thermally saturated.
MIN_SATURATION_MINUTES = 5.0

MeasurementContext = namedtuple(
    "MeasurementContext",
    ["nvpmodel_mode",       # e.g. "MODE_15W", or the raw nvpmodel id
     "jetson_clocks",       # bool: was jetson_clocks active?
     "soc_temp_start_c",    # SoC temperature at run start (deg C)
     "soc_temp_end_c",      # SoC temperature at run end (deg C)
     "thermal_state",       # "cold" or "saturated"
     "saturation_minutes",  # sustained load before the run (minutes)
     "concurrent_load"])    # what else was running (free text, must be stated)


def validate_context(ctx):
    """Return ``(ok, missing)`` — ``missing`` lists what disqualifies the run.

    Every field is mandatory. ``thermal_state`` must be exactly ``"cold"`` or
    ``"saturated"``; a ``"saturated"`` run must show at least
    ``MIN_SATURATION_MINUTES`` of prior sustained load (5 minutes), otherwise it
    is not actually saturated and the claim is invalid.
    """
    missing = []
    if not ctx.nvpmodel_mode:
        missing.append("nvpmodel_mode")
    if ctx.jetson_clocks is None:
        missing.append("jetson_clocks")
    if ctx.soc_temp_start_c is None:
        missing.append("soc_temp_start_c")
    if ctx.soc_temp_end_c is None:
        missing.append("soc_temp_end_c")
    if ctx.thermal_state not in ("cold", "saturated"):
        missing.append("thermal_state (must be 'cold' or 'saturated')")
    if ctx.thermal_state == "saturated":
        if ctx.saturation_minutes is None \
                or ctx.saturation_minutes < MIN_SATURATION_MINUTES:
            missing.append("saturation_minutes >= %.0f (sustained load before "
                           "a 'saturated' run)" % MIN_SATURATION_MINUTES)
    if ctx.concurrent_load is None:
        missing.append("concurrent_load (state it, even if 'nothing')")
    return (not missing), missing
