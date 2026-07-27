"""Collect the mandatory measurement context off the SoC.

This is adapter code: it touches the platform (nvpmodel, jetson_clocks, thermal
zones). Off a Jetson it degrades to ``None`` for the fields it cannot read —
and ``core.timing.validate_context`` will then correctly refuse the run rather
than record a context-free (meaningless) number.

The three fields that cannot be sensed automatically — cold vs saturated,
minutes of sustained load, and what else was running — must be stated by the
operator and are passed into ``finish()``.
"""
import glob
import subprocess

from core.timing import MeasurementContext


def read_soc_temp_c():
    """Hottest thermal zone in deg C, or None if unavailable (off-Jetson)."""
    temps = []
    for path in sorted(glob.glob(
            "/sys/devices/virtual/thermal/thermal_zone*/temp")):
        try:
            with open(path) as fh:
                milli = int(fh.read().strip())
            temps.append(milli / 1000.0)
        except (OSError, ValueError):
            continue
    return max(temps) if temps else None


def read_nvpmodel_mode():
    """Current nvpmodel power mode string, or None if nvpmodel is absent."""
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True,
                             text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or "") + (out.stderr or "")
    mode_name = None
    mode_id = None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for idx, ln in enumerate(lines):
        low = ln.lower()
        if "power mode" in low:
            # format is often: "NV Power Mode: MODE_15W\n2"
            if ":" in ln:
                mode_name = ln.split(":", 1)[1].strip() or mode_name
            if idx + 1 < len(lines) and lines[idx + 1].isdigit():
                mode_id = lines[idx + 1]
    if mode_name and mode_id:
        return "%s (id %s)" % (mode_name, mode_id)
    return mode_name or mode_id


def read_jetson_clocks():
    """True/False if jetson_clocks state is readable, else None."""
    try:
        out = subprocess.run(["jetson_clocks", "--show"], capture_output=True,
                             text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or "").lower()
    if not text:
        return None
    # Heuristic: when clocks are maxed, min == max on the CPU governor lines.
    # We cannot be fully certain, so report None if we cannot tell.
    if "maxperf" in text or "active" in text:
        return True
    return None


class ContextCollector:
    """Snapshots SoC temperature at start; builds the context at ``finish``."""

    def __init__(self):
        self.soc_temp_start_c = read_soc_temp_c()

    def finish(self, thermal_state, concurrent_load, saturation_minutes=None,
               nvpmodel_mode=None, jetson_clocks=None):
        """Build a ``MeasurementContext``.

        Operator-supplied fields (``thermal_state``, ``concurrent_load``,
        ``saturation_minutes``) are required for a valid run. ``nvpmodel_mode``
        / ``jetson_clocks`` are auto-read when not overridden.
        """
        return MeasurementContext(
            nvpmodel_mode=(nvpmodel_mode if nvpmodel_mode is not None
                           else read_nvpmodel_mode()),
            jetson_clocks=(jetson_clocks if jetson_clocks is not None
                           else read_jetson_clocks()),
            soc_temp_start_c=self.soc_temp_start_c,
            soc_temp_end_c=read_soc_temp_c(),
            thermal_state=thermal_state,
            saturation_minutes=saturation_minutes,
            concurrent_load=concurrent_load,
        )
