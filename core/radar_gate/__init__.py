"""core.radar_gate — pure, cheap radar point relevance gating.

Removes the IWR1843's most-likely-manufactured returns (noise-floor specks,
out-of-zone points, isolated sidelobe/multipath ghosts) before clustering or
streaming, and reports the reduction so the efficiency win is measurable. Pure
(stdlib math only), MicroPython-friendly, testable off-hardware.

Honest by construction: the ``ReductionReport`` makes visible how many points
were cut and why, and the gates are deliberately conservative because every
ghost dropped risks a real weak target (corpus: CFAR already drives missed%).
"""
from .gate import gate_points, ReductionReport, MIN_ABS_DB, NEIGHBOR_EPS_M

__all__ = ["gate_points", "ReductionReport", "MIN_ABS_DB", "NEIGHBOR_EPS_M"]
