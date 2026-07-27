"""Elevation observability invariant.

CLAUDE.md trap #5: sigma_el ~= 12 deg on the IWR1843 — elevation is the
bottleneck for pixel-level accuracy.
CLAUDE.md trap #13: initialising a target from radar-only elevation drops the
optimiser into a local minimum, giving ~6 m height error at 30 m range.
CLAUDE.md trap #10: the 1.2 m cluster epsilon is the height budget that
per-obstacle residuals must beat.

With a per-observation angular sigma this large, elevation is only observable
if either (a) an external height prior pins it, or (b) enough independent
observations are averaged to pull the height standard error under the budget.
"""
import math

from . import InvariantResult

# Default geometry for the check (CLAUDE.md trap #13 quotes 30 m).
DEFAULT_RANGE_M = 30.0
# The height budget a per-obstacle residual must beat (cluster epsilon, trap #10).
DEFAULT_TARGET_M = 1.2


def elevation_observability(n_obs, sigma_el_deg, has_height_priors,
                            range_m=DEFAULT_RANGE_M, target_m=DEFAULT_TARGET_M):
    """Is target elevation observable given angular noise and observation count?

    A single observation at range ``range_m`` with angular sigma
    ``sigma_el_deg`` maps to a height standard deviation of
    ``range_m * tan(sigma_el_deg)``. Averaging ``n_obs`` independent
    observations scales that by ``1/sqrt(n_obs)``.

    Elevation is observable if a height prior is present, or if the averaged
    height std beats ``target_m``. Radar-only with few observations at this
    sigma is exactly the trap.

    Returns
    -------
    InvariantResult(ok, reason)
    """
    if n_obs < 1:
        return InvariantResult(
            False, "need at least 1 observation, got %r" % (n_obs,))
    if not 0.0 <= sigma_el_deg < 90.0:
        return InvariantResult(
            False, "sigma_el must be in [0, 90) deg, got %r" % (sigma_el_deg,))

    height_std_1 = range_m * math.tan(math.radians(sigma_el_deg))
    height_std_n = height_std_1 / math.sqrt(n_obs)

    if has_height_priors:
        return InvariantResult(
            True, "height prior present — elevation constrained despite "
                  "sigma_el=%.1f deg (single-obs height std %.2f m at %.0f m)"
                  % (sigma_el_deg, height_std_1, range_m))

    if height_std_n <= target_m:
        return InvariantResult(
            True, "radar-only: %d obs at sigma_el=%.1f deg give height std "
                  "%.2f m at %.0f m, within the %.2f m budget"
                  % (n_obs, sigma_el_deg, height_std_n, range_m, target_m))

    needed = math.ceil((height_std_1 / target_m) ** 2)
    return InvariantResult(
        False,
        "radar-only elevation is not observable: %d obs at sigma_el=%.1f deg "
        "give height std %.2f m at %.0f m (>%.2f m budget). Single-obs std is "
        "%.2f m — this is the ~6 m local-minimum trap. Need a height prior or "
        "~%d obs (CLAUDE.md traps #5, #13)."
        % (n_obs, sigma_el_deg, height_std_n, range_m, target_m,
           height_std_1, needed))
