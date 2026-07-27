"""Angular-resolution invariant.

CLAUDE.md trap #2: the angular resolution of a uniform linear array (ULA) is
NOT constant across the FOV. The effective aperture shrinks with the cosine of
the steering angle, so the beamwidth broadens as 1/cos(theta). Treating the
broadside resolution as valid off-boresight silently under-states the error at
the edges of the FOV.
"""
import math

from . import InvariantResult

# Rayleigh beamwidth constant for a uniformly weighted aperture (~0.886).
RAYLEIGH_K = 0.886
# Broadening factor above which "resolution is constant" is no longer a safe
# approximation. 1.1 == you are already 10% worse than broadside.
MAX_BROADENING = 1.1


def ula_resolution_at_azimuth(n_elements, spacing_lambda, theta_deg,
                              max_broadening=MAX_BROADENING):
    """Check the 1/cos(theta) broadening of a ULA at azimuth ``theta_deg``.

    The broadside beamwidth (radians) is ``RAYLEIGH_K / (n_elements *
    spacing_lambda)`` — wavelength cancels because ``spacing_lambda`` is in
    units of the wavelength. Off-boresight it broadens by ``1/cos(theta)``.

    ``ok`` is True while the broadening stays within ``max_broadening`` (i.e.
    a constant-resolution assumption is still defensible), and False once the
    1/cos(theta) growth is significant — the documented trap.

    Returns
    -------
    InvariantResult(ok, reason)
    """
    if n_elements < 2:
        return InvariantResult(
            False, "a ULA needs at least 2 elements, got %r" % (n_elements,))
    if spacing_lambda <= 0:
        return InvariantResult(
            False, "element spacing must be positive, got %r"
                   % (spacing_lambda,))
    if not -90.0 < theta_deg < 90.0:
        return InvariantResult(
            False, "azimuth %.1f deg is at or beyond +/-90 deg; the array has "
                   "no aperture there (1/cos(theta) -> infinity)" % theta_deg)

    theta = math.radians(theta_deg)
    res_broadside = RAYLEIGH_K / (n_elements * spacing_lambda)  # radians
    broadening = 1.0 / math.cos(theta)
    res_theta = res_broadside * broadening

    if broadening <= max_broadening:
        return InvariantResult(
            True, "resolution %.2f deg at broadside, %.2f deg at %.1f deg "
                  "(x%.2f) — within the %.2f constant-resolution budget"
                  % (math.degrees(res_broadside), math.degrees(res_theta),
                     theta_deg, broadening, max_broadening))
    return InvariantResult(
        False,
        "resolution broadens x%.2f at %.1f deg (%.2f deg -> %.2f deg). ULA "
        "resolution is not constant across the FOV; it grows as 1/cos(theta) "
        "(CLAUDE.md trap #2)."
        % (broadening, theta_deg, math.degrees(res_broadside),
           math.degrees(res_theta)))
