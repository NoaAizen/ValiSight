"""Doppler / velocity invariants.

CLAUDE.md trap #3: IWR1843, TDM-MIMO, 3TX, 16 loops => max unambiguous velocity
is only about +/-0.67 m/s. A pedestrian at 1.4 m/s aliases to ~0.06 m/s and is
then misread as static. The alias fold is a *known failure mode*, not an edge
case.

CLAUDE.md trap #4: the STATIC_V=0.25 Doppler gate is only valid while the radar
platform is stationary. On a moving platform the whole "static" assumption
collapses.
"""
from . import InvariantResult

# Speed of light (m/s). Pure constant, no hardware.
C = 299_792_458.0
# IWR1843 operates 76-81 GHz; the shipped configs start the chirp at ~77 GHz.
DEFAULT_CARRIER_HZ = 77.0e9
# Ego speed (m/s) below which the platform counts as stationary for the
# STATIC_V Doppler gate. Tight — anything above walking-jitter breaks trap #4.
EGO_STATIC_TOL = 0.05


def max_unambiguous_velocity(n_tx, n_loops, chirp_time_s,
                             carrier_hz=DEFAULT_CARRIER_HZ, v=None):
    """Max unambiguous velocity for a TDM-MIMO chirp, and an optional check.

    For TDM-MIMO the effective Doppler PRI is ``n_tx * chirp_time_s`` because
    the transmitters are time-multiplexed, so::

        v_max = lambda / (4 * n_tx * chirp_time_s)

    ``n_loops`` does not change ``v_max`` (it sets velocity *resolution*); it is
    reported so the caller can see the resolution budget too.

    If ``v`` is given, the check fails when ``|v| > v_max`` and the reason
    reports the aliased velocity that the hardware would actually report — the
    trap #3 fold.

    Returns
    -------
    InvariantResult(ok, reason). The reason always states v_max.
    """
    if n_tx < 1 or n_loops < 1 or chirp_time_s <= 0 or carrier_hz <= 0:
        return InvariantResult(
            False, "invalid chirp geometry: n_tx=%r n_loops=%r chirp_time_s=%r "
                   "carrier_hz=%r" % (n_tx, n_loops, chirp_time_s, carrier_hz))

    lam = C / carrier_hz
    v_max = lam / (4.0 * n_tx * chirp_time_s)
    v_res = lam / (2.0 * n_tx * n_loops * chirp_time_s)

    if v is None:
        return InvariantResult(
            True, "v_max = +/-%.3f m/s, v_res = %.3f m/s (n_tx=%d, "
                  "n_loops=%d, Tc=%.1f us, fc=%.1f GHz)"
                  % (v_max, v_res, n_tx, n_loops, chirp_time_s * 1e6,
                     carrier_hz / 1e9))

    if abs(v) <= v_max:
        return InvariantResult(
            True, "|v|=%.3f m/s is within v_max=+/-%.3f m/s — unambiguous"
                  % (abs(v), v_max))
    folded = doppler_alias_fold(v, v_max)
    return InvariantResult(
        False,
        "|v|=%.3f m/s exceeds v_max=+/-%.3f m/s and aliases to %.3f m/s. "
        "This is the trap #3 fold: a target above v_max is reported at the "
        "wrong (often near-zero) velocity and can be misclassified as static."
        % (abs(v), v_max, folded))


def doppler_alias_fold(v_true, v_max):
    """Return the velocity a +/-v_max Doppler window would actually report.

    Folds ``v_true`` into ``[-v_max, v_max)`` with period ``2 * v_max``.
    Example (trap #3): fold(1.4, 0.67) == 0.06 m/s.

    Returns a float (this is a helper, not a pass/fail invariant).
    """
    if v_max <= 0:
        raise ValueError("v_max must be positive, got %r" % (v_max,))
    span = 2.0 * v_max
    return ((v_true + v_max) % span) - v_max


def static_assumption_valid(ego_speed, tol=EGO_STATIC_TOL):
    """The STATIC_V Doppler gate is valid only on a stationary platform.

    Returns False as soon as the ego platform is moving: on a moving platform a
    world-static object has non-zero radial Doppler and the STATIC_V=0.25 gate
    is meaningless (CLAUDE.md trap #4).

    Returns
    -------
    InvariantResult(ok, reason)
    """
    if ego_speed is None:
        return InvariantResult(
            False, "ego speed is unknown; cannot trust a Doppler-based static "
                   "gate without knowing the platform is still")
    if abs(ego_speed) <= tol:
        return InvariantResult(
            True, "ego speed %.3f m/s <= %.3f m/s — platform effectively "
                  "stationary, STATIC_V gate applies" % (abs(ego_speed), tol))
    return InvariantResult(
        False,
        "ego speed %.3f m/s > %.3f m/s: the platform is moving, so world-static "
        "objects carry ego-induced Doppler and the STATIC_V=0.25 gate no longer "
        "means 'not moving' (CLAUDE.md trap #4). Compensate for ego-motion "
        "before gating on Doppler." % (abs(ego_speed), tol))
