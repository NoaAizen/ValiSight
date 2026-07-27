"""Intensity-vs-range invariant.

CLAUDE.md trap #1: a monostatic point target falls off as 1/R**4, not 1/R**2.
The 1/R**2 form is the one-way / bistatic-illumination intuition and it has
passed human review before. This invariant fits the measured falloff in
log-log space and refuses anything that is not a fourth-power law.
"""
import math

from . import InvariantResult

# Expected log-log slope of a monostatic point-target return.
MONOSTATIC_EXPONENT = -4.0
# How far the fitted slope may drift before we call it a violation. Wide enough
# to absorb measurement noise, tight enough that a 1/R**2 law (slope -2) fails.
EXPONENT_TOL = 0.15


def monostatic_falloff_exponent(range_m, intensity,
                                expected=MONOSTATIC_EXPONENT,
                                tol=EXPONENT_TOL):
    """Verify that ``intensity`` falls off as ``range_m ** expected``.

    Fits ``log(intensity) = slope * log(range_m) + b`` by ordinary least
    squares and checks ``|slope - expected| <= tol``.

    Parameters
    ----------
    range_m, intensity : sequences of matching length, strictly positive.
    expected : target log-log slope (default -4.0, the monostatic law).
    tol : allowed absolute deviation of the fitted slope.

    Returns
    -------
    InvariantResult(ok, reason)
    """
    r = list(range_m)
    i = list(intensity)
    if len(r) != len(i):
        return InvariantResult(
            False, "range_m and intensity differ in length "
                   "(%d vs %d)" % (len(r), len(i)))
    if len(r) < 2:
        return InvariantResult(
            False, "need at least 2 samples to fit a falloff slope, got %d"
                   % len(r))
    if any(v <= 0 for v in r) or any(v <= 0 for v in i):
        return InvariantResult(
            False, "range and intensity must be strictly positive for a "
                   "log-log fit (a zero/negative sample means the data is not "
                   "a clean point-target return)")

    xs = [math.log(v) for v in r]
    ys = [math.log(v) for v in i]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return InvariantResult(
            False, "all ranges are equal; cannot estimate a falloff exponent")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx

    if abs(slope - expected) <= tol:
        return InvariantResult(
            True, "falloff exponent %.3f within %.2f of monostatic %.1f"
                  % (slope, tol, expected))
    return InvariantResult(
        False,
        "falloff exponent %.3f is not the monostatic %.1f (tol %.2f). "
        "A slope near -2 means someone modelled 1/R^2 for a monostatic point "
        "target — it is 1/R^4 (CLAUDE.md trap #1)." % (slope, expected, tol))
