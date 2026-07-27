"""Geoid liveness invariant.

CLAUDE.md trap #7: pyproj does NOT raise when the EGM2008 grid is missing — it
silently returns the input height unchanged. In Israel that fabricates a
constant 17-18 m offset that looks exactly like a calibration error. Every path
that touches geoid height must go through a live-guard.

This invariant lives in the pure layer, so it cannot load a grid itself. The
adapter injects a ``separation_probe(lon, lat) -> metres`` callable that asks
pyproj for the geoid-ellipsoid separation at a point. The *physics* check —
"is the undulation actually non-zero where it must be?" — stays here. If the
grid is missing, pyproj returns ~0 and this invariant catches it.
"""
from . import InvariantResult

# A reference point in Israel where the EGM2008 undulation is ~17-18 m.
ISRAEL_REF_LON = 34.8
ISRAEL_REF_LAT = 32.1
# If |separation| is below this at the reference point, the grid did not load:
# a live EGM2008 grid never returns a near-zero undulation in Israel.
MIN_SEPARATION_M = 5.0
# Plausible EGM2008 undulation band for Israel — a live grid should land here.
ISRAEL_SEP_RANGE_M = (15.0, 20.0)


def geoid_is_live(separation_probe,
                  ref_lon=ISRAEL_REF_LON, ref_lat=ISRAEL_REF_LAT,
                  min_separation_m=MIN_SEPARATION_M,
                  expected_range_m=ISRAEL_SEP_RANGE_M):
    """Confirm an EGM2008 grid is actually loaded, not silently absent.

    ``separation_probe(lon, lat)`` must return the geoid-ellipsoid separation
    (metres) at that point, as reported by the adapter's pyproj transform.
    A near-zero result at the Israel reference means the grid never loaded and
    pyproj passed the height straight through (trap #7).

    Returns
    -------
    InvariantResult(ok, reason)
    """
    if separation_probe is None:
        return InvariantResult(
            False, "no separation probe supplied — cannot prove the EGM2008 "
                   "grid is loaded, so geoid height must not be trusted")
    try:
        sep = separation_probe(ref_lon, ref_lat)
    except Exception as exc:  # a probe that raises is a clear 'not live'
        return InvariantResult(
            False, "separation probe raised at (%.3f, %.3f): %r — grid not "
                   "usable" % (ref_lon, ref_lat, exc))

    if sep is None:
        return InvariantResult(
            False, "separation probe returned None at (%.3f, %.3f)"
                   % (ref_lon, ref_lat))

    if abs(sep) < min_separation_m:
        return InvariantResult(
            False,
            "geoid separation %.3f m at Israel ref (%.3f, %.3f) is ~0: the "
            "EGM2008 grid did not load and pyproj returned the height "
            "unchanged. This mimics a 17-18 m calibration offset (CLAUDE.md "
            "trap #7). Route through _assert_geoid_live()."
            % (sep, ref_lon, ref_lat))

    lo, hi = expected_range_m
    if not lo <= abs(sep) <= hi:
        return InvariantResult(
            True,
            "geoid grid live (separation %.3f m at Israel ref) but outside the "
            "expected %.1f-%.1f m band — verify the reference/datum."
            % (sep, lo, hi))
    return InvariantResult(
        True, "geoid grid live: separation %.3f m at Israel ref (%.3f, %.3f)"
              % (sep, ref_lon, ref_lat))
