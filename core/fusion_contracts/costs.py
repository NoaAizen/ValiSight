"""Unit-tagged association costs.

Every cost component in an association carries a declared unit: meters, radians,
pixels, or dimensionless. Summing components of different units without a
declared normalisation factor is a runtime error, not a warning — because a
"cost" that adds azimuth-degrees to meters to a dimensionless prior (exactly
what the legacy greedy matcher did) is meaningless.

A dimensionless cost — ``PERSON_METAL_COST`` is the canonical one — must
explicitly declare what it is normalised against; a bare dimensionless number
with no stated reference is rejected.
"""
from collections import namedtuple

METERS = "meters"
RADIANS = "radians"
PIXELS = "pixels"
DIMENSIONLESS = "dimensionless"
UNITS = frozenset({METERS, RADIANS, PIXELS, DIMENSIONLESS})


class UnitError(Exception):
    pass


# scale: the value (in the term's unit) that maps to 1.0 dimensionless.
# reference: human-readable statement of what that scale represents.
Normalizer = namedtuple("Normalizer", ["scale", "reference"])

CostTerm = namedtuple("CostTerm", ["name", "value", "unit", "normalizer"])


def cost_term(name, value, unit, normalizer=None):
    if unit not in UNITS:
        raise UnitError("unknown unit %r for cost %r (allowed: %s)"
                        % (unit, name, ", ".join(sorted(UNITS))))
    return CostTerm(name=name, value=value, unit=unit, normalizer=normalizer)


def person_metal_cost_term(value, normalized_against):
    """A dimensionless PERSON_METAL_COST term that MUST state its reference.

    ``normalized_against`` is the explicit declaration of what the number is
    normalised against; empty/None is rejected.
    """
    if not normalized_against:
        raise UnitError(
            "PERSON_METAL_COST is dimensionless and must declare what it is "
            "normalised against (normalized_against=...)")
    return CostTerm(name="PERSON_METAL_COST", value=value, unit=DIMENSIONLESS,
                    normalizer=Normalizer(scale=1.0,
                                          reference=normalized_against))


def to_dimensionless(term):
    """Reduce one term to a dimensionless number via its declared normalizer.

    Raises ``UnitError`` if the term has no declared normalizer, or (for a
    dimensionless term) no declared reference.
    """
    if term.unit == DIMENSIONLESS:
        if term.normalizer is None or not term.normalizer.reference:
            raise UnitError(
                "dimensionless cost %r must declare what it is normalised "
                "against" % (term.name,))
        return term.value
    if term.normalizer is None:
        raise UnitError(
            "cost %r is in %s with no declared normalizer; it cannot be summed "
            "with costs of another unit" % (term.name, term.unit))
    if not term.normalizer.reference:
        raise UnitError(
            "normalizer for cost %r must state the reference it normalises "
            "against" % (term.name,))
    if term.normalizer.scale == 0:
        raise UnitError("normalizer scale for %r is zero" % (term.name,))
    return term.value / term.normalizer.scale


def sum_costs(terms):
    """Sum association cost terms.

    * Homogeneous terms (all the same physical unit, none dimensionless) sum
      directly — meters + meters is fine.
    * Any mix of units, or the presence of a dimensionless term, forces every
      term through its declared normalizer to dimensionless first. A term
      without a declared normalizer raises ``UnitError`` — this is the guard
      against silently adding meters to pixels.
    """
    terms = list(terms)
    if not terms:
        return 0.0
    physical_units = {t.unit for t in terms if t.unit != DIMENSIONLESS}
    has_dimensionless = any(t.unit == DIMENSIONLESS for t in terms)
    homogeneous = len(physical_units) <= 1 and not has_dimensionless
    if homogeneous:
        return sum(t.value for t in terms)
    return sum(to_dimensionless(t) for t in terms)
