"""physics_invariants — pure, hardware-free checks for the physics that has
already fooled a human reviewer once (see CLAUDE.md, "מלכודות פיזיקה מתועדות").

Contract
--------
Every invariant is a plain function that returns an ``InvariantResult`` —
a 2-tuple ``(ok: bool, reason: str)``:

* ``ok`` is ``True`` when the physics holds and ``False`` when it is violated.
* ``reason`` is a human-readable explanation. On failure it must say *what*
  broke and *why*, in terms a human reviewer can act on. Never a bare assert,
  never a silent bool.

``InvariantResult`` subclasses ``tuple``, so callers may unpack it as
``ok, why = invariant(...)`` or read ``.ok`` / ``.reason``.

This package is pure. It imports nothing but the standard library and never
touches hardware, files or the network. That is deliberate: the whole point is
to be runnable in CI, on a laptop, and inside a unit test with no rig attached.

The one exception is :func:`geoid_is_live`, which cannot load an EGM2008 grid
from inside a pure layer — so it takes a *probe* callable injected by the
adapter. The physics check (is the undulation actually non-zero?) still lives
here; only the I/O lives in the adapter.
"""
from collections import namedtuple

InvariantResult = namedtuple("InvariantResult", ["ok", "reason"])

from .intensity import monostatic_falloff_exponent
from .angle import ula_resolution_at_azimuth
from .velocity import (
    max_unambiguous_velocity,
    doppler_alias_fold,
    static_assumption_valid,
)
from .elevation import elevation_observability
from .geoid import geoid_is_live

__all__ = [
    "InvariantResult",
    "monostatic_falloff_exponent",
    "ula_resolution_at_azimuth",
    "max_unambiguous_velocity",
    "doppler_alias_fold",
    "static_assumption_valid",
    "elevation_observability",
    "geoid_is_live",
]
