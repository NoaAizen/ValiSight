# core/physics_invariants

Pure, hardware-free checks for the physics that has already fooled a human
reviewer at least once. Each entry in `CLAUDE.md` ("מלכודות פיזיקה מתועדות")
should map to an invariant here.

**This package is pure.** Standard library only. No hardware, no I/O, no
network. It must run in CI, on a laptop, and inside a unit test with no rig
attached. If a check needs the outside world (files, pyproj, a UART), inject it
as a callable — see `geoid_is_live`, which takes a `separation_probe` from the
adapter while keeping the physics check here.

## The contract

Every invariant is a function that returns an `InvariantResult`, a
`(ok: bool, reason: str)` tuple:

```python
ok, why = monostatic_falloff_exponent(ranges, intensity)
if not ok:
    print(why)   # human-readable: what broke and why
```

* `ok is True`  — the physics holds.
* `ok is False` — the physics is violated; `reason` explains it in terms a
  human reviewer can act on.

No bare `assert`. No silent bool. The `reason` string is the product — it is
what a reviewer or a failing test reads.

(`doppler_alias_fold` is the one exception: it is a numeric helper, not a
pass/fail invariant, so it returns a `float`.)

## Current invariants

| Function | CLAUDE.md trap |
|---|---|
| `monostatic_falloff_exponent` | #1 — falloff is 1/R⁴, not 1/R² |
| `ula_resolution_at_azimuth`   | #2 — resolution broadens as 1/cos θ |
| `max_unambiguous_velocity`    | #3 — TDM-MIMO v_max, alias flag |
| `doppler_alias_fold`          | #3 — the fold itself (helper) |
| `static_assumption_valid`     | #4 — STATIC_V gate needs a still platform |
| `elevation_observability`     | #5, #13 — σ_el ≈ 12° is the bottleneck |
| `geoid_is_live`               | #7 — EGM2008 grid must actually be loaded |

## How to add a new invariant

1. **Start from a documented bug.** An invariant only earns its place if it
   catches a real failure. Point it at a specific `CLAUDE.md` trap (add the
   trap first if it is new). If you cannot name the bug it catches, do not add
   it.

2. **Write the function** in the matching module (`intensity.py`, `angle.py`,
   `velocity.py`, `elevation.py`, `geoid.py`) or a new module if it is a new
   axis of physics.
   - Import `InvariantResult` from `core.physics_invariants`.
   - Return `InvariantResult(True, "...")` on pass and
     `InvariantResult(False, "...")` on fail.
   - Name every physical constant at module top in `UPPER_CASE` (the physics
     lint blocks magic numbers inside formulas — see `scripts/physics_lint.py`).
   - Validate inputs and fail closed: bad/insufficient input returns
     `ok=False` with a reason, never a crash and never a false `True`.
   - Cite the trap number in the failure `reason`.

3. **Export it** from `__init__.py` (add to the imports and to `__all__`).

4. **Add two tests** in `tests/test_invariants.py`:
   - one *positive* case with correct physics → `ok is True`;
   - one *negative* case that **reconstructs the real bug** from `CLAUDE.md`
     → `ok is False`. This is the important one. An invariant that does not
     fail on the wrong input is worthless.

5. **Run** `python -m pytest tests/test_invariants.py -q`. The `Stop` hook runs
   exactly this, so keep it fast.

## Testing philosophy (from CLAUDE.md)

Unit tests here are pass/fail. They are *not* characterization sweeps — a check
that emits an error curve or a budget belongs in a sweep, reported separately.
And a green test that calls an internal and skips the full pipeline is a broken
test. Keep the negative cases anchored to inputs the real sensor can actually
produce.
