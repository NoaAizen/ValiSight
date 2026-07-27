# core/fusion_contracts

Pure, hardware-free contracts that keep the fusion math honest. Standard library
only (may import `core` siblings, e.g. `core.physics_invariants.geoid_is_live`).

Three mechanisms:

## 1. Named frame registry (`frames.py`)

Every reference frame has an explicit name (`radar_spherical`, `radar_body`,
`thermal_pixel`, `thermal_camera`, `platform_body`, `enu_local`,
`wgs84_ellipsoidal`, `egm2008_orthometric`). `FrameGraph.transform(p, src, dst)`
applies the registered chain.

- An unregistered transform **raises** `UnregisteredTransform` — it never falls
  back to identity. (Same-frame `transform` is the only identity.)
- `validate()` rejects cycles and **two distinct paths** between the same pair
  of frames — that duplicate-path shape is exactly the two parallel fusion
  stacks the project is collapsing.
- The `wgs84_ellipsoidal <-> egm2008_orthometric` edge is registered via
  `register_geoid_edge(graph, separation_probe)` and goes through
  `geoid_is_live`; a dead grid raises `GeoidNotLive` (CLAUDE.md trap #7).

## 2. Unit-tagged costs (`costs.py`)

Association cost components are `CostTerm(name, value, unit, normalizer)` with
`unit ∈ {meters, radians, pixels, dimensionless}`.

- `sum_costs` sums homogeneous same-unit terms directly, but any mix of units
  (or any dimensionless term) forces every term through a declared `Normalizer`
  to dimensionless first. A term with no declared normalizer **raises**
  `UnitError` — the guard against silently adding meters to pixels.
- `PERSON_METAL_COST` is dimensionless and **must** declare what it is
  normalised against: `person_metal_cost_term(value, normalized_against=...)`.

## 3. Uncertainty propagation (`projection.py`)

`project_with_covariance(point, cov_in) -> Projection(pixel, cov_px,
vertical_sigma_m)` propagates a radar spherical covariance into the thermal
plane (Jacobian by central differences) and returns an **ellipse**, not a pixel.

- `elevation_uncertainty_is_honest(projection)` rejects a projected vertical
  uncertainty smaller than a human. With `sigma_el ≈ 12°` at 30 m that is
  ~6.4 m; a sub-human value means the elevation spread was collapsed to a point.

## Tests

`tests/test_fusion_contracts.py` — one positive + one bug-reproducing negative
per mechanism (missing transform → identity, meters + pixels, a 30 m projection
returning uncertainty smaller than a human).

## Adding a contract

Start from a real fusion bug, put the check here (raise on violation, or return
`(ok, reason)` for a soft contract), export it from `__init__.py`, and add a
positive + a bug-reproducing negative test. A contract that does not fail on the
real bug is worthless.
