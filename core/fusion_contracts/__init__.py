"""core.fusion_contracts — pure, hardware-free contracts for sensor fusion.

Three mechanisms that make the fusion math honest:

1. **Named frame registry** (``frames``): every reference frame is named, every
   transform declares src/dst, unregistered transforms raise instead of falling
   back to identity, the graph is checked for cycles and duplicate paths, and
   the geoid height edge is guarded by ``geoid_is_live``.

2. **Unit-tagged costs** (``costs``): association cost components carry a
   declared unit; summing mixed units without a declared normaliser raises.

3. **Uncertainty propagation** (``projection``): a radar point projected into
   the thermal plane returns an uncertainty ellipse, and a contract check
   rejects a projected vertical uncertainty smaller than a human (the silent
   single-pixel collapse).

Pure: standard library only, no hardware. It may import ``core`` siblings
(e.g. ``core.physics_invariants.geoid_is_live``) — that is core->core, allowed.
"""
from .frames import (
    FrameGraph,
    FrameError,
    UnknownFrame,
    UnregisteredTransform,
    AmbiguousTransformGraph,
    GeoidNotLive,
    make_geoid_transforms,
    register_geoid_edge,
    KNOWN_FRAMES,
    RADAR_SPHERICAL, RADAR_BODY, THERMAL_PIXEL, THERMAL_CAMERA,
    PLATFORM_BODY, ENU_LOCAL, WGS84_ELLIPSOIDAL, EGM2008_ORTHOMETRIC,
)
from .costs import (
    METERS, RADIANS, PIXELS, DIMENSIONLESS, UNITS,
    UnitError, Normalizer, CostTerm,
    cost_term, person_metal_cost_term, to_dimensionless, sum_costs,
)
from .projection import (
    Projection,
    project_with_covariance,
    elevation_uncertainty_is_honest,
    spherical_cov,
    SIGMA_EL_DEG, HUMAN_HEIGHT_M,
    THERMAL_W, THERMAL_H,
)

__all__ = [
    # frames
    "FrameGraph", "FrameError", "UnknownFrame", "UnregisteredTransform",
    "AmbiguousTransformGraph", "GeoidNotLive", "make_geoid_transforms",
    "register_geoid_edge", "KNOWN_FRAMES",
    "RADAR_SPHERICAL", "RADAR_BODY", "THERMAL_PIXEL", "THERMAL_CAMERA",
    "PLATFORM_BODY", "ENU_LOCAL", "WGS84_ELLIPSOIDAL", "EGM2008_ORTHOMETRIC",
    # costs
    "METERS", "RADIANS", "PIXELS", "DIMENSIONLESS", "UNITS",
    "UnitError", "Normalizer", "CostTerm",
    "cost_term", "person_metal_cost_term", "to_dimensionless", "sum_costs",
    # projection
    "Projection", "project_with_covariance", "elevation_uncertainty_is_honest",
    "spherical_cov", "SIGMA_EL_DEG", "HUMAN_HEIGHT_M", "THERMAL_W", "THERMAL_H",
]
