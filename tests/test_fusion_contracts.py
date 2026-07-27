"""Each fusion contract must CATCH its documented bug.

Per mechanism: one positive case (correct behaviour) and one negative case that
reconstructs a real bug:
  * a missing transform that silently became identity,
  * summing meters with pixels,
  * a 30 m projection returning an uncertainty smaller than a human -> wrong.
"""
import math

import pytest

from core.fusion_contracts import (
    FrameGraph, UnregisteredTransform, AmbiguousTransformGraph, GeoidNotLive,
    register_geoid_edge,
    RADAR_SPHERICAL, RADAR_BODY, PLATFORM_BODY, THERMAL_PIXEL, ENU_LOCAL,
    WGS84_ELLIPSOIDAL, EGM2008_ORTHOMETRIC,
    METERS, PIXELS, Normalizer,
    cost_term, person_metal_cost_term, sum_costs, UnitError,
    spherical_cov, project_with_covariance, elevation_uncertainty_is_honest,
)


# --- mechanism 1: frame registry ---------------------------------------------

def _tag(name):
    return lambda p: (name, p)


def test_registered_transform_applies_and_chains():
    g = FrameGraph()
    g.add_transform(RADAR_SPHERICAL, RADAR_BODY, _tag("cart"))
    g.add_transform(RADAR_BODY, PLATFORM_BODY, _tag("plat"))
    g.validate()
    assert g.transform(("sph",), RADAR_SPHERICAL, RADAR_BODY) == \
        ("cart", ("sph",))
    assert g.transform(("sph",), RADAR_SPHERICAL, PLATFORM_BODY) == \
        ("plat", ("cart", ("sph",)))
    # same frame is the only legitimate identity
    assert g.transform(("x",), RADAR_BODY, RADAR_BODY) == ("x",)


def test_missing_transform_raises_not_identity():
    g = FrameGraph()
    g.add_transform(RADAR_SPHERICAL, RADAR_BODY, _tag("cart"))
    # no path THERMAL_PIXEL -> ENU_LOCAL: must raise, NOT return the point
    with pytest.raises(UnregisteredTransform):
        g.transform(("x",), THERMAL_PIXEL, ENU_LOCAL)


def test_two_paths_between_frames_is_rejected():
    g = FrameGraph()
    g.add_transform(RADAR_SPHERICAL, RADAR_BODY, _tag("a"))
    g.add_transform(RADAR_BODY, PLATFORM_BODY, _tag("b"))
    g.add_transform(RADAR_SPHERICAL, PLATFORM_BODY, _tag("c"))  # 2nd path
    with pytest.raises(AmbiguousTransformGraph):
        g.validate()


def test_geoid_edge_live_converts_height():
    g = FrameGraph()
    register_geoid_edge(g, lambda lon, lat: 17.5)   # live EGM2008 undulation
    g.validate()
    out = g.transform((32.1, 34.8, 100.0),
                      WGS84_ELLIPSOIDAL, EGM2008_ORTHOMETRIC)
    assert out == (32.1, 34.8, 100.0 - 17.5)


def test_geoid_edge_dead_grid_raises():
    g = FrameGraph()
    register_geoid_edge(g, lambda lon, lat: 0.0)    # grid missing -> ~0
    with pytest.raises(GeoidNotLive):
        g.transform((32.1, 34.8, 100.0),
                    WGS84_ELLIPSOIDAL, EGM2008_ORTHOMETRIC)


# --- mechanism 2: unit-tagged costs ------------------------------------------

def test_homogeneous_and_normalized_sums_work():
    # same unit sums directly
    assert sum_costs([cost_term("a", 1.0, METERS),
                      cost_term("b", 2.0, METERS)]) == 3.0
    # mixed units, each normalised to dimensionless, then summed
    m = cost_term("range_err", 3.0, METERS,
                  Normalizer(10.0, "10 m range gate"))
    px = cost_term("az_err", 20.0, PIXELS,
                   Normalizer(40.0, "40 px azimuth gate"))
    pm = person_metal_cost_term(5.0, normalized_against="the 10 m range gate")
    assert sum_costs([m, px, pm]) == pytest.approx(0.3 + 0.5 + 5.0)


def test_summing_meters_with_pixels_raises():
    with pytest.raises(UnitError):
        sum_costs([cost_term("range", 1.0, METERS),
                   cost_term("az", 2.0, PIXELS)])


def test_person_metal_cost_needs_declared_reference():
    with pytest.raises(UnitError):
        person_metal_cost_term(5.0, normalized_against="")


# --- mechanism 3: uncertainty propagation ------------------------------------

def test_projection_preserves_elevation_uncertainty():
    point = (0.0, 0.0, 30.0)                          # boresight, 30 m
    cov = spherical_cov(sigma_az_deg=1.0, sigma_el_deg=12.0, sigma_rng_m=0.1)
    proj = project_with_covariance(point, cov)
    ok, why = elevation_uncertainty_is_honest(proj)
    assert ok, why
    assert proj.vertical_sigma_m == pytest.approx(30.0 * math.radians(12.0),
                                                  rel=1e-6)
    assert proj.cov_px[1][1] > 0.0                    # a real ellipse, not a dot


def test_collapsed_projection_is_flagged():
    point = (0.0, 0.0, 30.0)
    cov = spherical_cov(sigma_az_deg=1.0, sigma_el_deg=0.0, sigma_rng_m=0.1)
    proj = project_with_covariance(point, cov)        # elevation collapsed
    ok, why = elevation_uncertainty_is_honest(proj)
    assert not ok
    assert "human height" in why
