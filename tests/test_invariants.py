"""Every physics invariant must CATCH its documented bug.

For each invariant there is one positive case (correct physics -> ok) and one
negative case that reconstructs the *real* failure from CLAUDE.md (the wrong
physics -> not ok). An invariant that does not fail on the wrong input is
worthless, so the negative cases are the point of this file.

conftest.py puts the repo root on sys.path, so `core` imports cleanly.
"""
import math

from core.physics_invariants import (
    monostatic_falloff_exponent,
    ula_resolution_at_azimuth,
    max_unambiguous_velocity,
    doppler_alias_fold,
    static_assumption_valid,
    elevation_observability,
    geoid_is_live,
)


# --- trap #1: monostatic falloff is 1/R^4, not 1/R^2 -------------------------

def test_monostatic_falloff_accepts_fourth_power():
    ranges = [2.0, 4.0, 8.0, 16.0, 32.0]
    intensity = [1.0 / (r ** 4) for r in ranges]        # correct physics
    ok, why = monostatic_falloff_exponent(ranges, intensity)
    assert ok, why


def test_monostatic_falloff_catches_inverse_square_bug():
    ranges = [2.0, 4.0, 8.0, 16.0, 32.0]
    intensity = [1.0 / (r ** 2) for r in ranges]        # the trap #1 mistake
    ok, why = monostatic_falloff_exponent(ranges, intensity)
    assert not ok
    assert "1/R^4" in why or "monostatic" in why


# --- trap #2: ULA resolution broadens as 1/cos(theta) ------------------------

def test_ula_resolution_ok_near_broadside():
    ok, why = ula_resolution_at_azimuth(n_elements=8, spacing_lambda=0.5,
                                        theta_deg=5.0)
    assert ok, why


def test_ula_resolution_catches_constant_assumption_at_fov_edge():
    # 60 deg -> 1/cos = 2.0x broadening: cannot be treated as constant.
    ok, why = ula_resolution_at_azimuth(n_elements=8, spacing_lambda=0.5,
                                        theta_deg=60.0)
    assert not ok
    assert "cos(theta)" in why


# --- trap #3: TDM-MIMO max unambiguous velocity + alias fold -----------------

# Chirp time chosen so a 3TX chirp gives v_max ~= 0.67 m/s (the documented value).
_LAMBDA = 299_792_458.0 / 77.0e9
_TC_FOR_067 = _LAMBDA / (4.0 * 3 * 0.67)                 # ~485 us


def test_v_max_accepts_target_below_limit():
    ok, why = max_unambiguous_velocity(n_tx=3, n_loops=16,
                                       chirp_time_s=_TC_FOR_067, v=0.5)
    assert ok, why


def test_v_max_catches_pedestrian_aliasing_to_static():
    # Pedestrian at 1.4 m/s exceeds +/-0.67 m/s -> flagged as aliasing.
    ok, why = max_unambiguous_velocity(n_tx=3, n_loops=16,
                                       chirp_time_s=_TC_FOR_067, v=1.4)
    assert not ok
    assert "alias" in why


def test_doppler_alias_fold_identity_inside_window():
    # positive: a velocity already inside the window is unchanged
    assert abs(doppler_alias_fold(0.3, 0.67) - 0.3) < 1e-9


def test_doppler_alias_fold_reproduces_pedestrian_fold():
    # the documented fold: 1.4 m/s -> ~0.06 m/s at v_max = 0.67
    assert abs(doppler_alias_fold(1.4, 0.67) - 0.06) < 1e-9


# --- trap #4: STATIC_V gate is only valid on a stationary platform -----------

def test_static_assumption_ok_when_platform_still():
    ok, why = static_assumption_valid(ego_speed=0.0)
    assert ok, why


def test_static_assumption_catches_moving_platform():
    ok, why = static_assumption_valid(ego_speed=1.4)     # platform driving
    assert not ok
    assert "moving" in why


# --- traps #5/#13: elevation observability -----------------------------------

def test_elevation_observable_with_height_prior():
    ok, why = elevation_observability(n_obs=4, sigma_el_deg=12.0,
                                      has_height_priors=True)
    assert ok, why


def test_elevation_catches_radar_only_local_minimum():
    # radar-only, few obs, sigma_el 12 deg at 30 m -> ~6 m single-obs std.
    ok, why = elevation_observability(n_obs=4, sigma_el_deg=12.0,
                                      has_height_priors=False)
    assert not ok
    assert "observable" in why


# --- trap #7: geoid must be live, not silently returning input ---------------

def test_geoid_live_when_probe_reports_israel_undulation():
    ok, why = geoid_is_live(lambda lon, lat: 17.5)       # real EGM2008 value
    assert ok, why


def test_geoid_catches_silent_missing_grid():
    # pyproj with no grid returns the height unchanged -> ~0 separation.
    ok, why = geoid_is_live(lambda lon, lat: 0.0)
    assert not ok
    assert "trap #7" in why or "did not load" in why
