"""How a fix decays between map corrections, and what dominates the decay."""

from __future__ import annotations

import math

import pytest

from mapinit.nav.propagation import (
    MAX_HORIZON_S,
    MEASURED,
    RIG_ERROR_MODEL,
    STANDARD_GRAVITY,
    DeadReckoner,
    ImuErrorModel,
    NavState,
    NoiseTerm,
    PropagationError,
    SpeedAiding,
    advance,
    propagate,
)


def clean_term(name: str, value: float, unit: str = "") -> NoiseTerm:
    """A measured constant, for models built to isolate one error source."""
    return NoiseTerm(name, value, unit, MEASURED, "test fixture")


def model_with(**overrides) -> ImuErrorModel:
    """The rig model with every term zeroed except the ones named.

    Isolating one term is the only way to check a term's own arithmetic; with
    all five live, any of them could be producing the total.
    """
    fields = {
        "tilt_sigma_deg": clean_term("tilt_sigma", 0.0, "deg"),
        "accel_bias_mg": clean_term("accel_bias", 0.0, "mg"),
        "accel_sigma_mg": clean_term("accel_sigma", 0.0, "mg"),
        "gyro_bias_sigma_dps": clean_term("gyro_bias_sigma", 0.0, "deg/s"),
        "gyro_arw_dps_sqrt_s": clean_term("gyro_arw", 0.0, "deg/sqrt(s)"),
    }
    for name, value in overrides.items():
        unit = fields[name].unit
        fields[name] = clean_term(fields[name].name, value, unit)
    return ImuErrorModel(**fields)


# -- heading growth ---------------------------------------------------------


def test_heading_sigma_matches_the_rig_formula():
    """The growth law must reproduce what the rig reports, not improve on it."""
    model = model_with(gyro_bias_sigma_dps=0.05, gyro_arw_dps_sqrt_s=0.15)
    for moving_s in (1.0, 10.0, 60.0):
        expected = math.sqrt((0.05 * moving_s) ** 2 + 0.15 ** 2 * moving_s)
        assert model.heading_sigma_deg(moving_s) == pytest.approx(expected)


def test_heading_does_not_drift_while_static():
    """A yaw that is not integrated cannot drift, so zero moving time is zero."""
    assert RIG_ERROR_MODEL.heading_sigma_deg(0.0) == 0.0


def test_random_walk_dominates_early_and_bias_dominates_late():
    """The crossover is real and worth asserting: it decides which one to fix."""
    model = model_with(gyro_bias_sigma_dps=0.05, gyro_arw_dps_sqrt_s=0.15)
    walk_at = lambda t: model.gyro_arw_dps_sqrt_s.value ** 2 * t  # noqa: E731
    bias_at = lambda t: (model.gyro_bias_sigma_dps.value * t) ** 2  # noqa: E731
    assert walk_at(1.0) > bias_at(1.0)
    assert bias_at(60.0) > walk_at(60.0)


def test_negative_moving_time_is_refused():
    with pytest.raises(PropagationError):
        RIG_ERROR_MODEL.heading_sigma_deg(-1.0)


# -- the inertial regime ----------------------------------------------------


def test_tilt_leakage_is_half_g_theta_t_squared():
    """The term that decides the unaided horizon, checked against its closed form."""
    model = model_with(tilt_sigma_deg=0.2)
    budget = DeadReckoner(model).budget_at(10.0)
    term = next(t for t in budget.terms if t.name == "tilt_leakage")
    expected = 0.5 * STANDARD_GRAVITY * math.sin(math.radians(0.2)) * 100.0
    assert term.metres == pytest.approx(expected)


def test_inertial_error_is_quadratic_in_time():
    """Doubling the horizon quadruples the error: the signature of two integrations."""
    reckoner = DeadReckoner(model_with(tilt_sigma_deg=0.2))
    assert reckoner.budget_at(20.0).total_m == pytest.approx(
        4.0 * reckoner.budget_at(10.0).total_m
    )


def test_white_noise_is_negligible_against_bias():
    """Averages down at 200 Hz, so it is not what to go and fix."""
    budget = DeadReckoner().budget_at(60.0)
    noise = next(t for t in budget.terms if t.name == "accel_noise")
    assert noise.metres < 0.01 * budget.total_m


def test_the_rig_holds_about_ten_seconds_unaided():
    """The project's headline claim, now computed rather than asserted in prose."""
    horizon = DeadReckoner().horizon_for(1.0)
    assert 4.0 < horizon < 12.0


# -- the aided regime -------------------------------------------------------


def test_speed_aiding_changes_the_regime_not_just_the_constant():
    """Radar ego-velocity is worth an order of magnitude, which is the whole case."""
    inertial = DeadReckoner().budget_at(60.0).total_m
    aided = DeadReckoner(aiding=SpeedAiding(5.0, 0.05)).budget_at(60.0).total_m
    assert aided < inertial / 5.0


def test_cross_track_accumulates_rather_than_using_the_final_heading_error():
    """A heading error steers, so displacement builds up as the error grows.

    Multiplying the final sigma by the whole distance would overstate it: the
    heading was better than that for the entire run up to the end.
    """
    reckoner = DeadReckoner(aiding=SpeedAiding(5.0, 0.0))
    budget = reckoner.budget_at(60.0)
    cross = math.sqrt(sum(
        term.metres ** 2 for term in budget.terms if term.name.startswith("cross_track")
    ))
    naive = 5.0 * 60.0 * math.radians(reckoner.heading_sigma_deg(60.0))
    assert cross < naive


def test_a_gyro_bias_displaces_as_t_squared_and_random_walk_as_t_to_the_three_halves():
    """The two heading errors grow at different powers, so they want different fixes."""
    speed, horizon = 5.0, 30.0
    bias_only = model_with(gyro_bias_sigma_dps=0.05)
    walk_only = model_with(gyro_arw_dps_sqrt_s=0.15)

    bias = DeadReckoner(bias_only, SpeedAiding(speed, 0.0)).budget_at(horizon)
    walk = DeadReckoner(walk_only, SpeedAiding(speed, 0.0)).budget_at(horizon)

    assert bias.total_m == pytest.approx(
        speed * math.radians(0.05) * horizon ** 2 / 2.0
    )
    assert walk.total_m == pytest.approx(
        speed * math.radians(0.15) * horizon ** 1.5 / math.sqrt(3.0)
    )


def test_random_walk_cross_track_is_not_the_integral_of_its_sigma():
    """Integrating the running sigma treats samples as perfectly correlated.

    They are not: a random walk's increments are independent, and the honest
    coefficient is 1/sqrt(3), not the 2/3 an integral of sigma would give --
    a 15 percent overstatement that would make the budget disagree with the
    filter over the same rig.
    """
    model = model_with(gyro_arw_dps_sqrt_s=0.15)
    horizon = 30.0
    exact = 5.0 * math.radians(0.15) * horizon ** 1.5 / math.sqrt(3.0)
    integral_of_sigma = 5.0 * math.radians(0.15) * (2.0 / 3.0) * horizon ** 1.5
    budget = DeadReckoner(model, SpeedAiding(5.0, 0.0)).budget_at(horizon)
    assert budget.total_m == pytest.approx(exact)
    assert budget.total_m < integral_of_sigma


def test_a_stationary_aided_rig_does_not_drift():
    """Speed zero is displacement zero, whatever the heading is doing."""
    budget = DeadReckoner(aiding=SpeedAiding(0.0, 0.0)).budget_at(600.0)
    assert budget.total_m == pytest.approx(0.0)


def test_initial_heading_sigma_dominates_a_short_run():
    """A manual pin's heading error is there from t=0 and swamps early drift."""
    pinned = DeadReckoner(
        aiding=SpeedAiding(5.0, 0.0), initial_heading_sigma_deg=5.0
    )
    perfect = DeadReckoner(aiding=SpeedAiding(5.0, 0.0))
    assert pinned.budget_at(10.0).total_m > 10.0 * perfect.budget_at(10.0).total_m


# -- reporting --------------------------------------------------------------


def test_the_dominant_term_is_named():
    """"35 m after a minute" is not actionable; naming the term is."""
    budget = DeadReckoner().budget_at(60.0)
    assert budget.dominant is not None
    assert budget.dominant.metres == max(t.metres for t in budget.terms)


def test_assumed_constants_are_reported_per_regime():
    """A gyro constant against an inertial budget sends the reader to fix the wrong thing."""
    inertial = DeadReckoner().budget_at(60.0)
    aided = DeadReckoner(aiding=SpeedAiding(5.0, 0.05)).budget_at(60.0)
    assert {term.name for term in inertial.assumed} == {"tilt_sigma", "accel_bias"}
    assert {term.name for term in aided.assumed} == {"gyro_bias_sigma", "gyro_arw"}


def test_a_model_resting_on_assumptions_says_so():
    """The rig model is a prediction today, and must not be quoted as measured."""
    check = next(
        c for c in DeadReckoner().checks() if c.name == "propagation.constants_measured"
    )
    assert not check.passed
    assert "tilt_sigma" in check.detail


def test_a_fully_measured_model_passes_the_provenance_check():
    reckoner = DeadReckoner(model_with(tilt_sigma_deg=0.2))
    check = next(
        c for c in reckoner.checks() if c.name == "propagation.constants_measured"
    )
    assert check.passed


def test_horizon_check_fails_when_the_rig_cannot_hold_the_requirement():
    checks = DeadReckoner().checks(limit_m=1.0, required_horizon_s=30.0)
    assert not next(c for c in checks if c.name == "propagation.horizon").passed


def test_an_unreachable_limit_returns_the_bound_rather_than_raising():
    """"Longer than an hour" is good news, and must not arrive as an exception."""
    reckoner = DeadReckoner(model_with(), aiding=SpeedAiding(5.0, 0.0))
    assert reckoner.horizon_for(1.0) == MAX_HORIZON_S


def test_a_non_positive_limit_is_refused():
    with pytest.raises(PropagationError):
        DeadReckoner().horizon_for(0.0)


# -- propagating a pose -----------------------------------------------------


def test_a_straight_run_north_goes_north():
    state = NavState.at_fix(heading_deg=0.0, sigma_position_m=1.0, sigma_heading_deg=1.0)
    moved = propagate(state, [(0.1, 5.0, 0.0)] * 100)
    assert moved.north_m == pytest.approx(50.0)
    assert moved.east_m == pytest.approx(0.0, abs=1e-9)


def test_a_quarter_turn_ends_up_heading_east():
    state = NavState.at_fix(heading_deg=0.0, sigma_position_m=0.0, sigma_heading_deg=0.0)
    turned = propagate(state, [(0.1, 0.0, 9.0)] * 100)
    assert turned.heading_deg == pytest.approx(90.0)


def test_position_uncertainty_grows_with_distance_not_time():
    """A heading error becomes a position error only once the rig moves."""
    state = NavState.at_fix(heading_deg=0.0, sigma_position_m=1.0, sigma_heading_deg=5.0)
    parked = propagate(state, [(0.1, 0.0, 0.0)] * 600)
    driven = propagate(state, [(0.1, 5.0, 0.0)] * 600)
    assert parked.sigma_horizontal_m == pytest.approx(state.sigma_horizontal_m)
    assert driven.sigma_horizontal_m > 3.0 * state.sigma_horizontal_m


def test_heading_error_lands_across_track_not_along_it():
    """Driving north with a heading error displaces you east, not further north."""
    state = NavState.at_fix(0.0, 0.0, 5.0, model=model_with())
    driven = propagate(state, [(0.1, 5.0, 0.0)] * 600, model=model_with())
    assert driven.sigma_east_m > 10.0 * max(driven.sigma_north_m, 1e-9)


def test_a_static_step_freezes_heading_and_its_uncertainty():
    state = NavState.at_fix(heading_deg=10.0, sigma_position_m=1.0, sigma_heading_deg=1.0)
    held = advance(state, 60.0, speed_mps=0.0, yaw_rate_dps=0.0, is_static=True)
    assert held.heading_deg == pytest.approx(10.0)
    assert held.sigma_heading_deg == pytest.approx(1.0)
    assert held.moving_s == 0.0
    assert held.elapsed_s == pytest.approx(60.0)


def test_a_static_step_does_not_heal_position_uncertainty():
    """Standing still stops new error; it does not undo error already made."""
    state = NavState.at_fix(heading_deg=0.0, sigma_position_m=4.0, sigma_heading_deg=1.0)
    held = advance(state, 60.0, speed_mps=0.0, yaw_rate_dps=0.0, is_static=True)
    assert held.sigma_horizontal_m == pytest.approx(state.sigma_horizontal_m)


def test_the_fixs_own_heading_error_survives_propagation():
    """A 5 degree pin stays at least 5 degrees uncertain, drift only adds."""
    state = NavState.at_fix(heading_deg=0.0, sigma_position_m=1.0, sigma_heading_deg=5.0)
    later = propagate(state, [(0.1, 5.0, 0.0)] * 100)
    assert later.sigma_heading_deg >= 5.0


def test_propagated_heading_sigma_agrees_with_the_closed_form():
    """The propagator and the budget must not disagree about the same rig."""
    state = NavState.at_fix(heading_deg=0.0, sigma_position_m=0.0, sigma_heading_deg=0.0)
    later = propagate(state, [(0.1, 5.0, 0.0)] * 100)
    assert later.sigma_heading_deg == pytest.approx(
        RIG_ERROR_MODEL.heading_sigma_deg(10.0), rel=1e-6
    )


def test_covariance_stays_symmetric():
    state = NavState.at_fix(heading_deg=30.0, sigma_position_m=1.0, sigma_heading_deg=3.0)
    later = propagate(state, [(0.1, 5.0, 2.0)] * 200)
    for row in range(4):
        for column in range(4):
            assert later.covariance[row][column] == pytest.approx(
                later.covariance[column][row]
            )


def test_a_negative_step_is_refused():
    state = NavState.at_fix(heading_deg=0.0, sigma_position_m=1.0, sigma_heading_deg=1.0)
    with pytest.raises(PropagationError):
        advance(state, -0.1, 5.0, 0.0)


def test_negative_speed_aiding_is_refused():
    with pytest.raises(PropagationError):
        SpeedAiding(speed_mps=-1.0, sigma_speed_mps=0.1)


def test_a_static_step_admits_no_speed_noise():
    """A speed that is not integrated cannot displace, however badly it is known."""
    state = NavState.at_fix(heading_deg=0.0, sigma_position_m=1.0, sigma_heading_deg=1.0)
    held = advance(
        state, 60.0, speed_mps=0.0, yaw_rate_dps=0.0,
        sigma_speed_mps=0.5, is_static=True,
    )
    assert held.sigma_horizontal_m == pytest.approx(state.sigma_horizontal_m)


def test_speed_error_enters_along_the_direction_of_travel():
    """Not knowing your speed makes you unsure how far, not how far sideways.

    The gyro terms are zeroed so the cross-track axis is left with nothing but
    speed error to show, which is the only way to see that it shows none.
    """
    clean = model_with()
    state = NavState.at_fix(0.0, 0.0, 0.0, model=clean)
    driven = propagate(state, [(0.1, 5.0, 0.0)] * 100, model=clean, sigma_speed_mps=0.5)
    assert driven.sigma_north_m > 0.1
    assert driven.sigma_east_m == pytest.approx(0.0, abs=1e-9)


def test_speed_error_is_a_bias_so_it_grows_linearly_with_time():
    """White noise would grow as sqrt(t) and understate a long run tenfold."""
    clean = model_with()
    state = NavState.at_fix(0.0, 0.0, 0.0, model=clean)
    driven = propagate(state, [(0.1, 5.0, 0.0)] * 100, model=clean, sigma_speed_mps=0.5)
    assert driven.sigma_north_m == pytest.approx(0.5 * 10.0, rel=1e-6)


def test_the_propagator_and_the_budget_agree_about_the_same_rig():
    """Two ways of answering one question; disagreement means one is wrong."""
    aiding = SpeedAiding(speed_mps=5.0, sigma_speed_mps=0.05)
    budget = DeadReckoner(aiding=aiding).budget_at(30.0)

    state = NavState.at_fix(heading_deg=0.0, sigma_position_m=0.0, sigma_heading_deg=0.0)
    driven = propagate(
        state,
        [(0.05, aiding.speed_mps, 0.0)] * 600,
        sigma_speed_mps=aiding.sigma_speed_mps,
    )
    assert driven.sigma_horizontal_m == pytest.approx(budget.total_m, rel=0.005)


def test_the_gyro_bias_is_carried_as_a_state_not_folded_into_heading_noise():
    """Folding it in would look identical in heading and be optimistic in position.

    A bias injected as process noise still produces the right heading sigma --
    which is exactly why the mistake survives review -- while under-stating the
    cross-track error it causes by sqrt(2/3). The bias state is what keeps the
    two consistent, so the filter's answer matches the closed form.
    """
    model = model_with(gyro_bias_sigma_dps=0.05)
    state = NavState.at_fix(0.0, 0.0, 0.0, model=model)
    assert state.sigma_gyro_bias_dps == pytest.approx(0.05)

    driven = propagate(state, [(0.05, 5.0, 0.0)] * 600, model=model)
    exact = 5.0 * math.radians(0.05) * 30.0 ** 2 / 2.0
    assert driven.sigma_east_m == pytest.approx(exact, rel=0.01)


def test_the_bias_uncertainty_itself_does_not_grow():
    """It is a constant over any horizon this propagates across, not a walk."""
    state = NavState.at_fix(0.0, 1.0, 1.0)
    later = propagate(state, [(0.1, 5.0, 0.0)] * 600)
    assert later.sigma_gyro_bias_dps == pytest.approx(state.sigma_gyro_bias_dps)
