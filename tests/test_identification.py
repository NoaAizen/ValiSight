#!/usr/bin/env python3
"""Fitting sensor constants from a recording, and agreeing with the rig.

Two things are guarded here. The first is that the fit does what it claims:
the Allan estimator recovers a coefficient it was given, and the constants it
marks measured are the ones a stationary recording can actually settle.

The second matters more. ``propagation`` reproduces the growth law inside
``yael_api.imu.AttitudeEstimator``, so that the uncertainty this package
predicts and the uncertainty the rig reports are the same number. Nothing
enforces that but a test: the two files are in different packages, owned by
different people, and a change to either one is invisible from the other. The
cross-check below runs against the real estimator and fails the moment they
part company.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mapinit.check import CheckFailed
from mapinit.nav.identification import (
    ARW_TAU_S,
    GRAVITY_MG,
    RecordingError,
    allan_deviation,
    angle_random_walk_dps_sqrt_s,
    identification_checks,
    identify,
    load_recording,
)
from mapinit.nav.propagation import (
    ASSUMED,
    MEASURED,
    DeadReckoner,
    PropagationError,
    RIG_ERROR_MODEL,
    SpeedAiding,
)

REPO = Path(__file__).resolve().parent.parent
RECORDING = REPO / "bridge" / "recordings" / "imu_holds_2026-08-18"
SAMPLE_RATE_HZ = 200.0

needs_recording = pytest.mark.skipif(
    not RECORDING.exists(), reason=f"{RECORDING.name} not present"
)


@pytest.fixture(scope="module")
def recording():
    if not RECORDING.exists():
        pytest.skip(f"{RECORDING.name} not present")
    return load_recording(RECORDING)


# ---------------------------------------------------------------------------
# the estimator itself
# ---------------------------------------------------------------------------


def _white_noise(count: int, sigma_dps: float, seed: int = 7):
    """Deterministic white rate noise, so the assertion is reproducible."""
    import random

    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma_dps) for _ in range(count)]


def test_allan_recovers_the_coefficient_it_was_given():
    # White rate noise of standard deviation s at rate f is angle random walk
    # with coefficient s / sqrt(f), and the Allan deviation at one second must
    # return that coefficient rather than the raw standard deviation.
    sigma_dps = 0.3
    expected = sigma_dps / math.sqrt(SAMPLE_RATE_HZ)

    measured = angle_random_walk_dps_sqrt_s(
        _white_noise(60 * int(SAMPLE_RATE_HZ), sigma_dps), SAMPLE_RATE_HZ
    )

    assert measured == pytest.approx(expected, rel=0.15)


def test_allan_curve_falls_as_one_over_root_tau():
    curve = allan_deviation(_white_noise(20_000, 0.3), SAMPLE_RATE_HZ)
    early = next(point for point in curve if point[0] >= 0.1)
    late = next(point for point in curve if point[0] >= 1.0)

    # Four times the averaging time halves the deviation on this slope.
    ratio = early[1] / late[1]
    assert ratio == pytest.approx(math.sqrt(late[0] / early[0]), rel=0.2)


def test_allan_refuses_a_series_too_short_to_average():
    with pytest.raises(PropagationError, match="at least 3 samples"):
        allan_deviation([0.1, 0.2], SAMPLE_RATE_HZ)


def test_allan_refuses_a_nonsense_sample_rate():
    with pytest.raises(PropagationError, match="sample rate must be positive"):
        allan_deviation([0.1] * 100, 0.0)


# ---------------------------------------------------------------------------
# against the real recording
# ---------------------------------------------------------------------------


@needs_recording
def test_recording_loads_with_its_holds(recording):
    assert len(recording.holds) == 8
    assert recording.duration_s == pytest.approx(110.0, abs=1.0)
    assert recording.longest_hold().label == "hold_02"


@needs_recording
def test_gravity_norm_is_near_nominal_in_every_hold(recording):
    # Placement-independent: however the rig is set down, the magnitude of
    # gravity is the same, so a deviation here is the sensor's and not the
    # placer's. Anything past 20 mg would mean the recording caught motion.
    for hold in recording.holds:
        assert abs(hold.norm_error_mg) < 20.0, hold.label
        assert hold.accel_norm_mg == pytest.approx(GRAVITY_MG, abs=20.0)


@needs_recording
def test_fit_settles_the_gyro_terms_and_leaves_tilt_alone(recording):
    fitted = identify(recording)

    assert fitted.gyro_arw_dps_sqrt_s.provenance == MEASURED
    assert fitted.gyro_bias_sigma_dps.provenance == MEASURED
    assert fitted.accel_bias_mg.provenance == MEASURED
    # A still rig cannot separate its own tilt error from the surface under it.
    assert fitted.tilt_sigma_deg.provenance == ASSUMED
    assert fitted.assumed_terms() == [fitted.tilt_sigma_deg]


@needs_recording
def test_the_shipped_random_walk_is_far_too_conservative(recording):
    # The placeholder was chosen to be safe rather than right. It is more than
    # an order of magnitude above what this gyro does, and those metres argued
    # for corrections the hardware does not need.
    fitted = identify(recording)

    assert fitted.gyro_arw_dps_sqrt_s.value < 0.02
    assert fitted.gyro_arw_dps_sqrt_s.value < RIG_ERROR_MODEL.gyro_arw_dps_sqrt_s.value / 10.0


@needs_recording
def test_measured_constants_shrink_the_aided_budget(recording):
    aiding = SpeedAiding(5.0, 0.1)
    horizon_s = 60.0
    before = DeadReckoner(RIG_ERROR_MODEL, aiding=aiding).budget_at(horizon_s)
    after = DeadReckoner(identify(recording), aiding=aiding).budget_at(horizon_s)

    # Aided, the budget is made of gyro terms, and both of those were assumed
    # high. Measuring them takes the total down and moves speed error to the top.
    assert after.total_m < before.total_m
    assert before.rests_on_assumption
    assert not after.rests_on_assumption
    assert after.dominant.name == "along_track.speed"


@needs_recording
def test_measuring_makes_the_unaided_budget_worse(recording):
    """Measurement is not the same as improvement, and the report must not blur it.

    Unaided, the budget is dominated by the accelerometer, whose shipped bias
    was optimistic rather than conservative. Fitting it doubles the number. A
    model that only ever moved in the reassuring direction would be evidence it
    was being tuned rather than measured.
    """
    horizon_s = 60.0
    before = DeadReckoner(RIG_ERROR_MODEL).budget_at(horizon_s)
    after = DeadReckoner(identify(recording)).budget_at(horizon_s)

    assert after.total_m > before.total_m
    assert after.dominant.name == "accel_bias"


@needs_recording
def test_fit_separates_what_the_record_supports_from_what_it_does_not(recording):
    fitted = identify(recording)
    checks = {check.name: check for check in identification_checks(recording, fitted)}

    # Thirty seconds is ample for the random walk, read at one second...
    assert checks["identification.arw_supported"].passed
    # ...and nowhere near enough for the bias floor, which sits two orders of
    # magnitude further out. The fit says so instead of passing a bound off as
    # a measurement.
    assert not checks["identification.bias_floor_supported"].passed
    assert checks["identification.tilt_still_assumed"].passed
    with pytest.raises(CheckFailed):
        checks["identification.bias_floor_supported"].raise_if_failed()


@needs_recording
def test_fit_refuses_an_axis_that_is_not_an_axis(recording):
    with pytest.raises(PropagationError, match="yaw_axis must be"):
        identify(recording, yaw_axis=3)


@needs_recording
def test_fit_refuses_when_too_few_holds_are_long_enough(recording):
    with pytest.raises(RecordingError, match="to estimate bias spread"):
        identify(recording, min_hold_s=1000.0)


def test_missing_recording_names_the_file_rather_than_KeyError(tmp_path):
    with pytest.raises(RecordingError, match="imu.csv does not exist"):
        load_recording(tmp_path)


# ---------------------------------------------------------------------------
# agreement with the rig's own estimator
# ---------------------------------------------------------------------------


def test_heading_sigma_matches_the_rig_exactly():
    """The rig's uncertainty and ours must be the same number, not merely close.

    ``AttitudeEstimator`` accrues drift only while moving, so the gyro is fed
    above the rig's own static threshold to keep it integrating. If this ever
    fails, one of the two formulas has changed and the two halves of the system
    have started reporting different confidence in the same fix.
    """
    imu = pytest.importorskip(
        "yael_api.imu", reason="yael_api is not on the path; pull branch IMU"
    )

    estimator = imu.AttitudeEstimator()
    model = RIG_ERROR_MODEL.__class__(
        tilt_sigma_deg=RIG_ERROR_MODEL.tilt_sigma_deg,
        accel_bias_mg=RIG_ERROR_MODEL.accel_bias_mg,
        accel_sigma_mg=RIG_ERROR_MODEL.accel_sigma_mg,
        gyro_bias_sigma_dps=RIG_ERROR_MODEL.gyro_bias_sigma_dps.__class__(
            "gyro_bias_sigma", imu.BIAS_SIGMA_DPS, "deg/s", ASSUMED, "read from the rig"
        ),
        gyro_arw_dps_sqrt_s=RIG_ERROR_MODEL.gyro_arw_dps_sqrt_s.__class__(
            "gyro_arw", imu.ARW_DPS_SQRT_S, "deg/sqrt(s)", ASSUMED, "read from the rig"
        ),
    )
    reckoner = DeadReckoner(model)

    dt = 1.0 / SAMPLE_RATE_HZ
    moving_gyro = (0.0, 0.0, imu.STATIC_GYRO_DPS + 2.0)
    for step in range(1, 2001):
        estimator.feed(step * dt, (1000.0, 0.0, 0.0), moving_gyro)

    assert estimator.t_moving_s > 5.0, "the rig never left its static gate"
    assert estimator.last["yaw_sigma_deg"] == pytest.approx(
        reckoner.heading_sigma_deg(estimator.t_moving_s), abs=1e-12
    )


def test_our_constants_are_the_rig_s_constants():
    """A drift budget built on numbers the rig does not use predicts nothing."""
    imu = pytest.importorskip(
        "yael_api.imu", reason="yael_api is not on the path; pull branch IMU"
    )

    assert RIG_ERROR_MODEL.gyro_bias_sigma_dps.value == imu.BIAS_SIGMA_DPS
    assert RIG_ERROR_MODEL.gyro_arw_dps_sqrt_s.value == imu.ARW_DPS_SQRT_S


@needs_recording
def test_the_rig_would_call_every_hold_static():
    """The windows we fit from must be the same ones the rig calls static.

    Fitting a bias from samples the rig would reject means measuring a constant
    the rig never uses, so the two must agree on what "still" means.
    """
    imu = pytest.importorskip(
        "yael_api.imu", reason="yael_api is not on the path; pull branch IMU"
    )
    recording = load_recording(RECORDING)

    for hold in recording.holds:
        norm = math.sqrt(sum(component ** 2 for component in hold.accel_mg))
        assert abs(norm - GRAVITY_MG) < imu.STATIC_ACC_TOL_MG, hold.label
        assert max(abs(rate) for rate in hold.gyro_bias_dps) < imu.STATIC_GYRO_DPS, hold.label
