"""Tests for the IMU-to-camera rotation calibration.

Every case builds a rig whose true rotation is chosen in advance and checks
that the solve recovers it. Two of them pin corrections to earlier mistakes:

* tilting repeatedly about a single axis is sufficient, not degenerate — two
  gravity directions that are not parallel determine all three degrees of
  freedom, since their cross product supplies the axis neither constrains
* the magnitude test barely detects horizontal acceleration, because gravity
  and lateral motion add in quadrature; a contaminated sample has to be caught
  by disagreeing with the others instead
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = pytest.importorskip("numpy")

from mapinit.calibration.imu import (  # noqa: E402
    GRAVITY_TOLERANCE_MG,
    MIN_TILT_SEPARATION_DEG,
    ImuCalibrationFailed,
    ImuCameraConstraint,
    RigOrientation,
    solve_imu_camera,
    tilt_separation_deg,
)

#: The rotation every scene is built around, a few degrees off nominal.
TRUE_RVEC = (0.05, -0.03, 0.08)

#: A layout that determines all three axes comfortably.
GOOD_TILTS = [((1, 0, 0), 0), ((1, 0, 0), 40), ((0, 1, 0), 40), ((1, 1, 0), 35), ((0, 1, 0), -40)]


def rotation(rvec):
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = np.asarray(rvec, dtype=float) / theta
    k = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)


def build_scene(tilts=None, seed=11, noise_mg=3.6, lateral_mg=None, lateral_index=None):
    """Synthesise rig orientations with a known IMU-to-camera rotation.

    ``lateral_mg`` adds horizontal acceleration to one orientation, standing in
    for a reading taken while the vehicle was braking or turning.
    """
    generator = np.random.default_rng(seed)
    true_rotation = rotation(TRUE_RVEC)
    orientations = []

    for index, (axis, degrees) in enumerate(tilts or GOOD_TILTS):
        direction = np.asarray(axis, dtype=float)
        rig = rotation(direction / np.linalg.norm(direction) * math.radians(degrees))
        # The accelerometer measures the reaction to gravity, so at rest it
        # points UP. World up is +z; a level rig reads +1 g on whichever axis
        # faces the sky.
        up_imu = rig.T @ np.array([0.0, 0.0, 1.0])

        accel = up_imu * 1000.0 + generator.normal(0, noise_mg, 3)
        if lateral_mg is not None and index == lateral_index:
            accel = accel + np.array([lateral_mg, 0.0, 0.0])

        orientations.append(RigOrientation(
            label=f"P{index + 1}",
            imu_accel_mg=tuple(accel),
            camera_up=tuple(true_rotation @ up_imu),
        ))
    return orientations


def rotation_error_deg(solved) -> float:
    relative = np.asarray(solved).T @ rotation(TRUE_RVEC)
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(relative) - 1.0) / 2.0))))


# --------------------------------------------------------------------------
# The solve
# --------------------------------------------------------------------------


def test_a_level_rig_reads_positive_g_on_its_upward_axis():
    """The physical convention, pinned against the hardware.

    This board lying flat reads (+1002, 4, 15) mg, so +X faces the sky. An
    accelerometer measures the reaction to gravity, not gravity, and the
    difference is a sign that nothing downstream would catch.
    """
    flat = RigOrientation("flat", (1002.0, 4.0, 15.0), (0.0, 0.0, 1.0))
    up = flat.imu_up
    assert up[0] > 0.99, "the axis facing the sky reads positive"
    assert flat.is_static


def test_flipping_one_vector_and_not_the_other_is_caught_by_the_residual():
    """Getting the up/down sense wrong on one side cannot hide.

    Negating every camera direction would need R = -R_true to fit, and in three
    dimensions the negative of a rotation has determinant -1: it is a
    reflection, not a rotation. The SVD refuses reflections, so it returns the
    nearest proper rotation instead and the misfit shows up as a large
    residual. The convention is still worth stating loudly -- but if it does get
    reversed, the number says so rather than a plausible answer coming back.
    """
    good = build_scene()
    flipped = [
        RigOrientation(o.label, o.imu_accel_mg, tuple(-c for c in o.camera_up))
        for o in good
    ]

    honest = solve_imu_camera(good)
    wrong = solve_imu_camera(flipped)

    assert honest.residual_rms_deg < 1.0
    assert rotation_error_deg(honest.rotation) < 1.0

    assert wrong.residual_rms_deg > 10.0, "a flip must not fit cleanly"
    assert rotation_error_deg(wrong.rotation) > 45.0


def test_solver_recovers_a_known_rotation():
    solution = solve_imu_camera(build_scene())
    assert rotation_error_deg(solution.rotation) < 1.0
    assert solution.residual_rms_deg < 1.0


def test_two_orientations_are_enough():
    """Two non-parallel directions fix all three degrees of freedom."""
    solution = solve_imu_camera(build_scene([((1, 0, 0), 0), ((1, 0, 0), 90)]))
    assert rotation_error_deg(solution.rotation) < 1.5


def test_tilting_about_a_single_axis_is_not_degenerate():
    """Correction: what matters is that the directions differ, not the axes.

    Rotating the rig repeatedly about one axis sweeps gravity through a plane,
    and a plane already contains two non-parallel directions.
    """
    single_axis = [((1, 0, 0), 0), ((1, 0, 0), 20), ((1, 0, 0), 45), ((1, 0, 0), -35)]
    solution = solve_imu_camera(build_scene(single_axis))
    assert rotation_error_deg(solution.rotation) < 1.5
    assert solution.observability > 0.05


def test_solution_is_a_proper_rotation():
    """Orthonormal with determinant +1: the SVD must not return a reflection."""
    solved = np.asarray(solve_imu_camera(build_scene()).rotation)
    assert np.allclose(solved @ solved.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(solved) == pytest.approx(1.0)


def test_solve_is_deterministic():
    scene = build_scene()
    first, second = solve_imu_camera(scene), solve_imu_camera(scene)
    assert np.allclose(np.asarray(first.rotation), np.asarray(second.rotation))


# --------------------------------------------------------------------------
# Observability
# --------------------------------------------------------------------------


@pytest.mark.parametrize("separation_deg", [25.0, 45.0, 90.0])
def test_observability_matches_its_closed_form(separation_deg):
    """Two directions t apart determine the weakest axis by (1 - cos t) / 2.

    Pinning it to the analytic value keeps the number interpretable: a reader
    can convert a tilt angle to an expected observability in their head, rather
    than treating the metric as an opaque score.
    """
    scene = build_scene([((1, 0, 0), 0), ((1, 0, 0), separation_deg)], noise_mg=0.0)
    expected = (1 - math.cos(math.radians(separation_deg))) / 2
    assert solve_imu_camera(scene).observability == pytest.approx(expected, abs=0.01)


def test_wider_tilts_determine_the_rotation_more_strongly():
    narrow = solve_imu_camera(build_scene([((1, 0, 0), 0), ((1, 0, 0), 20)], noise_mg=0.0))
    wide = solve_imu_camera(build_scene([((1, 0, 0), 0), ((1, 0, 0), 80)], noise_mg=0.0))
    assert wide.observability > 5 * narrow.observability


def test_tilt_separation_is_the_widest_pair():
    scene = build_scene([((1, 0, 0), 0), ((1, 0, 0), 10), ((1, 0, 0), 60)], noise_mg=0.0)
    assert tilt_separation_deg(scene) == pytest.approx(60.0, abs=0.5)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_a_single_orientation_is_refused():
    with pytest.raises(ImuCalibrationFailed, match="at least 2"):
        solve_imu_camera(build_scene([((1, 0, 0), 20)]))


def test_orientations_that_barely_differ_are_refused():
    """Gravity 3 degrees apart twice is one orientation recorded twice."""
    with pytest.raises(ImuCalibrationFailed, match="span only"):
        solve_imu_camera(build_scene([((1, 0, 0), 0), ((1, 0, 0), 3)], noise_mg=0.0))


def test_the_refusal_explains_what_to_do_about_it():
    with pytest.raises(ImuCalibrationFailed) as excinfo:
        solve_imu_camera(build_scene([((1, 0, 0), 0), ((1, 0, 0), 2)], noise_mg=0.0))
    assert "tilt" in str(excinfo.value).lower()


# --------------------------------------------------------------------------
# Motion contamination
# --------------------------------------------------------------------------


def test_lateral_acceleration_barely_moves_the_magnitude():
    """Why the magnitude test is weak: gravity and motion add in quadrature.

    300 mg sideways tilts the apparent vertical by 17 degrees while raising the
    magnitude by 44 mg, which is inside the tolerance. The reading looks static
    and is not.
    """
    braking = RigOrientation("braking", (300.0, 0.0, -1000.0), (0.0, 0.0, -1.0))

    assert braking.magnitude_mg == pytest.approx(1044.0, abs=1.0)
    assert braking.magnitude_mg - 1000.0 < GRAVITY_TOLERANCE_MG
    assert braking.is_static, "the magnitude test does not catch this, by construction"

    tilt = math.degrees(math.atan2(300.0, 1000.0))
    assert tilt > 15.0, "yet the apparent vertical is well off true"


def test_a_contaminated_orientation_is_caught_by_disagreeing_with_the_others():
    """What the magnitude test misses, the residual finds."""
    scene = build_scene(lateral_mg=300.0, lateral_index=2)
    constraint = ImuCameraConstraint(scene)
    checks = {c.name.split(".")[-1]: c for c in constraint.validate()}

    # Either it was discarded as non-static, or it survived and shows up as the
    # orientation that disagrees. One of the two must have flagged it.
    discarded = checks["static_orientations"].measured < len(scene)
    flagged = not checks.get("worst_orientation", checks["residual"]).passed
    assert discarded or flagged, "a braking sample must not pass unnoticed"


def test_clean_orientations_all_report_static():
    constraint = ImuCameraConstraint(build_scene())
    checks = {c.name.split(".")[-1]: c for c in constraint.validate()}
    assert checks["static_orientations"].measured == len(GOOD_TILTS)


# --------------------------------------------------------------------------
# The constraint, as the calibration stage sees it
# --------------------------------------------------------------------------


def test_a_good_set_passes_every_check():
    checks = ImuCameraConstraint(build_scene()).validate()
    assert all(c.passed for c in checks), [str(c) for c in checks if not c.passed]


def test_the_solution_is_available_after_validation():
    constraint = ImuCameraConstraint(build_scene())
    assert constraint.solution is None
    constraint.validate()
    assert constraint.solution is not None
    assert rotation_error_deg(constraint.solution.rotation) < 1.0


def test_too_few_orientations_fails_the_count_check():
    constraint = ImuCameraConstraint(build_scene([((1, 0, 0), 0), ((1, 0, 0), 40)]), min_orientations=3)
    counts = [c for c in constraint.validate() if c.name.endswith("static_orientations")]
    assert counts and not counts[0].passed


def test_observations_carry_the_accelerometer_angular_noise():
    """3.6 mg on a 1000 mg vector is about 0.2 degrees."""
    observations = ImuCameraConstraint(build_scene()).observations()
    assert len(observations) == len(GOOD_TILTS)
    assert all(o.kind == "gravity_direction" for o in observations)
    assert observations[0].sigma == pytest.approx(0.206, abs=0.01)


def test_summary_names_the_numbers_that_qualify_the_answer():
    text = str(solve_imu_camera(build_scene()))
    assert "observability" in text
    assert "tilt spread" in text
    assert "residual" in text


def test_minimum_separation_threshold_is_stated_in_degrees():
    assert MIN_TILT_SEPARATION_DEG == pytest.approx(15.0)


def test_a_missing_package_is_a_bad_install_not_a_bad_capture():
    """The exception type is what a consumer branches on, and it decides who is called.

    Every other ImuCalibrationFailed says the captured orientations were not
    enough. This one says the capture was never read. Reporting it as the
    former sends someone back out to the rig to repeat a session that was fine.
    """
    from mapinit.calibration.imu import CalibrationDependencyMissing, ImuCalibrationFailed

    failure = CalibrationDependencyMissing("numpy", "solve the IMU rotation")
    assert isinstance(failure, ImuCalibrationFailed)
    assert failure.name == "numpy"
    assert "pip install numpy" in str(failure)
