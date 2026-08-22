#!/usr/bin/env python3
"""Solve the fixed rotation between the IMU and the thermal camera.

The two are soldered to the same board, so the rotation between them never
changes and is measured once. It is what lets the IMU's attitude say where the
camera is pointing, which is what seats a camera image onto a map.

**Rotation only.** They sit about 3 cm apart, and a rigid body has the same
orientation at every point on it, so the offset never enters. Three unknowns,
not six.

**How it is observed.** The rig is held still in several orientations. In each:

* the accelerometer measures the *reaction* to gravity, so at rest the axis
  pointing skyward reads +1 g. The measured direction is therefore **up**, not
  down. Confirmed against this board: lying flat it reads (+1002, 4, 15) mg,
  which is +X up.
* the camera observes surveyed targets, which fixes its orientation in the
  world, and therefore where up lies in camera coordinates

Both vectors must use the same sense. Flipping one and not the other yields a
rotation wrong by 180 degrees about a horizontal axis, and Wahba's problem
reports it with a clean residual because it fits the flipped data perfectly.

Two readings of one direction in two frames, repeated. The rotation that
reconciles them is Wahba's problem, solved in closed form by SVD.

**Why more than one orientation is required.** A single gravity vector fixes
only two degrees of freedom: rotating the rig about the gravity direction
itself changes nothing the accelerometer can see. Two directions that are not
parallel fix all three, since their cross product supplies the axis neither of
them constrains. Tilting repeatedly about the same axis is therefore fine —
what matters is that the gravity directions differ, not that the tilt axes do.

``observability`` reports how strongly the weakest rotational axis is pinned,
and it grows with tilt angle: a pair of directions 25 degrees apart determines
that axis about ten times less strongly than a pair 90 degrees apart.

**On detecting motion.** The accelerometer measures gravity plus whatever else
the rig is doing, and the magnitude test below is a poor detector of the
horizontal case: braking at 300 mg tilts the apparent vertical by 17 degrees
while raising the magnitude only 44 mg, because the two add in quadrature. The
magnitude test catches gross vertical motion; what actually catches a
contaminated sample is that it disagrees with the others, which is why the
per-orientation residual is checked and not just the aggregate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Sequence, Tuple

from ..check import Check
from .constraints import CalibrationConstraint, Observation

#: Accelerometer noise at rest, in milli-g. Measured on this board over 5977
#: stationary samples: |a| = 1004 mg with a standard deviation of 3.6 mg,
#: which is about 0.2 degrees of angular noise.
ACCEL_SIGMA_MG = 3.6
STANDARD_GRAVITY_MG = 1000.0

#: Below this the accelerometer is not measuring gravity alone. On a moving
#: vehicle, braking at 0.3 g tilts the apparent vertical by 17 degrees, so a
#: sample taken under acceleration is not a gravity reading at all.
GRAVITY_TOLERANCE_MG = 60.0

#: Minimum angle between two gravity directions for the pair to say anything
#: about rotation around the first one. Below this the second orientation is a
#: repeat of the first, however different it looks in the log.
MIN_TILT_SEPARATION_DEG = 15.0

#: Smallest eigenvalue of the rotation information matrix, per orientation.
#: Two directions separated by an angle t give (1 - cos t) / 2, so 25 degrees
#: gives 0.05 and 90 degrees gives 0.5. Below this floor the weakest axis of
#: the solution is determined too feebly to report as a measurement.
MIN_OBSERVABILITY = 0.02


class ImuCalibrationFailed(RuntimeError):
    """Raised when the orientations cannot determine the rotation."""


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImuCalibrationFailed("numpy is required to solve the IMU rotation") from exc
    return np


@dataclass(frozen=True)
class RigOrientation:
    """One static pose: what the IMU read, and where the camera was pointing.

    ``camera_up`` is which way is up, expressed in camera coordinates. It comes
    from the camera's orientation against the surveyed targets, not from the
    IMU, which is the whole point — the two must be independent for their
    comparison to mean anything.

    Both directions point **up**, matching what the accelerometer physically
    reports at rest. The naming is laboured on purpose: this is a sign that can
    be got wrong silently.
    """

    label: str
    #: Raw accelerometer reading in IMU coordinates, milli-g. At rest the axis
    #: pointing skyward reads about +1000.
    imu_accel_mg: Tuple[float, float, float]
    #: Unit vector pointing up, in camera coordinates. Same sense as the
    #: accelerometer reading, not its negation.
    camera_up: Tuple[float, float, float]

    @property
    def magnitude_mg(self) -> float:
        return math.sqrt(sum(component * component for component in self.imu_accel_mg))

    @property
    def is_static(self) -> bool:
        """Whether the reading is gravity alone rather than gravity plus motion."""
        return abs(self.magnitude_mg - STANDARD_GRAVITY_MG) <= GRAVITY_TOLERANCE_MG

    @property
    def imu_up(self) -> Tuple[float, float, float]:
        """Accelerometer reading normalised: a unit vector pointing up.

        Up rather than down, because that is what the sensor measures. Naming
        it after gravity invites a negation that nothing downstream would catch.
        """
        magnitude = self.magnitude_mg
        if magnitude < 1e-6:
            raise ImuCalibrationFailed(f"Orientation {self.label!r} has no measurable acceleration")
        return tuple(component / magnitude for component in self.imu_accel_mg)


@dataclass
class ImuCameraSolution:
    """The IMU-to-camera rotation, and the evidence for how well it is pinned."""

    #: Rotation taking a direction in IMU coordinates into camera coordinates.
    rotation: "object"
    roll_deg: float
    pitch_deg: float
    yaw_deg: float

    #: Angle between measured and predicted gravity, per orientation, degrees.
    residuals_deg: List[float] = field(default_factory=list)
    #: Fraction of the third rotational axis the tilts actually determine.
    observability: float = 0.0
    #: Largest angle between any two gravity directions, degrees.
    tilt_separation_deg: float = 0.0
    orientations: List[str] = field(default_factory=list)

    @property
    def residual_rms_deg(self) -> float:
        if not self.residuals_deg:
            return 0.0
        return math.sqrt(sum(r * r for r in self.residuals_deg) / len(self.residuals_deg))

    def __str__(self) -> str:
        return (
            f"IMU->camera  RPY = ({self.roll_deg:+.3f}, {self.pitch_deg:+.3f}, "
            f"{self.yaw_deg:+.3f}) deg\n"
            f"             residual {self.residual_rms_deg:.3f} deg over "
            f"{len(self.orientations)} orientations, observability "
            f"{self.observability:.3f}, tilt spread {self.tilt_separation_deg:.1f} deg"
        )


def _euler_from_rotation(matrix) -> Tuple[float, float, float]:
    """Rotation matrix to (roll, pitch, yaw) in degrees, ZYX intrinsic."""
    np = _require_numpy()
    m = np.asarray(matrix, dtype=float)
    pitch = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(m[2, 0]) < 0.999999:
        roll = math.atan2(m[2, 1], m[2, 2])
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-m[0, 1], m[1, 1])
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def tilt_separation_deg(orientations: Sequence[RigOrientation]) -> float:
    """Largest angle between any two gravity directions in the set.

    This is what makes a second orientation informative. Two tilts about the
    same axis leave rotation about gravity as unconstrained as one tilt did.
    """
    np = _require_numpy()
    vectors = [np.array(o.imu_up) for o in orientations]
    widest = 0.0
    for i, first in enumerate(vectors):
        for second in vectors[i + 1:]:
            cosine = max(-1.0, min(1.0, float(first @ second)))
            widest = max(widest, math.degrees(math.acos(cosine)))
    return widest


def _observability(np, orientations: Sequence[RigOrientation]) -> float:
    """How strongly the weakest rotational axis is determined, per orientation.

    Turning the solution by a small angle about an axis moves each predicted
    direction by the cross product of the two, so an axis parallel to a gravity
    direction moves it not at all. Summing that sensitivity over the set gives
    the information matrix ``N*I - sum(g g^T)``, whose smallest eigenvalue is
    the worst-determined axis. Normalising by N makes the number comparable
    between runs with different numbers of orientations.
    """
    directions = [np.array(o.imu_up) for o in orientations]
    scatter = sum(np.outer(g, g) for g in directions)
    information = len(directions) * np.eye(3) - scatter
    return float(np.linalg.eigvalsh(information)[0] / len(directions))


def solve_imu_camera(orientations: Sequence[RigOrientation]) -> ImuCameraSolution:
    """Solve the IMU-to-camera rotation from gravity observed in both frames.

    Wahba's problem: find the rotation minimising the angle between each
    measured pair. The SVD solution is exact and has no starting guess to get
    wrong, which is worth a great deal here — a Gauss-Newton pass over the same
    data can converge to a reflection instead.
    """
    np = _require_numpy()

    usable = [o for o in orientations if o.is_static]
    if len(usable) < 2:
        moving = [o.label for o in orientations if not o.is_static]
        raise ImuCalibrationFailed(
            f"Need at least 2 static orientations, got {len(usable)}. "
            f"A single gravity direction leaves rotation about it undetermined. "
            + (f"Discarded as non-static: {moving}" if moving else "")
        )

    separation = tilt_separation_deg(usable)
    if separation < MIN_TILT_SEPARATION_DEG:
        raise ImuCalibrationFailed(
            f"Gravity directions span only {separation:.1f} degrees across "
            f"{len(usable)} orientations. Tilting about nearly the same axis "
            f"twice adds no information; tilt about a different axis."
        )

    # Attitude profile matrix. Its SVD gives the rotation that best reconciles
    # the two sets of directions, and its singular values say how much of the
    # rotation the directions actually constrain.
    profile = np.zeros((3, 3))
    for orientation in usable:
        camera = np.array(orientation.camera_up, dtype=float)
        camera = camera / np.linalg.norm(camera)
        profile += np.outer(camera, np.array(orientation.imu_up))

    u, _, vt = np.linalg.svd(profile)
    # The middle term forbids a reflection, which would fit the data equally
    # well while being physically impossible
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
    rotation = u @ correction @ vt

    observability = _observability(np, usable)

    residuals = []
    for orientation in usable:
        predicted = rotation @ np.array(orientation.imu_up)
        measured = np.array(orientation.camera_up, dtype=float)
        measured = measured / np.linalg.norm(measured)
        cosine = max(-1.0, min(1.0, float(predicted @ measured)))
        residuals.append(math.degrees(math.acos(cosine)))

    roll, pitch, yaw = _euler_from_rotation(rotation)
    return ImuCameraSolution(
        rotation=rotation,
        roll_deg=roll,
        pitch_deg=pitch,
        yaw_deg=yaw,
        residuals_deg=residuals,
        observability=observability,
        tilt_separation_deg=separation,
        orientations=[o.label for o in usable],
    )


class ImuCameraConstraint(CalibrationConstraint):
    """Validates that a set of rig orientations can pin the IMU rotation."""

    name: ClassVar[str] = "imu_camera"

    def __init__(
        self,
        orientations: Sequence[RigOrientation],
        min_orientations: int = 3,
        min_tilt_separation_deg: float = MIN_TILT_SEPARATION_DEG,
        min_observability: float = MIN_OBSERVABILITY,
        max_residual_deg: float = 2.0,
    ) -> None:
        self.orientations = list(orientations)
        self.min_orientations = min_orientations
        self.min_tilt_separation_deg = min_tilt_separation_deg
        self.min_observability = min_observability
        self.max_residual_deg = max_residual_deg
        self._solution: Optional[ImuCameraSolution] = None

    @property
    def solution(self) -> Optional[ImuCameraSolution]:
        """The solved rotation, once validate() has run successfully."""
        return self._solution

    def validate(self) -> List[Check]:
        static = [o for o in self.orientations if o.is_static]

        checks = [
            Check.in_range(
                self.check_name("static_orientations"),
                len(static),
                self.min_orientations,
                float("inf"),
                unit="poses",
                detail=(
                    f"{len(static)} of {len(self.orientations)} orientations are static; "
                    f"the rest read acceleration beyond gravity and were discarded"
                ),
            )
        ]

        if len(static) < 2:
            return checks

        separation = tilt_separation_deg(static)
        checks.append(
            Check.in_range(
                self.check_name("tilt_separation"),
                separation,
                self.min_tilt_separation_deg,
                float("inf"),
                unit="deg",
                detail=(
                    f"gravity directions span {separation:.1f} deg; below "
                    f"{self.min_tilt_separation_deg:.0f} the tilts repeat each other"
                ),
            )
        )

        try:
            solution = solve_imu_camera(static)
        except ImuCalibrationFailed as exc:
            checks.append(Check.that(self.check_name("solvable"), False, str(exc)))
            return checks

        self._solution = solution

        checks.append(
            Check.in_range(
                self.check_name("observability"),
                solution.observability,
                self.min_observability,
                float("inf"),
                detail=(
                    f"the tilts determine {solution.observability:.3f} of the third "
                    f"rotational axis; near zero means rotation about gravity is free"
                ),
            )
        )
        checks.append(
            Check.in_range(
                self.check_name("residual"),
                solution.residual_rms_deg,
                0.0,
                self.max_residual_deg,
                unit="deg",
                detail=(
                    f"gravity is reproduced to {solution.residual_rms_deg:.3f} deg "
                    f"across {len(static)} orientations"
                ),
            )
        )

        # The magnitude test misses horizontal acceleration almost entirely, so
        # a sample taken while the rig was braking or turning arrives here
        # looking static. It shows up instead as one orientation that disagrees
        # with the rest, which is what this check is for.
        worst = max(zip(solution.residuals_deg, solution.orientations))
        checks.append(
            Check.in_range(
                self.check_name("worst_orientation"),
                worst[0],
                0.0,
                max(self.max_residual_deg * 2.0, 3.0),
                unit="deg",
                detail=(
                    f"orientation {worst[1]!r} departs by {worst[0]:.2f} deg; a sample "
                    f"taken under acceleration lands here rather than failing the "
                    f"magnitude test"
                ),
            )
        )
        return checks

    def observations(self) -> List[Observation]:
        """One gravity direction per static orientation.

        Sigma is angular: 3.6 mg of noise on a 1000 mg vector is about 0.2
        degrees, which is why roll and pitch from this IMU are trustworthy in
        a way its yaw never is.
        """
        angular_sigma = math.degrees(math.atan2(ACCEL_SIGMA_MG, STANDARD_GRAVITY_MG))
        return [
            Observation(
                kind="gravity_direction",
                source=f"{self.name}:{orientation.label}",
                sigma=angular_sigma,
                payload={
                    "imu_up": orientation.imu_up,
                    "camera_up": orientation.camera_up,
                },
            )
            for orientation in self.orientations
            if orientation.is_static
        ]
