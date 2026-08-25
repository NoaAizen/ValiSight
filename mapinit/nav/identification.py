#!/usr/bin/env python3
"""Turn a stationary IMU recording into measured sensor constants.

``propagation`` computes a drift budget from an ``ImuErrorModel``, and says so
when that model rests on constants nobody measured. This module is the other
half: it reads a recording off the rig and returns the same model with those
constants filled in from data, so the budget stops being a prediction.

**Why the numbers move so much.** The constants shipped as placeholders were
chosen to be safe rather than right, and safe is not free: an angle random walk
assumed 35 times larger than the sensor's own puts metres into the budget that
the hardware never contributes, and those metres argue for corrections the rig
does not actually need. A conservative guess is not the cautious choice when it
changes what gets built.

**What a still rig can and cannot tell you.**

* *Angle random walk* — yes. It is read off the Allan deviation at one second,
  where the curve still falls as one over root tau. Thirty seconds of samples
  resolve that slope perfectly well.
* *Bias instability* — only partly. It is the flat floor of the same curve,
  reached at taus this recording is too short to average over. What the spread
  of per-hold biases gives instead is an upper bound, and it is reported as one.
* *Accelerometer scale and bias* — only through the norm. Gravity has the same
  magnitude in every orientation, so ``|a|`` should read 1000 mg however the
  rig is placed, and a deviation is the sensor's error rather than the placer's.
  The individual axes cannot be separated this way: a board resting a degree off
  square moves one axis by the same amount a bias would, and nothing in a
  recording distinguishes them. That separation needs a levelled reference, and
  ``tilt_sigma`` stays assumed until one exists.

The recording format is the one ``bridge/imu_holds.py`` writes: an ``imu.csv``
of raw samples and a ``holds.json`` marking the stationary windows.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..check import Check
from .propagation import (
    ASSUMED,
    MEASURED,
    ImuErrorModel,
    NoiseTerm,
    PropagationError,
    RIG_ERROR_MODEL,
)

#: Nominal gravity as the accelerometer reports it, in mg. The norm of a
#: stationary reading is compared against this, and the difference is the
#: sensor's error rather than the placement's.
GRAVITY_MG = 1000.0

#: Tau at which the Allan deviation is read to get the angle random walk, in
#: seconds. At this tau the curve is still on its one-over-root-tau slope for
#: any MEMS gyro, and sigma(1 s) is the coefficient by definition.
ARW_TAU_S = 1.0

#: Where the Allan floor that sets bias instability typically sits for a MEMS
#: gyro, in seconds. Below this the curve is still falling on its random-walk
#: slope, so a shorter record cannot see the floor at all.
BIAS_FLOOR_TAU_S = 100.0

#: How long a record needs to be to average that floor rather than glimpse it.
#: A handful of clusters at the longest tau is noise, not an estimate.
BIAS_FLOOR_RECORD_S = 10.0 * BIAS_FLOOR_TAU_S

#: A hold shorter than this contributes a bias estimate too noisy to trust and
#: is excluded from the spread. The tilted holds in the August recording run
#: two to six seconds, and the two shortest sit visibly off the others.
MIN_HOLD_S = 3.0

#: Ticks are microseconds when the recording declares ts_src=0.
TICKS_PER_SECOND = 1e6


class RecordingError(RuntimeError):
    """Raised when a recording cannot answer what is being asked of it."""


@dataclass(frozen=True)
class Hold:
    """One stationary window, and what the sensors read during it."""

    label: str
    duration_s: float
    samples: int
    #: Mean raw gyro over the window, deg/s. With the rig still, this is the
    #: bias: earth rate is 0.004 deg/s and sits far below the noise here.
    gyro_bias_dps: Tuple[float, float, float]
    #: Mean accelerometer reading over the window, mg.
    accel_mg: Tuple[float, float, float]

    @property
    def accel_norm_mg(self) -> float:
        return math.sqrt(sum(component ** 2 for component in self.accel_mg))

    @property
    def norm_error_mg(self) -> float:
        """How far the magnitude of gravity reads from nominal, in mg.

        Independent of how the rig was placed, which is what makes it a
        measurement of the sensor rather than of the table.
        """
        return self.accel_norm_mg - GRAVITY_MG


@dataclass(frozen=True)
class Recording:
    """Raw samples and the stationary windows marked within them."""

    times_s: List[float]
    gyro_dps: List[Tuple[float, float, float]]
    accel_mg: List[Tuple[float, float, float]]
    holds: List[Hold]
    sample_rate_hz: float
    path: Path

    @property
    def duration_s(self) -> float:
        return self.times_s[-1] - self.times_s[0] if self.times_s else 0.0

    def longest_hold(self) -> Hold:
        if not self.holds:
            raise RecordingError(f"{self.path} marks no stationary holds")
        return max(self.holds, key=lambda hold: hold.samples)

    def window(self, hold: Hold) -> Tuple[int, int]:
        """Index range of one hold's samples, as a half-open slice."""
        for start, end, label in self._spans:
            if label == hold.label:
                return start, end
        raise RecordingError(f"no samples for hold {hold.label!r}")


def allan_deviation(
    rates_dps: Sequence[float],
    sample_rate_hz: float,
    growth: float = 1.3,
) -> List[Tuple[float, float, int]]:
    """Overlapping Allan deviation of a rate series.

    Returns ``(tau_s, sigma_dps, clusters)`` per averaging time. Overlapping
    rather than plain because a thirty-second record has few independent
    clusters at the taus that matter, and the overlapping estimator uses every
    one of them instead of discarding the remainder.
    """
    count = len(rates_dps)
    if count < 3:
        raise PropagationError(f"need at least 3 samples, got {count}")
    if sample_rate_hz <= 0:
        raise PropagationError(f"sample rate must be positive, got {sample_rate_hz}")

    # Integrating rate gives angle, and the Allan variance is the mean squared
    # second difference of angle. Doing it this way costs one pass instead of
    # one per tau.
    angle = []
    total = 0.0
    for rate in rates_dps:
        total += rate / sample_rate_hz
        angle.append(total)

    curve: List[Tuple[float, float, int]] = []
    stride = 1
    while stride <= (count - 1) // 2:
        tau = stride / sample_rate_hz
        clusters = count - 2 * stride
        accumulated = 0.0
        for index in range(clusters):
            second_difference = (
                angle[index + 2 * stride] - 2.0 * angle[index + stride] + angle[index]
            )
            accumulated += second_difference * second_difference
        curve.append((tau, math.sqrt(accumulated / (2.0 * clusters * tau * tau)), clusters))
        stride = max(stride + 1, int(math.ceil(stride * growth)))
    return curve


def angle_random_walk_dps_sqrt_s(
    rates_dps: Sequence[float],
    sample_rate_hz: float,
    tau_s: float = ARW_TAU_S,
) -> float:
    """Angle random walk coefficient, deg/sqrt(s), from the Allan curve.

    On the one-over-root-tau slope, ``sigma(tau) = N / sqrt(tau)``, so any tau
    on that slope recovers N. Reading at a tau the record cannot reach would
    return the noise of a handful of clusters rather than the coefficient, so
    the nearest available tau is used and the caller can check the record is
    long enough.
    """
    curve = allan_deviation(rates_dps, sample_rate_hz)
    if not curve:
        raise PropagationError("Allan curve is empty; recording is too short")
    tau, sigma, _ = min(curve, key=lambda point: abs(point[0] - tau_s))
    return sigma * math.sqrt(tau)


def load_recording(directory: Path, sample_rate_hz: float = 200.0) -> Recording:
    """Read an ``imu.csv`` / ``holds.json`` pair written by ``imu_holds.py``."""
    directory = Path(directory)
    csv_path, holds_path = directory / "imu.csv", directory / "holds.json"
    for path in (csv_path, holds_path):
        if not path.exists():
            raise RecordingError(f"{path} does not exist")

    times_s: List[float] = []
    gyro: List[Tuple[float, float, float]] = []
    accel: List[Tuple[float, float, float]] = []
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            times_s.append(int(row["ts_ticks"]) / TICKS_PER_SECOND)
            gyro.append(
                tuple(float(row[f"g{axis}_mdps"]) / 1000.0 for axis in "xyz")  # type: ignore[arg-type]
            )
            accel.append(
                tuple(float(row[f"a{axis}_mg"]) for axis in "xyz")  # type: ignore[arg-type]
            )
    if not times_s:
        raise RecordingError(f"{csv_path} holds no samples")

    holds: List[Hold] = []
    spans: List[Tuple[int, int, str]] = []
    for entry in json.loads(holds_path.read_text())["holds"]:
        start_s, end_s = float(entry["t_start_s"]), float(entry["t_end_s"])
        indices = [i for i, t in enumerate(times_s) if start_s <= t <= end_s]
        if not indices:
            continue
        start, end = indices[0], indices[-1] + 1
        spans.append((start, end, entry["label"]))
        holds.append(
            Hold(
                label=entry["label"],
                duration_s=end_s - start_s,
                samples=end - start,
                gyro_bias_dps=_mean(gyro[start:end]),
                accel_mg=_mean(accel[start:end]),
            )
        )

    recording = Recording(times_s, gyro, accel, holds, sample_rate_hz, directory)
    object.__setattr__(recording, "_spans", spans)
    return recording


def identify(
    recording: Recording,
    yaw_axis: int = 0,
    baseline: ImuErrorModel = RIG_ERROR_MODEL,
    min_hold_s: float = MIN_HOLD_S,
) -> ImuErrorModel:
    """Fit the constants a stationary recording can settle.

    ``yaw_axis`` is the IMU axis that points up on this rig, since that is the
    one yaw integrates around and therefore the one whose noise reaches heading.
    It defaults to X, matching ``DEFAULT_R_RIG_IMU`` on the delivered rig.

    ``tilt_sigma`` is carried over from the baseline untouched. Nothing a still
    rig records can separate its own tilt error from the tilt of whatever it is
    resting on, and quietly marking it measured would be the one lie this module
    could tell that nobody would catch.
    """
    if not 0 <= yaw_axis <= 2:
        raise PropagationError(f"yaw_axis must be 0, 1 or 2, got {yaw_axis}")

    hold = recording.longest_hold()
    start, end = recording.window(hold)
    axis_rates = [sample[yaw_axis] for sample in recording.gyro_dps[start:end]]
    arw = angle_random_walk_dps_sqrt_s(axis_rates, recording.sample_rate_hz)

    usable = [h for h in recording.holds if h.duration_s >= min_hold_s]
    if len(usable) < 2:
        raise RecordingError(
            f"need 2 holds of at least {min_hold_s:g} s to estimate bias spread, "
            f"got {len(usable)} of {len(recording.holds)}"
        )
    bias_spread = _stdev([h.gyro_bias_dps[yaw_axis] for h in usable])

    worst = max(recording.holds, key=lambda h: abs(h.norm_error_mg))

    return ImuErrorModel(
        tilt_sigma_deg=baseline.tilt_sigma_deg,
        accel_bias_mg=NoiseTerm(
            "accel_bias",
            abs(worst.norm_error_mg),
            "mg",
            MEASURED,
            f"largest |a| deviation from {GRAVITY_MG:g} mg over "
            f"{len(recording.holds)} holds, worst {worst.label}",
        ),
        accel_sigma_mg=baseline.accel_sigma_mg,
        gyro_bias_sigma_dps=NoiseTerm(
            "gyro_bias_sigma",
            bias_spread,
            "deg/s",
            MEASURED,
            f"spread of per-hold bias on axis {'xyz'[yaw_axis]} over "
            f"{len(usable)} holds of at least {min_hold_s:g} s; an upper bound, "
            "since the record is too short to reach the Allan floor",
        ),
        gyro_arw_dps_sqrt_s=NoiseTerm(
            "gyro_arw",
            arw,
            "deg/sqrt(s)",
            MEASURED,
            f"Allan deviation at tau={ARW_TAU_S:g} s on {hold.label}, "
            f"{hold.duration_s:.1f} s",
        ),
    )


def identification_checks(
    recording: Recording,
    fitted: ImuErrorModel,
    baseline: ImuErrorModel = RIG_ERROR_MODEL,
) -> List[Check]:
    """Whether the recording supports the constants fitted from it."""
    hold = recording.longest_hold()
    checks = [
        Check.in_range(
            "identification.arw_supported",
            hold.duration_s,
            10.0 * ARW_TAU_S,
            float("inf"),
            unit="s",
            detail=(
                f"{hold.label} runs {hold.duration_s:.1f} s, at least ten times the "
                f"tau={ARW_TAU_S:g} s the random walk is read at"
            ),
        ),
        Check.in_range(
            "identification.bias_floor_supported",
            hold.duration_s,
            BIAS_FLOOR_RECORD_S,
            float("inf"),
            unit="s",
            detail=(
                f"{hold.label} runs {hold.duration_s:.1f} s; the Allan floor that "
                f"sets bias instability sits past tau={BIAS_FLOOR_TAU_S:g} s and "
                f"needs about {BIAS_FLOOR_RECORD_S:g} s to average, so "
                f"gyro_bias_sigma is an upper bound rather than a measurement"
            ),
        ),
        Check.that(
            "identification.tilt_still_assumed",
            fitted.tilt_sigma_deg.is_assumed,
            "tilt_sigma is carried over as assumed, as a still rig cannot "
            "separate its own tilt error from the surface it rests on",
        ),
    ]
    for fitted_term, baseline_term in zip(fitted.terms(), baseline.terms()):
        if fitted_term.provenance != MEASURED or baseline_term.value == 0:
            continue
        ratio = fitted_term.value / baseline_term.value
        checks.append(
            Check.that(
                f"identification.{fitted_term.name}_moved",
                True,
                f"{fitted_term.name}: {baseline_term.value:g} [{baseline_term.provenance}] "
                f"-> {fitted_term.value:.4g} [measured], {ratio:.2f}x",
            )
        )
    return checks


def _mean(rows: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    count = len(rows)
    if not count:
        raise RecordingError("cannot average an empty window")
    return tuple(sum(row[axis] for row in rows) / count for axis in range(3))  # type: ignore[return-value]


def _stdev(values: Sequence[float]) -> float:
    count = len(values)
    if count < 2:
        raise RecordingError(f"need at least 2 values for a spread, got {count}")
    mean = sum(values) / count
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (count - 1))
