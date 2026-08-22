#!/usr/bin/env python3
"""How fast a fix decays once the map stops correcting it.

The project's headline numbers — the rig holds ten seconds before it is a metre
out, thirty-five metres after a minute — were estimates carried in prose. This
module computes them instead, from named sensor constants, and reports which
term produced the answer. That last part is the point: "35 m after a minute" is
not actionable, while "35 m, of which 34 is gravity leaking through a tilt error
you have never measured" says exactly what to go and fix.

**Two regimes, and the gap between them is the argument for radar.**

*Inertial only.* Position comes from integrating acceleration twice, so every
error is multiplied by t squared. The dominant term is not the accelerometer's
noise, which averages away at 200 Hz, but its *tilt error*: the estimated
vertical is off by some small angle, so a slice of gravity is mistaken for
horizontal acceleration. A fifth of a degree of tilt leaks 3.5 mg of phantom
acceleration -- 0.034 m/s^2 -- and integrating that twice over a minute is
62 metres.

*Speed aided.* With radar ego-velocity there is no acceleration to integrate,
only a heading to be wrong about. Position error becomes speed times the
accumulated heading error, which grows as t squared but with a constant three
orders of magnitude smaller. This is the whole case for ``radar_ego_velocity()``
and it is quantified in ``DriftBudget``.

**Heading uncertainty is deliberately the rig's own formula.** The growth law
here reproduces ``yael_api.imu.AttitudeEstimator``: variance from a residual
gyro bias grows as (b*t)^2, angle random walk grows as A^2*t, and neither
accrues while the rig is static because a yaw that is not being integrated
cannot drift. Matching the producer matters more than improving on it. Where
this module disagrees with the number the rig reports, one of them is wrong, and
that is a question worth surfacing rather than an inconsistency worth hiding.

**Assumed constants are labelled.** Every constant carries where its value came
from, and ``ImuErrorModel.assumed_terms()`` lists the ones nobody has measured.
A drift budget resting on an assumed constant is a prediction, not a
measurement, and ``checks()`` says so out loud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..check import Check

#: Standard gravity, m/s^2. The constant that turns a tilt error into an
#: acceleration error, and therefore the one that sets the inertial horizon.
STANDARD_GRAVITY = 9.80665

#: Sample interval of the IMU, seconds. 200 Hz as measured on the rig over 190
#: seconds with zero dropped samples. Only white noise cares about this.
IMU_SAMPLE_INTERVAL_S = 1.0 / 200.0

#: Longest horizon any of these functions will search, seconds. Beyond an hour
#: the quadratic terms are meaningless anyway and a bisection needs a bound.
MAX_HORIZON_S = 3600.0


class PropagationError(RuntimeError):
    """Raised when a drift budget is asked for something it cannot answer."""


# ---------------------------------------------------------------------------
# constants, and where they came from
# ---------------------------------------------------------------------------


#: Provenance values, ordered from strongest to weakest evidence.
MEASURED = "measured"
DATASHEET = "datasheet"
ASSUMED = "assumed"


@dataclass(frozen=True)
class NoiseTerm:
    """One sensor constant, carrying where its value came from.

    A drift budget is only as good as its weakest constant, and the weakest
    constant is invisible once every number is a bare float. Keeping provenance
    attached means a report can say which of its inputs is a guess.
    """

    name: str
    value: float
    unit: str
    provenance: str
    detail: str = ""

    @property
    def is_assumed(self) -> bool:
        return self.provenance == ASSUMED

    def __str__(self) -> str:
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{self.name} = {self.value:g} {self.unit} [{self.provenance}]{suffix}"


@dataclass(frozen=True)
class ImuErrorModel:
    """The sensor constants that decide how fast an unaided fix decays."""

    #: Angle between the estimated vertical and the true one, degrees, one
    #: sigma. The single most important number here, and the one nobody has
    #: measured: it needs a levelled reference, not a still rig.
    tilt_sigma_deg: NoiseTerm
    #: Residual accelerometer bias after whatever compensation is applied,
    #: milli-g. Enters position the same way tilt does, one integration later.
    accel_bias_mg: NoiseTerm
    #: Per-sample accelerometer noise, milli-g, one sigma.
    accel_sigma_mg: NoiseTerm
    #: Residual gyro bias after a static re-estimate, deg/s, one sigma.
    gyro_bias_sigma_dps: NoiseTerm
    #: Angle random walk, deg/sqrt(s).
    gyro_arw_dps_sqrt_s: NoiseTerm

    def terms(self) -> List[NoiseTerm]:
        return [
            self.tilt_sigma_deg,
            self.accel_bias_mg,
            self.accel_sigma_mg,
            self.gyro_bias_sigma_dps,
            self.gyro_arw_dps_sqrt_s,
        ]

    def assumed_terms(self, regime: Optional[str] = None) -> List[NoiseTerm]:
        """The constants resting on nothing measured.

        Filtered by regime when one is named, because listing a gyro constant
        against an inertial-only budget that never consulted it invites the
        reader to go and measure the wrong thing.
        """
        used = {
            "inertial-only": (
                self.tilt_sigma_deg, self.accel_bias_mg, self.accel_sigma_mg,
            ),
            "speed-aided": (
                self.gyro_bias_sigma_dps, self.gyro_arw_dps_sqrt_s,
            ),
        }.get(regime, tuple(self.terms()))
        return [term for term in used if term.is_assumed]

    # -- heading ----------------------------------------------------------

    def heading_sigma_deg(self, moving_s: float) -> float:
        """Heading uncertainty after this much *moving* time since the last reset.

        Moving time, not elapsed time, because yaw is integrated only while the
        rig moves. A rig parked for an hour has not accumulated an hour of
        drift; it has accumulated none, having integrated nothing.

        Reproduces the rig's own formula so the two agree by construction.
        """
        if moving_s < 0:
            raise PropagationError(f"moving time cannot be negative, got {moving_s}")
        bias = self.gyro_bias_sigma_dps.value * moving_s
        walk = self.gyro_arw_dps_sqrt_s.value ** 2 * moving_s
        return math.sqrt(bias * bias + walk)

    def cross_track_terms_m(
        self,
        speed_mps: float,
        moving_s: float,
        initial_heading_sigma_deg: float = 0.0,
    ) -> List[Tuple[str, float, str]]:
        """Cross-track displacement error, one entry per source of heading error.

        Split by source because the three grow at different powers of time and
        want different fixes. A heading error persists rather than resampling,
        so it does not merely add error, it steers: every second spent on the
        wrong bearing adds its own displacement.

        * a *fixed* heading offset -- what the initial fix got wrong -- displaces
          linearly, v * sigma * t
        * a *constant* gyro bias integrates into a heading that grows linearly,
          so displacement grows as v * sigma_b * t^2 / 2
        * *angle random walk* is a Wiener process; integrating it once more gives
          variance v^2 A^2 t^3 / 3, hence v * A * t^1.5 / sqrt(3)

        The random-walk coefficient is exact, not the square root of a variance
        integral. Integrating the running sigma instead -- treating each
        instant's heading error as if it were the same draw -- *understates*
        this term by 13 percent, which is the dangerous direction: it is an
        optimistic uncertainty, and an optimistic uncertainty is the failure
        this package exists to prevent.
        """
        if moving_s <= 0 or speed_mps <= 0:
            return []
        return [
            (
                "initial_heading",
                speed_mps * math.radians(initial_heading_sigma_deg) * moving_s,
                f"{initial_heading_sigma_deg:g} deg of heading error at the fix",
            ),
            (
                "gyro_bias",
                speed_mps
                * math.radians(self.gyro_bias_sigma_dps.value)
                * moving_s ** 2 / 2.0,
                f"{self.gyro_bias_sigma_dps.value:g} deg/s of residual bias, "
                f"integrating into heading",
            ),
            (
                "angle_random_walk",
                speed_mps
                * math.radians(self.gyro_arw_dps_sqrt_s.value)
                * moving_s ** 1.5 / math.sqrt(3.0),
                f"{self.gyro_arw_dps_sqrt_s.value:g} deg/sqrt(s) of random walk",
            ),
        ]

    # -- derived ----------------------------------------------------------

    @property
    def tilt_leakage_mps2(self) -> float:
        """Phantom horizontal acceleration produced by the tilt error."""
        return STANDARD_GRAVITY * math.sin(math.radians(self.tilt_sigma_deg.value))

    @property
    def accel_bias_mps2(self) -> float:
        return self.accel_bias_mg.value / 1000.0 * STANDARD_GRAVITY

    @property
    def velocity_random_walk_mps_sqrt_s(self) -> float:
        """Velocity random walk from per-sample accelerometer noise.

        White noise integrated once is a random walk whose sigma grows as the
        square root of time, with coefficient sigma_a * sqrt(dt).
        """
        sigma_mps2 = self.accel_sigma_mg.value / 1000.0 * STANDARD_GRAVITY
        return sigma_mps2 * math.sqrt(IMU_SAMPLE_INTERVAL_S)


#: The rig as it stands on 21 August 2026.
#:
#: Two of the five constants are measured, three are not, and the unmeasured
#: ones are the ones that dominate. That is the honest state of this model.
RIG_ERROR_MODEL = ImuErrorModel(
    tilt_sigma_deg=NoiseTerm(
        "tilt_sigma", 0.2, "deg", ASSUMED,
        "needs a levelled reference; a still rig measures noise, not tilt bias",
    ),
    accel_bias_mg=NoiseTerm(
        "accel_bias", 4.0, "mg", ASSUMED,
        "|a| reads 1004 mg at rest, so 4 mg is scale or bias, undetermined which",
    ),
    accel_sigma_mg=NoiseTerm(
        "accel_sigma", 3.6, "mg", MEASURED,
        "5977 stationary samples on this board",
    ),
    gyro_bias_sigma_dps=NoiseTerm(
        "gyro_bias_sigma", 0.05, "deg/s", ASSUMED,
        "assumed post-static-re-estimate value in yael_api.imu",
    ),
    gyro_arw_dps_sqrt_s=NoiseTerm(
        "gyro_arw", 0.15, "deg/sqrt(s)", ASSUMED,
        "conservative placeholder in yael_api.imu; 200 Hz data would settle it",
    ),
)


@dataclass(frozen=True)
class SpeedAiding:
    """Speed from an external source, and how well it is known.

    Radar ego-velocity is the intended supplier. Its presence changes the
    regime rather than merely improving it: position stops being a double
    integral of acceleration and becomes a single integral of a known speed
    along an uncertain heading.
    """

    speed_mps: float
    sigma_speed_mps: float
    source: str = "radar_ego_velocity"

    def __post_init__(self) -> None:
        if self.speed_mps < 0:
            raise PropagationError(f"speed cannot be negative, got {self.speed_mps}")
        if self.sigma_speed_mps < 0:
            raise PropagationError("speed sigma cannot be negative")


# ---------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftTerm:
    """One contributor to position error, in metres at the stated horizon."""

    name: str
    metres: float
    detail: str

    def __str__(self) -> str:
        return f"{self.name:<30} {self.metres:8.2f} m   {self.detail}"


@dataclass(frozen=True)
class DriftBudget:
    """Position error at one horizon, decomposed by where it came from."""

    horizon_s: float
    regime: str
    terms: List[DriftTerm] = field(default_factory=list)
    #: Constants in the model that nobody has measured.
    assumed: List[NoiseTerm] = field(default_factory=list)

    @property
    def total_m(self) -> float:
        """Root-sum-square of the terms, which are independent error sources."""
        return math.sqrt(sum(term.metres ** 2 for term in self.terms))

    @property
    def dominant(self) -> Optional[DriftTerm]:
        """The term to attack first, or None when there is nothing to report."""
        return max(self.terms, key=lambda term: term.metres) if self.terms else None

    @property
    def rests_on_assumption(self) -> bool:
        return bool(self.assumed)

    def __str__(self) -> str:
        head = (
            f"{self.regime} drift at {self.horizon_s:g} s: "
            f"{self.total_m:.2f} m total"
        )
        lines = [head] + [f"  {term}" for term in sorted(
            self.terms, key=lambda t: -t.metres
        )]
        if self.assumed:
            names = ", ".join(term.name for term in self.assumed)
            lines.append(f"  rests on assumed constants: {names}")
        return "\n".join(lines)


class DeadReckoner:
    """Predicts how a fix decays, and how long it survives a stated limit."""

    def __init__(
        self,
        model: ImuErrorModel = RIG_ERROR_MODEL,
        aiding: Optional[SpeedAiding] = None,
        initial_heading_sigma_deg: float = 0.0,
    ) -> None:
        self.model = model
        self.aiding = aiding
        #: Heading uncertainty already present at the fix, degrees. An initial
        #: fix from a manual pin carries several degrees of it before the gyro
        #: has drifted at all, and it is the term that dominates a short run.
        self.initial_heading_sigma_deg = initial_heading_sigma_deg

    @property
    def regime(self) -> str:
        return "speed-aided" if self.aiding is not None else "inertial-only"

    def heading_sigma_deg(self, moving_s: float) -> float:
        """Total heading uncertainty: what the fix started with, plus drift."""
        drift = self.model.heading_sigma_deg(moving_s)
        return math.hypot(self.initial_heading_sigma_deg, drift)

    def budget_at(self, horizon_s: float) -> DriftBudget:
        """Decompose position error at this horizon, assuming continuous motion."""
        if horizon_s < 0:
            raise PropagationError(f"horizon cannot be negative, got {horizon_s}")
        terms = (
            self._aided_terms(horizon_s)
            if self.aiding is not None
            else self._inertial_terms(horizon_s)
        )
        return DriftBudget(
            horizon_s=horizon_s,
            regime=self.regime,
            terms=terms,
            assumed=self.model.assumed_terms(self.regime),
        )

    def _inertial_terms(self, t: float) -> List[DriftTerm]:
        """Double integration: everything systematic is multiplied by t squared."""
        half_t_squared = 0.5 * t * t
        walk = self.model.velocity_random_walk_mps_sqrt_s
        return [
            DriftTerm(
                "tilt_leakage",
                self.model.tilt_leakage_mps2 * half_t_squared,
                f"{self.model.tilt_sigma_deg.value:g} deg of tilt reads as "
                f"{self.model.tilt_leakage_mps2 * 1000 / STANDARD_GRAVITY:.1f} mg",
            ),
            DriftTerm(
                "accel_bias",
                self.model.accel_bias_mps2 * half_t_squared,
                f"{self.model.accel_bias_mg.value:g} mg of residual bias",
            ),
            DriftTerm(
                # A random walk integrated once more: sigma grows as t^1.5,
                # divided by sqrt(3) from integrating a Wiener process.
                "accel_noise",
                walk * (t ** 1.5) / math.sqrt(3.0),
                f"white noise at {1 / IMU_SAMPLE_INTERVAL_S:.0f} Hz, averages down",
            ),
        ]

    def _aided_terms(self, t: float) -> List[DriftTerm]:
        """Single integration: speed is known, only the direction is uncertain."""
        aiding = self.aiding
        assert aiding is not None  # guarded by the caller
        terms = [
            DriftTerm(f"cross_track.{name}", metres, detail)
            for name, metres, detail in self.model.cross_track_terms_m(
                aiding.speed_mps, t, self.initial_heading_sigma_deg
            )
        ]
        terms.append(
            DriftTerm(
                "along_track.speed",
                aiding.sigma_speed_mps * t,
                f"{aiding.sigma_speed_mps:g} m/s of speed error from {aiding.source}",
            )
        )
        return terms

    def horizon_for(self, limit_m: float) -> float:
        """How long until position error reaches this limit, seconds.

        Returns ``MAX_HORIZON_S`` when the limit is not reached inside the
        search bound rather than raising: "longer than an hour" is a legitimate
        answer for a well-aided rig, and an exception would make the caller
        treat good news as an error.
        """
        if limit_m <= 0:
            raise PropagationError(f"limit must be positive, got {limit_m}")
        if self.budget_at(MAX_HORIZON_S).total_m < limit_m:
            return MAX_HORIZON_S

        # Total error is monotonic in time in both regimes, so bisection is
        # exact to whatever tolerance it is run to.
        low, high = 0.0, MAX_HORIZON_S
        for _ in range(80):
            middle = 0.5 * (low + high)
            if self.budget_at(middle).total_m < limit_m:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    def checks(self, limit_m: float = 1.0, required_horizon_s: float = 10.0) -> List[Check]:
        """Whether the rig holds the stated limit for the stated time."""
        horizon = self.horizon_for(limit_m)
        budget = self.budget_at(required_horizon_s)
        dominant = budget.dominant
        results = [
            Check.in_range(
                "propagation.horizon",
                horizon,
                required_horizon_s,
                MAX_HORIZON_S,
                unit="s",
                detail=(
                    f"{self.regime}: {limit_m:g} m reached after {horizon:.1f} s, "
                    f"against a {required_horizon_s:g} s requirement"
                ),
            )
        ]
        if dominant is not None:
            results.append(
                Check.that(
                    "propagation.dominant_term",
                    True,
                    f"at {required_horizon_s:g} s the budget is led by "
                    f"{dominant.name} at {dominant.metres:.2f} m of "
                    f"{budget.total_m:.2f} m total",
                )
            )
        # Not a failure: a model resting on assumed constants is usable, it just
        # must not be quoted as if it had been measured.
        results.append(
            Check.that(
                "propagation.constants_measured",
                not budget.rests_on_assumption,
                "every constant is measured" if not budget.rests_on_assumption
                else "prediction, not measurement -- assumed constants: "
                + ", ".join(str(term) for term in budget.assumed),
            )
        )
        return results


# ---------------------------------------------------------------------------
# propagating an actual pose
# ---------------------------------------------------------------------------


def _matmul(left, right):
    """Multiply two small square matrices held as tuples of tuples."""
    size = len(left)
    return tuple(
        tuple(
            sum(left[row][k] * right[k][column] for k in range(size))
            for column in range(size)
        )
        for row in range(size)
    )


def _transpose(matrix):
    return tuple(zip(*matrix))


@dataclass(frozen=True)
class NavState:
    """A pose on the local tangent plane, with the covariance that qualifies it.

    East and north in metres from the fix rather than latitude and longitude:
    over the tens of metres this survives, the tangent plane is exact to well
    under the noise, and a metric frame keeps the covariance in one unit.

    **The gyro bias is part of the state.** It is never estimated here -- with
    no measurement to estimate it from, its mean stays where the fix left it --
    but its *uncertainty* has to be carried, because it is the one heading error
    that does not resample.

    A filter that injects it as process noise on heading instead is not merely
    a little optimistic, it is optimistic by a factor that grows without bound.
    Process noise makes heading variance grow as b^2 * t, so cross-track error
    grows as t^1.5; a bias that is genuinely constant makes heading grow as b*t
    and cross-track as t^2. The ratio between them is 2/sqrt(3t), measured here
    at 0.37 after ten seconds and 0.15 after sixty -- so after a minute at 5 m/s
    the naive filter reports 1.2 m where the honest answer is 7.8 m. Correlation
    lost is optimism gained, and an optimistic sigma is the exact failure this
    project exists to prevent.

    Carrying the bias also makes the heading variance come out as
    sigma_b^2 * t^2 + A^2 * t on its own, which is the rig's own formula,
    reproduced rather than asserted.

    ``moving_s`` is carried because heading uncertainty is a function of moving
    time, not of elapsed time, and nothing else in the state can reconstruct it.
    """

    east_m: float
    north_m: float
    heading_deg: float
    #: 4x4 covariance over (east, north, heading, gyro bias), in m^2, deg^2 and
    #: (deg/s)^2. The bias row is what keeps the cross-track term honest.
    covariance: Tuple[Tuple[float, ...], ...]
    moving_s: float = 0.0
    elapsed_s: float = 0.0

    #: Index of each state in the covariance, named so the blocks stay readable.
    EAST, NORTH, HEADING, BIAS = 0, 1, 2, 3

    @classmethod
    def at_fix(
        cls,
        heading_deg: float,
        sigma_position_m: float,
        sigma_heading_deg: float,
        model: "ImuErrorModel" = None,
    ) -> "NavState":
        """The state at the moment of a fix, before any propagation.

        The gyro bias starts at the model's own bias uncertainty, since that is
        precisely what "we re-estimated the bias while static" leaves behind.
        """
        position_variance = sigma_position_m ** 2
        bias_sigma = (model or RIG_ERROR_MODEL).gyro_bias_sigma_dps.value
        zero = [0.0, 0.0, 0.0, 0.0]
        covariance = [list(zero) for _ in range(4)]
        covariance[cls.EAST][cls.EAST] = position_variance
        covariance[cls.NORTH][cls.NORTH] = position_variance
        covariance[cls.HEADING][cls.HEADING] = sigma_heading_deg ** 2
        covariance[cls.BIAS][cls.BIAS] = bias_sigma ** 2
        return cls(
            east_m=0.0,
            north_m=0.0,
            heading_deg=heading_deg,
            covariance=tuple(tuple(row) for row in covariance),
        )

    @property
    def sigma_east_m(self) -> float:
        return math.sqrt(max(0.0, self.covariance[self.EAST][self.EAST]))

    @property
    def sigma_north_m(self) -> float:
        return math.sqrt(max(0.0, self.covariance[self.NORTH][self.NORTH]))

    @property
    def sigma_heading_deg(self) -> float:
        return math.sqrt(max(0.0, self.covariance[self.HEADING][self.HEADING]))

    @property
    def sigma_gyro_bias_dps(self) -> float:
        return math.sqrt(max(0.0, self.covariance[self.BIAS][self.BIAS]))

    @property
    def sigma_horizontal_m(self) -> float:
        """One number for horizontal uncertainty: the RSS of the two axes."""
        return math.hypot(self.sigma_east_m, self.sigma_north_m)

    def __str__(self) -> str:
        return (
            f"{self.east_m:+.2f} E, {self.north_m:+.2f} N m "
            f"(+/-{self.sigma_horizontal_m:.2f} m), "
            f"heading {self.heading_deg:.2f} +/-{self.sigma_heading_deg:.2f} deg, "
            f"{self.elapsed_s:.1f} s since fix, {self.moving_s:.1f} s moving"
        )


def advance(
    state: NavState,
    dt_s: float,
    speed_mps: float,
    yaw_rate_dps: float,
    model: ImuErrorModel = RIG_ERROR_MODEL,
    sigma_speed_mps: float = 0.0,
    is_static: bool = False,
) -> NavState:
    """Propagate a state forward one step, carrying its covariance with it.

    Standard covariance propagation, P <- F P F' + Q, over the four states. The
    only entries worth reading closely are the two that couple heading into
    position -- they are why a heading error becomes a position error -- and the
    one that couples the gyro bias into heading, which is why that error keeps
    pointing the same way instead of averaging out.

    Position advances along the heading at the start of the step, plain Euler.
    At 200 Hz the difference from a midpoint rule is far below the covariance
    this is carrying, and the simpler form is the one that can be read against
    the Jacobian.

    ``is_static`` freezes heading, its uncertainty, and the speed error, since
    a yaw that is not integrated cannot drift and a speed that is not integrated
    cannot displace. It does not *reduce* position uncertainty: the rig may be
    still, but a position error already made does not heal by standing there.
    """
    if dt_s < 0:
        raise PropagationError(f"time step cannot be negative, got {dt_s}")

    east_i, north_i, heading_i, bias_i = (
        NavState.EAST, NavState.NORTH, NavState.HEADING, NavState.BIAS
    )
    moving_s = state.moving_s + (0.0 if is_static else dt_s)
    heading_rad = math.radians(state.heading_deg)
    travelled = 0.0 if is_static else speed_mps * dt_s

    # Jacobian. Heading is held in degrees, so the position derivatives carry
    # the degree-to-radian factor with them.
    per_degree = math.pi / 180.0
    d_east_d_heading = travelled * math.cos(heading_rad) * per_degree
    d_north_d_heading = -travelled * math.sin(heading_rad) * per_degree

    transition = [[1.0 if row == column else 0.0 for column in range(4)] for row in range(4)]
    transition[east_i][heading_i] = d_east_d_heading
    transition[north_i][heading_i] = d_north_d_heading
    # An unknown bias b makes the integrated heading wrong by -b*dt every step,
    # and it is the same b every step. This entry is the whole point of the
    # fourth state.
    transition[heading_i][bias_i] = 0.0 if is_static else -dt_s
    transition = tuple(tuple(row) for row in transition)

    covariance = _matmul(_matmul(transition, state.covariance), _transpose(transition))
    covariance = [list(row) for row in covariance]

    if not is_static:
        # Angle random walk: white gyro noise integrates to variance A^2 per
        # second of heading. Nothing is added for the bias, which is constant
        # over any horizon this propagates across.
        covariance[heading_i][heading_i] += (
            model.gyro_arw_dps_sqrt_s.value ** 2 * dt_s
        )
        # Speed error is a bias too, not white noise: radar ego-velocity errs
        # through scale and mounting alignment, which do not resample between
        # frames. A bias displaces by sigma_v * t, so its variance is
        # sigma_v^2 * t^2 and one step adds the difference of two squares.
        along_track = sigma_speed_mps ** 2 * (
            2.0 * state.moving_s * dt_s + dt_s * dt_s
        )
        covariance[east_i][east_i] += along_track * math.sin(heading_rad) ** 2
        covariance[north_i][north_i] += along_track * math.cos(heading_rad) ** 2
        cross = along_track * math.sin(heading_rad) * math.cos(heading_rad)
        covariance[east_i][north_i] += cross
        covariance[north_i][east_i] += cross

    return NavState(
        east_m=state.east_m + travelled * math.sin(heading_rad),
        north_m=state.north_m + travelled * math.cos(heading_rad),
        heading_deg=state.heading_deg + (0.0 if is_static else yaw_rate_dps * dt_s),
        covariance=tuple(tuple(row) for row in covariance),
        moving_s=moving_s,
        elapsed_s=state.elapsed_s + dt_s,
    )


def propagate(
    state: NavState,
    samples: Sequence[Tuple[float, float, float]],
    model: ImuErrorModel = RIG_ERROR_MODEL,
    sigma_speed_mps: float = 0.0,
) -> NavState:
    """Run a state through a sequence of ``(dt_s, speed_mps, yaw_rate_dps)``."""
    for dt_s, speed_mps, yaw_rate_dps in samples:
        state = advance(
            state, dt_s, speed_mps, yaw_rate_dps,
            model=model, sigma_speed_mps=sigma_speed_mps,
        )
    return state
