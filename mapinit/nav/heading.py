#!/usr/bin/env python3
"""Recover heading by matching an image against what the map predicts.

Heading has no other source. Roll and pitch come from gravity and do not drift;
position has GNSS, a manual pin, and eventually radar against the map. Heading
has a gyro that drifts and an initial value someone typed in. This module is the
only thing in the system that can correct it, which is why it is worth building
before the sensor that would make it easier arrives.

**Why heading alone, and why now.** The full problem is three-dimensional --
latitude, longitude and heading against the map -- and its natural observation
is radar returns from walls, which is blocked on ``radar_detections_all()``
reaching us in the field. Heading on its own is not blocked: it needs one
thermal frame and a position good to a few metres, both of which exist today.
It is also the term that matters most, because a heading error is multiplied by
range while a position error is not: at 20 m, one degree of heading is 35 cm of
error on every predicted wall in the frame.

**What is matched.** Building corners, as bearings. A corner is a vertical line
in the world, it projects to a vertical line in the image, and where it falls
across the frame depends only on the footprint and the pose -- not on the
building's height, which the map mostly does not know. The observation is
therefore a list of relative bearings, and the prediction is the same list from
``ViewPredictor``, so matching is a one-dimensional alignment problem.

**Scoring is soft, not nearest-neighbour.** Every observed bearing votes for
every predicted one, weighted by a Gaussian in the angular difference. Hard
correspondence would need a decision per edge before the heading is known, which
is the wrong way round; the soft version lets a wrong correspondence be
outvoted instead of committed to.

**Ambiguity is reported, not resolved.** A street of similar facades produces a
score curve with several near-equal peaks, and the right output there is not the
tallest peak but the statement that the peaks are not separable. A confident
wrong heading is worse than an admitted unknown, because everything downstream
multiplies it by range. ``HeadingFix.ambiguous`` is the flag, and it is set
whenever the runner-up peak is within a stated fraction of the best.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..check import Check

#: Angular tolerance of the match kernel, degrees, one sigma. Set from what the
#: sensor can resolve: 53.4 degrees across 160 columns is 0.33 deg/px, and a
#: corner is not localised better than a couple of pixels in a 160-wide thermal
#: frame. Widening it past the real localisation error blurs distinct peaks
#: together and hides ambiguity rather than reporting it.
DEFAULT_MATCH_SIGMA_DEG = 0.75

#: Step of the heading search, degrees. Finer than the match kernel, so the peak
#: is never straddled, and coarse enough that a full circle stays cheap.
DEFAULT_SEARCH_STEP_DEG = 0.25

#: How far from the best peak another peak must be before it counts as a rival
#: rather than as the same peak's shoulder.
PEAK_LOCKOUT_DEG = 5.0

#: Above this ratio of runner-up score to best score, the two are not separable.
AMBIGUITY_RATIO = 0.85

#: Below this score the best peak is not a match at all, it is the best of a bad
#: set. A score is a mean over observations, so 0.3 means the typical observed
#: edge sits more than one kernel sigma from anything predicted.
MIN_ACCEPTABLE_SCORE = 0.30


class HeadingMatchError(RuntimeError):
    """Raised when a heading match is asked for something it cannot answer."""


def bearings_from_columns(
    columns: Sequence[float],
    fx: float,
    cx: float,
) -> List[float]:
    """Convert pixel columns to bearings relative to boresight, degrees.

    Uses the measured focal length and principal point rather than a field of
    view, because the two agree only for an ideal lens. Positive is to the
    right of boresight, matching the rig's azimuth convention.

    This assumes the columns have already had lens distortion removed. On this
    rig that is not automatic: the thermal frames arrive raw, and the measured
    radial coefficients are large enough to matter -- k1 = -0.366 moves a corner
    pixel roughly 10 columns, which is over 3 degrees. Feeding raw columns in
    here produces a confident and wrong heading.
    """
    if fx <= 0:
        raise HeadingMatchError(f"focal length must be positive, got {fx}")
    return [math.degrees(math.atan2(float(column) - cx, fx)) for column in columns]


def angular_difference_deg(first: float, second: float) -> float:
    """Signed difference between two bearings, wrapped to [-180, 180)."""
    return (first - second + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class HeadingCandidate:
    """One heading the search evaluated, and how well it explained the image."""

    heading_deg: float
    #: Mean over observations of the best Gaussian vote each one received, so
    #: the score is comparable between frames with different edge counts.
    score: float
    #: Observations whose best predicted match is inside the kernel's sigma.
    inliers: int
    #: Predicted edges in frame at this heading, whether or not they matched.
    predicted_in_frame: int


@dataclass(frozen=True)
class HeadingFix:
    """A heading recovered from one frame, with the caveats that qualify it."""

    heading_deg: float
    sigma_deg: float
    score: float
    inliers: int
    observed: int
    predicted_in_frame: int
    #: The next-best peak outside the lockout window, if the search found one.
    runner_up_deg: Optional[float] = None
    runner_up_score: Optional[float] = None
    ambiguous: bool = False
    #: Every peak above the acceptance floor, best first. A caller carrying a
    #: prior can pick from these instead of taking the fix's own choice.
    peaks: List[HeadingCandidate] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """Whether this fix should be used to correct a heading at all."""
        return (
            not self.ambiguous
            and self.score >= MIN_ACCEPTABLE_SCORE
            and self.inliers >= 2
        )

    def checks(self) -> List[Check]:
        """The fix's own validation, in the same form as every other guard."""
        results = [
            Check.in_range(
                "heading_match.score",
                self.score,
                MIN_ACCEPTABLE_SCORE,
                1.0,
                detail=(
                    f"{self.inliers} of {self.observed} observed edges matched "
                    f"within the kernel, against {self.predicted_in_frame} "
                    f"predicted in frame"
                ),
            ),
            Check.that(
                "heading_match.unambiguous",
                not self.ambiguous,
                f"heading {self.heading_deg:.2f} deg is separable from the rest"
                if not self.ambiguous
                else (
                    f"heading {self.heading_deg:.2f} deg (score {self.score:.3f}) "
                    f"is not separable from {self.runner_up_deg:.2f} deg "
                    f"(score {self.runner_up_score:.3f}); the scene repeats"
                ),
            ),
        ]
        return results

    def __str__(self) -> str:
        verdict = "accepted" if self.accepted else "rejected"
        line = (
            f"heading {self.heading_deg:.2f} +/-{self.sigma_deg:.2f} deg "
            f"[{verdict}] score {self.score:.3f}, "
            f"{self.inliers}/{self.observed} edges matched"
        )
        if self.ambiguous and self.runner_up_deg is not None:
            line += f", rival peak at {self.runner_up_deg:.2f} deg"
        return line


class HeadingMatcher:
    """Searches heading against a map prediction from a known position."""

    def __init__(
        self,
        match_sigma_deg: float = DEFAULT_MATCH_SIGMA_DEG,
        search_step_deg: float = DEFAULT_SEARCH_STEP_DEG,
        ambiguity_ratio: float = AMBIGUITY_RATIO,
    ) -> None:
        if match_sigma_deg <= 0:
            raise HeadingMatchError("match sigma must be positive")
        if search_step_deg <= 0:
            raise HeadingMatchError("search step must be positive")
        self.match_sigma_deg = match_sigma_deg
        self.search_step_deg = search_step_deg
        self.ambiguity_ratio = ambiguity_ratio

    # -- scoring ----------------------------------------------------------

    def score_heading(
        self,
        observed_relative_deg: Sequence[float],
        predicted_absolute_deg: Sequence[float],
        heading_deg: float,
        hfov_deg: float,
    ) -> HeadingCandidate:
        """Score one candidate heading against the map's absolute bearings.

        The prediction is kept in absolute bearings and rotated by the candidate
        rather than re-predicted per heading: which corners exist and what hides
        them depends on where the camera stands, not where it points, so the
        expensive half of the work is done once for the whole search.
        """
        half_fov = hfov_deg / 2.0
        in_frame = [
            bearing for bearing in predicted_absolute_deg
            if abs(angular_difference_deg(bearing, heading_deg)) <= half_fov
        ]
        if not observed_relative_deg:
            raise HeadingMatchError("no observed edges to match")

        two_sigma_squared = 2.0 * self.match_sigma_deg ** 2
        total = 0.0
        inliers = 0
        for relative in observed_relative_deg:
            absolute = heading_deg + relative
            best = 0.0
            closest = None
            for bearing in in_frame:
                delta = abs(angular_difference_deg(absolute, bearing))
                if closest is None or delta < closest:
                    closest = delta
                vote = math.exp(-(delta * delta) / two_sigma_squared)
                if vote > best:
                    best = vote
            total += best
            if closest is not None and closest <= self.match_sigma_deg:
                inliers += 1

        return HeadingCandidate(
            heading_deg=heading_deg,
            score=total / len(observed_relative_deg),
            inliers=inliers,
            predicted_in_frame=len(in_frame),
        )

    # -- the search -------------------------------------------------------

    def match(
        self,
        observed_relative_deg: Sequence[float],
        predicted_absolute_deg: Sequence[float],
        hfov_deg: float,
        prior_heading_deg: Optional[float] = None,
        prior_sigma_deg: Optional[float] = None,
    ) -> HeadingFix:
        """Recover heading from observed edge bearings and the map's prediction.

        With no prior the whole circle is searched, which is the honest default:
        a gyro heading an hour old carries no information a search should trust.
        A prior narrows the search to plus or minus three sigma, which both
        costs less and, more importantly, excludes distant look-alike peaks that
        would otherwise make an unambiguous fix look ambiguous.
        """
        if not observed_relative_deg:
            raise HeadingMatchError("no observed edges to match")
        if not predicted_absolute_deg:
            raise HeadingMatchError(
                "the map predicts no building corners from this position; "
                "a heading cannot be recovered from an empty prediction"
            )

        headings = self._search_headings(prior_heading_deg, prior_sigma_deg)
        candidates = [
            self.score_heading(
                observed_relative_deg, predicted_absolute_deg, heading, hfov_deg
            )
            for heading in headings
        ]
        peaks = self._peaks(candidates)
        best = peaks[0]

        runner_up = peaks[1] if len(peaks) > 1 else None
        ambiguous = bool(
            runner_up is not None
            and best.score > 0
            and runner_up.score / best.score >= self.ambiguity_ratio
        )

        return HeadingFix(
            heading_deg=best.heading_deg % 360.0,
            sigma_deg=self._sigma_from(best),
            score=best.score,
            inliers=best.inliers,
            observed=len(observed_relative_deg),
            predicted_in_frame=best.predicted_in_frame,
            runner_up_deg=None if runner_up is None else runner_up.heading_deg % 360.0,
            runner_up_score=None if runner_up is None else runner_up.score,
            ambiguous=ambiguous,
            peaks=[peak for peak in peaks if peak.score >= MIN_ACCEPTABLE_SCORE],
        )

    def _search_headings(
        self,
        prior_heading_deg: Optional[float],
        prior_sigma_deg: Optional[float],
    ) -> List[float]:
        if prior_heading_deg is None or prior_sigma_deg is None:
            count = int(round(360.0 / self.search_step_deg))
            return [self.search_step_deg * index for index in range(count)]

        # Three sigma each way. Beyond that the prior says the answer is not
        # there, and searching anyway only invites a look-alike peak.
        span = 3.0 * prior_sigma_deg
        if span >= 180.0:
            count = int(round(360.0 / self.search_step_deg))
            return [self.search_step_deg * index for index in range(count)]
        count = int(round(2.0 * span / self.search_step_deg)) + 1
        return [
            prior_heading_deg - span + self.search_step_deg * index
            for index in range(count)
        ]

    def _peaks(self, candidates: Sequence[HeadingCandidate]) -> List[HeadingCandidate]:
        """Local maxima of the score curve, best first, one per lockout window.

        Taking the top N samples instead would return the same peak N times,
        which is exactly the failure that makes an ambiguity test useless.
        """
        ordered = sorted(candidates, key=lambda candidate: -candidate.score)
        kept: List[HeadingCandidate] = []
        for candidate in ordered:
            if all(
                abs(angular_difference_deg(candidate.heading_deg, peak.heading_deg))
                > PEAK_LOCKOUT_DEG
                for peak in kept
            ):
                kept.append(candidate)
            if len(kept) >= 8:
                break
        return kept

    def _sigma_from(self, best: HeadingCandidate) -> float:
        """Uncertainty of the recovered heading, degrees.

        The kernel width divided by the square root of the inlier count: each
        matched edge is an independent measurement of the same heading, so they
        average down. Floored at the search step, below which the search itself
        cannot resolve, and left at the kernel width when nothing matched, since
        a fix with no inliers has learned nothing and should not claim to.
        """
        if best.inliers <= 0:
            return self.match_sigma_deg
        return max(
            self.search_step_deg,
            self.match_sigma_deg / math.sqrt(best.inliers),
        )


def bearings_from_view(view: "object") -> List[float]:
    """Absolute bearings of the visible corners in a PredictedView.

    Absolute rather than relative, because the matcher rotates the prediction
    itself and a relative bearing has already had one heading baked into it.
    """
    return [edge.bearing_deg for edge in view.visible_edges]
