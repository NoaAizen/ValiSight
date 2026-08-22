"""Recovering heading from one frame matched against the map's building corners."""

from __future__ import annotations

import math

import pytest

from mapinit.nav.heading import (
    DEFAULT_MATCH_SIGMA_DEG,
    MIN_ACCEPTABLE_SCORE,
    HeadingMatchError,
    HeadingMatcher,
    angular_difference_deg,
    bearings_from_columns,
)

#: A scene with corners at irregular spacings, which is what makes a heading
#: recoverable: even spacing is exactly the case that repeats.
SCENE = [12.0, 18.5, 41.0, 44.2, 67.0, 95.5, 130.0, 212.0, 288.5]

HFOV = 53.4


def observed_from(scene, heading, hfov=HFOV):
    """What a camera at this heading would see of this scene, as the image sees it."""
    relative = [angular_difference_deg(bearing, heading) for bearing in scene]
    return sorted(value for value in relative if abs(value) <= hfov / 2.0)


# -- pixels to bearings -----------------------------------------------------


def test_the_frame_centre_is_boresight():
    assert bearings_from_columns([79.5], fx=159.27, cx=79.5) == [0.0]


def test_the_frame_edge_matches_the_measured_field_of_view():
    """160 columns at fx=159.27 span 53.4 degrees, not the datasheet's 57.

    Measured from the sensor's outer edges rather than the outermost pixel
    centres, which is the convention the rig's reported hfov uses; between
    centres the span is a third of a degree narrower.
    """
    left, right = bearings_from_columns([-0.5, 159.5], fx=159.27, cx=79.5)
    assert right - left == pytest.approx(53.34, abs=0.05)


def test_positive_columns_are_to_the_right():
    """Matching the rig's azimuth convention; a flipped sign is silent otherwise."""
    assert bearings_from_columns([120.0], fx=159.27, cx=79.5)[0] > 0


def test_a_non_positive_focal_length_is_refused():
    with pytest.raises(HeadingMatchError):
        bearings_from_columns([80.0], fx=0.0, cx=79.5)


# -- the search -------------------------------------------------------------


def test_the_true_heading_is_recovered():
    matcher = HeadingMatcher()
    fix = matcher.match(observed_from(SCENE, 40.0), SCENE, HFOV)
    assert fix.heading_deg == pytest.approx(40.0, abs=0.3)
    assert fix.accepted


#: Corners all the way round, with gaps deliberately uneven: clusters and
#: voids, no spacing that recurs. Even or quasi-even spacing would make several
#: headings equally good answers, which is the ambiguity test's job, not this
#: one's. Every heading exercised below sees at least three of these.
CIRCLE = [
    0.0, 7.0, 29.6, 50.4, 60.0, 74.9, 88.8, 107.1, 128.5, 134.6, 139.2,
    161.6, 175.1, 195.9, 199.9, 213.7, 233.6, 242.6, 267.4, 291.2, 295.9,
    300.5, 316.4, 341.0, 353.4,
]


def test_recovery_works_all_the_way_round_the_compass():
    """Bearing wrap is the classic place a matcher quietly breaks."""
    matcher = HeadingMatcher()
    for truth in (0.0, 5.0, 90.0, 180.0, 270.0, 355.0):
        observed = observed_from(CIRCLE, truth)
        assert observed, f"the test scene shows nothing at {truth} deg"
        fix = matcher.match(observed, CIRCLE, HFOV)
        assert abs(angular_difference_deg(fix.heading_deg, truth)) < 0.5


def test_noise_of_a_pixel_or_two_does_not_move_the_answer():
    """0.33 deg per column, so a couple of pixels of corner localisation error."""
    matcher = HeadingMatcher()
    observed = [value + offset for value, offset in zip(
        observed_from(SCENE, 40.0), (0.3, -0.4, 0.2, -0.25, 0.35)
    )]
    fix = matcher.match(observed, SCENE, HFOV)
    assert fix.heading_deg == pytest.approx(40.0, abs=1.0)


def test_more_matched_edges_means_a_tighter_heading():
    """Independent measurements of one heading average down as sqrt(n)."""
    matcher = HeadingMatcher()
    sparse = matcher.match(observed_from([12.0, 18.5], 15.0), [12.0, 18.5], HFOV)
    dense = matcher.match(observed_from(SCENE, 40.0), SCENE, HFOV)
    assert dense.inliers > sparse.inliers
    assert dense.sigma_deg < sparse.sigma_deg


def test_a_fix_that_matched_nothing_does_not_claim_precision():
    matcher = HeadingMatcher()
    fix = matcher.match([0.0, 3.0], [90.0, 200.0], HFOV)
    assert fix.sigma_deg == pytest.approx(DEFAULT_MATCH_SIGMA_DEG)
    assert not fix.accepted


# -- ambiguity --------------------------------------------------------------


def test_a_repeating_scene_is_reported_ambiguous_rather_than_guessed():
    """Corners every 30 degrees look identical from four headings.

    The tallest peak here is an artefact of noise, and reporting it as the
    answer is the failure this flag exists to prevent: everything downstream
    multiplies heading by range.
    """
    repeating = [step * 30.0 for step in range(12)]
    fix = HeadingMatcher().match(observed_from(repeating, 60.0), repeating, HFOV)
    assert fix.ambiguous
    assert not fix.accepted
    assert fix.runner_up_deg is not None


def test_an_irregular_scene_is_not_ambiguous():
    fix = HeadingMatcher().match(observed_from(SCENE, 40.0), SCENE, HFOV)
    assert not fix.ambiguous


def test_rival_peaks_are_distinct_headings_not_one_peaks_shoulder():
    """Taking the top N samples would return the same peak N times."""
    fix = HeadingMatcher().match(observed_from(SCENE, 40.0), SCENE, HFOV)
    for first in range(len(fix.peaks)):
        for second in range(first + 1, len(fix.peaks)):
            separation = abs(angular_difference_deg(
                fix.peaks[first].heading_deg, fix.peaks[second].heading_deg
            ))
            assert separation > 5.0


def test_a_prior_resolves_an_ambiguity_the_full_search_cannot():
    """A gyro heading with a few degrees of drift excludes the look-alikes."""
    repeating = [step * 30.0 for step in range(12)]
    observed = observed_from(repeating, 60.0)
    matcher = HeadingMatcher()
    assert matcher.match(observed, repeating, HFOV).ambiguous
    guided = matcher.match(
        observed, repeating, HFOV, prior_heading_deg=61.0, prior_sigma_deg=3.0
    )
    assert not guided.ambiguous
    assert guided.heading_deg == pytest.approx(60.0, abs=0.5)


def test_a_hopeless_prior_falls_back_to_the_whole_circle():
    """Past three sigma of 60 degrees the prior says nothing, so do not pretend."""
    matcher = HeadingMatcher()
    fix = matcher.match(
        observed_from(SCENE, 40.0), SCENE, HFOV,
        prior_heading_deg=200.0, prior_sigma_deg=90.0,
    )
    assert fix.heading_deg == pytest.approx(40.0, abs=0.5)


# -- refusing to answer -----------------------------------------------------


def test_an_empty_prediction_is_an_error_not_a_heading():
    """The map showing no corners here is a real state, and it has no answer."""
    with pytest.raises(HeadingMatchError):
        HeadingMatcher().match([0.0, 5.0], [], HFOV)


def test_an_empty_observation_is_an_error_not_a_heading():
    with pytest.raises(HeadingMatchError):
        HeadingMatcher().match([], SCENE, HFOV)


def test_edges_from_a_different_place_score_below_the_floor():
    """Standing somewhere the map does not describe must not yield a fix."""
    fix = HeadingMatcher().match([-20.0, -8.0, 3.0, 19.0], SCENE, HFOV)
    assert fix.score < MIN_ACCEPTABLE_SCORE or fix.ambiguous
    assert not fix.accepted


def test_a_fix_carries_its_own_checks():
    fix = HeadingMatcher().match(observed_from(SCENE, 40.0), SCENE, HFOV)
    names = {check.name for check in fix.checks()}
    assert names == {"heading_match.score", "heading_match.unambiguous"}
    assert all(check.passed for check in fix.checks())


def test_an_ambiguous_fixs_check_explains_which_headings_collided():
    repeating = [step * 30.0 for step in range(12)]
    fix = HeadingMatcher().match(observed_from(repeating, 60.0), repeating, HFOV)
    check = next(c for c in fix.checks() if c.name == "heading_match.unambiguous")
    assert not check.passed
    assert "repeats" in check.detail


def test_a_non_positive_kernel_is_refused():
    with pytest.raises(HeadingMatchError):
        HeadingMatcher(match_sigma_deg=0.0)


# -- scoring ----------------------------------------------------------------


def test_only_corners_inside_the_field_of_view_can_be_matched():
    """A corner behind the camera is not a weak observation, it is no observation."""
    matcher = HeadingMatcher()
    candidate = matcher.score_heading([0.0], SCENE, heading_deg=180.0, hfov_deg=HFOV)
    in_frame = [b for b in SCENE if abs(angular_difference_deg(b, 180.0)) <= HFOV / 2]
    assert candidate.predicted_in_frame == len(in_frame)


def test_the_score_is_a_mean_so_frames_are_comparable():
    """A frame with more edges must not outscore a better-matched sparse one."""
    matcher = HeadingMatcher()
    perfect_sparse = matcher.score_heading([12.0], [42.0], 30.0, HFOV)
    assert perfect_sparse.score == pytest.approx(1.0)


def test_score_falls_off_smoothly_with_misalignment():
    """Smooth is what lets a wrong correspondence be outvoted instead of committed to."""
    matcher = HeadingMatcher()
    scores = [
        matcher.score_heading(observed_from(SCENE, 40.0), SCENE, 40.0 + offset, HFOV).score
        for offset in (0.0, 0.5, 1.0, 2.0)
    ]
    assert scores == sorted(scores, reverse=True)


def test_angular_difference_wraps_the_short_way():
    assert angular_difference_deg(359.0, 1.0) == pytest.approx(-2.0)
    assert angular_difference_deg(1.0, 359.0) == pytest.approx(2.0)
