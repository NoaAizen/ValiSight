"""Research-based physics invariants for the radar pipeline.

Doppler ambiguity in FMCW radar: velocity comes from chirp-to-chirp phase,
which is only unambiguous in [-v_max, +v_max), v_max = lambda/(4*Tc)
(TI, "Introduction to mmWave sensing"; Skolnik, Radar Handbook). The rig's
current cfg gives v_max ~= 0.67 m/s (the TDM-MIMO 3-TX cost), so an average
pedestrian at 1.4 m/s (gait literature) folds to ~0.06 m/s and reads as
near-static — the exact failure mode of item 7 / tests 10-12 in the team
plan. fold/unfold below encode that modular arithmetic.

validate_points() encodes "impossible physics = parsing bug" canaries:
  * |v| beyond v_max cannot come out of the radar's Doppler FFT — if we see
    it, our units or offsets are wrong, not the radar
  * range beyond the cfg max, NaN/inf coordinates -> corrupted floats
"""
import pytest
from frame_builder import build_frame
from mmwave_parser import (fold_velocity, parse_frame, unfold_candidates,
                           validate_points)

VMAX = 0.67  # m/s, current radar cfg


def test_pedestrian_folds_to_near_static():
    # 1.4 m/s walker -> 1.4 - 2*0.67 = 0.06 m/s: "the invisible pedestrian"
    assert fold_velocity(1.4, VMAX) == pytest.approx(0.06, abs=1e-9)


def test_in_band_velocity_unchanged():
    assert fold_velocity(0.3, VMAX) == pytest.approx(0.3)
    assert fold_velocity(-0.5, VMAX) == pytest.approx(-0.5)


def test_fold_is_symmetric():
    assert fold_velocity(-1.4, VMAX) == pytest.approx(-0.06, abs=1e-9)


def test_fold_always_lands_in_band():
    for v in (-5.0, -1.4, -0.7, 0.0, 0.66, 0.68, 1.4, 3.1):
        folded = fold_velocity(v, VMAX)
        assert -VMAX <= folded < VMAX


def test_unfold_recovers_true_velocity():
    measured = fold_velocity(1.4, VMAX)
    cands = unfold_candidates(measured, VMAX, kmax=2)
    assert any(c == pytest.approx(1.4, abs=1e-9) for c in cands)


def test_validator_flags_out_of_band_velocity():
    pts = parse_frame(build_frame(points=[(1.0, 2.0, 0.0, 1.4)]))['points']
    assert validate_points(pts, vmax=VMAX)


def test_validator_flags_impossible_range():
    pts = parse_frame(build_frame(points=[(0.0, 80.0, 0.0, 0.1)]))['points']
    assert validate_points(pts, max_range=50.0)


def test_validator_flags_non_finite_floats():
    pts = parse_frame(
        build_frame(points=[(float('nan'), 2.0, 0.0, 0.1)]))['points']
    assert validate_points(pts)


def test_validator_passes_sane_frame():
    good = [(0.5, 2.5, 0.1, 0.25), (-1.0, 4.0, -0.2, -0.6)]
    pts = parse_frame(build_frame(points=good))['points']
    assert validate_points(pts, vmax=VMAX, max_range=50.0) == []
