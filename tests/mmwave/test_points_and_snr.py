"""Stage 4-5 gates: TLV type 1 point parsing, TLV type 7 SNR joined by index.

Physical grounding for the reflector test: a corner reflector / metal plate
has an RCS orders of magnitude above a person's (~-10..0 dBsm), so its SNR
must separate cleanly from background — that is exactly the Stage 5 bench
gate, encoded here on synthetic data.
"""
import pytest
from frame_builder import build_frame
from mmwave_parser import parse_frame

PTS = [(1.5, 3.0, 0.25, 0.4), (-0.5, 2.0, 0.0, -0.3), (0.1, 6.5, -0.4, 0.0)]
SIDE = [(180, 90), (95, 85), (60, 88)]  # (snr, noise) in 0.1 dB units


def test_point_values_roundtrip():
    pts = parse_frame(build_frame(points=PTS))['points']
    assert len(pts) == 3
    for got, exp in zip(pts, PTS):
        assert got['x'] == pytest.approx(exp[0], rel=1e-6)
        assert got['y'] == pytest.approx(exp[1], rel=1e-6)
        assert got['z'] == pytest.approx(exp[2], rel=1e-6)
        assert got['v'] == pytest.approx(exp[3], rel=1e-6)


def test_point_count_matches_header():
    out = parse_frame(build_frame(points=PTS))
    assert len(out['points']) == out['header']['num_detected_obj']


def test_snr_joined_by_index():
    pts = parse_frame(build_frame(points=PTS, side_info=SIDE))['points']
    for got, (snr, noise) in zip(pts, SIDE):
        assert got['snr'] == snr
        assert got['noise'] == noise


def test_reflector_has_highest_snr():
    side = [(60, 90), (430, 85), (55, 88)]  # strong return on point #1
    pts = parse_frame(build_frame(points=PTS, side_info=side))['points']
    best = max(pts, key=lambda p: p['snr'])
    assert best['x'] == pytest.approx(PTS[1][0], rel=1e-6)


def test_missing_side_info_gives_none_snr():
    pts = parse_frame(build_frame(points=PTS))['points']
    assert all(p['snr'] is None and p['noise'] is None for p in pts)


def test_short_side_info_does_not_crash():
    pts = parse_frame(build_frame(points=PTS, side_info=SIDE[:2]))['points']
    assert pts[0]['snr'] == SIDE[0][0]
    assert pts[1]['snr'] == SIDE[1][0]
    assert pts[2]['snr'] is None
