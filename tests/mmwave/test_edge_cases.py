"""Edge cases beyond the basic stage gates — the "what if" suite.

Each test encodes a specific real-world failure mode: an empty scene, a
header that lies, a corrupt length field, values at the exact physical
boundary. If one of these ever happens on the bench, the behavior is
already defined and tested here.
"""
import struct
import pytest
from frame_builder import build_frame, MAGIC
from mmwave_parser import (FrameSync, fold_velocity, parse_frame,
                           validate_points)

VMAX = 0.67


# --- empty / partial scenes ---------------------------------------------

def test_empty_scene_frame_parses():
    # radar sees nothing: 0 points, 0 TLVs — must be a normal frame, not
    # an error (happens constantly in an empty room)
    out = parse_frame(build_frame())
    assert out['points'] == []
    assert out['header']['num_detected_obj'] == 0


def test_side_info_without_points():
    out = parse_frame(build_frame(side_info=[(100, 90), (80, 85)]))
    assert out['points'] == []  # SNR with no points to join: no crash


def test_unknown_tlv_type_is_skipped_not_fatal():
    # future SDK versions may add TLV types we don't know
    out = parse_frame(build_frame(points=[(1.0, 2.0, 0.0, 0.1)],
                                  extra_tlvs=[(250, b'\xab' * 12)]))
    assert len(out['points']) == 1
    assert 250 in out['tlv_types']


def test_zero_length_tlv_payload():
    out = parse_frame(build_frame(extra_tlvs=[(250, b'')]))
    assert out['tlv_types'] == [250]


# --- lying / corrupt headers ---------------------------------------------

def test_header_lying_about_point_count():
    # header claims 5 objects, payload holds 2 — the payload is the truth
    # (we parse what is actually there; the mismatch is visible upstream)
    out = parse_frame(build_frame(points=[(1, 2, 0, 0), (3, 4, 0, 0)],
                                  num_detected_obj=5))
    assert len(out['points']) == 2
    assert out['header']['num_detected_obj'] == 5


def test_side_info_longer_than_points():
    out = parse_frame(build_frame(
        points=[(1, 2, 0, 0), (3, 4, 0, 0)],
        side_info=[(10, 1), (20, 2), (30, 3), (40, 4), (50, 5)]))
    assert [p['snr'] for p in out['points']] == [10, 20]


def test_framesync_rejects_absurd_total_and_recovers():
    bad = bytearray(build_frame(points=[(1, 2, 0, 0)]))
    struct.pack_into('<I', bad, 12, 100000)  # totalPacketLen = 100 KB
    good = build_frame(frame_number=2, points=[(5, 6, 0, 0)])
    s = FrameSync()
    out = s.feed(bytes(bad)) + s.feed(good)
    assert [bytes(o) for o in out] == [good]
    assert s.resync_count >= 1


def test_framesync_rejects_tiny_total_and_recovers():
    bad = bytearray(build_frame(points=[(1, 2, 0, 0)]))
    struct.pack_into('<I', bad, 12, 10)  # totalPacketLen < minimum frame
    good = build_frame(frame_number=2, points=[(5, 6, 0, 0)])
    s = FrameSync()
    out = s.feed(bytes(bad)) + s.feed(good)
    assert [bytes(o) for o in out] == [good]


# --- stream slicing edge cases -------------------------------------------

def test_padded_frames_back_to_back():
    f = build_frame(points=[(1, 2, 0, 0)], pad_to_32=True)
    out = FrameSync().feed(f + f)
    assert [bytes(o) for o in out] == [f, f]


def test_magic_split_across_two_chunks():
    f = build_frame(points=[(1, 2, 0, 0)])
    s = FrameSync()
    assert s.feed(f[:4]) == []          # first half of the magic word only
    assert [bytes(o) for o in s.feed(f[4:])] == [f]


def test_feed_empty_bytes_is_noop():
    s = FrameSync()
    assert s.feed(b'') == []
    assert s.dropped_bytes == 0


# --- physical boundary values ---------------------------------------------

def test_fold_at_exact_vmax_wraps_to_negative():
    # +v_max and -v_max are the same Doppler FFT bin: the band is
    # [-v_max, +v_max), so exactly +v_max reads as -v_max
    assert fold_velocity(VMAX, VMAX) == pytest.approx(-VMAX)
    assert -VMAX <= fold_velocity(VMAX, VMAX) < VMAX


def test_validator_accepts_exact_vmax_reading():
    # float32 rounding of a legal boundary reading must not be flagged
    pts = parse_frame(build_frame(points=[(1.0, 2.0, 0.0, VMAX)]))['points']
    assert validate_points(pts, vmax=VMAX) == []


def test_validator_flags_huge_finite_range():
    # 1e30 m is finite, so it passes the NaN/inf check — the range check
    # must catch it (classic byte-offset bug symptom)
    pts = parse_frame(build_frame(points=[(1e30, 2.0, 0.0, 0.1)]))['points']
    assert validate_points(pts, max_range=50.0)
