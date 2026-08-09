"""Stage 3 gates: header decode + a TLV walk that closes the frame exactly.

The classic SDK-version trap is encoded here: does the TLV `length` field
include the 8-byte TLV header or only the payload? walk_tlvs() must decide
empirically per frame — both variants are fed and must yield identical
payloads. The real demo also zero-pads packets to a 32-byte multiple; the
walk must tolerate that without misreading padding as a TLV.
"""
import pytest
from frame_builder import build_frame, TLV_POINTS, TLV_SIDE_INFO
from mmwave_parser import parse_header, walk_tlvs

PTS = [(1.5, 3.0, 0.25, 0.4), (-0.5, 2.0, 0.0, -0.3)]
SIDE = [(210, 90), (55, 80)]


def test_header_fields_roundtrip():
    f = build_frame(frame_number=42, points=PTS, side_info=SIDE)
    h = parse_header(f)
    assert h['frame_number'] == 42
    assert h['total_len'] == len(f)
    assert h['num_detected_obj'] == 2
    assert h['num_tlvs'] == 2
    assert h['platform'] == 0x000A1843


def test_parse_header_rejects_non_frame():
    with pytest.raises(ValueError):
        parse_header(b'\x00' * 64)


def test_walk_payload_only_semantics():
    f = build_frame(points=PTS, side_info=SIDE,
                    length_includes_tlv_header=False)
    tlvs, mode = walk_tlvs(f)
    assert mode == 'payload'
    assert [t for t, _ in tlvs] == [TLV_POINTS, TLV_SIDE_INFO]
    assert len(tlvs[0][1]) == 2 * 16
    assert len(tlvs[1][1]) == 2 * 4


def test_walk_includes_header_semantics():
    a = build_frame(points=PTS, side_info=SIDE,
                    length_includes_tlv_header=False)
    b = build_frame(points=PTS, side_info=SIDE,
                    length_includes_tlv_header=True)
    tlvs_a, _ = walk_tlvs(a)
    tlvs_b, mode = walk_tlvs(b)
    assert mode == 'includes_header'
    assert tlvs_a == tlvs_b  # same payloads whatever the length semantics


def test_walk_tolerates_sdk_zero_padding():
    f = build_frame(points=PTS, side_info=SIDE, pad_to_32=True)
    assert parse_header(f)['total_len'] % 32 == 0
    tlvs, _ = walk_tlvs(f)
    assert [t for t, _ in tlvs] == [TLV_POINTS, TLV_SIDE_INFO]


def test_walk_tolerates_sdk_garbage_padding():
    # Real SDK 3.6 firmware pads from uninitialized memory, not zeros:
    # 385/451 frames in aliasing_walk_20260810/radar_raw.bin carry
    # non-zero padding. Only the pad SIZE may be validated, never content.
    # 3 points -> 116-byte frame -> 12 bytes of padding (2 points give 96,
    # already a 32-multiple, so no pad region to corrupt).
    pts3 = PTS + [(0.5, 4.0, -0.1, 0.7)]
    side3 = SIDE + [(120, 70)]
    unpadded = build_frame(points=pts3, side_info=side3)
    f = bytearray(build_frame(points=pts3, side_info=side3, pad_to_32=True))
    assert len(f) > len(unpadded)
    f[len(unpadded):] = b'\xa5' * (len(f) - len(unpadded))
    tlvs, _ = walk_tlvs(bytes(f))
    assert [t for t, _ in tlvs] == [TLV_POINTS, TLV_SIDE_INFO]


def test_corrupt_tlv_length_raises():
    f = bytearray(build_frame(points=PTS))
    # first TLV sits right after magic(8)+header(32); its length is at 44..48
    f[44:48] = (0xFFFF).to_bytes(4, 'little')
    with pytest.raises(ValueError):
        walk_tlvs(bytes(f))
