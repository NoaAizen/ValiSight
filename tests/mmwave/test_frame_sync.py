"""Stage 2 gates: byte stream -> whole frames, resilient to corruption.

Plan gates encoded here:
  * every sliced frame's length == totalPacketLen from its own header
  * frameNumber never goes backwards across recovered frames
  * bounded buffering under a garbage flood (no unbounded growth)
  * clean streams produce zero resyncs; corruption recovers within a frame
  * a magic word INSIDE a payload must not split a frame — framing is
    validated via totalPacketLen, never via the magic alone
"""
import pytest
from frame_builder import MAGIC, build_frame
from mmwave_parser import FrameSync, parse_header


def test_single_frame_single_chunk():
    f = build_frame(frame_number=7, points=[(1.0, 2.0, 0.5, 0.1)])
    s = FrameSync()
    out = s.feed(f)
    assert [bytes(o) for o in out] == [f]
    assert s.resync_count == 0


def test_frame_arrives_byte_by_byte():
    f = build_frame(points=[(1.0, 2.0, 0.5, 0.1), (0.3, 4.0, -0.2, -0.4)])
    s = FrameSync()
    out = []
    for i in range(len(f)):
        out += s.feed(f[i:i + 1])
    assert [bytes(o) for o in out] == [f]


def test_two_frames_one_chunk_no_resync():
    f1 = build_frame(frame_number=1, points=[(1, 2, 0, 0)])
    f2 = build_frame(frame_number=2, points=[(1, 2, 0, 0), (3, 1, 0, 0.2)])
    s = FrameSync()
    out = s.feed(f1 + f2)
    assert [bytes(o) for o in out] == [f1, f2]
    assert s.resync_count == 0


def test_leading_garbage_then_frame():
    f = build_frame(points=[(1, 2, 0, 0)])
    s = FrameSync()
    out = s.feed(b'\x55' * 300 + f)
    assert [bytes(o) for o in out] == [f]
    assert s.dropped_bytes >= 300


def test_magic_inside_payload_does_not_split_frame():
    evil = b'\x11\x22' + MAGIC + b'\x33\x44'
    f1 = build_frame(frame_number=1, extra_tlvs=[(250, evil)])
    f2 = build_frame(frame_number=2, points=[(1, 2, 0, 0)])
    out = FrameSync().feed(f1 + f2)
    assert [bytes(o) for o in out] == [f1, f2]


def test_truncated_frame_recovers_on_next():
    truncated = build_frame(frame_number=1, points=[(5, 5, 0, 0)])[:-20]
    good = build_frame(frame_number=2, points=[(1, 2, 0, 0)])
    s = FrameSync()
    out = s.feed(truncated + good)
    assert bytes(out[-1]) == good
    assert s.resync_count >= 1


def test_frame_numbers_never_go_backwards():
    frames = [build_frame(frame_number=n, points=[(1, 2, 0, 0)])
              for n in (10, 11, 13, 14)]  # a gap is allowed, regression isn't
    s = FrameSync()
    nums = []
    for f in frames:
        for out in s.feed(f):
            nums.append(parse_header(bytes(out))['frame_number'])
    assert nums == sorted(nums)


def test_buffer_bounded_under_garbage_flood():
    s = FrameSync(max_buffer=64 * 1024)
    fed = 0
    chunk = bytes(range(256)) * 64  # 16 KB, deterministic, contains no magic
    for _ in range(64):              # 1 MB total
        assert s.feed(chunk) == []
        fed += len(chunk)
    assert s.dropped_bytes >= fed - 64 * 1024
    # and the stream still recovers afterwards
    f = build_frame(points=[(1, 2, 0, 0)])
    assert [bytes(o) for o in s.feed(f)] == [f]
