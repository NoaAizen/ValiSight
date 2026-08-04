"""Host side of the dual thermal+RGB recording bridge, driven offline.

Synthetic n6_dual_stream protocol lines go through the SAME parsing and
recording path view_thermal_rgb.py uses live — the real src/recorder.py and
the real frame_clock.Unwrapper, not reimplementations. The device half can
only be verified on the board; what is testable off-board is exactly this:
that stamped payloads land in the session layout the registration analyst
reads, on a mono_us axis that survives the 2**30 ticks wrap.
"""
import json
import os

import _path  # noqa: F401  — src/ on sys.path

import frame_clock
import view_thermal_rgb as vtr

RAW_THERMAL = bytes(range(256)) * 75          # 19200 B, non-trivial content
FAKE_JPEG = b"\xff\xd8fake-jpeg-payload\xff\xd9"


def test_split_stamped_formats():
    assert vtr.split_stamped(b"12345:QUJD") == (12345, b"QUJD")
    # old device script: no stamp — payload passes through for viewing only
    assert vtr.split_stamped(b"QUJDIGZvbw==") == (None, b"QUJDIGZvbw==")
    # base64 containing ':' cannot occur, but a non-digit head must not crash
    assert vtr.split_stamped(b"abc:def")[0] is None


def test_decode_thermal_raw_roundtrip():
    import base64
    img, raw = vtr.decode_thermal(base64.b64encode(RAW_THERMAL))
    assert raw == RAW_THERMAL
    assert img.shape == (vtr.THERMAL_H, vtr.THERMAL_W)
    assert img[0, 5] == RAW_THERMAL[5]


def test_records_both_streams_across_tick_wrap(tmp_path):
    dr = vtr.DualRecorder(save_frames="both", root=str(tmp_path), note="unit")
    near_wrap = frame_clock.TICKS_PERIOD - 500

    dr.frame("thermal", near_wrap, RAW_THERMAL)
    dr.frame("rgb", near_wrap + 200, FAKE_JPEG)
    dr.frame("thermal", 700, RAW_THERMAL)          # counter wrapped on-board

    sess = dr.rec.dir
    frames = [json.loads(l) for l in
              open(os.path.join(sess, "frames.jsonl"), encoding="utf-8")]
    assert [f["sensor"] for f in frames] == ["thermal", "rgb", "thermal"]

    # the unwrapped axis is monotonic THROUGH the wrap, and preserves true dt
    mono = [f["mono_us"] for f in frames]
    assert mono == sorted(mono)
    assert mono[2] - mono[0] == 1200
    assert all(f["epoch"] == 1 for f in frames)

    # payloads land in the exact layout the recorder documents
    t_file = os.path.join(sess, frames[0]["file"])
    assert frames[0]["file"] == os.path.join("thermal", "000001.bin")
    assert os.path.getsize(t_file) == 19200
    assert open(t_file, "rb").read() == RAW_THERMAL
    r_file = os.path.join(sess, frames[1]["file"])
    assert frames[1]["file"] == os.path.join("rgb", "000001.jpg")
    assert open(r_file, "rb").read() == FAKE_JPEG

    meta = json.load(open(os.path.join(sess, "meta.json"), encoding="utf-8"))
    assert meta["camera"] == "dual"
    assert meta["session_note"] == "unit"   # NOT "note" — Recorder owns that key
    assert meta["dead_rows"]        # the analyst excludes these rows; absence
    assert meta["offset_rows"]      # would silently poison every centroid


def test_new_epoch_restarts_the_time_axis(tmp_path):
    """A device-stream restart resets the board's ticks counter; the recorder
    must bump epoch and open a fresh axis, or the Unwrapper would 'unwrap' the
    backwards jump into a fabricated forward step (frame_clock.stamp doc)."""
    dr = vtr.DualRecorder(save_frames="none", root=str(tmp_path))
    dr.frame("thermal", 1_000_000, RAW_THERMAL)
    dr.new_epoch()
    dr.frame("thermal", 500, RAW_THERMAL)      # rebooted: ticks near zero again

    frames = [json.loads(l) for l in
              open(os.path.join(dr.rec.dir, "frames.jsonl"), encoding="utf-8")]
    assert [f["epoch"] for f in frames] == [1, 2]
    assert frames[1]["mono_us"] == 500         # fresh axis, not stitched to old


def test_rgb_dims_stamped_from_frame_not_from_request(tmp_path):
    """The board answers a QVGA (320x240) request with 320x200 JPEGs, so the
    meta claim must come from the recorded frame itself. Re-stamping meta
    mid-session must not shift the session-start clock."""
    import cv2
    import numpy as np

    dr = vtr.DualRecorder(save_frames="both", root=str(tmp_path))
    meta_path = os.path.join(dr.rec.dir, "meta.json")
    started = json.load(open(meta_path, encoding="utf-8"))["started_host_wall"]

    jpeg = cv2.imencode(".jpg", np.zeros((200, 320, 3), np.uint8))[1].tobytes()
    dr.frame("rgb", 1000, jpeg)

    meta = json.load(open(meta_path, encoding="utf-8"))
    assert "320x200" in meta["rgb"]
    assert meta["started_host_wall"] == started


def test_unstamped_frames_are_skipped_not_guessed(tmp_path):
    dr = vtr.DualRecorder(save_frames="both", root=str(tmp_path))
    dr.frame("thermal", None, RAW_THERMAL)
    assert dr.dropped_unstamped == 1
    assert dr.seq == {"thermal": 0, "rgb": 0}
    assert not os.path.exists(os.path.join(dr.rec.dir, "frames.jsonl"))
    assert not os.listdir(os.path.join(dr.rec.dir, "thermal"))
    assert "WARNING" in dr.summary()
