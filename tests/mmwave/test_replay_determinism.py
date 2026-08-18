# Plan item 8: replay determinism over the REAL field recording.
#
# The contract under test: same recording => bit-identical canonical
# output, run after run, no matter how the byte stream is sliced into
# chunks. Chunk-slicing invariance is the strong form — live UART delivers
# arbitrary chunk boundaries, so any boundary-dependent behavior is a real
# parser bug that would appear "sometimes" in the field.

import hashlib
import json
import os
import subprocess
import sys

import pytest

import replay

_REC_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..',
    'data', 'recordings', 'aliasing_walk_20260810'))
RAW = os.path.join(_REC_DIR, 'radar_raw.bin')
GOLDEN = os.path.join(_REC_DIR, 'radar_frames.jsonl')

needs_recording = pytest.mark.skipif(
    not os.path.exists(RAW), reason='field recording not present')


def _digest(out):
    return hashlib.sha256(out).hexdigest()


@needs_recording
def test_two_runs_are_bit_identical():
    a, _ = replay.replay_file(RAW, chunk=4096)
    b, _ = replay.replay_file(RAW, chunk=4096)
    assert a == b
    assert len(a) > 0


@needs_recording
def test_chunk_slicing_invariance():
    """1-byte drip, UART-ish, large, and two seeded-random slicings must
    all produce the same canonical bytes."""
    outputs = {}
    for label, kwargs in [
        ('bytewise', dict(chunk=1)),
        ('uart64', dict(chunk=64)),
        ('big4096', dict(chunk=4096)),
        ('rand1217', dict(random_seed=1217)),
        ('rand42', dict(random_seed=42)),
    ]:
        out, stats = replay.replay_file(RAW, **kwargs)
        outputs[label] = _digest(out)
        assert stats['frames'] > 0, label
    assert len(set(outputs.values())) == 1, outputs


@needs_recording
def test_fresh_process_reproduces_digest():
    """Guards against per-process state (hash seed, import order, env):
    a brand-new interpreter must land on the identical digest."""
    out, _ = replay.replay_file(RAW, chunk=4096)
    lib = os.path.dirname(replay.__file__)
    proc = subprocess.run(
        [sys.executable, os.path.join(lib, 'replay.py'), RAW,
         '--chunk', '4096'],
        capture_output=True, text=True, check=True)
    assert proc.stdout.split()[1] == _digest(out)


@needs_recording
def test_recording_parses_essentially_clean():
    """The recording was captured over USB with zero live resyncs; replay
    must not invent parse failures."""
    out, stats = replay.replay_file(RAW, chunk=4096)
    assert stats['bad_frames'] == 0
    assert stats['frames'] >= 400


@needs_recording
def test_agrees_with_golden_jsonl_structure():
    """Cross-check against the jsonl captured live: same frame-number
    sequence, same per-frame point counts, same velocity values (golden
    rounds to 4 decimals; its x,y,z are in a rotated axis convention, so
    geometry is compared only per-point velocity here)."""
    if not os.path.exists(GOLDEN):
        pytest.skip('golden jsonl not present')
    out, _ = replay.replay_file(RAW, chunk=4096)
    ours = [json.loads(line) for line in out.decode().splitlines()]
    golden = [json.loads(line) for line in open(GOLDEN)]

    # The live recorder came up two frames late: the raw stream holds
    # 71811+ but the jsonl starts at 71813. Align on the golden's first
    # frame; from there the sequences must agree exactly.
    start = [f['frame'] for f in ours].index(golden[0]['frame'])
    ours = ours[start:]
    assert [f['frame'] for f in ours] == [g['frame'] for g in golden]
    for f, g in zip(ours, golden):
        assert len(f['points']) == len(g['points']), f['frame']
        for p, gp in zip(f['points'], g['points']):
            assert round(p[3], 4) == pytest.approx(gp[3], abs=1e-9), \
                f['frame']
