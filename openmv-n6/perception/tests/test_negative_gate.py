#!/usr/bin/env python3
"""The per-frame veto on "this session was empty".

Synthetic sessions with a known answer, plus the property the whole gate rests
on: it only ever moves NEGATIVE towards UNKNOWN. If that direction ever
reverses, a warm frame becomes supervision saying a person is nobody, which is
what poisoned v3 and v4.
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)

from perception.dataset import LiveSession                     # noqa: E402
from perception.export import negative_gate as ng              # noqa: E402
from perception.export.negative_gate import (                  # noqa: E402
    NegativeGateError, occupied_frames, runs_of)

H, W = ng.THERMAL_H, ng.THERMAL_W
C_PER_LSB = 60.0 / 255.0        # the 0:60 window run_live.sh pins


def write_session(path, frames, c_per_lsb=C_PER_LSB, dtype=np.uint8):
    os.makedirs(path, exist_ok=True)
    arr = np.asarray(frames, dtype=dtype)
    with open(os.path.join(path, 'thermal.bin'), 'wb') as f:
        f.write(arr.tobytes())
    nbytes = arr[0].nbytes
    with open(os.path.join(path, 'frames.jsonl'), 'w') as f:
        for i in range(len(arr)):
            f.write(json.dumps({
                'i': i, 't_mono': 100.0 + i * 0.114,
                'thermal_off': i * nbytes, 'thermal_len': nbytes,
                'view': 'visible', 'clean': True}) + '\n')
    with open(os.path.join(path, 'radar.jsonl'), 'w'):
        pass
    meta = {'thermal_frame_bytes': nbytes,
            'thermal_dtype': 'uint8' if dtype == np.uint8 else 'uint16_le'}
    if c_per_lsb is not None:
        meta.update({'c_per_lsb': c_per_lsb, 'tmin': 0, 'tmax': 60})
    with open(os.path.join(path, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    return path


def empty_room(n, base_c=22.0, rng=None):
    """A flat wall with sensor noise, at `base_c` degrees."""
    rng = rng or np.random.default_rng(7)
    codes = base_c / C_PER_LSB
    return np.clip(rng.normal(codes, 1.2, (n, H, W)), 0, 255).astype(np.uint8)


class GateTest(unittest.TestCase):
    def gate(self, frames, **kw):
        with tempfile.TemporaryDirectory() as td:
            p = write_session(os.path.join(td, 's'), frames, **kw)
            return occupied_frames(LiveSession(p))

    def test_empty_room_is_empty(self):
        occ, stats = self.gate(empty_room(400))
        self.assertEqual(occ, set())
        self.assertEqual(stats['frames_occupied'], 0)

    def test_a_person_walking_through_is_found(self):
        frames = empty_room(400)
        # A 12x30 px body at 33 C - roughly what a person at 8 m subtends on
        # this sensor - crossing the frame between 150 and 200.
        for k, i in enumerate(range(150, 200)):
            x = 20 + 2 * k
            frames[i, 45:75, x:x + 12] = int(33.0 / C_PER_LSB)
        occ, stats = self.gate(frames)
        self.assertTrue(occ, 'a person crossing an empty room was not noticed')
        # Every frame the body is in must be demoted; the dilation may add more.
        self.assertTrue(set(range(150, 200)) <= occ)
        self.assertNotIn(0, occ)

    def test_whole_scene_warming_is_not_a_person(self):
        """The sun on a patio lifts every pixel together. radar3-negative1 is
        a real session that a whole-session background flags 27.5% of."""
        frames = empty_room(600).astype(np.float32)
        drift = np.linspace(0, 8.0 / C_PER_LSB, len(frames))
        frames = np.clip(frames + drift[:, None, None], 0, 255).astype(np.uint8)
        occ, _ = self.gate(frames)
        self.assertEqual(occ, set())

    def test_a_thin_hot_edge_is_not_a_person(self):
        """A sunlit roof edge is 1-3 px wide and spans the frame; the erosion
        is what separates it from a torso."""
        frames = empty_room(400)
        frames[:, 30:32, :] = int(34.0 / C_PER_LSB)
        occ, _ = self.gate(frames)
        self.assertEqual(occ, set())

    def test_no_temperature_scale_is_refused_not_guessed(self):
        with self.assertRaisesRegex(NegativeGateError, 'c_per_lsb'):
            self.gate(empty_room(120), c_per_lsb=None)

    def test_fused_session_with_no_thermal_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p = write_session(os.path.join(td, 's'), empty_room(20))
            with open(os.path.join(p, "frames.jsonl")) as src:
                rows = [json.loads(line) for line in src]
            with open(os.path.join(p, 'frames.jsonl'), 'w') as f:
                for r in rows:
                    r.pop('thermal_off')
                    f.write(json.dumps(r) + '\n')
            with self.assertRaisesRegex(NegativeGateError, 'no thermal'):
                occupied_frames(LiveSession(p))


class DirectionTest(unittest.TestCase):
    """The gate's safety argument, as a test rather than a comment."""

    def test_gate_only_removes_negatives_never_adds_them(self):
        from perception.export.export_shards import (
            LABEL_NEGATIVE, LABEL_UNKNOWN, label_state)
        # What export_session computes, reduced to the one expression that
        # matters: verified AND not occupied.
        for verified in (False, True):
            for occupied in (False, True):
                state = label_state(False, bool(verified and not occupied))
                if occupied:
                    self.assertEqual(state, LABEL_UNKNOWN,
                                     'an occupied frame must never be negative')
                elif verified:
                    self.assertEqual(state, LABEL_NEGATIVE)

    def test_runs_are_contiguous_and_inclusive(self):
        flags = np.zeros(20, bool)
        flags[3:7] = True
        flags[12] = True
        self.assertEqual(runs_of(flags), [(3, 6), (12, 12)])
        self.assertEqual(runs_of(flags, min_len=2), [(3, 6)])

    def test_dilation_reaches_both_sides(self):
        flags = np.zeros(30, bool)
        flags[15] = True
        out = ng._dilate(flags, width=3)
        self.assertEqual(set(np.flatnonzero(out)), set(range(12, 19)))


if __name__ == '__main__':
    unittest.main()
