#!/usr/bin/env python3
"""Synthetic contract tests for student_data.py; no PyTorch required."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from perception.student_data import (  # noqa: E402
    LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNKNOWN,
    detection_target, load_split, load_train_val,
)


class StudentDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = self.tmp.name
        self.manifest = {
            "version": "test-v2",
            "thermal": {"w": 160, "h": 120},
            "radar": {"k": 4},
            "split": {"train": ["a", "b"], "val": ["v"]},
            "sessions": {},
        }
        self._write_session("a", 3, with_ra=True)
        self._write_session("b", 2, with_ra=False)
        self._write_session("v", 2, with_ra=True)
        self._write_manifest()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_manifest(self):
        with open(os.path.join(self.data, "manifest.json"), "w") as f:
            json.dump(self.manifest, f)

    def _write_session(self, name, n, with_ra=False, calibrated=True):
        thermal = np.arange(n * 120 * 160, dtype=np.uint16).reshape(n, 120, 160)
        radar = np.zeros((n, 4, 6), np.float32)
        radar[:, 0] = [2.0, 0.1, 0.0, 0.2, 15.0, 4.0]
        th_boxes = np.zeros((n, 8, 5), np.float32)
        rgb_boxes = np.zeros((n, 8, 5), np.float32)
        th_boxes[1, 0] = [16, 12, 32, 24, 80]
        rgb_boxes[1, 0] = [64, 40, 128, 80, 0.9]
        arrays = {
            "thermal": thermal,
            "dt_ms": np.array([-1] + [114] * (n - 1), np.float32),
            "radar": radar,
            "n_radar": np.ones(n, np.int16),
            "th_boxes": th_boxes,
            "n_th_boxes": np.array([0, 1] + [0] * (n - 2), np.int16),
            "rgb_boxes": rgb_boxes,
            "n_rgb_boxes": np.array([0, 1] + [0] * (n - 2), np.int16),
            "thermal_label_state": np.array(
                [LABEL_NEGATIVE, LABEL_POSITIVE]
                + [LABEL_UNKNOWN] * (n - 2), np.int8),
            "radar_label_state": np.array(
                [LABEL_NEGATIVE, LABEL_POSITIVE]
                + [LABEL_UNKNOWN] * (n - 2), np.int8),
        }
        if with_ra:
            arrays["range_angle"] = np.ones((n, 4, 5), np.float16)
            arrays["ra_valid"] = np.ones(n, np.bool_)
        shard = f"{name}-000.npz"
        np.savez(os.path.join(self.data, shard), **arrays)
        info = {
            "shards": [shard], "frames": n,
            "thermal_counts_max": 65535,
            "provenance": {
                "lepton_gain": "high", "warp_lut_sha256": "warp",
                "radar_calib_sha256": "radar", "radar_cfg_sha256": "cfg",
                "detector_model": "yolov10n", "detector_engine_sha256": "eng",
            },
        }
        if calibrated:
            info.update(tmin=10.0, tmax=20.0, c_per_lsb=10.0 / 65535)
        self.manifest["sessions"][name] = info

    def test_scale_and_session_safe_history(self):
        split = load_split(self.data, "train")
        self.assertEqual(len(split), 5)
        self.assertAlmostEqual(float(split.arrays["thermal"][0, 0, 0]), 10.0)
        self.assertEqual(split.temporal.prev1[1], 0)
        self.assertEqual(split.temporal.prev2[2], 0)
        # b starts at combined index 3. Its dt is irrelevant across a boundary.
        self.assertEqual(split.temporal.prev1[3], -1)
        seq, valid = split.sequence("thermal", 3)
        self.assertEqual(valid.tolist(), [True, False, False])
        self.assertFalse(seq[1].any())

    def test_optional_stream_is_zero_filled_with_validity(self):
        split = load_split(self.data, "train")
        self.assertIn("range_angle", split.arrays)
        self.assertTrue(split.arrays["ra_valid"][:3].all())
        self.assertFalse(split.arrays["ra_valid"][3:].any())
        self.assertFalse(split.arrays["range_angle"][3:].any())
        seq, valid = split.sequence("range_angle", 3, "ra_valid")
        self.assertEqual(valid.tolist(), [False, False, False])
        self.assertFalse(seq.any())

    def test_targets_use_the_students_own_plane(self):
        split = load_split(self.data, "train")
        thermal = detection_target(split, 1, "thermal", 8)
        radar = detection_target(split, 1, "radar", 8)
        np.testing.assert_allclose(thermal["gt_boxes"][0], [0.2, 0.2, 0.2, 0.2])
        np.testing.assert_allclose(radar["gt_boxes"][0], [0.2, 0.2, 0.2, 0.2])
        self.assertEqual(thermal["gt_confidence"][0], 1.0)
        self.assertAlmostEqual(float(radar["gt_confidence"][0]), 0.9, places=5)

    def test_unknown_is_not_converted_to_negative(self):
        split = load_split(self.data, "train")
        target = detection_target(split, 2, "thermal", 8)
        self.assertEqual(target["supervision_state"], LABEL_UNKNOWN)
        self.assertFalse(target["gt_presence"].any())

    def test_train_and_val_modes_must_match(self):
        self.manifest["sessions"]["v"].pop("c_per_lsb")
        self.manifest["sessions"]["v"].pop("tmin")
        self.manifest["sessions"]["v"].pop("tmax")
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "thermal modes differ"):
            load_train_val(self.data)

    def test_train_dense_stream_missing_from_val_is_masked(self):
        self._write_session("v", 2, with_ra=False)
        self._write_manifest()
        _, train, val = load_train_val(self.data)
        self.assertIn("range_angle", train.arrays)
        self.assertIn("range_angle", val.arrays)
        self.assertFalse(val.arrays["ra_valid"].any())
        self.assertFalse(val.arrays["range_angle"].any())

    def test_incompatible_provenance_fails_closed(self):
        self.manifest["sessions"]["b"]["provenance"]["radar_cfg_sha256"] = "other"
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "radar_cfg_sha256"):
            load_split(self.data, "train")


if __name__ == "__main__":
    unittest.main()

