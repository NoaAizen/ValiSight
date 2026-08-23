#!/usr/bin/env python3
"""Regression checks for thermal framing and tri-state supervision."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)

from perception.dataset import LiveSession, THERMAL_PIXELS  # noqa: E402
from perception.autolabel.thermal_check import to_uint8_equivalent  # noqa: E402
from perception.export.export_shards import (              # noqa: E402
    LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNKNOWN, label_state,
    session_provenance, session_thermal_meta, write_training_bundle)
from perception.export import export_shards                 # noqa: E402


class ThermalFormatTest(unittest.TestCase):
    def make_session(self, dtype, declared=None, truncate=0):
        td = tempfile.TemporaryDirectory()
        frame = np.arange(THERMAL_PIXELS, dtype=dtype).reshape(120, 160)
        raw = frame.astype(dtype).tobytes()
        if truncate:
            raw = raw[:-truncate]
        with open(os.path.join(td.name, 'thermal.bin'), 'wb') as f:
            f.write(raw)
        with open(os.path.join(td.name, 'radar.jsonl'), 'w'):
            pass
        row = {'i': 0, 't_mono': 1.0, 'thermal_off': 0,
               'thermal_len': len(raw), 'view': 'visible', 'clean': True}
        with open(os.path.join(td.name, 'frames.jsonl'), 'w') as f:
            f.write(json.dumps(row) + '\n')
        if declared is not None:
            with open(os.path.join(td.name, 'meta.json'), 'w') as f:
                json.dump({'thermal_dtype': declared,
                           'thermal_frame_bytes': len(raw)}, f)
        return td, row, frame

    def test_legacy_uint8_is_inferred(self):
        td, row, expected = self.make_session(np.uint8)
        with td:
            got = LiveSession(td.name).thermal_frame(row)
            self.assertEqual(got.dtype, np.dtype('u1'))
            np.testing.assert_array_equal(got, expected)

    def test_uint16_le_is_inferred_without_offset_drift(self):
        td, row, expected = self.make_session(np.dtype('<u2'))
        with td:
            got = LiveSession(td.name).thermal_frame(row)
            self.assertEqual(got.dtype, np.dtype('<u2'))
            np.testing.assert_array_equal(got, expected)

    def test_declared_dtype_must_match_frame_length(self):
        td, row, _ = self.make_session(np.uint8, declared='uint16_le')
        with td:
            with self.assertRaisesRegex(ValueError, 'requires 38400'):
                LiveSession(td.name).thermal_frame(row)

    def test_truncated_frame_fails_closed(self):
        td, row, _ = self.make_session(np.uint8, truncate=1)
        with td:
            with self.assertRaisesRegex(ValueError, 'cannot infer'):
                LiveSession(td.name).thermal_frame(row)


    def test_uint16_autolabel_delta_uses_uint8_equivalent_counts(self):
        raw = np.array([0, 32768, 65535], dtype=np.uint16)
        got = to_uint8_equivalent(raw, 65535)
        np.testing.assert_allclose(got, [0, 127.50195, 255], atol=1e-4)

    def test_autolabel_rejects_counts_above_declared_scale(self):
        with self.assertRaisesRegex(ValueError, "exceed declared"):
            to_uint8_equivalent(np.array([256], dtype=np.uint16), 255)


class SupervisionTest(unittest.TestCase):
    def test_missing_teacher_is_unknown(self):
        self.assertEqual(label_state(False), LABEL_UNKNOWN)

    def test_positive_evidence_is_positive(self):
        self.assertEqual(label_state(True), LABEL_POSITIVE)

    def test_verified_empty_overrides_teacher_false_positive(self):
        self.assertEqual(label_state(True, verified_negative=True), LABEL_NEGATIVE)

    def test_session_manifest_preserves_uint16_scale(self):
        with tempfile.TemporaryDirectory() as td:
            old = export_shards.CAPTURES
            export_shards.CAPTURES = td
            try:
                sess = os.path.join(td, 's')
                os.makedirs(sess)
                with open(os.path.join(sess, 'meta.json'), 'w') as f:
                    json.dump({'tmin': 0, 'tmax': 60,
                               'thermal_dtype': 'uint16_le',
                               'thermal_counts_max': 65535,
                               'thermal_encoding': 'linear_set_range'}, f)
                rows = [{'thermal': np.zeros((120, 160), np.dtype('<u2'))}]
                got = session_thermal_meta('s', rows)
                self.assertEqual(got['thermal_dtype'], 'uint16_le')
                self.assertAlmostEqual(got['c_per_lsb'], 60 / 65535)
            finally:
                export_shards.CAPTURES = old

    def test_source_provenance_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            sess = os.path.join(td, "s")
            os.makedirs(sess)
            with open(os.path.join(sess, "meta.json"), "w") as f:
                json.dump({
                    "schema_version": 2,
                    "lepton_gain": "high",
                    "warp_lut_sha256": "warp",
                    "radar_calib_sha256": "calib",
                    "radar_cfg_stamp": {"sha256": "cfg",
                                        "name": "people.cfg"},
                }, f)
            old = export_shards.CAPTURES
            export_shards.CAPTURES = td
            try:
                got = session_provenance("s")
            finally:
                export_shards.CAPTURES = old
            self.assertEqual(got["radar_cfg_sha256"], "cfg")
            self.assertEqual(got["lepton_gain"], "high")


    def test_training_bundle_is_versioned_with_the_export(self):
        with tempfile.TemporaryDirectory() as td:
            got = write_training_bundle(td)
            self.assertEqual(
                got['entrypoint'], 'python -m perception.train_students')
            for rel, digest in got['sha256'].items():
                path = os.path.join(
                    td, rel if rel.endswith('.ipynb') else 'code/' + rel)
                self.assertTrue(os.path.exists(path), path)
                self.assertEqual(len(digest), 64)
            self.assertTrue(os.path.exists(
                os.path.join(td, 'train_students_colab.ipynb')))

    def test_range_profile_is_exported_without_ra_or_rd(self):
        from unittest import mock
        from perception.export import radar_heatmaps
        with tempfile.TemporaryDirectory() as td:
            sess = os.path.join(td, 's')
            os.makedirs(sess)
            open(os.path.join(sess, 'radar.bin'), 'wb').close()
            frame = {
                'frame': 7, 'ra': None, 'rd': None,
                'range_profile': np.arange(8, dtype=np.float32),
            }
            old = export_shards.CAPTURES
            export_shards.CAPTURES = td
            try:
                with mock.patch.object(
                        radar_heatmaps, 'frames', return_value=iter([frame])):
                    got = export_shards.load_heatmaps('s')
            finally:
                export_shards.CAPTURES = old
            self.assertIn(7, got)
            self.assertIsNone(got[7]['ra'])
            self.assertIsNone(got[7]['rd'])
            self.assertEqual(got[7]['rp'].dtype, np.float16)
            np.testing.assert_array_equal(got[7]['rp'], np.arange(8))

if __name__ == "__main__":
    unittest.main()
