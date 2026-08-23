#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest

TOOLS = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(TOOLS))

import detect
import trt_detect


class DetectorModelTest(unittest.TestCase):
    def test_registered_models_have_distinct_artifacts(self):
        engines = set()
        for model in ('yolov8n', 'yolov10n', 'yolo11n'):
            onnx, engine = trt_detect.model_paths(model)
            self.assertTrue(onnx.endswith('.onnx'))
            self.assertTrue(engine.endswith('.engine'))
            engines.add(engine)
        self.assertEqual(len(engines), 3)

    def test_build_command_is_argument_safe(self):
        cmd = trt_detect.build_command('yolo11n')
        self.assertIsInstance(cmd, list)
        self.assertIn('--fp16', cmd)
        self.assertTrue(any(x.endswith('yolo11n_nms.onnx') for x in cmd))
        self.assertTrue(any(x.endswith('yolo11n_nms_fp16.engine') for x in cmd))

    def test_static_nms_shape_is_accepted(self):
        trt_detect.validate_io_shapes((1, 3, 640, 640), (1, 300, 6))

    def test_raw_ultralytics_shape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'nms=True'):
            trt_detect.validate_io_shapes((1, 3, 640, 640),
                                          (1, 84, 8400))

    def test_missing_selected_engine_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, 'missing.engine')
            with self.assertRaisesRegex(FileNotFoundError, 'NMS engine'):
                detect.make_detector('gpu', model='yolo11n',
                                     engine=missing)


if __name__ == '__main__':
    unittest.main()
