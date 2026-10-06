#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
import numpy as np

TOOLS = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(TOOLS))

import detect
import trt_detect
from detector_models import MODEL_FILES, export_options, selected_engine


class DetectorModelTest(unittest.TestCase):
    def test_registered_models_have_distinct_artifacts(self):
        engines = set()
        for model in MODEL_FILES:
            onnx, engine = trt_detect.model_paths(model)
            self.assertTrue(onnx.endswith('.onnx'))
            self.assertTrue(engine.endswith('.engine'))
            engines.add(engine)
        self.assertEqual(len(engines), 8)

    def test_yolo26_export_selects_decoded_one_to_one_head(self):
        self.assertEqual(export_options('yolo26n'), {'nms': False})
        self.assertEqual(export_options('yolo11n'), {'nms': True})
        command = trt_detect.build_command('yolo26n')
        self.assertIn('--precisionConstraints=obey', command)
        self.assertIn('--noTF32', command)
        self.assertTrue(trt_detect.model_paths('yolo26n')[1].endswith('_mixed.engine'))

    def test_launcher_checks_selected_model_and_explicit_engine(self):
        for args in (['--detect-model', 'yolo26n'], ['--detect-model=yolo26n']):
            self.assertEqual(selected_engine(args), trt_detect.model_paths('yolo26n')[1])
        self.assertEqual(selected_engine(['--detect', 'all', '--view', 'visible', '--detect-model=yolo26n',
                                          '--detect-engine', '/tmp/custom model.engine']),
                         '/tmp/custom model.engine')

    def test_end_to_end_boxes_map_to_camera_and_reject_invalid_rows(self):
        detector = object.__new__(trt_detect.TrtDetector)
        detector.conf = 0.25
        detector.names = detect.COCO
        detector.classes = {'person'}
        detector._class_ids = None
        detector.h_out = np.array([[
            [10, 130, 110, 230, .9, 0],  # y padding must be removed
            [-10, 110, 650, 530, .8, 0],  # clamp to the full camera frame
            [10, 130, 110, 230, .9, 2],  # another class
            [10, 130, 110, 230, .1, 0],  # below threshold
            [10, 0, 110, 100, .9, 0],  # entirely in padding
            [float('nan'), 130, 110, 230, .9, 0],
            [10, 130, 110, 230, .9, -1],
            [10, 130, 110, 230, .9, 80],
            [10, 130, 110, 230, .9, .5],
        ]], dtype=np.float32)
        self.assertEqual(detector.decode((400, 640), 1, 0, 120), [
            {'cls': 'person', 'conf': .9, 'x': 10, 'y': 10, 'w': 100, 'h': 100},
            {'cls': 'person', 'conf': .8, 'x': 0, 'y': 0, 'w': 640, 'h': 400},
        ])
        detector.h_out = np.array([[[5, 65, 55, 115, .9, 0]]], dtype=np.float32)
        self.assertEqual(detector.decode((400, 640), .5, 0, 60)[0]['y'], 10)

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
