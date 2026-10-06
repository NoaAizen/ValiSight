import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from perception.external.thermal_objects_data import CLASSES, export


class ThermalObjectsDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for split, level in [('train', 50), ('val', 70)]:
            folder = self.root / split
            folder.mkdir()
            cv2.imwrite(str(folder / 'frame.png'), np.full((120, 160), level, np.uint8))
            annotations = {
                'images': [{'id': 1, 'file_name': 'frame.png', 'width': 160, 'height': 120}],
                'categories': [{'id': 7, 'name': 'car'}, {'id': 42, 'name': 'dog'}],
                'annotations': [{'image_id': 1, 'category_id': 7, 'bbox': [0, 0, 80, 60]},
                                {'image_id': 1, 'category_id': 42, 'bbox': [80, 60, 40, 30]}]}
            (folder / 'coco.json').write_text(json.dumps(annotations))

    def run_export(self):
        return export(self.root / 'train', self.root / 'train/coco.json',
                      self.root / 'val', self.root / 'val/coco.json', self.root / 'out')

    def test_preserves_cars_and_dogs_and_reports_untrained_animals(self):
        report = self.run_export()
        self.assertEqual(report['box_counts']['train']['car'], 1)
        self.assertEqual(report['box_counts']['train']['dog'], 1)
        self.assertIn('deer', report['missing_training_classes'])
        labels = next((self.root / 'out/labels/train').glob('*.txt')).read_text().splitlines()
        self.assertEqual(labels[0], f'{CLASSES.index("car")} 0.25000000 0.25000000 0.50000000 0.50000000')
        self.assertEqual(int(labels[1].split()[0]), CLASSES.index('dog'))

    def test_rejects_train_validation_leakage(self):
        cv2.imwrite(str(self.root / 'val/frame.png'), np.full((120, 160), 50, np.uint8))
        with self.assertRaisesRegex(ValueError, 'same image'):
            self.run_export()
        self.assertFalse((self.root / 'out').exists())

    def test_flir_motor_alias_and_selected_class_order(self):
        for split in ('train', 'val'):
            path = self.root / split / 'coco.json'
            data = json.loads(path.read_text())
            data['categories'][0]['name'] = 'motor'
            path.write_text(json.dumps(data))
        report = export(self.root / 'train', self.root / 'train/coco.json',
                        self.root / 'val', self.root / 'val/coco.json',
                        self.root / 'out', classes='motorcycle')
        self.assertEqual(report['classes'], ['motorcycle'])
        self.assertEqual(report['box_counts']['train'], {'motorcycle': 1})
        self.assertEqual(report['image_counts'], {'train': 1, 'val': 1})
        labels = next((self.root / 'out/labels/train').glob('*.txt')).read_text().splitlines()
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].split()[0], '0')

    def test_rejects_raw_radiometry(self):
        cv2.imwrite(str(self.root / 'train/frame.png'), np.full((120, 160), 30000, np.uint16))
        with self.assertRaisesRegex(ValueError, '8-bit'):
            self.run_export()

    def test_refuses_to_overwrite_an_export(self):
        self.run_export()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_export()

    def test_missing_source_does_not_make_an_empty_training_image(self):
        (self.root / 'train/frame.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or unreadable'):
            self.run_export()
        self.assertFalse((self.root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
