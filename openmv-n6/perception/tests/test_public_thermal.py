"""The public-dataset converter, checked against the loader that consumes it.

The interesting failures here are all label failures, not shape failures: a
frame whose people the crop removed becoming a NEGATIVE, an image with no
annotation file silently reading as empty, a 5:4 frame squashed into 4:3 so
every person is 7% wider than they are. Each of those trains the model on a
lie that no amount of epochs recovers from.
"""
import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

try:
    import cv2
except ImportError:                                   # pragma: no cover
    cv2 = None

from perception.external import public_thermal as pt
from perception.student_data import (
    LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNKNOWN, detection_target,
    load_split,
)


def _voc(path, boxes, name="person"):
    root = ET.Element("annotation")
    for (x, y, w, h) in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        bb = ET.SubElement(obj, "bndbox")
        for tag, v in (("xmin", x), ("ymin", y),
                       ("xmax", x + w), ("ymax", y + h)):
            ET.SubElement(bb, tag).text = str(int(v))
    ET.ElementTree(root).write(path)


@unittest.skipIf(cv2 is None, "cv2 is required to read the frames")
class PublicThermalTest(unittest.TestCase):

    def test_a_54_frame_is_cropped_not_squashed(self):
        # 1280x1024 is 5:4. Resizing it to 4:3 would widen every person by 7%,
        # and shape is the only thing that separates a person from a warm
        # rectangle in this data.
        crop, boxes = pt._boxes_to_thermal_plane(
            [(600, 400, 100, 300)], 1280, 1024)
        x0, y0, cw, ch = crop
        self.assertEqual((cw, ch), (1280, 960))       # trimmed top and bottom
        self.assertEqual(x0, 0)
        self.assertEqual(y0, 32)
        (bx, by, bw, bh), = boxes
        self.assertAlmostEqual(bw / bh, 100 / 300.0, places=2)

    def test_a_person_the_crop_cut_away_is_unknown_not_empty(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "infrared", "train"))
            os.makedirs(os.path.join(td, "Annotations"))
            img = np.full((1024, 1280), 80, np.uint8)
            cv2.imwrite(os.path.join(td, "infrared", "train", "a.jpg"), img)
            # a person in the strip the 4:3 crop removes
            _voc(os.path.join(td, "Annotations", "a.xml"), [(600, 2, 40, 20)])
            out = os.path.join(td, "out")
            stats = pt.convert(pt.read_llvip(td), out, "s")
            self.assertEqual(stats["negative"], 0, "cropped-away person "
                             "must never be exported as an empty frame")
            z = np.load(os.path.join(out, stats["shards"][0]))
            self.assertEqual(int(z["thermal_label_state"][0]), LABEL_UNKNOWN)

    def test_an_image_without_an_annotation_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "infrared", "train"))
            cv2.imwrite(os.path.join(td, "infrared", "train", "a.jpg"),
                        np.zeros((1024, 1280), np.uint8))
            with self.assertRaises(SystemExit):
                list(pt.read_llvip(td))

    def test_coco_frames_without_people_become_verified_negatives(self):
        # The premise, and the only reason these are usable where our own
        # unlabelled frames are not: the annotation is exhaustive.
        with tempfile.TemporaryDirectory() as td:
            data = os.path.join(td, "data")
            os.makedirs(data)
            images, anns = [], []
            for i in range(4):
                img = np.full((512, 640), 70, np.uint8)
                if i % 2:
                    img[150:370, 200:260] = 210
                    anns.append({"id": i, "image_id": i, "category_id": 1,
                                 "bbox": [200, 150, 60, 220], "iscrowd": 0})
                cv2.imwrite(os.path.join(data, "f%d.jpg" % i), img)
                images.append({"id": i, "file_name": "data/f%d.jpg" % i,
                               "width": 640, "height": 512})
            with open(os.path.join(td, "coco.json"), "w") as f:
                json.dump({"images": images, "annotations": anns,
                           "categories": [{"id": 1, "name": "person"},
                                          {"id": 2, "name": "car"}]}, f)
            out = os.path.join(td, "out")
            stats = pt.convert(pt.read_coco(td, "coco.json"), out, "flir")
            self.assertEqual((stats["positive"], stats["negative"]), (2, 2))

            z = np.load(os.path.join(out, stats["shards"][0]))
            states = list(z["thermal_label_state"])
            self.assertEqual(states, [LABEL_NEGATIVE, LABEL_POSITIVE,
                                      LABEL_NEGATIVE, LABEL_POSITIVE])
            self.assertEqual(z["thermal"].shape[1:], (120, 160))
            # No radar was within a hundred kilometres of these frames.
            self.assertTrue((z["radar_label_state"] == LABEL_UNKNOWN).all())

    def test_the_export_loads_through_the_real_loader(self):
        with tempfile.TemporaryDirectory() as td:
            data = os.path.join(td, "data")
            os.makedirs(data)
            images, anns = [], []
            for i in range(6):
                img = np.full((512, 640), 70, np.uint8)
                if i < 3:
                    img[150:370, 200:260] = 210
                    anns.append({"id": i, "image_id": i, "category_id": 1,
                                 "bbox": [200, 150, 60, 220], "iscrowd": 0})
                cv2.imwrite(os.path.join(data, "f%d.jpg" % i), img)
                images.append({"id": i, "file_name": "data/f%d.jpg" % i,
                               "width": 640, "height": 512})
            with open(os.path.join(td, "coco.json"), "w") as f:
                json.dump({"images": images, "annotations": anns,
                           "categories": [{"id": 1, "name": "person"}]}, f)
            out = os.path.join(td, "out")
            stats = pt.convert(pt.read_coco(td, "coco.json"), out, "flir",
                               per_shard=3)
            sessions, val = {}, set()
            for i, shard in enumerate(stats["shards"]):
                name = "flir-%03d" % i
                sessions[name] = {
                    "shards": [shard], "frames": None, "c_per_lsb": None,
                    "tmin": None, "tmax": None, "thermal_dtype": "uint8",
                    "thermal_counts_max": 255,
                    "thermal_encoding": "agc_8bit_unit", "provenance": {},
                }
                if i:
                    val.add(name)
            pt.write_manifest(out, sessions, val, {"dataset": "test"}, "test")

            split = load_split(out, "train")
            # Unit scale, not Celsius - and the loader must say so, because
            # mixing the two scales in one split is what this separation of
            # exports exists to prevent.
            self.assertEqual(split.thermal_mode, "unit")
            self.assertLessEqual(float(split.arrays["thermal"].max()), 1.0)

            target = detection_target(split, 0, "thermal", 8)
            self.assertEqual(target["supervision_state"], LABEL_POSITIVE)
            cx, cy, w, h = target["gt_boxes"][0]
            self.assertTrue(0.0 < cx < 1.0 and 0.0 < cy < 1.0)
            self.assertGreater(h, w, "a standing person is taller than wide")

            # Stills are not a temporal chain, and must not be linked into one.
            _, valid = split.sequence("thermal", 1)
            self.assertEqual([bool(v) for v in valid], [True, False, False])


if __name__ == "__main__":
    unittest.main()
