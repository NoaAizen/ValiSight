"""Thermal blob detection: synthetic frames with known ground truth.

The detector is pure, so unlike the aliasing sweep nothing here needs a
recording: every frame is built pixel-by-pixel around one question — does a
warm person become a Box, does thermal's failure mode become a weak/absent
Box, and do the synthetic (interpolated) dead rows stay out of the evidence.

Run:  cd <project root> && python3 -m unittest discover -s tests
"""
import sys
import unittest

import numpy as np

import _path                                # noqa: F401  (sys.path side effect)
if _path.ROOT not in sys.path:
    sys.path.insert(0, _path.ROOT)

import lepton_fix
from core.detect import thermal as td
from core.fusion import uncertainty as unc

H, W = 120, 160
RNG_SEED = 20260729


def _scene(rng, bg=90.0, noise=2.0):
    """A quiet indoor scene in AGC DN: flat background + sensor noise."""
    return np.clip(bg + rng.normal(0.0, noise, (H, W)), 0, 255).astype(np.uint8)


def _add_person(frame, cy, cx, hh=18, hw=6, dn=40.0):
    """Stamp a warm upright rectangle (a person-ish blob) onto a frame."""
    out = frame.astype(np.float32, copy=True)
    out[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw] += dn
    return np.clip(out, 0, 255).astype(np.uint8)


class TestDetection(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(RNG_SEED)

    def test_warm_person_becomes_one_confident_box(self):
        frame = _add_person(_scene(self.rng), cy=80, cx=60)
        boxes = td.detect(frame, lepton_fix.DEAD_ROWS)
        self.assertEqual(len(boxes), 1)
        b = boxes[0]
        self.assertGreater(b.confidence, 0.6)
        # the box actually covers the stamp
        self.assertTrue(b.x <= 54 and b.x + b.w >= 66)
        self.assertTrue(b.y <= 62 and b.y + b.h >= 98)

    def test_flat_field_is_thermal_blindness(self):
        """No contrast is the failure signature: no boxes, and the fusion
        seam turns that into a high-sigma thermal estimate."""
        frame = _scene(self.rng)
        self.assertEqual(td.detect(frame, lepton_fix.DEAD_ROWS), [])
        est = unc.thermal_estimate(td.best_box(frame, lepton_fix.DEAD_ROWS))
        self.assertEqual(est.sigma, unc.THERMAL_SIGMA_BLIND)

    def test_low_contrast_scores_below_high_contrast(self):
        hot = _add_person(_scene(self.rng), 60, 80, dn=45.0)
        dim = _add_person(_scene(self.rng), 60, 80, dn=12.0)
        c_hot = td.detect(hot, lepton_fix.DEAD_ROWS)[0].confidence
        c_dim = td.detect(dim, lepton_fix.DEAD_ROWS)[0].confidence
        self.assertGreater(c_hot, c_dim + 0.2)

    def test_blob_only_in_dead_rows_is_not_evidence(self):
        """A 'target' painted entirely into rows 55-63 is interpolation
        artefact territory: repair() invents those pixels, so nothing that
        exists only there may become a detection."""
        frame = _scene(self.rng).astype(np.float32)
        frame[55:64, 40:100] += 50.0                 # the 9-row dead run
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        self.assertEqual(td.detect(frame, lepton_fix.DEAD_ROWS), [])

    def test_person_straddling_dead_run_stays_one_discounted_box(self):
        """A person crossing rows 55-63 must not split in two, and the unseen
        band must cost confidence rather than add it."""
        clear = _add_person(_scene(self.rng), cy=90, cx=60, hh=14)
        split = _add_person(_scene(self.rng), cy=59, cx=60, hh=14)  # rows 45..72
        b_clear = td.detect(clear, lepton_fix.DEAD_ROWS)
        b_split = td.detect(split, lepton_fix.DEAD_ROWS)
        self.assertEqual(len(b_clear), 1)
        self.assertEqual(len(b_split), 1, "blob split across the dead-row run")
        self.assertLess(b_split[0].valid_fraction, 0.8)
        self.assertLess(b_split[0].confidence, b_clear[0].confidence)
        # the box still spans the full person, dead band included
        self.assertTrue(b_split[0].y <= 46 and b_split[0].y + b_split[0].h >= 72)

    def test_box_feeds_thermal_estimate_directly(self):
        frame = _add_person(_scene(self.rng), 80, 60)
        box = td.best_box(frame, lepton_fix.DEAD_ROWS)
        est = unc.thermal_estimate(box)
        self.assertEqual(est.sensor, unc.THERMAL)
        self.assertAlmostEqual(est.score, box.confidence, delta=1e-12)

    def test_confidences_bounded_on_random_frames(self):
        for _ in range(20):
            frame = self.rng.integers(0, 256, (H, W), dtype=np.uint8)
            for b in td.detect(frame, lepton_fix.DEAD_ROWS):
                self.assertTrue(0.0 <= b.confidence <= 1.0)
                self.assertGreaterEqual(b.area_px, td.MIN_AREA_PX)


if __name__ == "__main__":
    unittest.main()
