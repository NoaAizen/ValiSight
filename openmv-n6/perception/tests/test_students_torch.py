#!/usr/bin/env python3
"""Model smoke tests. Skipped on collection hosts without PyTorch."""
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from perception.students import (  # noqa: E402
        Derived2DChannels, RadarStudent, ThermalStudent, detection_loss,
    )


@unittest.skipUnless(torch is not None, "PyTorch is not installed on this host")
class StudentTorchTest(unittest.TestCase):
    def test_derived_channels_have_declared_shape(self):
        d = Derived2DChannels(20.0, 5.0, add_persistence=True)
        x = d(torch.rand(2, 3, 12, 16),
              torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0]),
              torch.tensor([114.0, 0.0]))
        self.assertEqual(x.shape, (2, len(d.channel_names), 12, 16))
        self.assertTrue(torch.isfinite(x).all())

    def test_thermal_forward_and_hungarian_loss(self):
        model = ThermalStudent(20.0, 5.0, max_objects=4)
        batch = {
            "thermal_seq": torch.rand(2, 3, 120, 160),
            "valid_prev1": torch.ones(2),
            "valid_prev2": torch.zeros(2),
            "dt1_ms": torch.full((2,), 114.0),
            "supervision_state": torch.tensor([-1, 1]),
            "gt_presence": torch.tensor(
                [[0, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float32),
            "gt_boxes": torch.tensor([
                [[0, 0, 0, 0]] * 4,
                [[0.5, 0.5, 0.2, 0.4], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0]],
            ], dtype=torch.float32),
            "gt_confidence": torch.ones(2, 4),
        }
        out = model(batch)
        self.assertEqual(out["boxes"].shape, (2, 4, 4))
        loss = detection_loss(out, batch)["total"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_radar_streams_and_family_ablation(self):
        model = RadarStudent(
            {"ra": (0.0, 1.0), "rd": (0.0, 1.0), "rp": (0.0, 1.0)},
            max_objects=4, use_ra=True, use_rd=True, use_rp=True)
        batch = {
            "radar_points": torch.rand(2, 8, 6),
            "n_radar": torch.tensor([5, 0]),
            "dt1_ms": torch.full((2,), 114.0),
        }
        for prefix, shape in (
            ("ra", (3, 16, 12)), ("rd", (3, 16, 8)),
            ("rp", (3, 32)),
        ):
            batch[f"{prefix}_seq"] = torch.rand((2,) + shape)
            batch[f"{prefix}_valid"] = torch.tensor([1.0, 0.0])
            batch[f"{prefix}_valid_prev1"] = torch.ones(2)
            batch[f"{prefix}_valid_prev2"] = torch.zeros(2)
        model.disable_family("points")
        out = model(batch)
        self.assertEqual(out["boxes"].shape, (2, 4, 4))
        self.assertFalse(out["point_latent"].any())


if __name__ == "__main__":
    unittest.main()

