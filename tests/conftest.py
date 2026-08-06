"""Shared test setup: project on sys.path + a deterministic flat baseline.

The flat Baseline has no curves, so every point scores  snr - 14.0 dB
regardless of range — tests control material outcomes purely via the snr
they put on each synthetic point:
    snr 30 -> score +16  (metal:  >= 10)
    snr 20 -> score  +6  (mid:    4..10)
    snr 12 -> score  -2  (fabric: <= 4)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import radar_material
from radar_calibration import Baseline

METAL_SNR = 30.0
MID_SNR = 20.0
SOFT_SNR = 12.0


@pytest.fixture(autouse=True)
def flat_baseline(monkeypatch):
    """Isolate every test from configs/refl_baseline.json."""
    monkeypatch.setattr(radar_material, "_baseline", Baseline())
