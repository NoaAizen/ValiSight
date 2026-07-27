"""Radar vital-signs DSP: recover a synthetic pulse, and REJECT the real
failure modes honestly.

The negative cases reconstruct the documented ways this goes wrong: bulk motion
swamping the ~0.5 mm heartbeat, a slow-time rate that cannot sample the heart
band, and a window too short to resolve it. An estimator that returns a
confident number in those cases would be worse than useless.
"""
import numpy as np

from core.vitals import (
    estimate_vitals, select_chest_bin, WAVELENGTH_M,
)


def synth_iq(fs, dur_s, resp_bpm, resp_mm, heart_bpm, heart_mm,
             noise=0.0, sway_mm=0.0, sway_hz=0.05, seed=0):
    """Complex slow-time series for a chest displacement of breathing + heart
    (+ optional bulk sway and complex noise)."""
    n = int(round(fs * dur_s))
    t = np.arange(n) / fs
    d = (resp_mm * 1e-3) * np.sin(2 * np.pi * (resp_bpm / 60.0) * t)
    d += (heart_mm * 1e-3) * np.sin(2 * np.pi * (heart_bpm / 60.0) * t)
    if sway_mm:
        d += (sway_mm * 1e-3) * np.sin(2 * np.pi * sway_hz * t)
    iq = np.exp(1j * 4.0 * np.pi * d / WAVELENGTH_M)
    if noise:
        rng = np.random.default_rng(seed)
        iq = iq + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return iq


# --- positive: recover breathing + heart rate --------------------------------

def test_recovers_breathing_and_heart_rate():
    fs = 20.0
    iq = synth_iq(fs, dur_s=20.0, resp_bpm=15.0, resp_mm=4.0,
                  heart_bpm=69.0, heart_mm=0.5, noise=0.02, seed=1)
    est = estimate_vitals(iq, fs)
    assert est.ok, est.reason
    assert abs(est.heart_bpm - 69.0) <= 4.0
    assert abs(est.resp_bpm - 15.0) <= 4.0
    assert est.confidence > 0.0


# --- negative: bulk motion swamps the heartbeat ------------------------------

def test_bulk_motion_is_rejected():
    fs = 20.0
    # 15 mm sway at 0.05 Hz -> ~30 mm p2p, far above the ~0.5 mm heartbeat
    iq = synth_iq(fs, dur_s=20.0, resp_bpm=15.0, resp_mm=4.0,
                  heart_bpm=69.0, heart_mm=0.5, noise=0.02, sway_mm=15.0, seed=2)
    est = estimate_vitals(iq, fs)
    assert not est.ok
    assert "motion" in est.reason.lower()


# --- negative: slow-time rate too low to sample the heart band ---------------

def test_low_frame_rate_is_rejected():
    fs = 3.0                                   # below MIN_FS_HZ (4 Hz)
    iq = synth_iq(fs, dur_s=20.0, resp_bpm=15.0, resp_mm=4.0,
                  heart_bpm=69.0, heart_mm=0.5, seed=3)
    est = estimate_vitals(iq, fs)
    assert not est.ok
    assert "rate" in est.reason.lower()


# --- negative: window too short for frequency resolution ---------------------

def test_short_window_is_rejected():
    fs = 20.0
    iq = synth_iq(fs, dur_s=4.0, resp_bpm=15.0, resp_mm=4.0,
                  heart_bpm=69.0, heart_mm=0.5, seed=4)
    est = estimate_vitals(iq, fs)
    assert not est.ok
    assert "window" in est.reason.lower()


# --- negative: pure noise must not yield a confident pulse -------------------

def test_pure_noise_is_not_a_pulse():
    fs = 20.0
    rng = np.random.default_rng(5)
    n = int(20.0 * fs)
    iq = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    est = estimate_vitals(iq, fs)
    assert not est.ok


# --- chest range-bin selection -----------------------------------------------

def test_select_chest_bin_picks_the_pulsing_bin():
    fs = 20.0
    n = int(20.0 * fs)
    n_bins = 16
    chest = 7
    frames = 0.1 * np.ones((n, n_bins), dtype=np.complex128)
    rng = np.random.default_rng(6)
    frames += 0.001 * (rng.standard_normal((n, n_bins))
                       + 1j * rng.standard_normal((n, n_bins)))
    pulse = synth_iq(fs, dur_s=n / fs, resp_bpm=15.0, resp_mm=4.0,
                     heart_bpm=69.0, heart_mm=0.5, seed=7)
    frames[:, chest] = pulse[:n]
    assert select_chest_bin(frames, fs) == chest
