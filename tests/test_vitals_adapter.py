"""Integration test for the radar vitals adapter chain.

Exercises the PUBLIC entry `analyze()` end-to-end — raw ADC cube -> range FFT ->
chest-bin selection -> core.vitals — on a synthetic point target whose phase is
modulated by a known breathing + heart rate. This tests the real pipeline, not
an internal shortcut: no hardware, but the same range-FFT + bin-select path a
live DCA1000 capture would take.
"""
import numpy as np

from radar_vitals_live import analyze, parse_cfg_params

CFG = "configs/iwr1843_vitals.cfg"
WAVELENGTH_M = 299_792_458.0 / 77.0e9


def _synth_adc_cube(cfg, target_m=0.5, dur_s=25.0,
                    resp_bpm=15.0, heart_bpm=69.0, seed=0):
    p = parse_cfg_params(cfg)
    fs, n_adc, rpb = p["fs_hz"], p["n_adc"], p["range_per_bin_m"]
    k = int(round(target_m / rpb))                 # target range bin
    nf = int(dur_s * fs)
    t = np.arange(nf) / fs
    d = (4e-3 * np.sin(2 * np.pi * (resp_bpm / 60.0) * t)
         + 0.5e-3 * np.sin(2 * np.pi * (heart_bpm / 60.0) * t))
    chest_phase = 4.0 * np.pi * d / WAVELENGTH_M
    beat = np.exp(1j * 2 * np.pi * k * np.arange(n_adc) / n_adc)
    rng = np.random.default_rng(seed)
    adc = np.empty((nf, 1, 1, n_adc), dtype=np.complex64)
    for f in range(nf):
        adc[f, 0, 0, :] = beat * np.exp(1j * chest_phase[f]) \
            + 0.05 * (rng.standard_normal(n_adc)
                      + 1j * rng.standard_normal(n_adc))
    return adc, k * rpb


def test_adapter_recovers_rate_through_range_fft():
    adc, target_range = _synth_adc_cube(CFG, target_m=0.5, heart_bpm=69.0)
    est, chest_range = analyze(adc, CFG)
    assert est.ok, est.reason
    assert abs(chest_range - target_range) <= 0.10        # within ~2 range bins
    assert abs(est.heart_bpm - 69.0) <= 4.0
