"""core.vitals — pure radar vital-signs DSP (no hardware, no I/O).

Estimates breathing and heart rate from the phase of a single radar range bin
sampled over slow time. This is the physically correct path for pulse (the
mmWave literature and our own corpus: raw ADC phase, NOT the CFAR point cloud,
which discards the sub-mm chest motion before the UART).

numpy is the only dependency — pure computation, fully testable off-hardware.
The adapter that feeds this from a live DCA1000 raw capture (range FFT -> chest
bin -> slow-time phase) lives outside core, at the app/adapter layer.

Honesty is built in: :func:`estimate_vitals` gates on slow-time rate, window
length, bulk motion and heart-band SNR, and returns ``ok=False`` with a reason
rather than a fabricated number when the pulse is not trustworthy.
"""
from .params import (
    WAVELENGTH_M, CARRIER_HZ, RESP_BAND_HZ, HEART_BAND_HZ,
    MIN_FS_HZ, MIN_WINDOW_S, MIN_HEART_SNR_DB, MIN_RESP_SNR_DB,
    MOTION_P2P_LIMIT_M,
)
from .dsp import (
    unwrap_phase, phase_to_displacement, detrend, bandpass_fft, dominant_rate,
)
from .estimate import VitalsEstimate, estimate_vitals, select_chest_bin

__all__ = [
    "WAVELENGTH_M", "CARRIER_HZ", "RESP_BAND_HZ", "HEART_BAND_HZ",
    "MIN_FS_HZ", "MIN_WINDOW_S", "MIN_HEART_SNR_DB", "MIN_RESP_SNR_DB",
    "MOTION_P2P_LIMIT_M",
    "unwrap_phase", "phase_to_displacement", "detrend", "bandpass_fft",
    "dominant_rate",
    "VitalsEstimate", "estimate_vitals", "select_chest_bin",
]
