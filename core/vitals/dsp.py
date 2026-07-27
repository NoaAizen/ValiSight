"""Pure DSP for radar vital signs. numpy only — no hardware, no I/O.

The chain, on a complex slow-time series (one range bin sampled once per frame):

    iq  ->  angle + unwrap  ->  displacement (m)  ->  detrend  ->
    band-limit  ->  dominant frequency + SNR

Everything here is testable with synthetic signals; nothing touches a radar.
"""
import numpy as np

from .params import WAVELENGTH_M


def unwrap_phase(iq):
    """Unwrapped phase (radians) of a complex slow-time series."""
    return np.unwrap(np.angle(np.asarray(iq, dtype=np.complex128)))


def phase_to_displacement(phase):
    """Chest displacement (m) from unwrapped phase: d = phase * lambda / (4 pi)."""
    return np.asarray(phase, dtype=np.float64) * WAVELENGTH_M / (4.0 * np.pi)


def detrend(x):
    """Remove DC + linear trend (slow bulk drift) via a first-order LS fit."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return x - (x.mean() if n else 0.0)
    t = np.arange(n, dtype=np.float64)
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)


def bandpass_fft(x, fs, band):
    """Zero-phase band-pass by masking rFFT bins outside ``band`` (Hz).

    FFT masking (not an IIR filter) keeps this dependency-free and exactly
    zero-phase, which matters when we later read a frequency off the result.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.fft.rfft(x)
    lo, hi = band
    spec[(freqs < lo) | (freqs > hi)] = 0.0
    return np.fft.irfft(spec, n=n)


def dominant_rate(x, fs, band):
    """Dominant frequency (Hz) within ``band``, its SNR (dB) and concentration.

    Returns ``(freq_hz, snr_db, concentration)``:

    * ``snr_db``  = peak-bin power over the median in-band power.
    * ``concentration`` = fraction of in-band energy in the peak bin (+/-1). A
      real narrowband pulse concentrates energy (-> ~1.0); white noise spreads
      it across the band (-> ~1/n_bins). This is the discriminator that keeps a
      random spectral peak from being read as a pulse — SNR alone cannot, since
      the max of ~24 random bins already sits several dB over the median.

    Returns ``(nan, -inf, 0.0)`` when the band holds no usable bins.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return float("nan"), float("-inf"), 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(x)) ** 2
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return float("nan"), float("-inf"), 0.0
    idx_in = np.flatnonzero(mask)
    local = power[idx_in]
    band_energy = float(np.sum(local))
    k = idx_in[int(np.argmax(local))]
    peak_power = power[k]

    noise = np.median(local)
    if noise <= 0.0:
        snr_db = float("inf") if peak_power > 0.0 else float("-inf")
    else:
        snr_db = 10.0 * np.log10(peak_power / noise)

    peak_energy = peak_power
    for j in (k - 1, k + 1):
        if j in idx_in:
            peak_energy += power[j]
    concentration = (peak_energy / band_energy) if band_energy > 0.0 else 0.0

    # Parabolic interpolation around the peak bin for a sub-bin frequency.
    if 0 < k < power.size - 1:
        a, b, c = power[k - 1], power[k], power[k + 1]
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    else:
        delta = 0.0
    df = freqs[1] - freqs[0] if freqs.size > 1 else 0.0
    freq = freqs[k] + delta * df
    return float(freq), float(snr_db), float(concentration)
