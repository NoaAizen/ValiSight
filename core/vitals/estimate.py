"""High-level vital-signs estimate with honest reliability gating (pure).

``estimate_vitals`` turns a complex slow-time series into breathing and heart
rates *with a confidence*, and refuses to emit a heart rate when the physics
says it cannot be trusted — the slow-time rate is too low, the window too short,
the subject moved (bulk motion swamps the ~0.5 mm heartbeat), or the heart-band
SNR is below threshold. A gated-out result reports ``ok=False`` and why, never a
fabricated number.
"""
from collections import namedtuple

import numpy as np

from .dsp import (
    unwrap_phase, phase_to_displacement, detrend, bandpass_fft, dominant_rate,
)
from .params import (
    RESP_BAND_HZ, HEART_BAND_HZ, MIN_FS_HZ, MIN_WINDOW_S,
    MIN_HEART_SNR_DB, MIN_RESP_SNR_DB, MIN_HEART_CONCENTRATION,
    MIN_RESP_CONCENTRATION, MOTION_P2P_LIMIT_M, SEC_PER_MIN,
)

VitalsEstimate = namedtuple(
    "VitalsEstimate",
    ["ok", "reason", "heart_bpm", "heart_snr_db",
     "resp_bpm", "resp_snr_db", "confidence", "fs_hz", "window_s"])


def _confidence(heart_snr_db):
    """Map heart-band SNR (dB) to a 0..1 confidence, saturating ~12 dB over gate."""
    span = 12.0
    c = (heart_snr_db - MIN_HEART_SNR_DB) / span
    return float(min(max(c, 0.0), 1.0))


def estimate_vitals(iq, fs):
    """Estimate breathing + heart rate from a complex slow-time series.

    Parameters
    ----------
    iq : complex slow-time samples of one range bin (one sample per frame).
    fs : slow-time sampling rate in Hz (the radar frame rate).

    Returns
    -------
    VitalsEstimate. ``ok`` is False (with a reason) whenever the estimate is not
    physically trustworthy.
    """
    iq = np.asarray(iq)
    n = iq.size
    window_s = (n / fs) if fs > 0 else 0.0

    def fail(reason, heart_bpm=None, heart_snr=float("-inf"),
             resp_bpm=None, resp_snr=float("-inf")):
        return VitalsEstimate(False, reason, heart_bpm, heart_snr,
                              resp_bpm, resp_snr, 0.0, fs, window_s)

    if fs < MIN_FS_HZ:
        return fail("slow-time rate %.2f Hz is below the %.1f Hz needed to "
                    "sample the heart band without aliasing" % (fs, MIN_FS_HZ))
    if window_s < MIN_WINDOW_S:
        return fail("window %.1f s is shorter than the %.1f s needed for usable "
                    "frequency resolution" % (window_s, MIN_WINDOW_S))

    disp = phase_to_displacement(unwrap_phase(iq))
    dtr = detrend(disp)

    p2p = float(np.max(dtr) - np.min(dtr)) if n else 0.0
    resp_sig = bandpass_fft(dtr, fs, RESP_BAND_HZ)
    heart_sig = bandpass_fft(dtr, fs, HEART_BAND_HZ)
    resp_f, resp_snr, resp_conc = dominant_rate(resp_sig, fs, RESP_BAND_HZ)
    heart_f, heart_snr, heart_conc = dominant_rate(heart_sig, fs, HEART_BAND_HZ)

    resp_bpm = resp_f * SEC_PER_MIN if np.isfinite(resp_f) else None
    heart_bpm = heart_f * SEC_PER_MIN if np.isfinite(heart_f) else None

    if p2p > MOTION_P2P_LIMIT_M:
        return fail("bulk motion detected: %.1f mm peak-to-peak chest "
                    "displacement (> %.0f mm) swamps the ~0.5 mm heartbeat — "
                    "subject must be still" % (p2p * 1e3, MOTION_P2P_LIMIT_M * 1e3),
                    heart_bpm, heart_snr, resp_bpm, resp_snr)
    if heart_snr < MIN_HEART_SNR_DB or heart_conc < MIN_HEART_CONCENTRATION:
        return fail("no reliable pulse: heart-band SNR %.1f dB (gate %.1f), "
                    "concentration %.2f (gate %.2f) — consistent with noise or "
                    "motion, not a narrowband pulse"
                    % (heart_snr, MIN_HEART_SNR_DB, heart_conc,
                       MIN_HEART_CONCENTRATION),
                    heart_bpm, heart_snr, resp_bpm, resp_snr)

    resp_reliable = (resp_bpm is not None and resp_snr >= MIN_RESP_SNR_DB
                     and resp_conc >= MIN_RESP_CONCENTRATION)
    reason = "heart %.0f bpm (SNR %.1f dB), resp %s" % (
        heart_bpm, heart_snr,
        "%.0f bpm (SNR %.1f dB)" % (resp_bpm, resp_snr)
        if resp_reliable else "not reliable")
    return VitalsEstimate(True, reason, heart_bpm, heart_snr,
                          resp_bpm, resp_snr, _confidence(heart_snr), fs,
                          window_s)


def select_chest_bin(frames, fs, bin_range=None):
    """Pick the range bin whose phase pulses the most — the chest.

    ``frames`` is a complex array shaped ``(slow_time, range_bins)`` (one range
    profile per frame), ``fs`` the frame rate. Returns the index of the bin with
    the largest *physiological-band* phase variance (band-limited so a broadband-
    noisy bin does not win), optionally restricted to ``bin_range=(lo, hi)``.
    """
    frames = np.asarray(frames, dtype=np.complex128)
    if frames.ndim != 2:
        raise ValueError("frames must be 2-D (slow_time, range_bins)")
    _, n_bins = frames.shape
    lo, hi = (0, n_bins) if bin_range is None else bin_range
    lo = max(0, lo)
    hi = min(n_bins, hi)
    vitals_band = (RESP_BAND_HZ[0], HEART_BAND_HZ[1])
    best_idx, best_score = lo, -1.0
    for b in range(lo, hi):
        dtr = detrend(unwrap_phase(frames[:, b]))
        band_sig = bandpass_fft(dtr, fs, vitals_band)
        score = float(np.var(band_sig))
        if score > best_score:
            best_score, best_idx = score, b
    return best_idx
