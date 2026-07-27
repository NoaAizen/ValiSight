"""Physical and DSP parameters for radar vital-signs estimation (pure).

All tunables are named here (no magic numbers inline in the DSP) so the physics
is auditable and the physics-lint stays quiet.
"""
C_M_S = 299_792_458.0
# IWR1843 chirp start is ~77 GHz (see configs/*.cfg profileCfg).
CARRIER_HZ = 77.0e9
WAVELENGTH_M = C_M_S / CARRIER_HZ            # ~3.9 mm at 77 GHz

SEC_PER_MIN = 60.0

# Physiological bands.
RESP_BAND_HZ = (0.1, 0.5)                    # 6-30 breaths/min
HEART_BAND_HZ = (0.8, 2.0)                   # 48-120 bpm

# Slow-time sampling must clear the heart-band Nyquist with margin.
MIN_FS_HZ = 4.0
# Need a long enough window for usable frequency resolution.
MIN_WINDOW_S = 8.0

# Reliability gates. Below these the estimate is not trustworthy and the
# pipeline reports "not reliable" instead of a number.
#
# SNR alone cannot reject noise: the max of the ~20-30 bins in a physiological
# band already sits ~6-7 dB over the median for white noise. Concentration (the
# fraction of in-band energy in the peak +/-1 bin) is the real discriminator —
# a narrowband pulse -> ~1.0, white noise -> ~1/n_bins. Both gates must pass.
MIN_HEART_SNR_DB = 8.0
MIN_RESP_SNR_DB = 6.0
MIN_HEART_CONCENTRATION = 0.30
MIN_RESP_CONCENTRATION = 0.30

# Peak-to-peak chest displacement (after detrend) above this means bulk motion,
# not vitals — the heartbeat is ~0.1-0.5 mm, breathing up to ~12 mm, so >2 cm
# of residual motion has swamped the signal we care about.
MOTION_P2P_LIMIT_M = 0.02
