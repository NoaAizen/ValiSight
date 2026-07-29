"""What an IWR1843 chirp configuration can and cannot measure. Pure maths.

No file I/O: `parse_cfg_text` takes the text of a .cfg, `geometry` takes numbers.
Reading the file is the caller's job, which keeps this importable from the pure
side of the project and from a replay that has only a recorded copy of the
config, not the file itself.

WHY THIS MODULE EXISTS: THE WAVELENGTH.

The obvious way to get lambda is `c / startFreq`, and it is wrong by 2.7%. The
demo firmware reports Doppler using the frequency at the CENTRE OF THE ADC
SAMPLING WINDOW, not at the start of the ramp. The ramp begins at startFreq and
climbs at `slope`; sampling begins `adcStartTime` into it and lasts
`numAdcSamples / digOutSampleRate`, so

    f_centre = startFreq + slope * (adcStartTime + dwell / 2)

On stock_iwr1843.cfg that is 79.21 GHz, not 77.00 -- a 2.9% shift that lands
directly on every velocity the system reports. Measured against 60144 points of
session 20260728_203833, where the Doppler axis is quantised onto exactly 16
levels 0.121715 m/s apart with a maximum magnitude of 0.973723:

    lambda at startFreq (77.00 GHz)   -> bin 0.125120   v_max 1.00096   (+2.8%)
    lambda at ADC centre (79.21 GHz)  -> bin 0.121713   v_max 0.973706  (-0.001%)

The second reproduces the hardware to one part in 10^5. This module is the only
place that formula lives, so it cannot drift back.

TI's firmware uses c = 3e8 rather than the exact speed of light. Using the exact
value here would reintroduce a 0.07% error against the measured bins, so C below
is deliberately 3e8. It is a property of the firmware, not of physics.

THE SECOND TRAP: CHIRP SLOTS ARE NOT TRANSMITTERS.

The Doppler sampling interval is set by how long the frame sequencer's loop
takes, and the loop walks `chirpStartIdx .. chirpEndIdx` from frameCfg. It spends
idle+rampEnd on every slot in that range whether or not the slot's TX survived
channelCfg's mask. So a config that masks a transmitter off WITHOUT shortening
the chirp range keeps paying for the dead slot: the interval between one TX's
successive chirps stays at slots*Tc, v_max stays where it was, and part of the
frame's active time transmits nothing. `n_slots` and `n_tx` are therefore
tracked separately here, and `slots_wasted` says when they disagree.

The Doppler bin count is `numLoops` regardless: each active TX fires once per
loop, so it collects one sample per loop no matter how long the loop is.
"""
import math

C = 3e8                     # as the TI firmware uses it; see the docstring


def _popcount(x):
    n = 0
    while x:
        n += x & 1
        x >>= 1
    return n


def geometry(start_ghz, slope_mhz_us, adc_start_us, n_adc, fs_ksps,
             idle_us, ramp_us, tx_mask, n_loops, n_slots, frame_ms):
    """Derived limits for one chirp configuration.

    `n_slots` is chirpEndIdx - chirpStartIdx + 1 from frameCfg -- the length of
    the sequencer's loop -- NOT the number of enabled transmitters. Pass both;
    the difference is a real and silent config bug.
    """
    slope = slope_mhz_us * 1e12                  # MHz/us -> Hz/s
    fs = fs_ksps * 1e3
    dwell = n_adc / fs                           # s of actual sampling
    f_centre = start_ghz * 1e9 + slope * (adc_start_us * 1e-6 + dwell / 2.0)
    lam = C / f_centre

    b_valid = slope * dwell                      # swept bandwidth ACTUALLY sampled
    tc = (idle_us + ramp_us) * 1e-6
    tc_eff = n_slots * tc                        # one TX's chirp-to-chirp interval
    n_tx = _popcount(tx_mask)

    v_max = lam / (4.0 * tc_eff)
    bin_mps = lam / (2.0 * n_loops * tc_eff)
    active = n_loops * n_slots * tc
    frame_s = frame_ms * 1e-3

    return {
        "lambda_m": lam,
        "f_centre_hz": f_centre,
        "range_res_m": C / (2.0 * b_valid),
        "r_max_m": fs * C / (2.0 * slope),
        "b_valid_hz": b_valid,
        "tc_us": tc * 1e6,
        "tc_eff_us": tc_eff * 1e6,
        "n_tx": n_tx,
        "n_slots": n_slots,
        # A slot whose TX is masked off still costs its idle+ramp. Non-zero here
        # means v_max is worse than the transmitter count would suggest.
        "slots_wasted": n_slots - n_tx,
        "n_bins": n_loops,
        "n_loops": n_loops,
        "v_max": v_max,
        "bin_mps": bin_mps,
        "frame_s": frame_s,
        "active_s": active,
        "duty": (active / frame_s) if frame_s else None,
    }


def parse_cfg_text(text):
    """Text of a .cfg -> the keyword arguments `geometry` wants.

    Tolerates the '%%'/'%' comment headers the project's configs carry, and
    ignores every line that is not one of the four that set the geometry.
    Returns None if any of the four is missing, because a partial answer here
    would be a confidently wrong v_max.
    """
    prof = frame = chan = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        key = parts[0]
        try:
            if key == "profileCfg" and len(parts) >= 12:
                prof = parts
            elif key == "frameCfg" and len(parts) >= 6:
                frame = parts
            elif key == "channelCfg" and len(parts) >= 3:
                chan = parts
        except (ValueError, IndexError):
            return None
    if not (prof and frame and chan):
        return None

    try:
        return {
            "start_ghz":    float(prof[2]),
            "idle_us":      float(prof[3]),
            "adc_start_us": float(prof[4]),
            "ramp_us":      float(prof[5]),
            "slope_mhz_us": float(prof[8]),
            "n_adc":        int(float(prof[10])),
            "fs_ksps":      float(prof[11]),
            "tx_mask":      int(chan[2]),        # channelCfg <rx> <tx> <cascade>
            "n_slots":      int(float(frame[2])) - int(float(frame[1])) + 1,
            "n_loops":      int(float(frame[3])),
            "frame_ms":     float(frame[5]),
        }
    except (ValueError, IndexError):
        return None


def from_cfg_text(text):
    """Convenience: text -> geometry dict, or None if it cannot be parsed."""
    kw = parse_cfg_text(text)
    return geometry(**kw) if kw else None


def summary_line(g):
    """The one-line form used in the `derived:` header of each .cfg."""
    return ("v_max %.2f m/s | v_res %.4f m/s | range_res %.2f cm | frame %.0f ms"
            % (g["v_max"], g["bin_mps"], g["range_res_m"] * 100,
               g["frame_s"] * 1e3))


# --- the stock geometry, which is what the module-level constants elsewhere in
# --- the project describe. Recomputed rather than copied, so it cannot drift.
STOCK = geometry(start_ghz=77.0, slope_mhz_us=70.0, adc_start_us=7.0,
                 n_adc=240, fs_ksps=4884.0, idle_us=267.0, ramp_us=57.14,
                 tx_mask=7, n_loops=16, n_slots=3, frame_ms=100.0)
