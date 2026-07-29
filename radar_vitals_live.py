"""Radar vital-signs adapter: DCA1000 raw ADC -> range FFT -> chest bin ->
core.vitals (breathing + heart rate).

This is the ADAPTER layer. The pure DSP + the honesty gating live in
`core/vitals/`; this file owns the hardware: it sends the vitals config to the
IWR1843, captures the raw ADC stream off the DCA1000, turns each frame into a
range profile, tracks the phase of the chest range bin over slow time, and
hands that to `core.vitals.estimate_vitals`.

Why this path and not the live point cloud: the TLV point cloud is CFAR-
thresholded on-chip, which throws away the sub-millimetre chest phase before it
ever reaches the UART. Vital signs need the raw ADC — exactly what the DCA1000
provides (and what our corpus says raw ADC unlocks).

Honest limits (enforced by core.vitals, surfaced here): the subject must be
nearly still, ~0.3-1.0 m away, chest toward the radar, alone in the range bin.
Any bulk motion swamps the ~0.5 mm heartbeat and the tool will report
"no reliable pulse" rather than a fabricated number.

Requires: DCA1000 on the LVDS connector, PC NIC static at 192.168.33.30, the
IWR1843 on its COM ports.

Usage:
    python radar_vitals_live.py --seconds 25          # live capture + estimate
    python radar_vitals_live.py --from-bin capture.bin  # offline, no hardware
    python radar_vitals_live.py --cfg-port COM8 --seconds 30
"""
import argparse
import os
import sys
import tempfile

import numpy as np

from core.vitals import estimate_vitals, select_chest_bin

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFG = os.path.join(_ROOT, "configs", "iwr1843_vitals.cfg")

C_M_S = 299_792_458.0
# Plausible chest range window (m) for a seated/standing subject in front of it.
CHEST_RANGE_M = (0.3, 1.5)


def parse_cfg_params(cfg_path):
    """Pull the geometry needed for the range axis and slow-time rate from a
    demo .cfg: chirp slope, ADC sample rate, ADC samples, and frame period."""
    slope_hz_s = adc_rate_sps = n_adc = frame_ms = None
    with open(cfg_path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "profileCfg":
                slope_hz_s = float(t[8]) * 1e6 / 1e-6      # MHz/us -> Hz/s
                n_adc = int(t[10])
                adc_rate_sps = float(t[11]) * 1e3          # ksps -> sps
            elif t[0] == "frameCfg":
                frame_ms = float(t[5])
    if None in (slope_hz_s, adc_rate_sps, n_adc, frame_ms):
        raise ValueError("could not parse profileCfg/frameCfg from %s" % cfg_path)
    fs_hz = 1000.0 / frame_ms
    range_per_bin_m = C_M_S * adc_rate_sps / (2.0 * slope_hz_s * n_adc)
    return {"fs_hz": fs_hz, "n_adc": n_adc,
            "range_per_bin_m": range_per_bin_m}


def slow_time_profiles(adc, chirp=0, rx=0):
    """(n_frames, n_chirps, n_rx, n_samples) complex ADC -> (n_frames, n_bins)
    complex range profiles for one chirp/rx, windowed range-FFT, positive half."""
    frames = np.asarray(adc)[:, chirp, rx, :]
    win = np.hanning(frames.shape[1])
    spec = np.fft.fft(frames * win, axis=1)
    half = frames.shape[1] // 2
    return spec[:, :half]


def analyze(adc, cfg_path, chest_range_m=CHEST_RANGE_M):
    """Full offline analysis: raw ADC array -> (VitalsEstimate, chest_range_m)."""
    p = parse_cfg_params(cfg_path)
    profiles = slow_time_profiles(adc)
    rpb = p["range_per_bin_m"]
    lo_bin = max(1, int(chest_range_m[0] / rpb))
    hi_bin = min(profiles.shape[1], int(chest_range_m[1] / rpb) + 1)
    chest = select_chest_bin(profiles, p["fs_hz"], bin_range=(lo_bin, hi_bin))
    iq = profiles[:, chest]
    est = estimate_vitals(iq, p["fs_hz"])
    return est, chest * rpb


def capture_live(cfg_path, seconds, cfg_port=None):
    """Send the vitals config, capture the DCA1000 raw stream, return the ADC
    array (n_frames, n_chirps, n_rx, n_samples). Needs the hardware."""
    import serial

    import iwr1843_uart
    from dca1000 import DCA1000, RawCapture, parse_adc

    if cfg_port is None:
        cfg_port, _ = iwr1843_uart.find_com_ports()
    if not cfg_port:
        raise SystemExit("CONFIG COM port not found; pass --cfg-port")

    print("sending %s on %s ..." % (os.path.basename(cfg_path), cfg_port))
    with serial.Serial(cfg_port, 115200, timeout=1) as cfg_ser:
        iwr1843_uart.send_config(cfg_ser, cfg_path)

    dca = DCA1000()
    try:
        dca.connect()
        print("DCA1000 FPGA %s — recording %.0fs (hold still)"
              % (dca.fpga_version(), seconds))
        dca.configure()
        tmp = os.path.join(tempfile.gettempdir(), "vitals_adc.bin")
        cap = RawCapture(tmp)
        cap.start()
        dca.start_record()
        import time
        time.sleep(seconds)
        dca.stop_record()
        cap.stop()
        print("capture stats:", cap.stats())
    finally:
        dca.close()
    return parse_adc(tmp, cfg_path)


def report(est, chest_range_m):
    print("\n=== radar vital signs ===")
    print("chest range bin: %.2f m" % chest_range_m)
    print("window: %.1f s @ %.1f Hz slow-time" % (est.window_s, est.fs_hz))
    if not est.ok:
        print("RESULT: no reliable pulse")
        print("  reason: %s" % est.reason)
        return
    print("RESULT: heart %.0f bpm (SNR %.1f dB, confidence %.2f)"
          % (est.heart_bpm, est.heart_snr_db, est.confidence))
    if est.resp_bpm is not None:
        print("        resp  %.0f bpm (SNR %.1f dB)"
              % (est.resp_bpm, est.resp_snr_db))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default=DEFAULT_CFG, help="radar vitals .cfg")
    ap.add_argument("--seconds", type=float, default=25.0,
                    help="live capture duration (>= 8 s; longer = better)")
    ap.add_argument("--from-bin", metavar="ADC.bin",
                    help="analyze an existing raw capture instead of going live")
    ap.add_argument("--cfg-port", help="radar CONFIG COM port (else auto)")
    ap.add_argument("--chest-range", nargs=2, type=float, metavar=("LO", "HI"),
                    default=CHEST_RANGE_M,
                    help="restrict the chest search to this range window (m). "
                         "Narrow it around the subject when a strong wall "
                         "behind them wins the bin selection (default %s %s)"
                         % CHEST_RANGE_M)
    args = ap.parse_args()

    if args.from_bin:
        from dca1000 import parse_adc
        adc = parse_adc(args.from_bin, args.cfg)
    else:
        adc = capture_live(args.cfg, args.seconds, args.cfg_port)

    est, chest_range_m = analyze(adc, args.cfg,
                                 chest_range_m=tuple(args.chest_range))
    report(est, chest_range_m)
    return 0 if est.ok else 1


if __name__ == "__main__":
    sys.exit(main())
