# core/vitals

Pure radar vital-signs DSP — breathing and heart rate from the phase of a
single radar range bin over slow time. numpy only; no hardware, no I/O. Fully
testable off-hardware with synthetic signals.

## Why radar, and why raw ADC (not the point cloud)

A heartbeat moves the chest wall ~0.1–0.5 mm; breathing ~1–12 mm. At 77 GHz
(λ ≈ 3.9 mm) that is a clean phase signal: 0.5 mm → ~1.6 rad. But the IWR1843
TLV **point cloud is CFAR-thresholded on-chip**, which discards that sub-mm
phase before the UART (our corpus, conclusion #2; EchoFusion). So vital signs
must come from the **raw ADC** (DCA1000) → range FFT → chest range-bin phase →
this module. The live point-cloud config physically cannot do it.

## The chain

```
iq (one range bin, one sample/frame)
  -> unwrap_phase        angle + np.unwrap
  -> phase_to_displacement   d = phase·λ/(4π)
  -> detrend             remove DC + linear bulk drift
  -> bandpass_fft        isolate resp (0.1–0.5 Hz) / heart (0.8–2.0 Hz)
  -> dominant_rate       peak freq, SNR (dB), spectral concentration
  -> estimate_vitals     rates + confidence, gated for honesty
```

## Honesty gating (the point of the module)

`estimate_vitals` returns `ok=False` with a reason instead of a number when the
physics says the estimate is untrustworthy:

- **slow-time rate < 4 Hz** — cannot sample the heart band without aliasing;
- **window < 8 s** — frequency resolution too coarse;
- **bulk motion** — >2 cm peak-to-peak chest displacement swamps the ~0.5 mm
  heartbeat (subject must be still);
- **low SNR / low concentration** — SNR alone cannot reject noise (the max of
  ~24 random band bins already sits ~6–7 dB over the median), so a **spectral
  concentration** gate (fraction of in-band energy in the peak ±1 bin) is the
  real discriminator: a narrowband pulse → ~1.0, white noise → ~1/n_bins.

The synthetic `test_pure_noise_is_not_a_pulse` exists specifically to keep a
random spectral peak from ever being reported as a pulse.

## Live use

The hardware adapter is `radar_vitals_live.py` (repo root): it sends
`configs/iwr1843_vitals.cfg` (LVDS streaming on, 20 Hz frames), captures the
DCA1000 raw stream, builds slow-time range profiles, picks the chest bin with
`select_chest_bin`, and calls `estimate_vitals`. It also has `--from-bin` to
analyze a recorded capture with no rig attached.

## Caveats (do not over-claim)

- Breathing is robust; **heart rate is the hard case** and needs a still
  subject at close range, alone in the range bin.
- Synthetic tests use pure sinusoids (no harmonics); real breathing has
  harmonics that can leak into the heart band — the concentration/SNR gates
  guard against false positives but a genuinely ambiguous scene should read as
  "not reliable", which is the correct answer, not a failure.

## Adding a step

Keep it pure (numpy, no hardware), name every physical constant in `params.py`
(the physics-lint blocks inline magic numbers), and add a synthetic positive +
a failure-mode negative test to `tests/test_vitals.py`.
