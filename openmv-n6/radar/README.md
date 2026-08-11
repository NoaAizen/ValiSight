# The mmWave radar path

An IWR1843BOOST alongside the thermal/visible pair, so a detection has a range
and a radial velocity as well as a temperature and a picture. This directory is
the parser and the config; the tools that drive them are in [../tools/](../tools/).

The radar reaches the system two different ways and both are supported, because
they answer different questions:

| path | wiring | what it is for |
|---|---|---|
| host | XDS110 USB → Jetson, CLI `/dev/ttyACM1`, DATA `/dev/ttyACM2` | works today; the calibration and all offline work happen here |
| board | radar DATA_TX → N6 `P13` (PE7, UART7_RX) @921600 | the product; the question there is whether the bytes survive, not whether they parse |

`mmwave.py` is the same file in both, and in the tests. No board imports, no
numpy, pure `struct`.

## Order of operations

```sh
cd ../tools
./test_radar.py                          # 57 assertions, no hardware
./send_radar_cfg.py                      # radar_10hz.cfg -> CLI UART
./radar_listen.py --stage1 --seconds 60  # THE GATE: passes only at lost=0
./radar_listen.py                        # live points
./radar_listen.py --record ../captures/radar_calib -n 300
```

Stage 1 is not a warm-up. Until the byte accounting is exact, a missing
detection and a dropped byte look identical, and every hour spent above a lossy
link is wasted. It is exact rather than statistical: each magic word declares
its own `totalPacketLen`, so the byte distance to the next magic must equal it,
and anything else is bytes lost or injected.

Measured 2026-08-09 over USB: **PASS, 201 frames, 200 good gaps, 0 bad, 0 bytes
lost**, 10.0 fps, frames 192–352 B. The board path has not been run — the radar
is not physically wired to the N6 yet, and `radar_stage1_n6.py` on unwired
hardware correctly reports `no bytes`.

## Three things measured here that the obvious reading gets wrong

**The 32-byte padding is not zeros.** `totalPacketLen` is rounded up to a
32-byte multiple and the leftover carries stale transmit-buffer content —
`00 03 13 00 00 00 00 00 79 de 00 00` in the frame that settled it. Validating
the tail as zero padding, which is what the first version of the parser did,
rejected **70% of good frames while stage 1 measured a flawless link**. The only
thing the tail may assert is its length.

**The live radar emits four TLVs, not two:** 1 (points), 7 (side info), 6
(stats), 9 (temperature). Anything that assumes points-and-side-info walks off
the end. Stats carry `interframe_margin_us` — the radar's own report that it is
overrunning the 100 ms frame period, which is a completely different failure
from a lossy link and needs distinguishing from it. Temperature is kept because
the range bias drifts with die temperature, so a calibration taken cold and used
warm is a slowly moving depth offset.

**A frame is only trusted once the next frame's magic confirms it.** The TLV
walk cannot see a byte lost from inside a payload: every declared length still
sums to `totalPacketLen`, and the frame quietly eats its successor's first byte.
Only the missing successor magic catches it. Where that costs latency,
`FrameSync` still stamps the frame with the time its **last byte arrived**, not
the time validation released it — otherwise every radar timestamp carries a
frame period of one-directional bias straight into the radar-to-camera offset.

## Coordinate frame, before anyone solves for extrinsics

TI reports **x = right, y = forward, z = up**. `mmwave.py` returns the *project*
frame, **x = forward, y = left, z = up**, which is what
`calibrate_radar_camera.solvePnP` and the archived classifier were built
against. `convention='ti'` gives the sensor's own numbers back.

This is worth stating in a file rather than a commit message because a frame
mix-up does not raise. It looks exactly like a 90° mounting error, `R` absorbs
it, the reprojection error stays small, and the rig is wrong in a way that only
shows up at a distance nobody calibrated at.

## What the config commits to

`configs/radar_10hz.cfg`, 10 fps, every line annotated. The four that decide
what this rig can do:

- **`profileCfg` hpfCornerFreq1/2 = 0/0 → 175 kHz → 0.375 m.** Below that the IF
  high-pass attenuates signal and noise alike. The working range starts at
  0.375 m regardless of what `cfarFovCfg` says, and the calibration target has
  to sit beyond it. Note the direction: this board is 2.4 dB *quieter* at
  0.04–0.48 m than at 2–3 m, so there is no close-range noise problem to fix —
  the problem is the filter.
- **`clutterRemoval` off.** The calibration target is a corner reflector on a
  tripod, and switchgear does not move either. Turning this on deletes the
  target from exactly the frames the calibration needs.
- **`extendedMaxVelocity` off, 3 TX.** v_max is ±0.649 m/s, so the alias period
  is 1.298 m/s and a walking person folds to |v| ≤ 0.10 m/s and reads as static.
  85.2% of logged points sit at exactly zero Doppler. `mmwave.fold_velocity`
  models it; shortening `idleTime` is what would buy velocity back, at the cost
  of range.
- **`compRangeBiasAndRxChanPhase` rangeBias = 0.0, i.e. uncalibrated.** Do this
  one *before* the radar↔camera extrinsics. A residual range bias is a constant
  depth offset, and `solvePnP` will absorb it into `t` as a fake lever arm that
  then breaks at every other distance.

## Hardware timestamping, if software pairing turns out not to be enough

`../tools/diag/radar_sync_probe.py` measured the board on 2026-08-09, with and
without both sensors streaming — the answers are the same either way:

- TIM2/3/4/5 are 32-bit at 400 MHz. TIM8, 12–17 are 16-bit. `pyb` does not
  expose TIM1 at all; the camera drives CSI_CLK from it.
- The only free header pins with a timer AF are **P9** (PG12, AF1_TIM17) and
  **P17/P18** (PB6/PB7, both AF7_TIM15).
- **P17 + P18 on TIM15 hold two input captures at once**, with the cameras up.
  One counter, so the two edges subtract directly instead of needing two clocks
  related to each other.
- TIM15 is 16-bit, so the prescaler is the design decision: at 8 µs/tick it
  wraps every 524 ms, comfortably longer than the 114 ms thermal frame. At the
  500 ns/tick that looks attractive it wraps every 32.7 ms — three times per
  thermal frame.
- `Timer.channel()` validates the **timer**, not the channel: TIM15 physically
  has two channels and accepted all four. Trust the timer column; a channel
  number needs a real edge to confirm.

P10 (PD6) is CSI_FSYNC and the board drives it, so it is a signal to jumper
*from*, never a pin to capture *on*.

## Files

| path | what |
|---|---|
| `mmwave.py` | framing, TLV walk, points + SNR join, stats, temperature, Doppler fold, physics validators |
| `configs/radar_10hz.cfg` | the 10 fps config, with the measurement behind each choice |
| `../tools/tests/test_radar.py` | 57 offline assertions |
| `../tools/send_radar_cfg.py` | config → CLI UART, waiting on the radar's own response per line |
| `../tools/diag/radar_listen.py` | host: stage-1 gate, live parse, recording |
| `../tools/diag/radar_stage1_n6.py` | board: the same gate over UART7, optionally with both cameras running |
| `../tools/diag/radar_sync_probe.py` | timer and pin inventory for hardware timestamping |
