# SCAN-MODES-PLAN — multi-mode scanning, and what belongs on the radar's own processor

Written 2026-08-09, before any of it has touched hardware. Everything here is a
derivation or a cited SDK fact until the gate in §6 says otherwise. The chirp
formulas were sanity-anchored first: they reproduce all three numbers measured
on this bench for `configs/radar_10hz.cfg` — v_max 0.649 m/s, v_res 0.0812,
R_max 11.16 m — so the derived tables below are trusted *as derivations*.

Goal envelope (unchanged): humans and vehicles, 2–15 m, close-urban, fused with
the RGB+thermal pair at 10 fps. Radar on USB (XDS110: CLI /dev/ttyACM1 @115200,
DATA /dev/ttyACM2 @921600), OOB demo firmware **SDK 3.04.00.03** (measured — it
rejected `calibData`, which dates it; the SDK-3.6 command tables cited below
must be re-checked against this build's `help` output).

The one sentence that shapes the whole plan: **the project's worst radar defect
— a walking person drawn ~9.6° off because v_max 0.649 m/s aliases — is fixed
by chirp timing, not by C.** The TDM-MIMO azimuth error is a k≠0 phenomenon;
raise v_max so every in-envelope target sits at k=0 and the error is simply
gone. Do not write firmware to fix what `idleTime` fixes.

---

## 1. The three scan modes

Angular performance is set by the antenna and is identical in every mode:
8 virtual azimuth elements (θ_res ≈ 14.3°, measured σ_az 1.10° median), 2
elevation elements (no resolution, σ ≈ 12° — the whisker stays honest in every
mode). A mode buys range/velocity trade-offs only.

### Mode P — people, 2–15 m (the default operational mode)

```
profileCfg 0 77 7 7 57 0 0 43.4 1 256 5209 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
chirpCfg 2 2 0 0 0 0 0 2
frameCfg 0 2 64 0 100 1 0
```

| quantity | value | trade taken |
|---|---|---|
| range resolution | 7.03 cm | coarser than today's 4.36 |
| R_max | 18.0 m | fold margin past the 15 m envelope |
| **v_max** | **±4.98 m/s** | walker/runner at k=0 — the fix |
| v_res | 0.156 m/s | separates slow walker from clutter |
| active / period | 12.3 ms / 100 ms | |
| HPF blind zone | **~0.60 m** (was 0.375) | corner moves with slope, as mmwave.py warns |
| radar cube | 768 KB | tight vs L3; **fallback: 48 loops** (v_res 0.208, 576 KB) |

CALIBRATION-PLAN stage 1 drafted this mode with idle 5 µs (v_max 5.25); 7 µs is
the conservative variant. If the CLI takes 5 and `interchirp_margin_us` (TLV 6)
stays positive, keep 5.25.

### Mode V — vehicle, to ~41 km/h at k=0

```
profileCfg 1 77 5 4 23 0 0 42 1 128 7000 0 0 30
(64 loops, 100 ms)
```

| quantity | value | trade taken |
|---|---|---|
| range resolution | 19.5 cm | coarse is fine for a car |
| R_max | 25.0 m | keeps 15–25 m vehicles from folding in |
| **v_max** | **±11.5 m/s = 41.4 km/h** | |
| v_res | 0.36 m/s | |
| radar cube | 393 KB | comfortable |

3 TX is retained on purpose: a 2 TX variant would double v_max but zero the
elevation aperture, turning every detection into a ray. For >41 km/h use
`extendedMaxVelocity 1` on this subframe first (§4, rank 0.5) — it is
config-only and also repairs the k=±1 azimuth compensation.

### Mode C — close/fine, 0.375–5 m (calibration and inspection heritage)

Today's config with **one number changed**: idleTime 429 → 7 µs.

```
profileCfg 0 77 7 7 57.14 0 0 70 1 256 5209 0 0 30
frameCfg 0 2 32 0 100 1 0
```

Range axis byte-identical to every existing capture (4.36 cm, R_max 11.16,
blind 0.375 — rampEnd 57.14 must not grow, the excursion is exactly 4000 MHz).
v_max becomes ±4.92 m/s, so even the calibration tripod scene stops aliasing a
passing person. Keep the CALIBRATION-PLAN discipline: `clutterRemoval 0` in the
calibration variant, `1` operationally.

---

## 2. Mechanism: two subframes interleaved, cfg files for operator switches

**Steady state: advanced frame with 2 subframes (P + V), 50 ms each — both
modes at 10 Hz, phase-locked to each other, zero switch latency.** Separate
.cfg files stay for operator-scale transitions (calibration vs operational,
Mode C sessions): a full sensorStop → flushCfg → resend → sensorStart is
estimated at 1–2.5 s of radar silence (10–25 camera frames); gate #4 replaces
that estimate with a measurement. `flushCfg` is mandatory before a new config,
and no command is optional (SDK user guide, verbatim).

SDK facts that make the subframe path config-only (SDK 3.6 UG; re-verify on
this 3.04 build):

- `dfeDataOutputMode 3` + `advFrameCfg` + per-subframe `subFrameCfg`, max 4
  subframes; demo restriction: one burst per subframe, numOfBurstLoops 1 —
  this design respects both. Software trigger only.
- `cfarCfg`, `aoaFovCfg`, `cfarFovCfg`, `clutterRemoval`, `multiObjBeamForming`,
  `extendedMaxVelocity` all take a leading `subFrameIdx` (-1 = all subframes),
  so per-mode CFAR threshold, range gate and clutter handling need no firmware.
  That subset is even changeable **while the sensor runs** — live threshold
  tuning from the viewer with no sensorStop.
- The 40-byte frame header ends with `subFrameNumber`, and `mmwave.py` already
  parses it (`parse_header` → `'subframe'`). Per-mode tagging is plumbing, not
  a parser change.

Sketch (periodicity units per this build's `help` — verify before trusting):

```
dfeDataOutputMode 3
profileCfg 0 ...Mode P...        profileCfg 1 ...Mode V...
chirpCfg 0..2 → profile 0        chirpCfg 3..5 → profile 1   (TX order 1/4/2)
advFrameCfg 2 0 0 1 0
subFrameCfg 0 -1 0 3 64 50 0 1 1 50
subFrameCfg 1 -1 3 3 64 50 0 1 1 50
```

**UART budget at 921600, TLVs [1,7,6,9]** (frame = 124 + 20·N bytes, padded to
32): two subframes at 10 fps each fit with margin at realistic point counts
(~30–120 pts people / 20–80 vehicle ≈ 36% of the link). Hard ceiling: 224
points per frame at 20 fps. Guards: per-subframe `cfarFovCfg` range gates
(P ≤16 m, V ≤22 m), `aoaFovCfg ±60°`, higher threshold on V, and watch
`transmit_out_us` / `interframe_margin_us` in TLV 6 — the radar's own overrun
report, distinct from host-side byte loss. Range profile stays off.

---

## 3. Host-side integration (the LIVE part)

Ordered so each step is testable offline against a recording before it touches
the bench.

1. **`radar/mmwave.py`**: split `CFG_10HZ` into per-mode limit dicts keyed by
   subframe number (P=0, V=1; legacy configs key None→mode). Without this,
   `validate_points` (built on v_max 0.649 / R_max 11.16) rejects every good
   Mode P/V point and `fold_velocity`'s default is wrong. Also settle on the
   first advanced-frame capture whether `frame_number` ticks per subframe or
   per advanced frame on this build — the t_host↔frame_number clock regression
   depends on it.
2. **`tools/recorder.py` / radar recording**: `radar.jsonl` rows already carry
   the parsed header — confirm `subframe` lands in them; CALIBRATION-PLAN §3
   lists it as a must-record field.
3. **`tools/radar_overlay.py`**: color per mode (P dots vs V dots), per-mode
   v_max for the alias marking (which becomes a guard that should never fire,
   not dead code to delete), per-mode range gate for the offscreen count.
4. **`tools/live.py`**: `/set?rmode=people|vehicle|close|dual` switches cfg
   files via the send_radar_cfg path (measured latency from gate #4 shown in
   the health row as "radar reconfiguring"); live `cfarCfg` nudging without
   sensorStop for threshold tuning; health row gains per-subframe frame rates.
   The recording tags the active mode set.
5. **Calibration impact**: the campaign math assumes 10 fps and one mode. Under
   P+V the DATA port carries 20 output frames/s; the beat-pairing arithmetic
   (114 ms thermal vs 100 ms radar) must pair against the *subframe of
   interest*, not every frame. Extrinsics are geometric and mode-independent —
   one solved R,t serves all modes; only the per-mode range grid needs the
   0.0436-m-style re-fit (gate #5).

---

## 4. C on the radar — what it actually buys, and the toolchain reality

### The honest ladder

| rank | what | where | effort | verdict for this project |
|---|---|---|---|---|
| 0 | **The §1 configs** | none | config | Kills the aliased-walker azimuth error and the clutterRemoval↔v_max coupling. Do this first; it obsoletes most of the C case |
| 0.5 | **`extendedMaxVelocity 1`** on the V subframe | none | config | Two-hypothesis TDM disambiguation → effective ±23 m/s ≈ 83 km/h AND correct k=±1 azimuth. Weakness: multiple targets in one range-Doppler cell. Verify with a drive-past recording |
| 1 | **gtrack on-chip** (tracks, not points) | R4F MSS; lib ships at `ti/alg/gtrack` (R4F and C674x both supported) | new-DPU class | **Near zero on the USB/Jetson path** — the Jetson should track on host from recorded points, where the algorithm can be iterated in Python/C without reflashing. Becomes item #1 only for the future N6-UART7 product path, where MicroPython drops bytes silently and ~100 B/frame of tracks vs KBs of points decides whether the link works |
| 2 | **Compact custom TLV** (int16 fields ≈ 10 B/pt vs 20) | R4F, `MmwDemo_transmitProcessedOutput` in mss_main.c | small patch | Halves link load; unneeded at 921600 on USB (§2 budget), same N6-path justification as gtrack but much cheaper |
| 3 | k≠0 Doppler-compensation fix beyond extendedMaxVelocity | C674x, `aoaprocdsp.c` | small patch → new-DPU | Only if targets routinely exceed the new v_max AND the single-target assumption measurably fails. Measure first |
| 4 | Range-dependent CFAR threshold | C674x, `cfarcaprocdsp.c` | small patch | Modest: per-mode CFAR is already config, and the measured noise floor is flat 24.3 dB |
| 5 | Per-mode clutter handling | — | config | `clutterRemoval` takes subFrameIdx; nothing to write |

Where the code lives when a rung is climbed: demo at
`packages/ti/demo/xwr18xx/mmw/` (MSS under `mss/`, detection DPC on the C674x
at `ti/datapath/dpc/objectdetection/objdethwa` driving HWA-based DPUs —
rangeproc/dopplerproc/cfarcaproc/aoaproc under `ti/datapath/`). The DPU layer
is TI's declared customization point (`DPU_<Name>_init/config/process/ioctl`).

### Toolchain reality (all cited, all verified 2026-08-09)

- **The Jetson cannot build or flash this firmware.** The mmWave SDK installer
  is a 32-bit x86 binary; the TI ARM/C6000 compilers ship x86-only; UniFlash is
  x86-only and the qemu-on-aarch64 workaround is known to fail (E2E). Same
  class of blocker as stedgeai.
- **Building needs no IDE**: SDK 3.06.02.00-LTS, `setenv.sh` + `gmake`,
  headless on x86 Linux (Ubuntu-class, i386 multilib enabled). CCS is only for
  JTAG debug.
- **Flashing is UART-only** (SOP 101 → ROM bootloader → UniFlash over the
  XDS110 user UART; no JTAG needed). No open-source Python flasher exists for
  the xWR1xxx ROM protocol (searched; pymmw and ROS drivers are capture-only).
- **The escape hatch: the SDK's Secondary Bootloader** (`ti/utils/sbl`). Flash
  the SBL once from an x86 box; from then on new application images are pushed
  **from the Jetson over UART** forever. If the C path is ever taken, this is
  step one, because it removes the x86 machine from the iteration loop.
- **Shortcut worth knowing**: TI's Traffic Monitoring lab (Industrial Toolbox,
  `labs/traffic_monitoring/18xx_...`, TIDEP-0090) ships **prebuilt** 1843
  binaries with gtrack already on the R4F — on-chip tracking without building
  anything. Different output protocol → mmwave.py grows a second parser. An
  x86 box is still needed once, to flash it.

### Firmware version decision

The bench board runs 3.04.00.03. Two of the §1–§2 mechanisms (`advFrameCfg`
semantics, per-subframe command set) are cited from the 3.6 docs; 3.04 has the
core subframe support but not the newer commands (`compressCfg`, `aoa2dproc`…).
Decision point at gate #1: if this build's CLI accepts the advanced-frame
sketch, stay on 3.04 for now (no x86 session needed); if not, the same x86
session that flashes 3.06.02 OOB should also flash the SBL, so it is the last
x86 session ever needed.

---

## 5. What this plan deliberately does not do

- No 2 TX high-v_max mode: elevation drops from "weak" to "none", and every
  radar↔camera correspondence would inherit an unbounded vertical error.
- No on-chip tracking for the USB path: host tracking iterates in minutes;
  firmware tracking iterates in flash cycles that require an x86 machine.
- No range profile TLV, no LVDS/DCA1000: the budget in §2 works and the old
  DCA1000 path stays in ~/archive/radar where commit 5e50051 put it.

---

## 6. Measurement gates, in order — nothing above survives contact unverified

1. **CLI census**: send `help` (and the `advFrameCfg` sketch) to this 3.04
   build; record what it accepts. This decides the firmware-version question.
2. **Mode P in legacy mode**: accepted without `Error` (watch for a silently
   rejected 768 KB cube → 48-loop fallback), stage-1 `lost=0` via
   `diag/radar_listen.py --stage1`, noise floor re-measured, then the plan's
   exit gate: a walker at 12 m detected in ≥90% of frames with v ≠ 0 and *no
   hollow (aliased) markers*.
3. **Advanced frame P+V**: `subframe` values present in radar.jsonl; settle
   whether `frame_number` ticks per subframe; confirm periodicity units;
   TLV 6 `interframe_margin_us` positive on both subframes.
4. **Time one full cfg switch** through send_radar_cfg.py — replaces the
   1–2.5 s estimate in §2.
5. **Re-fit the range grid per mode** (the 0.0436 m trick from the calibration
   campaign): 7.03 cm and 19.5 cm are derivations until a capture agrees.
6. **extendedMaxVelocity drive-past**: one vehicle recording ≥50 km/h; does the
   ±2·v_max hypothesis pick correctly, and does azimuth stay put through the
   k=±1 boundary?
