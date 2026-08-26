# Thermal + visible fusion on the OpenMV N6

Fused LWIR/visible imaging on an OpenMV N6 (STM32N657X0) for close-range
electrical and mechanical inspection at 0.5–2 m. A FLIR Lepton 3.5 (160x120)
and a PAG7936 (640x400 luma) run simultaneously off the board's two CSI ports.

The picture and the measurement are deliberately separate products. The image
exists to be looked at; the temperature is what a finding rests on, and the two
travel different paths over the same frame:

```
display      t_prep -> warp -> AGC -> guided filter -> palette + detail
measurement  t_prep -> warp -> temperature
```

The measurement path stops before the scene AGC, which is scene-relative, and
before the guided filter, which borrows the *visible* camera's edges to sharpen
the thermal layer. That sharpening is excellent for a picture and is not
something to quote a number off — so the number a user reads is never the pixel
they see, it is the thermal pixel behind it.

Full design notes and every measurement behind the decisions: [DESIGN.md](DESIGN.md)
(Hebrew). How the four streams — thermal, visible, radar and the detector — reach
the same frame, on which clock and with what left unsynchronised: [SYNC.md](SYNC.md)
(Hebrew).

## Layout

| path | what |
|---|---|
| `src/fusion.c`, `fusion.h` | the pipeline. Portable C99, no MicroPython, no floating point. One translation unit for host and Cortex-M55 |
| `src/py_fusion.c` | thin MicroPython binding; every decision lives in `fusion.c` |
| `host/` | harness that runs the same C over recorded frames, plus the test suites |
| `tools/` | the live pipeline: `capture.py` (board -> disk), `live.py` (live viewer; `--radar` overlay, `--record` session) and its modules. See [tools/README.md](tools/README.md) |
| `tools/board/` | board-side MicroPython, pushed whole via `mpx.py` |
| `tools/diag/` | bench diagnostics: radar bring-up probes, capture comparisons |
| `tools/tests/` | the offline test suites - no board, no radar needed |
| `tools/calib/` | OpenCV stereo calibration -> warp LUT, and a hand-wave fallback |
| `radar/` | the IWR1843 mmWave path: TLV parser, 10 fps config, link gate. [radar/README.md](radar/README.md) |
| `openmv-integration/` | the eleven lines that add the module to the firmware tree |
| `captures/handwave3/` | 18 real frame pairs, kept so the real-data tests can run |

The same `fusion.c` compiles on the host and into firmware. Algorithm work
happens on a workstation against recorded frames, where the rebuild-and-look
cycle is under a second, against minutes for a firmware flash.

## Build and test

```sh
cd host
make                # fuse (harness) + libfusion.so (for the viewer)
make test           # 33 synthetic assertions, under ASan + UBSan
make real           # the same pipeline over recorded frames
python3 ../tools/tests/test_live.py    # 57 viewer assertions, no board needed
```

`make test` proves the pipeline is *correct* on frames built to have a known
answer. `make real` is the only one that can catch a detector following the
scene instead of the sensor — which the dead-row detector did, in its first
version, while every synthetic test passed.

## Live viewer

```sh
cd tools
./send_radar_cfg.py                             # chirp config first, or --radar sees nothing
./live.py --radar /dev/ttyACM2 --radar-hfov 62.7 \
          --detect person --view visible        # http://localhost:8088
```

Three ports on this Jetson, and they are not interchangeable: `/dev/ttyACM0` is
the OpenMV N6, `/dev/ttyACM1` is the radar CLI at 115200 (where the `.cfg`
goes), `/dev/ttyACM2` is the radar data stream at 921600. `send_radar_cfg.py`
must run first — the IWR1843 emits nothing until it is configured, so a
`--radar` run against an unconfigured sensor looks like a dead link.

`--radar-hfov 62.7` overrides the 70° fallback with the measured value (f = 525
px at 640 wide, checkerboard against a tape measure). `--radar-calib` accepts
both calibration schemas (`K/R/t/dist` and
`K_rgb/R_cam_from_radar/t_cam_m/dist_rgb`); `run_live.sh` automatically loads
`calib-artifacts/radar_rgb_2026-08-18.json` when it is present.

`run_live.sh` also loads `calib-artifacts/warp.lut` when present. Without that
artifact the thermal layer is stretched, not registered, and every temperature
it quotes is marked `(unreg)` — see [tools/calib/](tools/calib/).

The board sends a hardware-JPEG of the visible frame plus the thermal frame
*uncompressed* — the thermal data is the measurement, and lossy compression on
radiometry is not a trade worth making for 17 KB. Fusion runs through
`libfusion.so`, the same `fusion.c` that compiles into firmware, so the viewer
shows what the board will do once flashed.

Hover for a temperature. The view selector (`blink` / `mix` / `edges`) exists
because **the fused image cannot tell you whether registration is right**: the
guided filter puts crisp edges in the right places even when the thermal layer
is offset, so a misregistered frame still looks sharp — it just colours the
wrong side of the edge. The health row reports what only a live stream can
show: dead rows rebuilt this frame, thermal coverage, tearing, the range the
sensor auto-picked.

## State

| # | milestone | |
|---|---|---|
| 0 | simultaneous RGB + thermal capture | done, 8/8 frames |
| 1 | RAW pair capture to host | done |
| 2 | `fusion.c` prototype | done, 8.8 ms/frame on the host |
| 3 | OpenCV calibration -> warp LUT | tooling done, 9/9 tests |
| 4 | firmware module | `firmware.bin` builds; flashing blocked on x86_64 |
| 5 | Helium optimisation | not started, and possibly unnecessary |
| 6 | radiometric API, palettes | done except the parallax slider |

The runtime now has thermal-visible and radar-visible calibration artifacts
under `calib-artifacts/`, and `run_live.sh` loads them automatically. A direct
`live.py` invocation without `--warp`/`--radar-calib` still uses explicit
placeholder geometry and reports that state in `/health`; mounting changes
invalidate the solved artifacts and require a new calibration.

## Two facts about this hardware worth knowing before reading the code

**The Lepton kills 14 fixed rows** — 6, 12, 28, 35, 36 and the run 55..63 —
carrying no scene at all. They are *not* stuck at 255, which is what they look
like and what the first implementation assumed: they sit at a fixed high offset
that clips at 255 only once the frame's own level rises, so a brightness
threshold misses them on 21 of 46 affected frames. The test that works is
flat-and-lifted (spread ≤ 24 codes, mean ≥ 64 above the frame median). Left
unrepaired they pinned the reported frame maximum at exactly the top of the
sensor range — a 10.3 °C error in the headline number, not merely a stripe.

It is also a *state*, not a permanent defect: every capture up to 15:23 on
2026-08-04 is clean, every capture from 21:51 has all 14 rows in 100% of frames.
Hence a detector rather than a hardcoded row list.

**Auto-range is worth roughly 4x in temperature resolution.**
`IOCTL_LEPTON_SET_RANGE(tmin, tmax)` maps `[tmin,tmax]` onto codes 0..255, so a
range picked per scene takes 0.588 down to 0.141 °C/LSB. Whatever range the
sensor is given must also be given to `fusion_set_range()`, or every reading is
scaled by the ratio of the two spans while looking entirely plausible.
