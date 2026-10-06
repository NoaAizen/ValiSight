# tools/ — what runs where

The root contains command-line entry points and shared sensor/recording modules.
Viewer components live in [viewer/](viewer/README.md), browser pages in
[web/](web/README.md), and transmitted MicroPython source in
[board/templates/](board/README.md). Calibration and bench diagnostics have
their own directories. Existing commands and `live.py` component imports remain
available.

## The pipeline (root)

| file | role |
|---|---|
| `live.py` | live viewer: board streams, host fuses, browser watches. `--radar` overlays the IWR1843, `--record` writes a session |
| `desktop_app.py` | desktop app window, startup progress and safe reuse of a running viewer; `--install` installs desktop/menu launchers |
| `capture.py` | synchronized RGB+thermal pairs, board → disk |
| `detect.py` | COCO detector over the visible luma, temperatures read per box |
| `radar_overlay.py` | RadarReader (owns the DATA port) + radar point → pixel |
| `recorder.py` | session recording: clean mp4, frames.jsonl, raw thermal.bin |
| `send_radar_cfg.py` | pushes the chirp config to the radar CLI UART — run this before `live.py --radar` |
| `mpx.py` | minimal raw-REPL runner; how everything in `board/` gets onto the board |

## Desktop application

Run `python3 tools/desktop_app.py --install` from `openmv-n6` once, then open
**Thermal Fusion** from the application menu or the desktop shortcut. The Hebrew
menu label is **מערכת צילום תרמי**. If the desktop asks to trust the shortcut,
choose **Allow Launching**. Re-run installation after moving the checkout.

The launcher uses the installed Chromium in app mode, without an address bar,
with a separate browser profile. Python Tkinter provides startup progress and
error messages. It reuses an existing viewer, otherwise starts `run_live.sh`
with its normal sensor and AI configuration. It waits for an existing acquisition
instead of interrupting it. `HTTP` can override the default local port, 8088.

Closing the app window **leaves acquisition and recordings running**. Stop the
sensor system explicitly with `tools/run_live.sh --stop` when finished.
Startup/window logs are in `~/.local/state/thermal-fusion/`; the underlying
sensor log remains `/tmp/live_radar.$USER.log` unless `LOG` was overridden.


The viewer's `operator` mode preserves visible luminance, adds adaptive thermal
colour and feathers the calibrated thermal footprint. It is the default in
`run_live.sh`; use `--view visible` for calibration picking and `--view fused`
when the rendered palette itself must remain a fixed thermal scale.

## Display channels

`--channel thermal|rgb_radar|thermal_radar|fusion|ai` chooses the operator
product independently of the calibration `--view`. The five channels are:
pure registered thermal; visible plus raw radar; thermal plus raw radar;
thermal+visible+radar fusion; and a clean AI view containing final detections
and tracks. Keys 1-5 and the first card in the browser switch them live.

The PAG7936 transport is still `GRAYSCALE`, so `rgb_radar` is labelled
**visible + radar** in the browser. Its stable API id is reserved for the true
RGB565 path; it must not be relabelled as colour until that path passes the
raw16 bandwidth and long-run sensor-stall gate.

## AI channels in the viewer

`live.py --students` runs the two trained person students beside the COCO
detector. Their thermal (orange), radar (cyan), and fusion (white) boxes are
intermediate evidence and are hidden in the AI product by default; the engines
continue feeding fusion and tracking. Enable **student evidence** (key `e`) only
when debugging those model outputs. The clean product uses one plain final box;
the jagged thermal **person outline** is opt-in with key `s`. **lock** (key `l`, on by default) keeps a person between frames: seen twice
they become `P1`, held through the frames no sensor found them in for up to
1.5 s and carried on their own velocity, drawn amber and dashed while coasting.
All three sensors feed one tracker, so a lock the detector loses can be held by
the thermal student — the letters after the id (D/T/R/F) name who is holding
it. A candidate the detector has never seen has to **move** before it is drawn:
in this lobby a lit glass door reads 31.7 °C and a person 30.9 °C, so warmth
cannot tell them apart and a door has never moved. Those are reported as
*static* beside the lock, not dropped in silence. `/ui` and `/ai` carry the tracks; `--no-lock` starts without it.

With **person outline** on (key `s`), a person is drawn as the warm shape the
thermal frame holds inside the box — for the students *and* for the detector's
green person boxes, which needs only `--warp`, not `--students`. Fusion draws
its ring around that shape. Each channel has a confidence slider on that card (it hides boxes, it does not
restart the engines — the count reads *shown of found*), boxes under 0.75 are
drawn as corner ticks rather than rectangles, and duplicate/nested boxes from
the same channel are dropped. Switch the channels from the card, with `t`/`r`/`c`,
or over `/set?ai_thermal=0&ai_radar=1&ai_fusion=1`; `--ai-channels` picks which
ones the session starts with, and `/ai` reports all three as JSON. The thermal
channel needs `--warp` to reach the visible plane and the radar channel needs
`--radar`; a channel that cannot draw says so on the card and in `/health`
rather than showing an empty count.

## Detector models on the Jetson

`live.py` accepts `--detect-model yolov10n|yolov8n|yolo11n|yolo26n|yolo26s|yolo26m|yolo26l|yolo26x`.
The repository loader requires one static TensorRT input and a decoded
`[1,max_det,6]` output; raw Ultralytics exports are rejected. YOLOv10n is
already installed on this rig. To prepare YOLO11n, export ONNX in an isolated
Ultralytics environment (this may be a workstation or Colab), copy the ONNX to
`~/archive/radar/models/`, then build the engine on the Jetson:

```sh
python3 tools/export_ultralytics_onnx.py --model yolo11n
python3 tools/trt_detect.py --build yolo11n --verbose
python3 tools/live.py --detect person --detect-backend gpu --detect-model yolo11n
```

For YOLO26n, use the [isolated export and deployment guide](../../deployment/yolo26/README.md).
`bash tools/run_yolo26.sh` selects YOLO26n on the GPU with the existing sensor
stack. YOLO26 uses its one-to-one head (`nms=False` in the pinned exporter),
whereas YOLOv8/11 use embedded NMS. `detector_models.py` is the shared artifact
registry used by export, runtime and launcher checks.

TensorRT engines are specific to the target GPU/TensorRT version; do not copy
an engine built on a different computer. Ultralytics code and pretrained models
are AGPL-3.0 by default, so check licensing before closed-source deployment.

## board/ — runs ON the N6

Pure MicroPython, pushed whole with `./mpx.py board/<script>.py`. The
`probe_*.py` scripts are the Milestone-0 diagnostics (bring-up, dual capture,
Lepton radiometry); `thermal_heatmap_n6.py` is the standalone on-board
heat-mapping demo.

## diag/ — bench diagnostics

Host-side, one question each. `radar_listen.py` (radar link sanity + radar-only
recording), `radar_stage1_n6.py` and `radar_sync_probe.py` (the deferred
radar-on-UART7 path: link gate and timer-capture inventory), `mono_compare.py`
(palette/gain tiling over a recorded capture).

## tests/ — no hardware needed

`test_live.py` (viewer + ctypes mirror against recorded frames),
`test_radar.py` (TLV parser), `test_radar_overlay.py` (projection/drawing).
All runnable from any directory; paths are script-relative.

## calib/

Thermal↔RGB stereo calibration → warp LUT, and the radar↔camera extrinsics
chain (correspondence picking → solver). Has its own tests, colocated.
