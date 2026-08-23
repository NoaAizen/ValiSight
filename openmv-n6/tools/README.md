# tools/ — what runs where

The root of this directory is **the live pipeline**: the files that import each
other and together put a fused picture on a screen. Everything that is not part
of that pipeline lives in a subdirectory named for what it is.

## The pipeline (root)

| file | role |
|---|---|
| `live.py` | live viewer: board streams, host fuses, browser watches. `--radar` overlays the IWR1843, `--record` writes a session |
| `capture.py` | synchronized RGB+thermal pairs, board → disk |
| `detect.py` | COCO detector over the visible luma, temperatures read per box |
| `radar_overlay.py` | RadarReader (owns the DATA port) + radar point → pixel |
| `recorder.py` | session recording: clean mp4, frames.jsonl, raw thermal.bin |
| `send_radar_cfg.py` | pushes the chirp config to the radar CLI UART — run this before `live.py --radar` |
| `mpx.py` | minimal raw-REPL runner; how everything in `board/` gets onto the board |


The viewer's `operator` mode preserves visible luminance, adds adaptive thermal
colour and feathers the calibrated thermal footprint. It is the default in
`run_live.sh`; use `--view visible` for calibration picking and `--view fused`
when the rendered palette itself must remain a fixed thermal scale.

## Detector models on the Jetson

`live.py` accepts `--detect-model yolov10n|yolov8n|yolo11n`. The repository
loader requires one static 640×640 TensorRT input and an NMS-included
`[1,max_det,6]` output; raw Ultralytics exports are rejected. YOLOv10n is
already installed on this rig. To prepare YOLO11n, export ONNX in an isolated
Ultralytics environment (this may be a workstation or Colab), copy the ONNX to
`~/archive/radar/models/`, then build the engine on the Jetson:

```sh
python3 tools/export_ultralytics_onnx.py --model yolo11n
python3 tools/trt_detect.py --build yolo11n --verbose
python3 tools/live.py --detect person --detect-backend gpu --detect-model yolo11n
```

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
