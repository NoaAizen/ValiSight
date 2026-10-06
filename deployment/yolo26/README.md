# YOLO26 on this Jetson

The first integration uses **YOLO26n**, COCO-80 detection, 640×640 static input,
batch 1, the one-to-one head and a locally built mixed FP16/FP32 TensorRT engine. It replaces
the visible detector only. Thermal students, radar processing, association and
tracking keep their existing interfaces. The current visible sensor transports
grayscale luma; the runtime replicates it into the model's three input channels.

The runtime accepts decoded `[1,300,6]` rows: `x0,y0,x1,y1,score,class`, in
letterboxed pixel coordinates. In Ultralytics 8.4.144, use `nms=False` for this
head. The default raw export has a different contract and is rejected. See the
[official YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/).

## Run

From the repository root, after engine creation:

```sh
bash openmv-n6/tools/run_yolo26.sh
# Additional normal viewer options:
bash openmv-n6/tools/run_yolo26.sh --record captures/yolo26-session
```

This starts the full hardware acquisition and can replace a running viewer.
The profile explicitly requires the GPU, so failure does not silently substitute
the CPU detector. The existing `run_live.sh` default remains YOLOv10n for
comparison. `--detect-model yolo26n --detect-backend gpu` also works with direct
`live.py` invocation. The launcher checks the selected engine before opening
sensor ports. `bash openmv-n6/tools/run_yolo26.sh --stop` uses the normal stop path.

## Reproduce export and engine build

The Dockerfile extends the existing local `thermal-objects-train:jp6-v1` image;
its source is in `openmv-n6/perception/training/thermal_objects/Dockerfile`.
It pins Ultralytics 8.4.144 and compatible export dependencies without modifying
the host Python installation. Docker requires access to the local daemon.

```sh
docker build --network=host -t thermal-fusion-yolo26-export:8.4.144 deployment/yolo26
export THERMAL_FUSION_MODEL_DIR="$HOME/archive/radar/models"
docker run --rm --network=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/openmv-n6/tools:/tools:ro" \
  -v "$THERMAL_FUSION_MODEL_DIR:/models" \
  thermal-fusion-yolo26-export:8.4.144 \
  python3 /tools/export_ultralytics_onnx.py --model yolo26n --out-dir /models
python3 openmv-n6/tools/trt_detect.py --build yolo26n --verbose
```

The first export downloads official weights. With weights cached in the model
directory, use `--network=none`. Export validates the COCO class order, ONNX
graph and static tensor shapes, and writes a JSON sidecar with export settings,
class names, version and ONNX SHA-256. The runtime does not require PyTorch or
Ultralytics. TensorRT engines must be rebuilt for a different GPU/TRT environment.

The registered YOLO26n engine is `yolo26n_end2end_mixed.engine`. Its build keeps
`/model.23/one2one_cv3*` and `/model.23/Sigmoid` in FP32, disables TF32, and uses
builder optimization level 0. This recipe passed the reference check below;
the initial unconstrained FP16 engine did not. `trt_detect.py --build yolo26n`
applies these settings automatically. Revalidate the recipe after changing
weights, exporter or TensorRT versions; its layer names belong to this graph.

Registered names also include `yolo26s/m/l/x`; only `yolo26n` is the initial
deployment candidate. Larger sizes require their own export, build and evaluation.
This visible detector exporter rejects custom class orders; thermal-trained
weights need their own class mapping and thermal preprocessing integration.

## Offline measurement

```sh
python3 openmv-n6/tools/benchmark_detector.py --model yolo26n \
  --images openmv-n6/captures/handwave3/*_rgb0.raw \
  --out openmv-n6/perception/out/yolo26/benchmark.json
```

Run the same command with `--model yolov10n` and a different output path to
compare the baseline. Timing includes preprocessing, transfers, synchronized
inference and decoded boxes. It excludes file reads, acquisition, fusion and
rendering. The report includes input/engine hashes, latency percentiles,
process peak RSS and detections. It does not measure precision or recall.

For numerical export verification, use a saved image containing clear objects:

```sh
# Inside the export container, with the model directory mounted at /models:
python3 /tools/validate_detector_export.py \
  --onnx /models/yolo26n_end2end.onnx --image /models/yolo26-smoke-bus.jpg \
  --reference /models/yolo26n-reference.npz
# On the host:
python3 openmv-n6/tools/validate_detector_export.py --model yolo26n \
  --reference "$THERMAL_FUSION_MODEL_DIR/yolo26n-reference.npz"
```

The sample used here is `bus.jpg` bundled in the installed Ultralytics package
(`ultralytics.utils.ASSETS`). Validation checks production preprocessing and
matches confident boxes by class/IoU, allowing FP16 score differences up to
0.03 and requiring IoU of at least 0.95. This is numerical parity on an image,
not a dataset accuracy evaluation.

## Measurements on 2026-09-08

Jetson Orin Nano Super, TensorRT 10.3.0. Sequential runs, each with 50 warmup
calls and 300 measured calls cycling through the 18 `handwave3` luma frames:

| Engine | Mean detector call | p95 | Peak process RSS |
| --- | --- | --- | --- |
| Existing YOLOv10n FP16 | 7.67 ms | 10.30 ms | 424.5 MiB |
| YOLO26n mixed FP16/FP32 | 11.13 ms | 12.88 ms | 409.1 MiB |

These are detector-call measurements, not camera FPS or full-pipeline latency.
Power/clock settings were not changed or locked; RSS is not total shared GPU
memory. This sample does not establish an accuracy improvement over YOLOv10n.
Full reports with artifact/input hashes are stored locally under
`openmv-n6/perception/out/yolo26/`.

On the bundled bus image converted to grayscale, four confident detections
were matched against ONNX CPU. Mixed precision achieved minimum box IoU 0.99626
and maximum absolute score error 0.01873. Unconstrained FP16 had score error
0.10565 and failed the 0.03 limit; strict FP32 passed with error 0.0000166.
The mixed recipe is the selected runtime candidate. Preserve the failed result
as evidence rather than interpreting engine creation alone as validation.

Live smoke test: `MAP=none bash openmv-n6/tools/run_yolo26.sh` successfully
started the connected N6, thermal/radar students, calibrated fusion and radar.
After 581 received frames the viewer reported 8.78 host FPS, 8.77 thermal FPS,
zero batch drops, stalls or starvation, and 25.5 ms detector time under the
full load. The radar had received 839 frames. The optional map service was off.
This is a short integration check, not a long-duration reliability test.

Health remained WARN for the chosen radiometric range/display AGC and, in one
snapshot, a thermal/radar candidate rejected by the existing geometry gate.
There were no AI processing errors. The scene had no accepted visible person
boxes in that snapshot, so this run does not establish live detection accuracy.
The saved telemetry is `openmv-n6/perception/out/yolo26/live-smoke.json`.

## Close-up filtering correction

A subsequent live inspection found a real near-camera person detected at 0.94
confidence but discarded by the existing large-landscape-box filter. The box
was `(190, 0, 449, 394)` in a 640×400 frame: the head/legs were outside the
image. An edge-clipped candidate with background remaining on one side is now
retained with `geometry_warning`, and contributes `det_weak` rather than instant
detector trust to tracking. The old 613×392 near-full-scene rejection remains.
Viewer regressions cover the observed box, its mirror and the original false
positive. This is a downstream filtering fix, not evidence that YOLO26 has
better dataset accuracy or that all close-range detections are now solved.
