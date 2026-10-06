# Verified project status — 2026-09-08

Scope: this `thermal-fusion` repository. The neighboring `ValiSight_yael`
repository is an optional map integration dependency and was not reorganized.

## Organization

The existing module split separates C fusion, board capture, host viewing,
web assets, perception models, reusable training and task-specific workflows.
Legacy entry points and compatibility exports remain in place. Root Make
targets, dependency lists, source checks and a documentation index provide one
entry point for development. Model paths can be overridden with
`THERMAL_FUSION_MODEL_DIR`; thermal training defaults follow the current user's
home directory. Python selection also reaches the C test helpers and launcher.

## Offline verification

- Source validation: 150 Python, shell/template and notebook files passed.
- Pytest: 82 passed, 3 skipped because host PyTorch is unavailable.
- C synthetic regression and ASan/UBSan/LeakSanitizer: passed outside the
  restricted ptrace environment.
- Radar, viewer/HTTP and calibration script suites: passed with loopback sockets
  available.
- Recorded regression sequence: all checks passed on 18 frames.

The machine identifies as Jetson Orin Nano Super. Host dependencies and the
TensorRT Python module are present. RGB YOLOv10n and V6 thermal/radar student
engine files, the thermal warp and solved radar calibration are present.
File presence does not prove engine compatibility or current sensor alignment.
The host interpreter lacks PyTorch, ONNX, ONNX Runtime and scikit-learn; another
environment or a training container may supply them.

## YOLO26 integration

YOLO26 model selection, checked ONNX export, TensorRT decoding and a GPU-only
launch profile have been added. The initial candidate is YOLO26n at 640×640.
See the [deployment guide](../deployment/yolo26/README.md) for commands and
validation. The original cleanup results above predate this integration.

YOLO26n is now exported, built and running in the live viewer. Plain FP16 failed
the numerical confidence check; the selected mixed FP16/FP32 engine passed.
The offline mean detector call was 11.13 ms (YOLOv10n baseline: 7.67 ms).
Live integration with thermal/radar ran at 8.78 FPS, with zero batch drops or
stream stalls over the first 581 frames; the sampled detector time under full
load was 25.5 ms. The map service was disabled for this smoke test. Health was
WARN, with range/display and fusion-gate warnings recorded in the deployment
guide. No accuracy improvement is claimed.

Integration verification: 154 source files passed, 85 pytest checks passed,
3 host PyTorch checks skipped, and the offline viewer/radar/calibration suites
passed. The sample-image ONNX/engine check and live telemetry are documented in
the deployment guide with their limits.

Before choosing a larger size or claiming improved accuracy:

1. Record the current live baseline: actual inputs, model versions, RAM,
   per-stage latency, dropped frames and end-to-end latency.
2. Evaluate the new detector on representative labeled recordings, including
   class IDs and boxes in the correct image plane.
3. Benchmark an RGB model on this Jetson before selecting a larger model.
4. Evaluate a thermal-trained detector on representative thermal recordings.
5. Validate timestamps, extrinsics and radar association before combining the
   detections into tracks; measure accuracy and latency of the whole pipeline.

The original cleanup did not run live acquisition, training, inference or
firmware flashing. The displayed visible stream currently
transports luma; calling the camera “RGB” does not make these frames color data.
