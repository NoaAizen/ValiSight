# Jetson thermal vehicle training

This trains a YOLOv8n baseline for person, car, truck, bus, motorcycle and
bicycle on the prepared FLIR thermal dataset. It does not enable a model in
the viewer, train animals, or establish accuracy on the Lepton sensor.

The isolated Docker image preserves the existing Jetson CUDA PyTorch 2.4.0 and
torchvision builds and adds Ultralytics 8.3.0. Host Python is unchanged.
Reference: [Ultralytics Jetson setup](https://docs.ultralytics.com/guides/nvidia-jetson/)
and [training configuration](https://docs.ultralytics.com/modes/train/).

From this directory:

```bash
./run.sh build
./run.sh smoke   # 64 training + 64 validation images, includes all six classes
./run.sh start   # requires a successful smoke checkpoint; runs detached
./run.sh status
./run.sh logs
./run.sh stop
./run.sh resume  # resumes baseline/weights/last.pt after stopping
```

Defaults: 320-pixel input, batch 1, no loader workers, no RAM image cache,
FP32, 30 maximum epochs, early stopping patience 8. Lazy CUDA loading and
early cuBLAS initialization are needed on this 8GB Orin Nano. The first
preflight attempts with stricter memory limits failed before completing an
epoch; these failures remain recorded in the smoke logs. Container CPU and
memory use are bounded; live acquisition should not be started alongside a
training job without checking available resources.

Paths on this machine:

- Dataset: `/home/moshe/datasets/FLIR_thermal_vehicles_yolo` (mounted read-only).
- Initialization: `/home/moshe/datasets/thermal-object-models/yolov8n.pt`.
  The adjacent source JSON records the official download URL and SHA-256.
- Results: `/home/moshe/datasets/thermal-object-runs`.
- Progress: `training-status.json`, `baseline/results.csv`, Docker logs.
- Checkpoints: `baseline/weights/last.pt`, `best.pt`, and periodic epoch files.

`THERMAL_DATA`, `THERMAL_MODELS`, and `THERMAL_RUNS` override those directories.
Status/checkpoint paths under `/results` refer to the mounted host results
directory. Closing SSH does not stop the detached training container. Stopping
mid-epoch loses that epoch's unsaved work; resume continues the last saved
checkpoint. A successfully completed smoke test checks the software path only;
its metrics are not a model acceptance result.

The test split remains untouched. Before deployment, compare per-class metrics
on validation, evaluate the held-out test and separate Lepton night recordings,
measure Jetson inference latency, and export a model with its preprocessing and
class-order contract. FLIR 640x512 intensity data and Lepton 160x120 Celsius
data are different domains; high FLIR scores alone do not validate the rig.

## Google Colab

Local training was stopped on 2026-09-08 before the first full epoch completed.
No full-run checkpoint exists; the cloud run starts from official YOLOv8n weights.

Upload `/home/moshe/datasets/thermal-colab/thermal-vehicles-colab.tar` and its
`.sha256` file to Google Drive under `My Drive/ThermalFusion`. Open
`thermal_vehicles_colab.ipynb` in Colab, select a GPU runtime, and Run all.
The notebook uses Ultralytics 8.3.200 with Colab CUDA PyTorch, keeps training data
on the VM, and persists last/best weights to Drive every epoch. Re-run after an
interruption to resume the last saved epoch. GPU runtime testing is pending
execution in the user’s Colab account.
