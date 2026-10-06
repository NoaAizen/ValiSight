# Thermal vehicles and animals

Status: the dataset is prepared and a one-epoch GPU smoke test has completed.
The local six-class vehicle/person run was stopped at the user’s request before
its first epoch completed; it has no baseline checkpoint. A Google Colab notebook
is available in `training/thermal_objects/thermal_vehicles_colab.ipynb`. No new model is
enabled in the live viewer or validated on Lepton recordings. The deployed V6
student is a person-only detector. Its confidence threshold cannot add classes.

## Received FLIR ADAS v2 audit

The Windows transfer at `/home/moshe/datasets/FLIR_ADAS_v2` contains 46,447
files and 12,737,914,590 bytes, matching the source Properties screenshot.
This is a size/count comparison, not a source-to-destination checksum check.
All images referenced by the three thermal COCO files exist. Their video IDs
are disjoint across train, validation and test. The detailed audit is saved at
`/home/moshe/datasets/FLIR_ADAS_v2_audit.json`.

| Thermal split | Images | Car boxes | Dog boxes | Deer boxes |
|---|---:|---:|---:|---:|
| Training | 10,742 | 73,623 | 4 | 8 |
| Validation | 1,144 | 7,133 | 0 | 0 |
| Test | 3,749 | 30,517 | 25 | 0 |

The animal coverage is insufficient for claiming reliable animal detection;
there is no animal validation set. Do not move test dogs into training just to
increase the count. Additional thermal animal recordings/annotations are needed.
Training has 2,110 explicitly night-labelled images and validation has 112;
many images have no time-of-day tag and must not be presumed day or night.

The vehicle baseline export uses `--classes person,car,truck,bus,motorcycle,bicycle`
and writes `/home/moshe/datasets/FLIR_thermal_vehicles_yolo`. The FLIR category
`motor` maps to `motorcycle`; the exporter preserves the declared output class
order. Source data and the test split are left intact. `report.json` and
`data.yaml` appear when export completes. Preparation does not train a model.

Host Python remains unchanged. Training uses the isolated
`thermal-objects-train:jp6-v1` Docker image with Jetson CUDA PyTorch 2.4.0 and
Ultralytics 8.3.0. See `training/thermal_objects/README.md` for progress,
checkpoint locations and resuming. ONNX/TensorRT export and live validation
remain separate follow-up steps after model quality is assessed.

## Exporting

The existing `external/public_thermal.py` conversion deliberately discards all
non-person boxes. Its exported FLIR shards cannot be reused as multiclass
training labels. Return to the original thermal images and COCO annotations.

Prepare separate training and validation sets from `openmv-n6`:

```bash
python3 -m perception.external.thermal_objects_data \
  --train-root /path/to/thermal/train \
  --train-annotations /path/to/train/coco.json \
  --val-root /path/to/thermal/val \
  --val-annotations /path/to/val/coco.json \
  --out /path/to/new/thermal-objects
```

This exports grayscale PNGs, normalized YOLO labels, `data.yaml` and a class
coverage report. It preserves person, car, truck, bus, motorcycle, bicycle,
dog, cat, horse, cow, sheep and deer annotations **when actually present**.
Zero-example classes are explicitly reported as missing. Listing a species
does not give a model the ability to detect it. Crowd/ignore labels require
explicit handling and are refused. Existing outputs, missing source images,
raw Celsius/16-bit inputs and identical train/validation frames are refused.
Keep entire recording sessions in one split; a pixel hash does not detect
near-duplicate frames from the same sequence.

Candidate sources inspected:

- [Teledyne FLIR ADAS](https://www.oem.flir.com/en-150/about/news/expanded-teledyne-flir-starter-thermal-dataset-for-adas-and-autonomous-vehicle-testing/): vehicle classes and dog, not all wildlife.
- [Thermal Object Detection models](https://huggingface.co/MAli-Farooq/Thermal_Object_Detection): author reports bicycle, bike, bus, car, dog, person and pole. The available model archive is about 1.23 GB; it has not been downloaded or tested here.

Before live integration, train/export a thermal multiclass detector and measure
per-class precision/recall, false positives and missed detections on held-out
Lepton 160x120 recordings, including night scenes, cold parked vehicles,
stationary animals and warm empty backgrounds. Check runtime on the Jetson.
Public automotive image scores do not establish performance on this camera.
The input normalization and class order must travel with the exported model.
Vehicles and animals must not pass through the existing person-height,
body-heat or person-lock filters. Thermal inference must remain independent
of visible-camera illumination and use the thermal-to-visible calibration only
for placing overlays. Without mapping, results need a native thermal view.
