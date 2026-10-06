# Perception code

Dataset and supervision contracts are in `dataset.py` and `student_data.py`.
`associate.py`, `project.py`, `decision.py` and `map_api.py` contain the sensor
association, geometry, decision and map boundaries.

Student implementation is split by responsibility:

| Module | Responsibility |
| --- | --- |
| `models/features.py` | Derived thermal/radar channels and point features |
| `models/students.py` | Encoders, detection heads and student architectures |
| `learning/data.py` | Augmentation, student datasets, loaders and stream statistics |
| `learning/losses.py` | Hungarian matching and supervision-weighted detection loss |
| `learning/engine.py` | Evaluation, training, checkpoint resume and optimizer state |
| `students.py` | Compatibility exports for the original public names |

`train_students.py` and `export_students_onnx.py` remain the command-line entry
points. See [TRAINING.md](TRAINING.md) for training and
[export/README.md](export/README.md) for export/Colab workflows.

The exporter snapshots both `models/` and `learning/` into each training bundle
and hashes their files. If a local training dependency is added, include it in
`export/export_shards.py:TRAINING_CODE_FILES`; the bundle dependency regression
test checks that imports can be resolved from the exported code. Notebook
preflight parses the entire copied Python package before starting training.

`autolabel/`, `external/` and `radar_ai/` retain their separate workflows.
`training/thermal_objects/` contains the thermal-object training workflow and
is distinct from the reusable optimization code in `learning/`.
