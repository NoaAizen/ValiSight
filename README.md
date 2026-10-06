# Thermal fusion

OpenMV N6 thermal/visible fusion, radar integration, calibration and student
model training. The project code lives in [openmv-n6/](openmv-n6/README.md).

נקודת הכניסה לפרויקט: כאן נמצאות פקודות העבודה ומפת התיקיות.
להתקנה, הגדרות ופתרון בעיות ראו [מדריך הפיתוח](docs/DEVELOPMENT.md).
למצב שנבדק ולהכנה לשילוב YOLO ראו [מצב הפרויקט](docs/STATUS.md).
כל מסמכי התכנון והכיול מרוכזים ב[אינדקס התיעוד](docs/README.md).

שילוב YOLO26: [הפעלה, ייצוא ומדידת ביצועים](deployment/yolo26/README.md).

## Work on the code

```sh
make               # list available commands
make doctor        # inspect dependencies and artifacts without opening sensors
make check         # Python/shell syntax and notebook document validation
make build          # host executable and shared C library
make test           # C sanitizers, Python, offline viewer/calibration, real frames
make test-python    # pytest suites only
make test-scripts   # script suites, including the live viewer's local HTTP server
```

Use `PYTHON=/path/to/python` for all Python targets, including the host C test
helpers and `tools/run_live.sh`. Python checks require NumPy, OpenCV, pyserial
and pytest; calibration also uses SciPy, and capture tools use Pillow. Model checks run when PyTorch is
installed and are reported as skipped otherwise. No test target opens hardware
ports. The viewer suite binds a temporary localhost port, and C tests need an
environment where ASan/UBSan/LeakSanitizer can run.

## Source map

| Directory | Responsibility |
| --- | --- |
| `docs/` | Development instructions, documentation index and verified project status |
| `scripts/` | Read-only project checks and dependency inventory |
| `requirements/` | Host, development and optional training dependency lists |
| `openmv-n6/src/` | Portable C pipeline and its MicroPython binding; one algorithm source for host and firmware |
| `openmv-n6/host/` | C build, synthetic fixtures and recorded-frame verification |
| `openmv-n6/tools/viewer/` | Native pipeline wrapper, display, rendering, serial streaming and telemetry |
| `openmv-n6/tools/` | CLI entry points, recording, sensor I/O, detection, tracking and host integration |
| `openmv-n6/tools/web/` | HTML assets for live viewing, labeling and radar calibration/diagnostics |
| `openmv-n6/tools/board/` | Standalone MicroPython probes and board code templates |
| `openmv-n6/tools/calib/` | Calibration solving, correspondence capture and validation |
| `openmv-n6/tools/diag/` | Host-side bench diagnostics |
| `openmv-n6/radar/` | Board-compatible mmWave parser and chirp configuration calculations |
| `openmv-n6/perception/` | Dataset contracts, association, decisions, map adapter and training/export entry points |
| `openmv-n6/perception/models/` | PyTorch feature builders and student architectures |
| `openmv-n6/perception/learning/` | Augmentation, loaders, losses, evaluation and training |
| `openmv-n6/perception/autolabel/` | Teacher inference, supervision and dataset preparation |
| `openmv-n6/perception/export/` | Shards, YOLO exports and portable Colab code bundles |
| `openmv-n6/perception/external/` | Public dataset conversion |
| `openmv-n6/perception/radar_ai/` | Radar clustering, features, tracking and classifier workflows |
| `openmv-n6/perception/training/` | Task-specific training workflows, including thermal objects |
| `openmv-n6/openmv-integration/` | Patch and instructions for integrating the C module into OpenMV |
| `openmv-n6/calib-artifacts/` | Measured calibration artifacts and supporting capture tools |
| `openmv-n6/firmware-out/` | Flashing instructions and local build artifacts |
| `captures/`, `openmv-n6/captures/`, `openmv-n6/perception/out/` | Recorded/generated data; `handwave3` is the checked-in regression fixture |

Existing entry points remain valid, including `tools/live.py`, `tools/capture.py`
and `python -m perception.train_students`. Deploy the complete source directories:
the host scripts load the neighboring `viewer/`, `web/` and `board/templates/`
resources. `perception.students` preserves the original model/helper imports;
new implementation code belongs in `models/` or `learning/`.
