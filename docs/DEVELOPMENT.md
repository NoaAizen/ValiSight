# Development

Run commands from the repository root. Python 3.10 is the currently checked
host interpreter; a C99 compiler, make, Git and Bash are used by the checks.

## Environment

```sh
make doctor
make check
make build
make test
```

`doctor` checks module availability and local artifact presence. It does not
load engines, open serial ports, configure the radar or validate calibration.
Required host/development dependencies affect its exit status; optional training
modules and deployment artifacts are reported separately.

For a fresh portable host environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/dev.txt
make doctor test PYTHON="$PWD/.venv/bin/python"
```

On this Jetson, the system interpreter already has the host dependencies and
TensorRT. Keep the working NVIDIA environment intact. If isolation is needed,
create the virtual environment with `--system-site-packages` so it can see
the installed system modules, then inspect it with `make doctor` before
installing missing dependencies. The requirement lists are not lock files;
they do not pin a JetPack/CUDA stack or reproduce a complete machine.

PyTorch, TensorRT and CUDA must match the target environment. Generic host
requirements intentionally do not install or upgrade them. For training use
the [student instructions](../openmv-n6/perception/TRAINING.md) or the existing
[thermal-object container](../openmv-n6/perception/training/thermal_objects/README.md).
`requirements/training.txt` lists supporting packages after that environment
has been selected; it is not a complete GPU installation recipe.

## Configuration and generated data

| Setting | Used by | Default |
| --- | --- | --- |
| `PYTHON` | Make targets and live launcher | `python3` |
| `THERMAL_FUSION_MODEL_DIR` | CPU/TensorRT RGB detectors and live launch check | `$HOME/archive/radar/models` |
| `STUDENT_ENGINES` | Live launcher | `openmv-n6/perception/out/gexport/v6/models` |
| `HTTP` | Live launcher and desktop window | `8088` |
| `RADAR_CFG` | Live launcher | `openmv-n6/radar/configs/radar_people.cfg` |
| `MAP`, `MAP_HEADING` | Live launcher | Existing measured site; heading unset |
| `THERMAL_DATA`, `THERMAL_MODELS`, `THERMAL_RUNS` | Thermal training launcher | Dataset directories under `$HOME/datasets` |

Use an absolute, expanded path for directory overrides. Export
`THERMAL_FUSION_MODEL_DIR` so both the shell launcher and Python subprocesses
see the same location. Existing defaults keep the current workstation usable;
moving the rig requires choosing its actual map position and checking calibration.

```sh
export THERMAL_FUSION_MODEL_DIR="$HOME/archive/radar/models"
# Hardware operation: starts acquisition and can replace a running viewer.
PYTHON=python3 openmv-n6/tools/run_live.sh
```

For all runtime flags see [tools](../openmv-n6/tools/README.md). The launcher
resolves USB identities; do not assume fixed `/dev/ttyACM*` numbering.
Keep `viewer/`, `web/` and `board/templates/` with the entry-point scripts.

Recordings, model weights, TensorRT engines and upstream firmware builds are
local artifacts. Keep recordings under the ignored capture/output directories,
and models in the configured external directory. The committed `handwave3`
sequence is the regression fixture. Calibration artifacts are source evidence:
retain them with their measurements and provenance.

## Checks and troubleshooting

`make test` includes source checks, C sanitizers, pytest, the script suites and
18 recorded frames. The script suites are explicit because several use `main()`
rather than pytest-discoverable tests. PyTorch checks skip when it is absent;
that does not validate model training. `make check` parses source without
executing it and validates notebook structure, not notebook execution.

LeakSanitizer requires a process environment without ptrace. The HTTP test
requires a temporary loopback socket. In a restricted runner, run these checks
in an environment that supports those features rather than disabling checks.
No offline test target opens hardware ports.

`make clean` removes only host build products and generated synthetic fixtures.
It does not delete recordings, calibration, trained models or firmware sources.

## Module boundaries

- Keep measurement algorithms in `src/`; host and firmware share the C source.
- Keep serial acquisition in `tools/viewer/streaming.py`, separate from rendering.
- Keep display enhancements out of radiometric measurements and model inputs.
- Keep feature/model definitions in `perception/models/`, training in `learning/`.
- Preserve `live.py` and `perception.students` compatibility exports when moving code.
- Update the training bundle file list when adding training dependencies; run
  `test_training_data.py` to verify exported bundles remain self-contained.

Add future detector integration through the detector boundary. Check tensor
layout, class mapping and coordinate transforms before feeding detections into
association; a different engine filename alone does not establish compatibility.
