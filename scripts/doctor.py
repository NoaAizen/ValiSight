#!/usr/bin/env python3
"""Read-only host dependency and artifact inventory; never opens sensor ports."""
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "openmv-n6"


def main():
    print(f"Project: {ROOT}\nPython: {sys.executable}\nHost: {platform.machine()}")
    model = Path("/proc/device-tree/model")
    if model.exists():
        print(f"Board: {model.read_text().rstrip(chr(0))}")
    missing = []
    for name in ("numpy", "cv2", "serial", "scipy", "PIL", "pytest"):
        found = importlib.util.find_spec(name) is not None
        print(f"{'OK' if found else 'MISSING'} Python {name}")
        if not found:
            missing.append(name)
    for name in ("make", "cc", "git", "bash"):
        found = shutil.which(name)
        print(f"{'OK' if found else 'MISSING'} {name}: {found or 'not on PATH'}")
        if not found:
            missing.append(name)
    for name in ("torch", "tensorrt", "onnx", "onnxruntime", "sklearn"):
        found = importlib.util.find_spec(name) is not None
        print(f"OPTIONAL {name}: {'installed' if found else 'not installed'}")
    models = Path(os.environ.get(
        "THERMAL_FUSION_MODEL_DIR", "~/archive/radar/models")).expanduser()
    students = Path(os.environ.get(
        "STUDENT_ENGINES", str(PROJECT / "perception/out/gexport/v6/models"))).expanduser()
    for path in (PROJECT / "host/libfusion.so",
                 PROJECT / "calib-artifacts/warp.lut",
                 PROJECT / "calib-artifacts/radar_rgb_2026-08-18.json",
                 models / "yolov10n_fp16.engine",
                 students / "thermal_student.engine",
                 students / "radar_student.engine"):
        print(f"ARTIFACT {'present' if path.is_file() and path.stat().st_size else 'missing'}: {path}")
    print("Optional modules/artifacts do not affect the exit status. "
          "Presence does not verify CUDA execution, calibration or sensor readiness.")
    return int(bool(missing))


if __name__ == "__main__":
    sys.exit(main())
