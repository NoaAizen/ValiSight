#!/usr/bin/env python3
"""Export YOLOv8n/YOLO11n to the static NMS ONNX contract used on the Jetson.

Run this in an isolated environment that has Ultralytics installed. The ONNX is
portable; copy it to the Jetson model directory and build the TensorRT engine
there, because TensorRT engines are tied to their target GPU/TRT version.

Ultralytics models/code are AGPL-3.0 by default. Check the applicable license
before embedding one in a closed-source or commercial product.
"""
import argparse
import os
import shutil
import sys

MODEL_FILES = {
    "yolov8n": ("yolov8n.pt", "yolov8n_nms.onnx"),
    "yolo11n": ("yolo11n.pt", "yolo11n_nms.onnx"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODEL_FILES),
                    default="yolo11n")
    ap.add_argument("--weights",
                    help="local .pt file; default lets Ultralytics obtain the official weight")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/archive/radar/models"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. Use an isolated environment, e.g. "
            "python3 -m venv /tmp/yolo-export && "
            "/tmp/yolo-export/bin/pip install ultralytics")

    default_weights, output_name = MODEL_FILES[args.model]
    weights = args.weights or default_weights
    exported = YOLO(weights).export(
        format="onnx",
        imgsz=640,
        batch=1,
        dynamic=False,
        nms=True,
        simplify=True,
        opset=args.opset,
    )
    exported = os.path.abspath(str(exported))
    os.makedirs(args.out_dir, exist_ok=True)
    destination = os.path.join(os.path.abspath(args.out_dir), output_name)
    if exported != destination:
        shutil.copy2(exported, destination)

    print(destination)
    print("On the Jetson, build the engine with:")
    print("  python3 tools/trt_detect.py --build %s" % args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
