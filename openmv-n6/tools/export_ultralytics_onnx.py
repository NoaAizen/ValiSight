#!/usr/bin/env python3
"""Export YOLO detectors to static [1,300,6] ONNX for the Jetson.

Run this in an isolated environment that has Ultralytics installed. The ONNX is
portable; copy it to the Jetson model directory and build the TensorRT engine
there, because TensorRT engines are tied to their target GPU/TRT version.

Ultralytics models/code are AGPL-3.0 by default. Check the applicable license
before embedding one in a closed-source or commercial product.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

from detector_models import MODEL_FILES, export_options



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODEL_FILES),
                    default="yolo26n")
    ap.add_argument("--weights",
                    help="local .pt file; default lets Ultralytics obtain the official weight")
    ap.add_argument("--out-dir", default=os.path.expanduser(os.environ.get(
        "THERMAL_FUSION_MODEL_DIR", "~/archive/radar/models")))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    try:
        import ultralytics
        from ultralytics import YOLO
        import onnx
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. Use an isolated environment, e.g. "
            "python3 -m venv /tmp/yolo-export && "
            "/tmp/yolo-export/bin/pip install ultralytics")

    output_name = MODEL_FILES[args.model][0]
    weights = args.weights or (args.model + ".pt")
    model = YOLO(weights)
    if model.task != "detect":
        raise SystemExit("This runtime requires an object detection model")
    # The host's class filters and association use the existing COCO-80 table.
    from detect import COCO
    aliases = {"motorbike": "motorcycle", "aeroplane": "airplane",
               "sofa": "couch", "pottedplant": "pottedplant",
               "tvmonitor": "tv"}
    def canonical(name):
        return aliases.get(name, name).replace("_", "").replace(" ", "")
    names = [model.names[i] for i in range(len(model.names))]
    if [canonical(n) for n in names] != [canonical(n) for n in COCO]:
        raise SystemExit("Model class order must match COCO-80 for the visible detector")
    options = dict(
        format="onnx",
        imgsz=640,
        batch=1,
        dynamic=False,
        **export_options(args.model),
        device="cpu",
        simplify=True,
        opset=args.opset,
    )
    exported = model.export(**options)
    exported = os.path.abspath(str(exported))
    graph = onnx.load(exported)
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise RuntimeError("Expected one ONNX input and one decoded output")
    def shape(tensor):
        return tuple(d.dim_value for d in tensor.type.tensor_type.shape.dim)
    from trt_detect import validate_io_shapes
    input_shape = shape(graph.graph.input[0])
    output_shape = shape(graph.graph.output[0])
    validate_io_shapes(input_shape, output_shape)
    os.makedirs(args.out_dir, exist_ok=True)
    destination = os.path.join(os.path.abspath(args.out_dir), output_name)
    if exported != destination:
        shutil.copy2(exported, destination)

    with open(destination, "rb") as source:
        digest = hashlib.sha256(source.read()).hexdigest()
    with open(destination + ".json", "w") as sidecar:
        json.dump({"model": args.model, "ultralytics": ultralytics.__version__,
                   "weights": str(weights), "options": options, "names": names,
                   "input_shape": input_shape, "output_shape": output_shape,
                   "onnx_sha256": digest}, sidecar, indent=2)

    print(destination)
    print("On the Jetson, build the engine with:")
    print("  python3 tools/trt_detect.py --build %s" % args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
