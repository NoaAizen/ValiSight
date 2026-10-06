"""Detector artifact names and export contracts; no GPU dependencies."""

MODEL_FILES = {
    "yolov10n": ("yolov10n.onnx", "yolov10n_fp16.engine"),
    "yolov8n": ("yolov8n_nms.onnx", "yolov8n_nms_fp16.engine"),
    "yolo11n": ("yolo11n_nms.onnx", "yolo11n_nms_fp16.engine"),
    **{f"yolo26{size}": (f"yolo26{size}_end2end.onnx",
                          f"yolo26{size}_end2end_fp16.engine")
       for size in "nsmlx"},
}
# Initial Jetson candidate: plain FP16 failed the confidence-parity check.
# The validated recipe retains FP32 in the classification head.
MODEL_FILES["yolo26n"] = ("yolo26n_end2end.onnx", "yolo26n_end2end_mixed.engine")


def export_options(model):
    """Select decoded xyxy/score/class output, never raw prediction planes."""
    if model not in MODEL_FILES:
        raise ValueError(f"unknown detector model: {model}")
    if model.startswith("yolo26") or model == "yolov10n":
        return {"nms": False}
    return {"nms": True}


def selected_engine(argv=None):
    """Resolve the same CLI model/engine override used by live.py, pre-launch."""
    import argparse
    import os
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--detect-model", choices=sorted(MODEL_FILES), default="yolov10n")
    parser.add_argument("--detect-engine")
    args, _ = parser.parse_known_args(argv)
    directory = os.path.expanduser(os.environ.get(
        "THERMAL_FUSION_MODEL_DIR", "~/archive/radar/models"))
    return args.detect_engine or os.path.join(directory, MODEL_FILES[args.detect_model][1])


if __name__ == "__main__":
    print(selected_engine())
