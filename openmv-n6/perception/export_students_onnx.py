#!/usr/bin/env python3
"""Export trained student checkpoints to ONNX for TensorRT on the Jetson.

Runs where PyTorch exists (Colab). The Jetson has no torch - it builds the
TensorRT engines from these ONNX files with trtexec and runs them in live.py.

Interfaces are frozen to what live.py can feed at runtime, batch of 1:

  thermal_student.onnx
    inputs : thermal_seq  (1,3,120,160) float32  [current, prev1, prev2],
             CELSIUS (raw counts * c_per_lsb + tmin - same scale as training)
             valid_prev1 (1,) float32  1.0 when prev1 is same-stream, dt<=300ms
             valid_prev2 (1,) float32
             dt1_ms      (1,) float32  ms between current and prev1
    outputs: scores (1,8) float32 sigmoid confidence per slot
             boxes  (1,8,4) float32 normalized cx,cy,w,h on the 160x120 plane

  radar_student.onnx
    inputs : radar_points (1,64,6) float32 [x,y,z,velocity,snr,noise],
             zero-padded, project frame - same as the export shards
             n_radar      (1,) int64 count of valid rows
    outputs: scores (1,8), boxes (1,8,4) normalized on the 640x400 RGB plane

Usage (Colab):
    PYTHONPATH=$DATA/code python3 -m perception.export_students_onnx --data $DATA
Then on the Jetson (see printed commands): trtexec --onnx=... --saveEngine=...
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from perception.students import RadarStudent, ThermalStudent

OPSET = 17


class ThermalOnnx(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, thermal_seq, valid_prev1, valid_prev2, dt1_ms):
        out = self.model({
            "thermal_seq": thermal_seq, "valid_prev1": valid_prev1,
            "valid_prev2": valid_prev2, "dt1_ms": dt1_ms,
        })
        return torch.sigmoid(out["object_logits"]), out["boxes"]


class RadarOnnx(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, radar_points, n_radar):
        out = self.model({"radar_points": radar_points, "n_radar": n_radar})
        return torch.sigmoid(out["object_logits"]), out["boxes"]


def rebuild_thermal(ck):
    model = ThermalStudent(
        ck["thermal_mean"], ck["thermal_std"], ck["max_objects"])
    if ck.get("disabled_channels"):
        model.derived.disable_channels(ck["disabled_channels"])
    model.load_state_dict(ck["model_state"])
    return model


def rebuild_radar(ck):
    streams = ck["streams"]
    model = RadarStudent(
        ck["stream_stats"], ck["max_objects"],
        use_ra=streams["ra"], use_rd=streams["rd"], use_rp=streams["rp"])
    for family, enabled in ck.get("family_enabled", {}).items():
        if not enabled and model.family_enabled.get(family, False):
            model.disable_family(family)
    for family, channels in (ck.get("channel_ablations") or {}).items():
        model.disable_channels(family, channels)
    model.load_state_dict(ck["model_state"])
    return model


def verify(onnx_path, torch_module, example_inputs, atol=2e-4):
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"  [WARN] onnxruntime not installed - skipping numeric check "
              f"of {os.path.basename(onnx_path)} (pip install onnxruntime)")
        return
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feeds = {
        inp.name: t.numpy() for inp, t in zip(sess.get_inputs(), example_inputs)
    }
    ort_out = sess.run(None, feeds)
    with torch.no_grad():
        torch_out = torch_module(*example_inputs)
    worst = 0.0
    for a, b in zip(torch_out, ort_out):
        worst = max(worst, float(np.abs(a.numpy() - b).max()))
    status = "OK" if worst <= atol else "MISMATCH"
    print(f"  numeric check vs torch: max abs diff {worst:.2e} [{status}]")
    if worst > atol:
        raise SystemExit(f"{onnx_path}: ONNX output diverges from torch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="gexport dir holding models/*.pt; ONNX lands there too")
    ap.add_argument("--max-points", type=int, default=64,
                    help="radar points per frame (export shard row width)")
    a = ap.parse_args()
    models = os.path.join(a.data, "models")
    torch.manual_seed(0)
    thermal_exported = False
    radar_exported = False

    ck_path = os.path.join(models, "thermal_student.pt")
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = ThermalOnnx(rebuild_thermal(ck)).eval()
        ex = (torch.rand(1, 3, 120, 160) * 40.0,
              torch.ones(1), torch.ones(1), torch.full((1,), 115.0))
        out = os.path.join(models, "thermal_student.onnx")
        torch.onnx.export(
            model, ex, out, opset_version=OPSET,
            input_names=["thermal_seq", "valid_prev1", "valid_prev2",
                         "dt1_ms"],
            output_names=["scores", "boxes"])
        print(f"exported {out}")
        verify(out, model, ex)
        thermal_exported = True
    else:
        print(f"[skip] {ck_path} not found")

    ck_path = os.path.join(models, "radar_student.pt")
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        streams = ck.get("streams", {})
        dense = [name for name in ("ra", "rd", "rp")
                 if streams.get(name, False)]
        if dense:
            # tools/trt_students.py currently has a point-cloud-only radar
            # input contract. Silently replacing trained dense streams with
            # zeros would create a valid-looking but behaviorally different
            # model, so refuse that lossy deployment conversion explicitly.
            print("[skip] radar ONNX: checkpoint uses dense stream(s) "
                  f"{dense}, while the Jetson runtime accepts points only")
        else:
            model = RadarOnnx(rebuild_radar(ck)).eval()
            pts = torch.zeros(1, a.max_points, 6)
            pts[0, :12] = torch.randn(12, 6) * torch.tensor(
                [2.0, 3.0, 0.5, 1.0, 10.0, 5.0])
            ex = (pts, torch.tensor([12], dtype=torch.long))
            out = os.path.join(models, "radar_student.onnx")
            torch.onnx.export(
                model, ex, out, opset_version=OPSET,
                input_names=["radar_points", "n_radar"],
                output_names=["scores", "boxes"])
            print(f"exported {out}")
            verify(out, model, ex)
            radar_exported = True
    else:
        print(f"[skip] {ck_path} not found")

    print("\nOn the Jetson, build the TensorRT engines with:")
    if thermal_exported:
        print("  /usr/src/tensorrt/bin/trtexec "
              "--onnx=models/thermal_student.onnx "
              "--saveEngine=models/thermal_student.engine --fp16")
    if radar_exported:
        print("  /usr/src/tensorrt/bin/trtexec "
              "--onnx=models/radar_student.onnx "
              "--saveEngine=models/radar_student.engine --fp16")


if __name__ == "__main__":
    main()
