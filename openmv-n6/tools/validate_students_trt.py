#!/usr/bin/env python3
"""Validate the TensorRT student engines against the val split on the Jetson.

Replicates perception.students.evaluate_model (same thresholds, same
Hungarian matching, same metric definitions) in numpy+scipy so no torch is
needed. If the numbers here match the training run's checkpoint metrics, the
whole exported chain - ONNX -> TensorRT engine -> this runtime - is faithful
to what was trained, and live.py can trust it.

Usage:
    python3 tools/validate_students_trt.py \
        [--data perception/out/gexport/v6] [--student both]
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), ".."))
from perception.student_data import load_manifest, load_split   # noqa: E402
import trt_students   # noqa: E402

LABEL_POSITIVE = 1


def frame_gt(arrays, i, plane, max_objects=8):
    """Mirror student_data.detection_target: GT centers normalized to plane."""
    if plane == "thermal":
        state_key, box_key, count_key = (
            "thermal_label_state", "th_boxes", "n_th_boxes")
        width, height = 160.0, 120.0
        conf_col = False
    else:
        state_key, box_key, count_key = (
            "radar_label_state", "rgb_boxes", "n_rgb_boxes")
        width, height = 640.0, 400.0
        conf_col = True
    state = int(arrays[state_key][i])
    centers = []
    if state == LABEL_POSITIVE:
        n = min(int(arrays[count_key][i]), max_objects)
        rows = np.asarray(arrays[box_key][i, :n], np.float32)
        order = np.argsort(-rows[:, 4]) if conf_col else np.arange(n)
        for row in rows[order[:max_objects]]:
            x, y, w, h = (float(v) for v in row[:4])
            centers.append((np.clip((x + 0.5 * w) / width, 0.0, 1.0),
                            np.clip((y + 0.5 * h) / height, 0.0, 1.0)))
    return state, centers


def evaluate(run_frame, arrays, n_frames, plane, width, height,
             threshold=0.5, progress_every=1000):
    u_err, v_err, count_err = [], [], []
    tp = fp = fn = tn = 0
    for i in range(n_frames):
        state, gt = frame_gt(arrays, i, plane)
        if state < 0:
            continue
        scores, boxes = run_frame(i)
        pred = [(float(boxes[k][0]), float(boxes[k][1]))
                for k in range(len(scores)) if scores[k] >= threshold]
        pred_frame, gt_frame = bool(pred), bool(gt)
        tp += int(pred_frame and gt_frame)
        fp += int(pred_frame and not gt_frame)
        fn += int(not pred_frame and gt_frame)
        tn += int(not pred_frame and not gt_frame)
        count_err.append(abs(len(pred) - len(gt)))
        if pred_frame and gt_frame:
            cost = np.abs(
                np.array(pred)[:, None, :] - np.array(gt)[None, :, :]
            ).sum(-1)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                u_err.append(abs(pred[r][0] - gt[c][0]) * width)
                v_err.append(abs(pred[r][1] - gt[c][1]) * height)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{n_frames} frames", file=sys.stderr)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    med = lambda x: float(np.median(x)) if x else float("nan")  # noqa: E731
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "count_mae": round(float(np.mean(count_err)), 3) if count_err
        else float("nan"),
        "median_u_px": round(med(u_err), 2),
        "median_v_px": round(med(v_err), 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    ap = argparse.ArgumentParser()
    default_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "perception", "out", "gexport", "v6")
    ap.add_argument("--data", default=default_data)
    ap.add_argument("--student", choices=("thermal", "radar", "both"),
                    default="both")
    ap.add_argument("--split", choices=("train", "val"), default="val")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--session", action="append", default=[],
        help="evaluate only this session from --split (repeatable)")
    a = ap.parse_args()

    manifest = load_manifest(a.data)
    if a.session:
        split_sessions = manifest["split"][a.split]
        unknown = [s for s in a.session if s not in split_sessions]
        if unknown:
            ap.error("--session is not in the selected split: "
                     + ", ".join(unknown))
        manifest = dict(manifest)
        manifest["split"] = dict(manifest["split"])
        manifest["split"][a.split] = a.session
    print(f"loading {a.split} split:", manifest["split"][a.split],
          file=sys.stderr)
    val = load_split(a.data, a.split, manifest)
    arrays, temporal = val.arrays, val.temporal
    n = len(val)
    print(f"{n} frames, thermal mode {val.thermal_mode}", file=sys.stderr)

    if a.student in ("thermal", "both"):
        model = trt_students.ThermalStudentTrt(conf=a.threshold)
        thermal = arrays["thermal"]

        def run_thermal(i):
            seq = np.zeros((3, 120, 160), np.float32)
            seq[0] = thermal[i]
            v1, v2 = float(temporal.valid1[i]), float(temporal.valid2[i])
            if v1 > 0:
                seq[1] = thermal[temporal.prev1[i]]
            if v2 > 0:
                seq[2] = thermal[temporal.prev2[i]]
            return model.infer_raw(seq, v1, v2, float(temporal.dt1_ms[i]))

        print("== THERMAL (TensorRT on Jetson) ==")
        m = evaluate(run_thermal, arrays, n, "thermal", 160.0, 120.0,
                     a.threshold)
        for k, v in m.items():
            print(f"  {k}: {v}")

    if a.student in ("radar", "both"):
        model = trt_students.RadarStudentTrt(conf=a.threshold)
        radar, n_radar = arrays["radar"], arrays["n_radar"]

        def run_radar(i):
            return model.infer_raw(radar[i], int(n_radar[i]))

        print("== RADAR (TensorRT on Jetson) ==")
        m = evaluate(run_radar, arrays, n, "radar", 640.0, 400.0, a.threshold)
        for k, v in m.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
