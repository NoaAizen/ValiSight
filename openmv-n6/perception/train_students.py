#!/usr/bin/env python3
"""Train thermal and radar person students from exported NPZ shards.

Example (Jetson or Colab):
    python3 -m perception.train_students \
        --data perception/out/gexport/v2 --student both --epochs 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone

import numpy as np


def _imports():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for student training. Use a Colab GPU runtime "
            "or install the Jetson-compatible NVIDIA PyTorch wheel."
        ) from exc
    from perception.student_data import estimate_scalar_stats, load_train_val
    from perception.students import (
        RadarStudent, ThermalStudent, build_loaders, radar_stream_stats,
        train_model,
    )
    return (torch, estimate_scalar_stats, load_train_val, RadarStudent,
            ThermalStudent, build_loaders, radar_stream_stats, train_model)


def parse_family_channels(rows):
    result = {}
    for row in rows:
        if ":" not in row:
            raise ValueError(
                f"{row!r}: expected FAMILY:channel[,channel], e.g. points:snr,noise"
            )
        family, values = row.split(":", 1)
        result.setdefault(family.strip().lower(), []).extend(
            x.strip() for x in values.split(",") if x.strip()
        )
    return result


def manifest_sha256(data_dir):
    path = os.path.join(data_dir, "manifest.json")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def checkpoint_payload(model, metrics, manifest, args, extra):
    return {
        "format": "thermal-fusion-student-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_version": manifest.get("version"),
        "manifest_sha256": manifest_sha256(args.data),
        "train_sessions": manifest["split"]["train"],
        "val_sessions": manifest["split"]["val"],
        "model_state": model.state_dict(),
        "metrics": metrics,
        "config": vars(args),
        **extra,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True,
                    help="directory containing manifest.json and NPZ shards")
    ap.add_argument("--out", help="checkpoint directory (default DATA/models)")
    ap.add_argument("--student", choices=("thermal", "radar", "both"),
                    default="both")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-objects", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-dt-ms", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto",
                    help="auto, cpu, cuda or a torch device such as cuda:0")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--disable-thermal", default="",
                    help="comma-separated derived thermal channels")
    ap.add_argument("--disable-radar-family", action="append", default=[],
                    choices=("points", "ra", "rd", "rp"),
                    help="repeat to ablate complete radar input families")
    ap.add_argument("--disable-radar-channel", action="append", default=[],
                    metavar="FAMILY:CH[,CH]",
                    help="repeatable, e.g. points:snr,noise or ra:dx")
    args = ap.parse_args()

    (torch, estimate_scalar_stats, load_train_val, RadarStudent,
     ThermalStudent, build_loaders, radar_stream_stats,
     train_model) = _imports()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    manifest, train, val = load_train_val(args.data, args.max_dt_ms)
    out_dir = args.out or os.path.join(args.data, "models")
    os.makedirs(out_dir, exist_ok=True)
    print(f"device: {device}")
    print(f"thermal scale: {train.thermal_mode}")
    print(f"train: {len(train)} frames from {list(train.session_names)}")
    print(f"val:   {len(val)} frames from {list(val.session_names)}")
    print("radar streams:",
          ["points"] + [p for k, p in (
              ("range_angle", "ra"), ("range_doppler", "rd"),
              ("range_profile", "rp")) if k in train.arrays])

    if args.student in ("thermal", "both"):
        mean, std = estimate_scalar_stats(train.arrays["thermal"])
        model = ThermalStudent(mean, std, args.max_objects)
        disabled = [x.strip() for x in args.disable_thermal.split(",")
                    if x.strip()]
        model.derived.disable_channels(disabled)
        train_loader, val_loader = build_loaders(
            train, val, "thermal", args.max_objects,
            args.batch_size, args.workers)
        model, metrics = train_model(
            model, train_loader, val_loader, 160, 120, device,
            epochs=args.epochs, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, use_amp=not args.no_amp,
            tag="THERMAL")
        path = os.path.join(out_dir, "thermal_student.pt")
        torch.save(checkpoint_payload(
            model, metrics, manifest, args, {
                "student": "thermal",
                "thermal_mean": mean,
                "thermal_std": std,
                "thermal_mode": train.thermal_mode,
                "channels": model.derived.channel_names,
                "disabled_channels": disabled,
                "max_objects": args.max_objects,
            }), path)
        print(f"saved {path}")
        print(json.dumps(metrics, indent=2))

    if args.student in ("radar", "both"):
        stats = radar_stream_stats(train)
        model = RadarStudent(
            stats, args.max_objects,
            use_ra="ra" in stats, use_rd="rd" in stats, use_rp="rp" in stats)
        for family in args.disable_radar_family:
            if not model.family_enabled.get(family, False):
                print(f"[WARN] cannot ablate absent radar family {family}")
            else:
                model.disable_family(family)
        channel_ablations = parse_family_channels(
            args.disable_radar_channel)
        for family, channels in channel_ablations.items():
            if family != "points" and not model.family_enabled.get(family, False):
                raise ValueError(
                    f"cannot disable channels in absent radar family {family}")
            model.disable_channels(family, channels)

        train_loader, val_loader = build_loaders(
            train, val, "radar", args.max_objects,
            args.batch_size, args.workers)
        model, metrics = train_model(
            model, train_loader, val_loader, 640, 400, device,
            epochs=args.epochs, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, use_amp=not args.no_amp,
            tag="RADAR")
        path = os.path.join(out_dir, "radar_student.pt")
        torch.save(checkpoint_payload(
            model, metrics, manifest, args, {
                "student": "radar",
                "stream_stats": stats,
                "streams": {
                    "points": True, "ra": model.use_ra,
                    "rd": model.use_rd, "rp": model.use_rp,
                },
                "family_enabled": model.family_enabled,
                "channel_ablations": channel_ablations,
                "max_objects": args.max_objects,
                "point_channels": model.point_builder.channel_names,
                "ra_channels": (model.ra_derived.channel_names
                                if model.use_ra else []),
                "rd_channels": (model.rd_derived.channel_names
                                if model.use_rd else []),
                "rp_channels": (model.rp_derived.channel_names
                                if model.use_rp else []),
            }), path)
        print(f"saved {path}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

