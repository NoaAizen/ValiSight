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


def initialise_from(model, path, tag):
    """Load pretrained weights into a fresh model, loudly.

    Reports what transferred and refuses a checkpoint that matched nothing.
    A silent no-op here looks exactly like pretraining that did not help, and
    would be diagnosed by rerunning the training rather than by reading one
    line of output.
    """
    import torch                      # lazy, like _imports() above
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload)
    own = model.state_dict()
    usable = {k: v for k, v in state.items()
              if k in own and own[k].shape == v.shape}
    if not usable:
        raise SystemExit(
            f"--init-from {path}: nothing in it fits this model (checkpoint "
            f"has {len(state)} tensors, none matching by name and shape). "
            f"Wrong student, or a checkpoint from a different architecture.")
    skipped = [k for k in own if k not in usable]
    model.load_state_dict(usable, strict=False)
    print(f"[{tag}] initialised {len(usable)}/{len(own)} tensors from "
          f"{os.path.basename(path)}"
          + (f" (from {payload.get('manifest_version')}, "
             f"{payload.get('thermal_mode', 'n/a')} scale)"
             if isinstance(payload, dict) else ""))
    if skipped:
        print(f"[{tag}] not in the checkpoint, staying random: "
              f"{', '.join(skipped[:6])}"
              + (f" (+{len(skipped) - 6} more)" if len(skipped) > 6 else ""))
    return model


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
    ap.add_argument("--no-augment", action="store_true",
                    help="train on the frames exactly as exported. The default "
                         "jitters ambient temperature and the radar point set, "
                         "which is what stops a 10 C change of scene from "
                         "collapsing the thermal student")
    ap.add_argument("--init-from", metavar="CKPT",
                    help="start from the weights in this checkpoint instead "
                         "of from scratch - the pretrain half of a "
                         "pretrain-then-fine-tune run. Only the learned "
                         "weights are taken; the thermal normalisation stats "
                         "are floats recomputed from whichever dataset is "
                         "being trained on, which is what lets a model "
                         "pretrained on 8-bit public thermal be fine-tuned on "
                         "our Celsius shards")
    ap.add_argument("--negative-weight", default="auto",
                    help="loss weight for verified-empty frames: 'auto' "
                         "balances them against positives (capped at 20), a "
                         "number sets it outright, 1 restores the unweighted "
                         "loss that learns to always answer 'person'")
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

    def check_val_is_measurable(plane):
        """A val split without BOTH classes cannot score the thing it claims.

        Measured the hard way: v3's first split held only empty sessions for
        the thermal plane, so precision and recall were 0/0, the false-positive
        rate read 0.999, and - worse - model selection ranks checkpoints by an
        F1 that was identically zero, which makes the saved "best" arbitrary.
        The mirror failure came first: a val where every frame holds a person
        gives F1 1.000 to a model that answers "person" always.
        """
        key = ("thermal_label_state" if plane == "thermal"
               else "radar_label_state")
        state = val.arrays[key]
        n_pos = int((state == 1).sum())
        n_neg = int((state == 0).sum())
        print(f"[{plane}] val: {n_pos} frames with a person, "
              f"{n_neg} verified empty")
        if n_pos == 0 or n_neg == 0:
            raise SystemExit(
                f"{plane}: the val split has "
                f"{'no frames with a person' if n_pos == 0 else 'no empty frames'}"
                f" - every metric it produces would be meaningless, and the "
                f"best-checkpoint choice with it. Fix manifest.json's split "
                f"so val holds both, then rerun.")

    def negative_weight_for(plane):
        """Balance verified-empty frames against positives, or take the flag."""
        if args.negative_weight != "auto":
            return float(args.negative_weight)
        state = train.arrays["thermal_label_state" if plane == "thermal"
                             else "radar_label_state"]
        n_pos = int((state == 1).sum())
        n_neg = int((state == 0).sum())
        if n_neg == 0:
            print(f"[{plane}] no verified-negative frames - negative weight 1.0 "
                  f"(the model cannot be taught what empty looks like; record "
                  f"an empty session and export it with --verified-negative)")
            return 1.0
        w = min(n_pos / n_neg, 20.0)
        print(f"[{plane}] {n_pos} positive vs {n_neg} verified-empty frames "
              f"-> negative weight {w:.1f}")
        return w
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
        if args.init_from:
            initialise_from(model, args.init_from, "THERMAL")
        disabled = [x.strip() for x in args.disable_thermal.split(",")
                    if x.strip()]
        model.derived.disable_channels(disabled)
        check_val_is_measurable("thermal")
        train_loader, val_loader = build_loaders(
            train, val, "thermal", args.max_objects,
            args.batch_size, args.workers, augment=not args.no_augment)
        resume = os.path.join(out_dir, "thermal_resume.pt")
        model, metrics = train_model(
            model, train_loader, val_loader, 160, 120, device,
            epochs=args.epochs, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, use_amp=not args.no_amp,
            tag="THERMAL", resume_path=resume,
            negative_weight=negative_weight_for("thermal"))
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
        if os.path.exists(resume):
            os.remove(resume)
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

        check_val_is_measurable("radar")
        train_loader, val_loader = build_loaders(
            train, val, "radar", args.max_objects,
            args.batch_size, args.workers, augment=not args.no_augment)
        resume = os.path.join(out_dir, "radar_resume.pt")
        model, metrics = train_model(
            model, train_loader, val_loader, 640, 400, device,
            epochs=args.epochs, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, use_amp=not args.no_amp,
            tag="RADAR", resume_path=resume,
            negative_weight=negative_weight_for("radar"))
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
        if os.path.exists(resume):
            os.remove(resume)
        print(f"saved {path}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

