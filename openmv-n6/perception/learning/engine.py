"""Evaluation and resumable optimization for thermal/radar students."""
import copy
import math
import os
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .losses import detection_loss

def move_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def evaluate_model(model, loader, image_width, image_height, device,
                   threshold=0.5):
    model.eval()
    u_errors, v_errors, count_errors = [], [], []
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch)
            probs = torch.sigmoid(out["object_logits"])
            boxes = out["boxes"]
            for b in range(len(boxes)):
                if int(batch["supervision_state"][b]) < 0:
                    continue
                pred_idx = torch.where(probs[b] >= threshold)[0]
                gt_idx = torch.where(batch["gt_presence"][b] > 0.5)[0]
                pred_frame, gt_frame = bool(len(pred_idx)), bool(len(gt_idx))
                tp += int(pred_frame and gt_frame)
                fp += int(pred_frame and not gt_frame)
                fn += int(not pred_frame and gt_frame)
                tn += int(not pred_frame and not gt_frame)
                count_errors.append(abs(len(pred_idx) - len(gt_idx)))
                if not pred_frame or not gt_frame:
                    continue
                cost = torch.cdist(
                    boxes[b, pred_idx, :2],
                    batch["gt_boxes"][b, gt_idx, :2], p=1)
                rows, cols = linear_sum_assignment(cost.cpu().numpy())
                for r, c in zip(rows, cols):
                    pbox = boxes[b, pred_idx[r]]
                    gbox = batch["gt_boxes"][b, gt_idx[c]]
                    u_errors.append(float(
                        torch.abs(pbox[0] - gbox[0]) * image_width))
                    v_errors.append(float(
                        torch.abs(pbox[1] - gbox[1]) * image_height))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    median = lambda x: float(np.median(x)) if x else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "false_positive_rate": fp / max(fp + tn, 1),
        "count_mae": float(np.mean(count_errors)) if count_errors else float("nan"),
        "median_u_px": median(u_errors),
        "median_v_px": median(v_errors),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def train_model(model, train_loader, val_loader, image_width, image_height,
                device, epochs=50, learning_rate=1e-3,
                weight_decay=1e-4, use_amp=True, tag="student",
                resume_path=None, negative_weight=1.0,
                hard_negative_weight=1.0):
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1))
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_state, best_score, best_metrics = None, float("inf"), None
    start_epoch = 0

    # Colab runtimes die without warning; a per-epoch resume file means an
    # interrupted run continues instead of restarting from zero. The file
    # lives next to the final checkpoint (on Drive in Colab) and is deleted
    # by the caller once the final .pt is safely written.
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        best_state = ckpt["best_state"]
        best_score = ckpt["best_score"]
        best_metrics = ckpt["best_metrics"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[{tag}] resuming from {resume_path} at epoch "
              f"{start_epoch + 1}/{epochs}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running, batches = 0.0, 0
        started = time.time()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(batch)
                losses = detection_loss(output, batch,
                                        negative_weight=negative_weight,
                                        hard_negative_weight=hard_negative_weight)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(losses["total"].item())
            batches += 1
        scheduler.step()

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            metrics = evaluate_model(
                model, val_loader, image_width, image_height, device)
            u = metrics["median_u_px"]
            v = metrics["median_v_px"]
            localization = (
                (u / image_width + v / image_height)
                if not math.isnan(u) and not math.isnan(v) else 2.0)
            score = (1.0 - metrics["f1"]) + localization
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = metrics
            print(
                f"[{tag}] {epoch + 1:03d}/{epochs} "
                f"loss={running / max(batches, 1):.4f} "
                f"F1={metrics['f1']:.3f} "
                f"u={metrics['median_u_px']:.1f}px "
                f"v={metrics['median_v_px']:.1f}px "
                f"{time.time() - started:.1f}s",
                flush=True,
            )
        else:
            print(
                f"[{tag}] {epoch + 1:03d}/{epochs} "
                f"loss={running / max(batches, 1):.4f} "
                f"{time.time() - started:.1f}s",
                flush=True,
            )

        if resume_path:
            tmp = resume_path + ".tmp"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_state": best_state,
                "best_score": best_score,
                "best_metrics": best_metrics,
            }, tmp)
            os.replace(tmp, resume_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_metrics is None:
        best_metrics = evaluate_model(
            model, val_loader, image_width, image_height, device)
    return model, best_metrics


