"""PyTorch thermal/radar students distilled from person detections.

The models consume the current export schema through student_data.py.  They
never consume the rendered fused image: thermal uses raw registered-source
counts/temperature, radar uses project-frame points and optional dense TLVs.
"""
from __future__ import annotations

import copy
import math
import os
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

from perception.student_data import (
    LABEL_POSITIVE, LoadedSplit, detection_target, estimate_scalar_stats,
)

EPS = 1e-6


class StudentDataset(Dataset):
    """Lazy three-frame histories over an in-memory exported split."""

    def __init__(self, split: LoadedSplit, plane: str, max_objects: int = 8,
                 supervised_only: bool = True):
        self.split = split
        self.plane = plane
        self.max_objects = int(max_objects)
        state_key = ("thermal_label_state" if plane == "thermal"
                     else "radar_label_state")
        state = split.arrays[state_key]
        keep = np.ones(len(split), dtype=bool)
        if supervised_only:
            keep &= state >= 0

        if plane == "radar":
            available = split.arrays["n_radar"] > 0
            for key, flag in (
                ("range_angle", "ra_valid"),
                ("range_doppler", "rd_valid"),
                ("range_profile", "rp_valid"),
            ):
                if key in split.arrays:
                    available |= split.arrays[flag]
            # Silence is a useful verified negative, but a positive with no
            # radar measurement is missing data rather than a hard example.
            keep &= (state != LABEL_POSITIVE) | available

        self.indices = np.flatnonzero(keep)
        if not len(self.indices):
            raise ValueError(f"{split.name}/{plane}: no usable supervised frames")

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _tensor(x, dtype=None):
        t = torch.from_numpy(np.ascontiguousarray(x)) if isinstance(x, np.ndarray) \
            else torch.tensor(x)
        return t.to(dtype=dtype) if dtype is not None else t

    def __getitem__(self, j):
        i = int(self.indices[j])
        s = self.split
        a = s.arrays
        target = detection_target(s, i, self.plane, self.max_objects)
        item = {
            "sample_index": torch.tensor(i, dtype=torch.long),
            "supervision_state": torch.tensor(
                target["supervision_state"], dtype=torch.long),
            "gt_presence": self._tensor(target["gt_presence"], torch.float32),
            "gt_boxes": self._tensor(target["gt_boxes"], torch.float32),
            "gt_confidence": self._tensor(
                target["gt_confidence"], torch.float32),
        }

        if self.plane == "thermal":
            seq, valid = s.sequence("thermal", i)
            item.update({
                "thermal_seq": self._tensor(seq, torch.float32),
                "valid_prev1": torch.tensor(valid[1], dtype=torch.float32),
                "valid_prev2": torch.tensor(valid[2], dtype=torch.float32),
                "dt1_ms": torch.tensor(
                    s.temporal.dt1_ms[i], dtype=torch.float32),
            })
            return item

        item.update({
            "radar_points": self._tensor(a["radar"][i], torch.float32),
            "n_radar": torch.tensor(int(a["n_radar"][i]), dtype=torch.long),
        })
        for key, prefix, valid_key in (
            ("range_angle", "ra", "ra_valid"),
            ("range_doppler", "rd", "rd_valid"),
            ("range_profile", "rp", "rp_valid"),
        ):
            if key not in a:
                continue
            seq, valid = s.sequence(key, i, valid_key)
            item[f"{prefix}_seq"] = self._tensor(seq, torch.float32)
            item[f"{prefix}_valid"] = torch.tensor(
                valid[0], dtype=torch.float32)
            item[f"{prefix}_valid_prev1"] = torch.tensor(
                valid[1], dtype=torch.float32)
            item[f"{prefix}_valid_prev2"] = torch.tensor(
                valid[2], dtype=torch.float32)
        item["dt1_ms"] = torch.tensor(
            s.temporal.dt1_ms[i], dtype=torch.float32)
        return item


class Derived2DChannels(nn.Module):
    BASE_NAMES = (
        "global_intensity", "frame_normalized", "dx", "dy",
        "gradient_magnitude", "gradient_cos", "gradient_sin",
        "d2x", "d2y", "laplacian", "local_contrast_small",
        "local_contrast_multiscale", "local_variance",
        "background_residual", "edge_strength", "temporal_diff",
        "temporal_rate", "previous_frame", "previous_frame_2",
    )

    def __init__(self, global_mean: float, global_std: float,
                 add_persistence: bool = False):
        super().__init__()
        self.global_mean = float(global_mean)
        self.global_std = max(float(global_std), 1e-6)
        self.add_persistence = bool(add_persistence)
        self.channel_names = list(self.BASE_NAMES)
        if self.add_persistence:
            self.channel_names.append("persistence")
        self.register_buffer(
            "channel_mask", torch.ones(len(self.channel_names), dtype=torch.float32)
        )
        kernels = {
            "kx": [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]],
            "ky": [[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]],
            "kxx": [[0, 0, 0], [1, -2, 1], [0, 0, 0]],
            "kyy": [[0, 1, 0], [0, -2, 0], [0, 1, 0]],
        }
        for name, values in kernels.items():
            self.register_buffer(
                name, torch.tensor(values, dtype=torch.float32)[None, None]
            )

    def reset_mask(self):
        self.channel_mask.fill_(1.0)

    def disable_channels(self, names: Iterable[str]):
        for name in names:
            if name not in self.channel_names:
                raise ValueError(f"unknown 2-D channel: {name}")
            self.channel_mask[self.channel_names.index(name)] = 0.0

    def disable_all(self):
        self.channel_mask.zero_()

    @staticmethod
    def _frame_normalize(x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
        return (x - mean) / std

    def forward(self, seq, valid_prev1, valid_prev2, dt1_ms):
        x0, x1, x2 = (seq[:, i:i + 1].float() for i in range(3))
        b = x0.shape[0]
        valid1 = valid_prev1.view(b, 1, 1, 1)
        valid2 = valid_prev2.view(b, 1, 1, 1)
        global0 = (x0 - self.global_mean) / self.global_std
        global1 = (x1 - self.global_mean) / self.global_std
        global2 = (x2 - self.global_mean) / self.global_std
        norm = self._frame_normalize(x0)

        gx = F.conv2d(norm, self.kx, padding=1)
        gy = F.conv2d(norm, self.ky, padding=1)
        grad_mag = torch.sqrt(gx.square() + gy.square() + EPS)
        d2x = F.conv2d(norm, self.kxx, padding=1)
        d2y = F.conv2d(norm, self.kyy, padding=1)
        mean3 = F.avg_pool2d(norm, 3, stride=1, padding=1)
        mean5 = F.avg_pool2d(norm, 5, stride=1, padding=2)
        mean9 = F.avg_pool2d(norm, 9, stride=1, padding=4)
        mean21 = F.avg_pool2d(norm, 21, stride=1, padding=10)
        mean_sq = F.avg_pool2d(norm.square(), 5, stride=1, padding=2)
        local_variance = (mean_sq - mean5.square()).clamp(min=0.0)
        edge_scale = (
            grad_mag.mean(dim=(-2, -1), keepdim=True)
            + grad_mag.std(dim=(-2, -1), keepdim=True)
        ).clamp(min=1e-4)
        temporal_diff = (global0 - global1) * valid1
        dt_seconds = (dt1_ms.view(b, 1, 1, 1) / 1000.0).clamp(min=1e-3)

        channels = [
            global0, norm, gx, gy, grad_mag,
            gx / grad_mag.clamp(min=1e-6),
            gy / grad_mag.clamp(min=1e-6),
            d2x, d2y, d2x + d2y,
            norm - mean3, mean3 - mean9, local_variance,
            norm - mean21, (grad_mag / edge_scale).clamp(0.0, 4.0) / 4.0,
            temporal_diff, temporal_diff / dt_seconds * valid1,
            global1 * valid1, global2 * valid2,
        ]
        if self.add_persistence:
            weight = 1.0 + 0.70 * valid1 + 0.49 * valid2
            persistence = (
                F.relu(global0)
                + 0.70 * F.relu(global1) * valid1
                + 0.49 * F.relu(global2) * valid2
            ) / weight.clamp(min=1.0)
            channels.append(persistence)

        x = torch.cat(channels, dim=1)
        return x * self.channel_mask.view(1, -1, 1, 1)


class DerivedRangeProfileChannels(nn.Module):
    CHANNEL_NAMES = (
        "global_intensity", "profile_normalized", "d_dr", "d2_dr2",
        "background_residual", "temporal_diff", "temporal_rate",
        "previous_profile", "previous_profile_2",
    )

    def __init__(self, global_mean, global_std):
        super().__init__()
        self.global_mean = float(global_mean)
        self.global_std = max(float(global_std), 1e-6)
        self.channel_names = list(self.CHANNEL_NAMES)
        self.register_buffer(
            "channel_mask", torch.ones(len(self.channel_names), dtype=torch.float32)
        )
        self.register_buffer(
            "kd1", torch.tensor([-0.5, 0, 0.5], dtype=torch.float32)[None, None])
        self.register_buffer(
            "kd2", torch.tensor([1, -2, 1], dtype=torch.float32)[None, None])

    def disable_channels(self, names):
        for name in names:
            if name not in self.channel_names:
                raise ValueError(f"unknown range-profile channel: {name}")
            self.channel_mask[self.channel_names.index(name)] = 0.0

    def disable_all(self):
        self.channel_mask.zero_()

    def forward(self, seq, valid_prev1, valid_prev2, dt1_ms):
        x0, x1, x2 = (seq[:, i:i + 1].float() for i in range(3))
        b = x0.shape[0]
        v1 = valid_prev1.view(b, 1, 1)
        v2 = valid_prev2.view(b, 1, 1)
        g0 = (x0 - self.global_mean) / self.global_std
        g1 = (x1 - self.global_mean) / self.global_std
        g2 = (x2 - self.global_mean) / self.global_std
        norm = (x0 - x0.mean(dim=-1, keepdim=True)) / \
            x0.std(dim=-1, keepdim=True).clamp(min=1e-4)
        d1 = F.conv1d(norm, self.kd1, padding=1)
        d2 = F.conv1d(norm, self.kd2, padding=1)
        bg = F.avg_pool1d(norm, 9, stride=1, padding=4)
        diff = (g0 - g1) * v1
        dt = (dt1_ms.view(b, 1, 1) / 1000.0).clamp(min=1e-3)
        x = torch.cat([
            g0, norm, d1, d2, norm - bg, diff, diff / dt * v1,
            g1 * v1, g2 * v2,
        ], dim=1)
        return x * self.channel_mask.view(1, -1, 1)


class PointFeatureBuilder(nn.Module):
    CHANNEL_NAMES = (
        "x", "y", "z", "velocity", "snr", "noise", "range_3d",
        "range_xy", "azimuth_sin", "azimuth_cos", "elevation_sin",
        "elevation_cos", "snr_minus_noise", "knn_mean_distance",
        "knn_min_distance", "local_density", "local_delta_velocity",
        "local_delta_snr", "local_delta_noise",
    )

    def __init__(self, knn=4):
        super().__init__()
        self.knn = int(knn)
        self.channel_names = list(self.CHANNEL_NAMES)
        self.register_buffer(
            "channel_mask", torch.ones(len(self.channel_names), dtype=torch.float32)
        )

    def disable_channels(self, names):
        for name in names:
            if name not in self.channel_names:
                raise ValueError(f"unknown point channel: {name}")
            self.channel_mask[self.channel_names.index(name)] = 0.0

    def disable_all(self):
        self.channel_mask.zero_()

    def forward(self, points, n_points):
        points = torch.nan_to_num(points.float())
        b, k, d = points.shape
        if d < 6:
            raise ValueError("radar points need [x,y,z,v,snr,noise]")
        idx = torch.arange(k, device=points.device)[None]
        valid = idx < n_points[:, None]
        x, y, z, velocity, snr, noise = (
            points[..., i] for i in range(6)
        )
        range_xy = torch.sqrt(x.square() + y.square() + EPS)
        range_3d = torch.sqrt(range_xy.square() + z.square() + EPS)
        azimuth = torch.atan2(y, x)
        elevation = torch.atan2(z, range_xy.clamp(min=1e-6))

        dist = torch.cdist(points[..., :3], points[..., :3])
        pair_valid = valid[:, :, None] & valid[:, None, :]
        pair_valid &= ~torch.eye(k, dtype=torch.bool, device=points.device)[None]
        dist = torch.where(pair_valid, dist, torch.full_like(dist, float("inf")))
        k_use = min(self.knn, max(k - 1, 1))
        nn_dist, nn_idx = torch.topk(
            dist, k=k_use, dim=-1, largest=False)
        nn_valid = torch.isfinite(nn_dist)
        count = nn_valid.sum(dim=-1).clamp(min=1)
        safe = torch.where(nn_valid, nn_dist, torch.zeros_like(nn_dist))
        mean_dist = safe.sum(dim=-1) / count
        # inf is representable in float16; a 1e9 literal is not (AMP overflow)
        min_dist = torch.where(
            nn_valid, nn_dist, torch.full_like(nn_dist, float("inf"))
        ).min(dim=-1).values
        min_dist = torch.where(
            torch.isinf(min_dist), torch.zeros_like(min_dist), min_dist)

        def neighbour_delta(value):
            expanded = value[:, None, :].expand(b, k, k)
            neighbours = torch.gather(expanded, 2, nn_idx)
            return (
                torch.abs(value[:, :, None] - neighbours) * nn_valid
            ).sum(dim=-1) / count

        features = torch.stack([
            x, y, z, velocity, snr, noise, range_3d, range_xy,
            torch.sin(azimuth), torch.cos(azimuth),
            torch.sin(elevation), torch.cos(elevation), snr - noise,
            mean_dist, min_dist, 1.0 / (mean_dist + 1e-3),
            neighbour_delta(velocity), neighbour_delta(snr),
            neighbour_delta(noise),
        ], dim=-1)
        features *= valid.unsqueeze(-1)
        features *= self.channel_mask.view(1, 1, -1)
        return features, valid


class Spatial2DEncoder(nn.Module):
    def __init__(self, in_channels, feat=160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96), nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d((6, 8)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 6 * 8, 320), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(320, feat),
            nn.LayerNorm(feat), nn.GELU(),
        )

    def forward(self, x):
        return self.fc(self.net(x))


class RangeProfileEncoder(nn.Module):
    def __init__(self, in_channels, feat=112):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 96, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(96 * 16, feat),
            nn.LayerNorm(feat), nn.GELU(),
        )

    def forward(self, x):
        return self.fc(self.net(x))


class PointSetEncoder(nn.Module):
    def __init__(self, in_features, feat=160):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_features)
        self.point_mlp = nn.Sequential(
            nn.Linear(in_features, 96), nn.GELU(),
            nn.Linear(96, 160), nn.GELU(),
            nn.Linear(160, feat), nn.GELU(),
        )
        self.scene = nn.Sequential(
            nn.Linear(feat * 2, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, feat), nn.GELU(),
        )

    def forward(self, features, valid):
        f = self.point_mlp(self.input_norm(features))
        # fill must fit the runtime dtype: under AMP f is float16, where a
        # literal -1e9 overflows Half and forward() crashes on GPU
        fill = torch.finfo(f.dtype).min
        fmax = f.masked_fill(~valid.unsqueeze(-1), fill).max(dim=1).values
        any_valid = valid.any(dim=1)
        fmax = torch.where(any_valid[:, None], fmax, torch.zeros_like(fmax))
        count = valid.sum(dim=1, keepdim=True).clamp(min=1)
        fmean = (f * valid.unsqueeze(-1)).sum(dim=1) / count
        return self.scene(torch.cat([fmax, fmean], dim=-1))


class DetectionHead(nn.Module):
    """Person-only unordered set prediction; matching is Hungarian."""

    def __init__(self, feat, max_objects=8):
        super().__init__()
        self.max_objects = int(max_objects)
        self.shared = nn.Sequential(
            nn.Linear(feat, 256), nn.GELU(), nn.Dropout(0.10))
        self.box_head = nn.Linear(256, self.max_objects * 5)

    def forward(self, latent):
        out = self.box_head(self.shared(latent)).view(
            -1, self.max_objects, 5)
        return {
            "object_logits": out[..., 0],
            "boxes": torch.sigmoid(out[..., 1:5]),
        }


class ThermalStudent(nn.Module):
    def __init__(self, thermal_mean, thermal_std, max_objects=8):
        super().__init__()
        self.derived = Derived2DChannels(thermal_mean, thermal_std)
        self.encoder = Spatial2DEncoder(
            len(self.derived.channel_names), feat=256)
        self.head = DetectionHead(256, max_objects)

    def forward(self, batch):
        x = self.derived(
            batch["thermal_seq"], batch["valid_prev1"],
            batch["valid_prev2"], batch["dt1_ms"])
        latent = self.encoder(x)
        result = self.head(latent)
        result["thermal_latent"] = latent
        return result


class RadarStudent(nn.Module):
    def __init__(self, stream_stats: Mapping[str, Sequence[float]],
                 max_objects=8, use_ra=False, use_rd=False, use_rp=False):
        super().__init__()
        self.point_builder = PointFeatureBuilder()
        self.point_encoder = PointSetEncoder(
            len(self.point_builder.channel_names), 160)
        self.use_ra, self.use_rd, self.use_rp = use_ra, use_rd, use_rp
        self.family_enabled = {"points": True, "ra": use_ra,
                               "rd": use_rd, "rp": use_rp}
        total = 160
        for prefix, enabled in (("ra", use_ra), ("rd", use_rd)):
            if enabled:
                mean, std = stream_stats[prefix]
                derived = Derived2DChannels(mean, std, add_persistence=True)
                encoder = Spatial2DEncoder(len(derived.channel_names), 160)
                setattr(self, f"{prefix}_derived", derived)
                setattr(self, f"{prefix}_encoder", encoder)
                total += 160
        if use_rp:
            mean, std = stream_stats["rp"]
            self.rp_derived = DerivedRangeProfileChannels(mean, std)
            self.rp_encoder = RangeProfileEncoder(
                len(self.rp_derived.channel_names), 112)
            total += 112
        self.fusion = nn.Sequential(
            nn.Linear(total, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(512, 320),
            nn.LayerNorm(320), nn.GELU(),
        )
        self.head = DetectionHead(320, max_objects)

    def disable_family(self, family):
        if family not in self.family_enabled:
            raise ValueError(f"unknown radar family: {family}")
        self.family_enabled[family] = False

    def disable_channels(self, family, channels):
        if family == "points":
            self.point_builder.disable_channels(channels)
        else:
            getattr(self, f"{family}_derived").disable_channels(channels)

    def forward(self, batch):
        point_features, valid = self.point_builder(
            batch["radar_points"], batch["n_radar"])
        point_latent = self.point_encoder(point_features, valid)
        if not self.family_enabled["points"]:
            point_latent = torch.zeros_like(point_latent)
        latents = [point_latent]
        result_latents = {"point_latent": point_latent}

        for prefix in ("ra", "rd"):
            if not getattr(self, f"use_{prefix}"):
                continue
            derived = getattr(self, f"{prefix}_derived")
            encoder = getattr(self, f"{prefix}_encoder")
            channels = derived(
                batch[f"{prefix}_seq"], batch[f"{prefix}_valid_prev1"],
                batch[f"{prefix}_valid_prev2"], batch["dt1_ms"])
            latent = encoder(channels) * batch[f"{prefix}_valid"].view(-1, 1)
            if not self.family_enabled[prefix]:
                latent = torch.zeros_like(latent)
            latents.append(latent)
            result_latents[f"{prefix}_latent"] = latent

        if self.use_rp:
            channels = self.rp_derived(
                batch["rp_seq"], batch["rp_valid_prev1"],
                batch["rp_valid_prev2"], batch["dt1_ms"])
            latent = self.rp_encoder(channels) * batch["rp_valid"].view(-1, 1)
            if not self.family_enabled["rp"]:
                latent = torch.zeros_like(latent)
            latents.append(latent)
            result_latents["rp_latent"] = latent

        radar_latent = self.fusion(torch.cat(latents, dim=1))
        result = self.head(radar_latent)
        result["radar_latent"] = radar_latent
        result.update(result_latents)
        return result


def detection_loss(output, batch, lambda_center=3.0, lambda_box=1.0,
                   lambda_object=1.0, negative_weight=1.0):
    pred_obj, pred_boxes = output["object_logits"], output["boxes"]
    presence = batch["gt_presence"]
    gt_boxes = batch["gt_boxes"]
    gt_conf = batch["gt_confidence"]
    states = batch["supervision_state"]
    bsz, queries = pred_obj.shape

    object_targets = torch.zeros_like(pred_obj)
    object_weights = (states >= 0).float().view(-1, 1).expand_as(pred_obj).clone()
    # Verified-empty frames are rare (v3: 17 positive frames per negative), and
    # an unweighted loss is minimised by calling "person" always - measured on
    # v3's first radar run: recall .99 but a 88% false-positive rate on the
    # held-out empty session. Scale the only frames that carry "there is nobody
    # here" so the two classes contribute comparably.
    if negative_weight != 1.0:
        neg = (states == LABEL_NEGATIVE).float().view(-1, 1)
        object_weights = object_weights * (1.0 + (negative_weight - 1.0) * neg)
    total_center = pred_obj.new_tensor(0.0)
    total_box = pred_obj.new_tensor(0.0)
    box_weight_sum = pred_obj.new_tensor(0.0)

    for b in range(bsz):
        if int(states[b]) != LABEL_POSITIVE:
            continue
        gt_idx = torch.where(presence[b] > 0.5)[0]
        if not len(gt_idx):
            continue
        gt = gt_boxes[b, gt_idx]
        cost = torch.cdist(pred_boxes[b], gt, p=1)
        cost -= 0.25 * torch.sigmoid(pred_obj[b])[:, None]
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        p_idx = torch.as_tensor(rows, device=pred_obj.device, dtype=torch.long)
        g_idx = gt_idx[torch.as_tensor(
            cols, device=pred_obj.device, dtype=torch.long)]
        conf = gt_conf[b, g_idx].clamp(0.25, 1.0)
        object_targets[b, p_idx] = 1.0
        object_weights[b, p_idx] = 4.0 * conf
        pbox, gbox = pred_boxes[b, p_idx], gt_boxes[b, g_idx]
        total_center += (torch.abs(pbox[:, :2] - gbox[:, :2]).mean(1)
                         * conf).sum()
        total_box += (F.smooth_l1_loss(
            pbox, gbox, reduction="none").mean(1) * conf).sum()
        box_weight_sum += conf.sum()

    object_raw = F.binary_cross_entropy_with_logits(
        pred_obj, object_targets, reduction="none")
    object_loss = (object_raw * object_weights).sum() \
        / object_weights.sum().clamp(min=1.0)
    center_loss = total_center / box_weight_sum.clamp(min=1.0)
    box_loss = total_box / box_weight_sum.clamp(min=1.0)
    total = (lambda_object * object_loss
             + lambda_center * center_loss + lambda_box * box_loss)
    return {"total": total, "object": object_loss,
            "center": center_loss, "box": box_loss}


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
                resume_path=None, negative_weight=1.0):
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
                                        negative_weight=negative_weight)
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


def build_loaders(train_split, val_split, plane, max_objects=8,
                  batch_size=32, num_workers=2):
    train_ds = StudentDataset(
        train_split, plane, max_objects, supervised_only=True)
    val_ds = StudentDataset(
        val_split, plane, max_objects, supervised_only=True)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=True, drop_last=False, **kwargs),
        DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs),
    )


def radar_stream_stats(train_split: LoadedSplit):
    stats = {}
    for key, prefix in (
        ("range_angle", "ra"),
        ("range_doppler", "rd"),
        ("range_profile", "rp"),
    ):
        if key in train_split.arrays:
            valid = train_split.arrays[
                {"ra": "ra_valid", "rd": "rd_valid", "rp": "rp_valid"}[prefix]]
            source = train_split.arrays[key][valid]
            if not len(source):
                continue
            stats[prefix] = estimate_scalar_stats(source)
    return stats

