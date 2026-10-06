"""Thermal and radar detection architectures, independent of training loops."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from .features import Derived2DChannels, DerivedRangeProfileChannels, PointFeatureBuilder

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
    def __init__(self, thermal_mean, thermal_std, max_objects=8,
                 radiometric_channels=True):
        super().__init__()
        self.derived = Derived2DChannels(
            thermal_mean, thermal_std,
            add_radiometric=bool(radiometric_channels))
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


