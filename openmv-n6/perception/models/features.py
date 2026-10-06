"""Derived sensor channels used by the thermal and radar encoders."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-6

class Derived2DChannels(nn.Module):
    BASE_NAMES = (
        "global_intensity", "frame_normalized", "dx", "dy",
        "gradient_magnitude", "gradient_cos", "gradient_sin",
        "d2x", "d2y", "laplacian", "local_contrast_small",
        "local_contrast_multiscale", "local_variance",
        "background_residual", "edge_strength", "temporal_diff",
        "temporal_rate", "previous_frame", "previous_frame_2",
    )

    RADIOMETRIC_NAMES = (
        "scene_delta_c", "warm_mask", "warm_persistence",
    )

    def __init__(self, global_mean: float, global_std: float,
                 add_persistence: bool = False,
                 add_radiometric: bool = False):
        super().__init__()
        self.global_mean = float(global_mean)
        self.global_std = max(float(global_std), 1e-6)
        self.add_persistence = bool(add_persistence)
        self.add_radiometric = bool(add_radiometric)
        self.channel_names = list(self.BASE_NAMES)
        if self.add_persistence:
            self.channel_names.append("persistence")
        if self.add_radiometric:
            self.channel_names.extend(self.RADIOMETRIC_NAMES)
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
        if self.add_radiometric:
            # Explicitly expose the physical distinction V6 had to infer from
            # normalized texture: absolute Celsius is already `global0`; these
            # channels add temperature above the current scene, a soft warm
            # mask, and whether that warm evidence persists across the three
            # frames. A mean is used here because it exports cleanly to ONNX;
            # the live safety gate uses the more robust median independently.
            bg0 = x0.mean(dim=(-2, -1), keepdim=True)
            delta0 = x0 - bg0
            warm0 = torch.sigmoid((delta0 - 2.0) / 0.5)

            bg1 = x1.mean(dim=(-2, -1), keepdim=True)
            bg2 = x2.mean(dim=(-2, -1), keepdim=True)
            warm1 = torch.sigmoid(((x1 - bg1) - 2.0) / 0.5) * valid1
            warm2 = torch.sigmoid(((x2 - bg2) - 2.0) / 0.5) * valid2
            persistence = (warm0 + 0.70 * warm1 + 0.49 * warm2) / (
                1.0 + 0.70 * valid1 + 0.49 * valid2).clamp(min=1.0)
            channels.extend([
                delta0 / self.global_std, warm0, persistence,
            ])

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


