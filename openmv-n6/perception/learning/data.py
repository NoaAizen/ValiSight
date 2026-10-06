"""Augmentation and PyTorch loaders for exported student splits."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from perception.student_data import (
    LABEL_POSITIVE, LoadedSplit, detection_target, estimate_scalar_stats,
)

def augment_thermal_seq(seq, rng, max_offset_c=10.0,
                        gain_range=(0.92, 1.08)):
    """Make the student invariant to AMBIENT temperature, not to people.

    The absolute-Celsius channels are normalised by one global mean/std taken
    from the training set, whose backgrounds all sit in 25-32 C. A 22 C dark
    room and 32.5 C sun-warmed ground both land outside everything the model
    ever saw, and it answers with its prior instead of with the picture -
    measured: full-frame boxes at conf 1.00 in both. A random offset and a
    mild contrast gain, drawn ONCE for the whole three-frame history because
    ambient shift is a property of the scene rather than of a frame, teach
    that the offset carries no information. A person stays warmer than what
    is behind them either way, and that difference is what should decide.
    """
    seq = np.asarray(seq, np.float32)
    centre = float(seq.mean())
    return ((seq - centre) * float(rng.uniform(*gain_range)) + centre
            + float(rng.uniform(-max_offset_c, max_offset_c)))


def augment_radar_points(points, n_points, rng, drop_p=0.15, min_keep=3,
                         pos_sigma_m=0.05, db_sigma=1.0):
    """Point-level jitter: the same person yields a different point set on
    every pass, so the student must not key on an exact constellation.

    Points are dropped and compacted rather than zeroed - n_radar is what
    marks a row valid downstream, and a zeroed row left inside the count is
    a phantom detection at the origin. Velocity is never jittered: doppler is
    the one channel that separates a person from furniture.
    """
    pts = np.array(points, np.float32, copy=True)
    n = int(n_points)
    if n > min_keep and rng.random() < 0.5:
        keep = rng.random(n) >= drop_p
        if int(keep.sum()) >= min_keep:
            kept = pts[:n][keep]
            pts[:] = 0.0
            pts[:len(kept)] = kept
            n = int(len(kept))
    if n:
        pts[:n, 0:3] += rng.normal(0.0, pos_sigma_m, (n, 3))
        pts[:n, 4] += rng.normal(0.0, db_sigma, n)
        pts[:n, 5] += rng.normal(0.0, db_sigma, n)
    return pts, n


class StudentDataset(Dataset):
    """Lazy three-frame histories over an in-memory exported split."""

    def __init__(self, split: LoadedSplit, plane: str, max_objects: int = 8,
                 supervised_only: bool = True, augment: bool = False):
        self.split = split
        self.plane = plane
        self.max_objects = int(max_objects)
        self.augment = bool(augment)
        self._worker_rng = None
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

    @property
    def rng(self):
        # Built on first use so each DataLoader worker draws its own stream;
        # a shared seed would hand every worker identical "random" offsets.
        if self._worker_rng is None:
            self._worker_rng = np.random.default_rng()
        return self._worker_rng

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
            hard = a.get("thermal_hard_negative")
            item["thermal_hard_negative"] = torch.tensor(
                bool(hard[i]) if hard is not None else False,
                dtype=torch.float32)

        if self.plane == "thermal":
            seq, valid = s.sequence("thermal", i)
            if self.augment:
                seq = augment_thermal_seq(seq, self.rng)
            item.update({
                "thermal_seq": self._tensor(seq, torch.float32),
                "valid_prev1": torch.tensor(valid[1], dtype=torch.float32),
                "valid_prev2": torch.tensor(valid[2], dtype=torch.float32),
                "dt1_ms": torch.tensor(
                    s.temporal.dt1_ms[i], dtype=torch.float32),
            })
            return item

        pts, n_pts = a["radar"][i], int(a["n_radar"][i])
        if self.augment:
            pts, n_pts = augment_radar_points(pts, n_pts, self.rng)
        item.update({
            "radar_points": self._tensor(pts, torch.float32),
            "n_radar": torch.tensor(n_pts, dtype=torch.long),
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


def build_loaders(train_split, val_split, plane, max_objects=8,
                  batch_size=32, num_workers=2, augment=True):
    # Augmentation is a training-set property: the val split has to stay the
    # same measurement run after run, or "it improved" means nothing.
    train_ds = StudentDataset(
        train_split, plane, max_objects, supervised_only=True,
        augment=augment)
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
