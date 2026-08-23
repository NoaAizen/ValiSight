"""Load exported NPZ shards for thermal/radar student training.

This is the executable replacement for the draft notebook's undefined
load_split().  It keeps session boundaries, tri-state supervision and per-session
thermal scale intact.  Derived channels remain a model-side operation; shards
contain only measurements that cannot be reconstructed.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

LABEL_UNKNOWN = -1
LABEL_NEGATIVE = 0
LABEL_POSITIVE = 1

REQUIRED_KEYS = (
    "thermal", "dt_ms", "radar", "n_radar",
    "th_boxes", "n_th_boxes", "rgb_boxes", "n_rgb_boxes",
    "thermal_label_state", "radar_label_state",
)
OPTIONAL_STREAMS = {
    "range_angle": "ra_valid",
    "range_doppler": "rd_valid",
    "range_profile": "rp_valid",
}
PROVENANCE_FIELDS = (
    "lepton_gain", "warp_lut_sha256", "radar_calib_sha256",
    "radar_cfg_sha256", "detector_model", "detector_engine_sha256",
)


@dataclass
class TemporalIndex:
    prev1: np.ndarray
    prev2: np.ndarray
    valid1: np.ndarray
    valid2: np.ndarray
    dt1_ms: np.ndarray


@dataclass
class LoadedSplit:
    name: str
    arrays: Dict[str, np.ndarray]
    session_names: Tuple[str, ...]
    thermal_mode: str
    temporal: TemporalIndex

    def __len__(self) -> int:
        return len(self.arrays["thermal"])

    def sequence(self, key: str, i: int,
                 valid_key: Optional[str] = None):
        """Return current/previous-two values and modality-specific validity."""
        current = np.asarray(self.arrays[key][i], dtype=np.float32)
        frames = [current]
        valid = [True]
        current_ok = True
        if valid_key is not None:
            current_ok = bool(self.arrays[valid_key][i])
            valid[0] = current_ok

        for which, common_valid in (
            (self.temporal.prev1, self.temporal.valid1),
            (self.temporal.prev2, self.temporal.valid2),
        ):
            p = int(which[i])
            ok = p >= 0 and bool(common_valid[i])
            if valid_key is not None:
                ok = ok and current_ok and bool(self.arrays[valid_key][p])
            frames.append(np.asarray(self.arrays[key][p], dtype=np.float32)
                          if ok else np.zeros_like(current))
            valid.append(ok)

        return (np.stack(frames, axis=0),
                np.asarray(valid, dtype=np.float32))


def load_manifest(data_dir: str) -> dict:
    path = os.path.join(data_dir, "manifest.json")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if "split" not in manifest or "sessions" not in manifest:
        raise ValueError(f"{path}: missing split/sessions")
    return manifest


def validate_provenance(manifest: Mapping, sessions: Iterable[str]) -> None:
    """Reject known-incompatible sources; unknown provenance remains a warning."""
    sessions = tuple(sessions)
    for field in PROVENANCE_FIELDS:
        values = {
            manifest["sessions"][s].get("provenance", {}).get(field)
            for s in sessions
        } - {None}
        if len(values) > 1:
            raise ValueError(
                f"incompatible session provenance for {field}: {sorted(values)}"
            )


def _thermal_mode(info: Mapping) -> str:
    return "celsius" if info.get("c_per_lsb") is not None else "unit"


def _scale_thermal(raw: np.ndarray, info: Mapping) -> np.ndarray:
    raw = raw.astype(np.float32)
    c_per_lsb = info.get("c_per_lsb")
    if c_per_lsb is not None:
        if info.get("tmin") is None:
            raise ValueError("c_per_lsb is present but tmin is missing")
        return raw * float(c_per_lsb) + float(info["tmin"])
    counts_max = info.get("thermal_counts_max")
    if counts_max is None or float(counts_max) <= 0:
        raise ValueError("uncalibrated thermal session needs thermal_counts_max")
    return raw / float(counts_max)


def _load_session(data_dir: str, manifest: Mapping, session: str):
    info = manifest["sessions"].get(session)
    if info is None:
        raise ValueError(f"session {session!r} is absent from manifest")
    paths = [os.path.join(data_dir, p) for p in info.get("shards", [])]
    if not paths:
        paths = sorted(glob.glob(os.path.join(data_dir, f"{session}-*.npz")))
    if not paths:
        raise ValueError(f"session {session!r} has no shards")

    chunks = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in REQUIRED_KEYS if k not in z]
            if missing:
                raise ValueError(
                    f"{path}: missing {missing}; re-export with the current v2 exporter"
                )
            chunks.append({k: np.array(z[k], copy=True) for k in z.files})

    keys = set.intersection(*(set(c) for c in chunks))
    arrays = {k: np.concatenate([c[k] for c in chunks], axis=0)
              for k in keys}
    n = len(arrays["thermal"])
    for key in REQUIRED_KEYS:
        if len(arrays[key]) != n:
            raise ValueError(f"{session}: {key} length does not match thermal")
    arrays["thermal"] = _scale_thermal(arrays["thermal"], info)
    return arrays, _thermal_mode(info)


def _optional_shapes(session_arrays: Sequence[Mapping[str, np.ndarray]]):
    shapes = {}
    for key in OPTIONAL_STREAMS:
        found = {tuple(a[key].shape[1:]) for a in session_arrays if key in a}
        if len(found) > 1:
            raise ValueError(f"incompatible {key} shapes: {sorted(found)}")
        if found:
            shapes[key] = found.pop()
    return shapes


def build_temporal_indices(session_index: np.ndarray,
                           dt_ms: np.ndarray,
                           max_dt_ms: float = 300.0) -> TemporalIndex:
    n = len(dt_ms)
    prev1 = np.full(n, -1, np.int64)
    prev2 = np.full(n, -1, np.int64)
    valid1 = np.zeros(n, np.float32)
    valid2 = np.zeros(n, np.float32)
    clean_dt = np.zeros(n, np.float32)

    for i in range(1, n):
        dt = float(dt_ms[i])
        if (session_index[i] == session_index[i - 1]
                and 0.0 < dt <= max_dt_ms):
            prev1[i] = i - 1
            valid1[i] = 1.0
            clean_dt[i] = dt
            if i >= 2 and valid1[i - 1] > 0:
                prev2[i] = i - 2
                valid2[i] = 1.0
    return TemporalIndex(prev1, prev2, valid1, valid2, clean_dt)


def load_split(data_dir: str, which: str,
               manifest: Optional[Mapping] = None,
               max_dt_ms: float = 300.0) -> LoadedSplit:
    manifest = dict(manifest or load_manifest(data_dir))
    sessions = tuple(manifest["split"].get(which, ()))
    if not sessions:
        raise ValueError(f"manifest split {which!r} is empty")
    validate_provenance(manifest, sessions)

    loaded = [_load_session(data_dir, manifest, s) for s in sessions]
    session_arrays = [x[0] for x in loaded]
    modes = {x[1] for x in loaded}
    if len(modes) != 1:
        raise ValueError(
            f"{which} mixes calibrated Celsius and unit thermal sessions: {modes}"
        )
    mode = modes.pop()
    shapes = _optional_shapes(session_arrays)

    base_keys = set(REQUIRED_KEYS)
    for key in ("i", "t_mono", "teacher_available", "radar_age_ms"):
        if all(key in a for a in session_arrays):
            base_keys.add(key)

    combined = {
        key: np.concatenate([a[key] for a in session_arrays], axis=0)
        for key in base_keys
    }
    session_index = np.concatenate([
        np.full(len(a["thermal"]), j, np.int32)
        for j, a in enumerate(session_arrays)
    ])
    combined["session_index"] = session_index

    for key, valid_key in OPTIONAL_STREAMS.items():
        if key not in shapes:
            continue
        values, validity = [], []
        shape = shapes[key]
        for a in session_arrays:
            n = len(a["thermal"])
            if key in a:
                values.append(a[key].astype(np.float32, copy=False))
                validity.append(
                    a.get(valid_key, np.ones(n, np.bool_)).astype(np.bool_)
                )
            else:
                values.append(np.zeros((n,) + shape, np.float32))
                validity.append(np.zeros(n, np.bool_))
        combined[key] = np.concatenate(values, axis=0)
        combined[valid_key] = np.concatenate(validity, axis=0)

    temporal = build_temporal_indices(
        session_index, combined["dt_ms"], max_dt_ms=max_dt_ms
    )
    return LoadedSplit(which, combined, sessions, mode, temporal)


def load_train_val(data_dir: str, max_dt_ms: float = 300.0):
    manifest = load_manifest(data_dir)
    validate_provenance(
        manifest, manifest["split"]["train"] + manifest["split"]["val"])
    train = load_split(data_dir, "train", manifest, max_dt_ms)
    val = load_split(data_dir, "val", manifest, max_dt_ms)

    # A dense radar TLV may exist in every training session but in none of the
    # held-out sessions. Keep the model input contract stable and make the
    # absence explicit instead of failing later in RadarStudent.forward().
    for key, valid_key in OPTIONAL_STREAMS.items():
        if key not in train.arrays:
            continue
        shape = train.arrays[key].shape[1:]
        if key in val.arrays:
            if val.arrays[key].shape[1:] != shape:
                raise ValueError(
                    f"train/val {key} shapes differ: {shape} vs "
                    f"{val.arrays[key].shape[1:]}"
                )
            continue
        val.arrays[key] = np.zeros((len(val),) + shape, np.float32)
        val.arrays[valid_key] = np.zeros(len(val), np.bool_)

    if train.thermal_mode != val.thermal_mode:
        raise ValueError(
            "train/val thermal modes differ: "
            f"{train.thermal_mode} vs {val.thermal_mode}"
        )
    return manifest, train, val


def estimate_scalar_stats(array: np.ndarray, max_frames: int = 512):
    x = np.asarray(array)
    if len(x) > max_frames:
        idx = np.linspace(0, len(x) - 1, max_frames).astype(np.int64)
        x = x[idx]
    x = x.astype(np.float32)
    mean, std = float(x.mean()), float(x.std())
    return mean, max(std, 1e-6)


def detection_target(split: LoadedSplit, i: int, plane: str,
                     max_objects: int):
    """Build person-only set targets in the plane the student actually sees."""
    if plane == "thermal":
        state_key, box_key, count_key = (
            "thermal_label_state", "th_boxes", "n_th_boxes"
        )
        width, height = 160.0, 120.0
        confidence_column = False       # fifth value is thermal delta, not conf
    elif plane == "radar":
        state_key, box_key, count_key = (
            "radar_label_state", "rgb_boxes", "n_rgb_boxes"
        )
        width, height = 640.0, 400.0
        confidence_column = True
    else:
        raise ValueError(f"unknown target plane {plane!r}")

    a = split.arrays
    state = int(a[state_key][i])
    presence = np.zeros(max_objects, np.float32)
    boxes = np.zeros((max_objects, 4), np.float32)
    confidence = np.ones(max_objects, np.float32)
    if state == LABEL_POSITIVE:
        n = min(int(a[count_key][i]), max_objects)
        if n <= 0:
            raise ValueError(
                f"{plane} frame {i} is positive but contains no target boxes"
            )
        rows = np.asarray(a[box_key][i, :n], np.float32)
        if confidence_column:
            order = np.argsort(-rows[:, 4])
        else:
            order = np.arange(n)
        for j, row in enumerate(rows[order[:max_objects]]):
            x, y, w, h = (float(v) for v in row[:4])
            presence[j] = 1.0
            boxes[j] = (
                np.clip((x + 0.5 * w) / width, 0.0, 1.0),
                np.clip((y + 0.5 * h) / height, 0.0, 1.0),
                np.clip(w / width, 0.0, 1.0),
                np.clip(h / height, 0.0, 1.0),
            )
            confidence[j] = (
                np.clip(float(row[4]), 0.25, 1.0)
                if confidence_column else 1.0
            )
    return {
        "supervision_state": np.int64(state),
        "gt_presence": presence,
        "gt_boxes": boxes,
        "gt_confidence": confidence,
    }

