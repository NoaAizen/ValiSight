"""Radar-track <-> camera-box association: horizontal only, gated Hungarian.

What is matched: image-u of the projected track centroid against the box
center. What is deliberately NOT matched:

  v (vertical)  - radar elevation is unusable (sigma ~12 deg) and the
                  extrinsic vertical is mechanical-geometry only.
  Doppler       - folded, and irrelevant to identity at one instant.

The projection is trusted in the 1.5-3.5 m extrinsic band (worst measured
error there 35 px, hence the +-50 px gate; V2 of the extrinsics tightens it
to +-20). Outside the band the pairing is still computed but flagged, never
counted in any quality number.

Silence is classified, not ignored (the decision layer feeds on this):
a track whose projection falls outside [0, W) is EXPLAINED - the camera
physically cannot see it; a track inside the frame with no box is a
CONTRADICTION - camera failure or radar invention, and which one is the
decision layer's problem, not ours.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from perception.project import RadarProjector

IMG_W = 640
GATE_PX = 50.0


@dataclass
class Pairing:
    track_idx: int
    box_idx: Optional[int]        # None = unmatched
    du_px: Optional[float]
    u_proj: float
    range_m: float
    in_band: bool
    status: str                   # MATCHED / SILENT_EXPLAINED / CONTRADICTION


class Associator:
    def __init__(self, projector=None, gate_px=GATE_PX, img_w=IMG_W):
        self.proj = projector or RadarProjector()
        self.gate = gate_px
        self.img_w = img_w

    def pair(self, track_xy: np.ndarray, boxes: List[dict]) -> List[Pairing]:
        """track_xy: (N, 2) ground-plane track positions (project frame).
        boxes: [{'x','y','w','h'}, ...] in visible pixels."""
        n = len(track_xy)
        if n == 0:
            return []
        pts3 = np.concatenate([track_xy, np.zeros((n, 1))], 1)
        uv, in_front = self.proj.project(pts3)
        rng = np.hypot(track_xy[:, 0], track_xy[:, 1])
        in_band = self.proj.in_calibrated_band(pts3)
        centers = np.array([b['x'] + b['w'] / 2 for b in boxes]) \
            if boxes else np.zeros(0)

        out = []
        cost = np.full((n, max(len(boxes), 1)), 1e6)
        for i in range(n):
            if not in_front[i]:
                continue
            for j in range(len(boxes)):
                d = abs(uv[i, 0] - centers[j])
                if d <= self.gate:
                    cost[i, j] = d
        ri, ci = linear_sum_assignment(cost)
        match = {i: j for i, j in zip(ri, ci) if cost[i, j] < 1e6}

        for i in range(n):
            u = float(uv[i, 0]) if in_front[i] else float('nan')
            if i in match:
                j = match[i]
                out.append(Pairing(i, j, float(abs(u - centers[j])), u,
                                   float(rng[i]), bool(in_band[i]),
                                   'MATCHED'))
            elif not in_front[i] or not (0 <= u < self.img_w):
                out.append(Pairing(i, None, None, u, float(rng[i]),
                                   bool(in_band[i]), 'SILENT_EXPLAINED'))
            else:
                out.append(Pairing(i, None, None, u, float(rng[i]),
                                   bool(in_band[i]), 'CONTRADICTION'))
        return out
