"""Per-frame clustering of radar points.

Connected components under a composite distance - cartesian (x, y) plus a
weighted FOLDED-Doppler axis - rather than textbook DBSCAN with min_samples:
after static-map suppression a person at 5 m is 1-3 points, and a min_samples
of 2+ would delete exactly the targets the mission cares about. z never
enters the metric (sigma_el ~12 deg makes it noise), but it is averaged into
the centroid for completeness.

The Doppler axis uses wrap-around distance with the alias period: two points
folded to +0.6 and -0.6 are 0.1 m/s apart, not 1.2. lambda_v defaults to 0
because every recorded session predates Mode P and its Doppler is dead;
turn it on when P1 data exists.
"""
from dataclasses import dataclass

import numpy as np

V_ALIAS = 1.298          # m/s, 3TX fold period (frozen radar cfg)


@dataclass
class Cluster:
    centroid: np.ndarray     # (3,) x, y, z
    n: int
    v_mean: float            # folded! sign untrustworthy
    v_spread: float          # intra-cluster spread, the micro-Doppler proxy
    snr_max: float
    snr_mean: float
    extent_xy: float         # max pairwise distance in the ground plane
    points: np.ndarray       # (n, 6) the raw members


def _folded_dv(a, b):
    d = np.abs(a - b) % V_ALIAS
    return np.minimum(d, V_ALIAS - d)


def cluster_frame(pts, eps=0.6, lambda_v=0.0):
    """pts: (N, 6) [x, y, z, v, snr, noise] -> list of Cluster."""
    n = len(pts)
    if n == 0:
        return []
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    d2 = dx * dx + dy * dy
    if lambda_v > 0:
        dv = _folded_dv(pts[:, 3:4], pts[:, 3:4].T)
        d2 = d2 + (lambda_v * dv) ** 2
    adj = d2 <= eps * eps

    label = np.full(n, -1, int)
    cur = 0
    for i in range(n):
        if label[i] >= 0:
            continue
        stack = [i]
        label[i] = cur
        while stack:
            j = stack.pop()
            for k in np.nonzero(adj[j])[0]:
                if label[k] < 0:
                    label[k] = cur
                    stack.append(k)
        cur += 1

    out = []
    for c in range(cur):
        m = pts[label == c]
        dxy = np.hypot(m[:, 0:1] - m[:, 0:1].T, m[:, 1:2] - m[:, 1:2].T)
        # circular mean of folded velocities
        ang = m[:, 3] * (2 * np.pi / V_ALIAS)
        v_mean = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
                       * (V_ALIAS / (2 * np.pi)))
        v_spread = float(np.median(_folded_dv(m[:, 3], v_mean)))
        out.append(Cluster(centroid=m[:, :3].mean(0), n=len(m),
                           v_mean=v_mean, v_spread=v_spread,
                           snr_max=float(m[:, 4].max()),
                           snr_mean=float(m[:, 4].mean()),
                           extent_xy=float(dxy.max()),
                           points=m))
    return out
