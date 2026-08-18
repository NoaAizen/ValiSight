"""Track-level features for the person/clutter classifier (gate 5's GBM).

The PERCEPTION-PLAN feature table, computed over a whole track: Doppler
statistics (spread = micro-Doppler proxy), range-compensated RCS, spatial
extent, and trajectory kinematics. Everything physical, everything
inspectable - when the GBM is wrong we get to see on WHICH feature.

Doppler features carry a caveat flag on pre-Mode-P data: with v_max 0.649
the fold destroys them, and training on them there teaches noise.
"""
import numpy as np


def track_features(track):
    h = track.history
    t = np.array([x[0] for x in h])
    xy = np.array([[x[1], x[2]] for x in h])
    clusters = [x[3] for x in h]

    rng = np.hypot(xy[:, 0], xy[:, 1])
    snr_max = np.array([c.snr_max for c in clusters])
    rcs = snr_max + 40.0 * np.log10(np.maximum(rng, 0.1))
    v_mean = np.array([c.v_mean for c in clusters])
    v_spread = np.array([c.v_spread for c in clusters])
    npts = np.array([c.n for c in clusters])
    ext = np.array([c.extent_xy for c in clusters])

    seg = np.diff(xy, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    dt = np.maximum(np.diff(t), 1e-3)
    speed = seg_len / dt
    path = float(seg_len.sum())
    disp = float(np.hypot(*(xy[-1] - xy[0])))

    return {
        # Doppler (folded pre-Mode-P: see caveat in the docstring)
        'v_abs_mean': float(np.abs(v_mean).mean()),
        'v_spread_med': float(np.median(v_spread)),
        'v_p90_10': float(np.percentile(v_mean, 90)
                          - np.percentile(v_mean, 10)),
        # RCS, range-compensated
        'rcs_mean': float(rcs.mean()),
        'rcs_max': float(rcs.max()),
        # spatial
        'extent_med': float(np.median(ext)),
        'npts_mean': float(npts.mean()),
        'npts_cv': float(npts.std() / max(npts.mean(), 1e-6)),
        # trajectory
        'duration_s': float(t[-1] - t[0]),
        'ground_speed_med': float(np.median(speed)) if len(speed) else 0.0,
        'ground_speed_p90': float(np.percentile(speed, 90)) if len(speed) else 0.0,
        'straightness': disp / path if path > 0.1 else 0.0,
        'range_med': float(np.median(rng)),
        'n_updates': len(h),
    }
