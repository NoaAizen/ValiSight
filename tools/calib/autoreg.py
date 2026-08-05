#!/usr/bin/env python3
"""Register the thermal frame to the visible one without a calibration target.

    ./autoreg.py ../../captures/person3 -o warp_auto.lut

Thermal and visible images do not share brightness - a warm face is bright in
one and dark in the other - so anything based on intensity difference or
correlation fails on this pair. They do share *structure*: the same object
boundaries. Mutual information measures exactly that statistical dependence
without assuming the intensities agree, which is why it is the standard tool for
multi-modal registration.

So instead of detecting a physical target in both cameras, search for the warp
that maximises mutual information between the warped thermal plane and the
visible luma. Frames with real thermal structure (people, warm equipment) carry
the signal; a flat wall carries none and is rejected.

What this recovers and what it does not:

  recovered      scale and translation, optionally rotation - the gross
                 misalignment that dominates the current output
  NOT recovered  lens distortion of either camera. A homography-shaped warp
                 cannot express it; only a real calibration can.
  depth-bound    the fit is valid near the distance of the scenes it was fitted
                 on. With a 12mm baseline the drift over 0.5-2m is about
                 +-1.3 thermal px, so one fit covers the working range.

Good enough to look at. Not good enough to measure against.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib  # noqa: E402  - LUT packing and geometry live there

RGB_W, RGB_H = calib.RGB_W, calib.RGB_H
TH_W, TH_H = calib.TH_W, calib.TH_H
LOW_W, LOW_H = calib.LOW_W, calib.LOW_H
BINS = 32


def decimate(y, factor=4):
    h, w = y.shape
    return y.reshape(h // factor, factor, w // factor, factor).mean((1, 3))


def grad_mag(a):
    """Register on gradient magnitude, not raw intensity.

    The two modalities agree on where edges are and disagree on which side is
    brighter, so gradients concentrate the shared information and drop the
    per-modality offset that would otherwise dilute the histogram."""
    gy, gx = np.gradient(a.astype(np.float64))
    return np.hypot(gx, gy)


def sample(thermal, p, shape=(LOW_H, LOW_W)):
    """Bilinear-sample the thermal frame onto the low-res grid under params p."""
    sx, sy, tx, ty, rot = p
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2.0, h / 2.0
    dx, dy = gx - cx, gy - cy
    c, s = np.cos(rot), np.sin(rot)
    u = (dx * c - dy * s) * sx + cx + tx
    v = (dx * s + dy * c) * sy + cy + ty

    ok = (u >= 0) & (v >= 0) & (u <= TH_W - 1.001) & (v <= TH_H - 1.001)
    uu, vv = np.clip(u, 0, TH_W - 1.001), np.clip(v, 0, TH_H - 1.001)
    x0, y0 = uu.astype(int), vv.astype(int)
    fx, fy = uu - x0, vv - y0
    t = thermal.astype(np.float64)
    out = (t[y0, x0] * (1 - fx) * (1 - fy) + t[y0, x0 + 1] * fx * (1 - fy) +
           t[y0 + 1, x0] * (1 - fx) * fy + t[y0 + 1, x0 + 1] * fx * fy)
    return out, ok


MIN_OVERLAP = 0.45          # fraction of the grid the warp must still cover
SAMPLES = 8000              # fixed sample count, see below


def nmi(a, b, mask, rng):
    """Mutual information with the sample-size bias taken out.

    Normalising by joint entropy is NOT enough on its own. As the warp shrinks
    the overlap the joint histogram gets sparser, and a sparse histogram scores
    *higher* - so an optimiser left to itself wins by shrinking the valid region
    instead of by aligning anything. Measured directly: NMI climbed monotonically
    from 1.009 to 1.018 as the scale ran to the edge of the search box, with no
    peak anywhere.

    Two fixes, both needed. A hard floor on overlap, and a fixed number of
    samples drawn from whatever overlap remains, so every candidate is scored
    with the same histogram density.
    """
    n_valid = int(mask.sum())
    if n_valid < MIN_OVERLAP * mask.size:
        return 0.0

    a, b = a[mask], b[mask]
    if a.size > SAMPLES:
        idx = rng.choice(a.size, SAMPLES, replace=False)
        a, b = a[idx], b[idx]
    if a.size < 500:
        return 0.0
    ra = np.clip(((a - a.min()) / max(a.ptp(), 1e-9) * (BINS - 1)), 0, BINS - 1).astype(int)
    rb = np.clip(((b - b.min()) / max(b.ptp(), 1e-9) * (BINS - 1)), 0, BINS - 1).astype(int)
    h = np.bincount(ra * BINS + rb, minlength=BINS * BINS).reshape(BINS, BINS).astype(float)
    h /= h.sum()
    pa, pb = h.sum(1), h.sum(0)

    def ent(p):
        p = p[p > 0]
        return -(p * np.log(p)).sum()

    hj = ent(h.ravel())
    return (ent(pa) + ent(pb)) / hj if hj > 1e-9 else 0.0


def load_pairs(d):
    out = []
    for jp in sorted(glob.glob(os.path.join(d, "*.json"))):
        stem = jp[:-5]
        rp = stem + "_rgb0.raw"
        if not os.path.exists(rp):
            rp = stem + "_rgb.raw"
        tp = stem + "_thermal.raw"
        if not (os.path.exists(rp) and os.path.exists(tp)):
            continue
        y = np.fromfile(rp, np.uint8).reshape(RGB_H, RGB_W)
        t = np.fromfile(tp, np.uint8).reshape(TH_H, TH_W)
        out.append((grad_mag(decimate(y)), grad_mag(t), os.path.basename(stem)))
    return out


def score(p, pairs, seed=12345):
    # same seed every call: the sampling must not add noise the optimiser chases
    rng = np.random.default_rng(seed)
    total = 0.0
    for ylow, t, _ in pairs:
        w, ok = sample(t, p)
        total += nmi(ylow, w, ok, rng)
    return -total / max(len(pairs), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="capture directories")
    ap.add_argument("-o", "--out", default="warp_auto.lut")
    ap.add_argument("--rotation", action="store_true", help="also fit a small rotation")
    ap.add_argument("--min-structure", type=float, default=1.02,
                    help="reject frames whose best NMI never beats this")
    args = ap.parse_args()

    pairs = []
    for d in args.dirs:
        pairs += load_pairs(d)
    if not pairs:
        raise SystemExit("no usable pairs found")
    print("loaded %d pair(s)" % len(pairs), file=sys.stderr)

    # Coarse sweep first. NMI over a warp is not convex - a local optimiser
    # started at the identity settles into whatever ridge it happens to be on.
    best, best_s = None, 1e9
    for sx in np.linspace(0.35, 1.05, 15):
        for tx in np.linspace(-30, 30, 13):
            for ty in np.linspace(-30, 30, 13):
                p = (sx, sx * (TH_H / TH_W) / (LOW_H / LOW_W), tx, ty, 0.0)
                s = score(p, pairs)
                if s < best_s:
                    best_s, best = s, p
    print("coarse: sx=%.3f sy=%.3f tx=%.1f ty=%.1f  NMI=%.4f" % (*best[:4], -best_s),
          file=sys.stderr)

    free = [0, 1, 2, 3] + ([4] if args.rotation else [])
    p0 = np.array(best, float)

    def wrapped(v):
        p = p0.copy()
        p[free] = v
        return score(p, pairs)

    res = minimize(wrapped, p0[free], method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 4000})
    p = p0.copy()
    p[free] = res.x
    print("refined: sx=%.4f sy=%.4f tx=%.2f ty=%.2f rot=%.4f rad  NMI=%.4f" % (*p, -res.fun),
          file=sys.stderr)

    # per-frame report: a flat scene contributes nothing and should be visible
    weak = 0
    for ylow, t, name in pairs:
        w, ok = sample(t, p)
        v = nmi(ylow, w, ok, np.random.default_rng(12345))
        flag = "" if v >= args.min_structure else "  (little structure - ignored by the fit)"
        if v < args.min_structure:
            weak += 1
        print("  %-22s NMI %.4f%s" % (name, v, flag), file=sys.stderr)
    if weak == len(pairs):
        raise SystemExit("every frame lacks thermal structure - capture a scene with "
                         "people or warm equipment in it, a flat wall cannot register")

    # bake the fitted warp into the LUT fusion.c consumes
    gy, gx = np.mgrid[0:LOW_H, 0:LOW_W].astype(np.float64)
    sx, sy, tx, ty, rot = p
    cx, cy = LOW_W / 2.0, LOW_H / 2.0
    dx, dy = gx - cx, gy - cy
    c, s = np.cos(rot), np.sin(rot)
    u = (dx * c - dy * s) * sx + cx + tx
    v = (dx * s + dy * c) * sy + cy + ty
    lut = calib.pack_lut(np.stack([u.ravel(), v.ravel()], -1))
    lut.tofile(args.out)

    cov = 100.0 * np.mean(lut[..., 0] != calib.FUSION_INVALID)
    json.dump({"sx": p[0], "sy": p[1], "tx": p[2], "ty": p[3], "rot": p[4],
               "nmi": -res.fun, "pairs": len(pairs), "coverage_percent": cov,
               "method": "mutual-information autoregistration",
               "caveat": "global transform only - no lens distortion, depth-bound"},
              open(os.path.splitext(args.out)[0] + ".json", "w"), indent=2)
    print("wrote %s (%d B, %.1f%% thermal coverage)" % (args.out, lut.nbytes, cov),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
