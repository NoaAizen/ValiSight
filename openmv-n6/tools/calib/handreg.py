#!/usr/bin/env python3
"""Register thermal to visible from a warm object moved around the frame.

    ./handreg.py ../../captures/handwave -o warp_hand.lut --preview

Mutual-information registration needs the thermal image to carry structure
spread across the frame. On ordinary indoor scenes it does not - a wall with two
warm faces gives a thermal frame that is almost featureless, and MI measured on
exactly that data came out flat at 1.009 (1.0 being statistical independence),
with no peak to optimise toward.

This sidesteps the problem. Move one warm object - a hand is enough - to a
spread of positions, and each frame yields a single unambiguous point in both
modalities:

    thermal   the hottest blob's centroid
    visible   the centroid of what moved, against the per-pixel median of the
              whole sequence (the hand moves, the room does not, so the median
              is a clean background plate with no separate shot needed)

That is a hard correspondence rather than a weak statistical one, and a handful
of well-spread positions pins down an affine fit. RANSAC drops the frames where
the hand left the thermal field, the arm dominated the motion mask, or a face
outshone the hand.

Still a global transform: no lens distortion, and tied to the depth it was
waved at. Use the printed target when the image has to be measured.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib  # noqa: E402

RGB_W, RGB_H = calib.RGB_W, calib.RGB_H
TH_W, TH_H = calib.TH_W, calib.TH_H
LOW_W, LOW_H = calib.LOW_W, calib.LOW_H


def repair_rows(t, flat=24, lift=64):
    """Interpolate away rows the sensor returns empty of scene.

    On this unit 14 fixed rows - 6, 12, 28, 35, 36 and the run 55..63 - come back
    carrying nothing. VoSPI tearing is random, so a defect pinned to fixed rows is
    the sensor itself, not the link. It has to be repaired here because such a row
    is hotter than anything in the scene, and the hottest-blob search below would
    lock onto it instead of the hand.

    This used to threshold on brightness (>=250 over 60% of the row), matching
    what the rows look like at a glance. Measured over 46 recorded frames, that
    misses them on 21: the rows are not stuck at 255, they sit at a fixed high
    offset that only clips at 255 once the frame's own level rises, and on the
    low-level frames of a sequence they read ~238 with a spread of 9-14 codes.

    So the test is whether the row holds any scene at all - flat to within `flat`
    codes, and `lift` codes above the frame median. Dead rows spread 0-14 and sit
    ~190 above; ordinary rows spread 55-90 and sit within ~5. fusion.c's
    deadrow_flat/deadrow_lift are the same test with the same defaults, so the
    calibration and the pipeline see the same frame.
    """
    out = t.copy()
    med = np.median(t)
    bad = (t.ptp(1) <= flat) & (t.mean(1) - med >= lift)

    # A scene that is genuinely flat and hot over most of the field is not a
    # defect, and interpolating it away would erase the warm object being tracked.
    if bad.sum() * 4 > len(bad):
        return out, 0
    if not bad.any():
        return out, 0
    good = np.nonzero(~bad)[0]
    for r in np.nonzero(bad)[0]:
        lo = good[good < r]
        hi = good[good > r]
        if len(lo) and len(hi):
            a, b = lo[-1], hi[0]
            w = (r - a) / float(b - a)
            out[r] = t[a] * (1 - w) + t[b] * w
        elif len(lo):
            out[r] = t[lo[-1]]
        elif len(hi):
            out[r] = t[hi[0]]
    return out, int(bad.sum())


def decimate(y, f=4):
    h, w = y.shape
    return y.reshape(h // f, f, w // f, f).mean((1, 3))


def hot_blob(t, min_px, max_px):
    """Find the warm object without trusting a fixed percentile.

    Two things defeat a fixed threshold here. Stuck-high pixels are hotter than
    a hand, so the top 1% of a raw frame is scattered dead pixels rather than the
    target - measured: 9 connected px at p99 versus ~900 at p90. And how much of
    the frame the object fills depends on how close it is held.

    So: repair the isolated pixels with a median first, then walk the threshold
    down until the largest connected region is a plausible size.
    """
    med = ndimage.median_filter(t, size=3)
    for pct in (99.0, 98.0, 97.0, 95.0, 93.0, 91.0, 89.0, 86.0, 82.0):
        c, area = biggest_blob(med > np.percentile(med, pct), min_px, max_px)
        if c is not None:
            return c, area, pct
    return None, 0, 0.0


def biggest_blob(mask, min_px, max_px):
    """Centroid of the largest connected region, if it is a plausible size."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return None, 0
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    area = int(sizes[k - 1])
    if area < min_px or area > max_px:
        return None, area
    cy, cx = ndimage.center_of_mass(mask, lab, k)
    return (cx, cy), area


def load(d):
    frames = []
    for jp in sorted(glob.glob(os.path.join(d, "*.json"))):
        st = jp[:-5]
        rp = st + "_rgb0.raw"
        if not os.path.exists(rp):
            rp = st + "_rgb.raw"
        tp = st + "_thermal.raw"
        if not (os.path.exists(rp) and os.path.exists(tp)):
            continue
        th, nbad = repair_rows(np.fromfile(tp, np.uint8).reshape(TH_H, TH_W).astype(float))
        if nbad:
            print("  %-10s repaired %d saturated row(s)"
                  % (os.path.basename(st), nbad), file=sys.stderr)
        frames.append((decimate(np.fromfile(rp, np.uint8).reshape(RGB_H, RGB_W)),
                       th, os.path.basename(st)))
    return frames


def correspondences(frames, hot_pct, move_thresh, verbose=True):
    ylow = np.stack([f[0] for f in frames])

    # Normalise each frame's level before differencing. The visible sensor runs
    # auto-exposure, so successive frames of an unchanged room differ by a global
    # offset - and that offset shows up across the whole difference image,
    # swamping the object we are trying to isolate. Measured: the background
    # structure of the entire scene appeared in the motion map, not just the hand.
    ylow = ylow - ylow.mean((1, 2), keepdims=True) + ylow.mean()
    bg = np.median(ylow, 0)                 # the room; the object averages out

    pts = []
    for i, (y, t, name) in enumerate(frames):
        tp, ta, used = hot_blob(t, 60, TH_W * TH_H // 3)

        # Same adaptive search as the thermal side. A std-scaled threshold is
        # self-defeating here: a frame with a big change has a big std, so the
        # threshold rises and nothing passes; a quiet frame has a small std and
        # everything passes. Measured, it produced only 0-pixel or 5600-pixel
        # blobs and never anything in between.
        diff = np.abs(ylow[i] - bg)
        vp, va = None, 0
        for pct in (99.5, 99.0, 98.0, 96.0, 94.0, 91.0, 88.0):
            c, area = biggest_blob(diff > max(move_thresh, np.percentile(diff, pct)),
                                   40, LOW_W * LOW_H // 4)
            if c is not None:
                vp, va = c, area
                break

        if tp and vp:
            pts.append((vp, tp, name))
            if verbose:
                print("  %-10s visible (%5.1f,%5.1f) px=%-5d  thermal (%5.1f,%5.1f) px=%-4d p%.0f"
                      % (name, vp[0], vp[1], va, tp[0], tp[1], ta, used), file=sys.stderr)
        elif verbose:
            why = []
            if not vp:
                why.append("no motion blob (%d px)" % va)
            if not tp:
                why.append("no hot blob (%d px)" % ta)
            print("  %-10s skip: %s" % (name, ", ".join(why)), file=sys.stderr)
    return pts


def fit_affine(src, dst):
    """Least-squares affine mapping src -> dst. src,dst are (N,2)."""
    A = np.hstack([src, np.ones((len(src), 1))])
    M, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return M                                  # (3,2)


def apply_affine(M, pts):
    return np.hstack([pts, np.ones((len(pts), 1))]) @ M


def ransac(src, dst, iters=2000, tol=6.0, seed=7):
    """Affine needs 3 points; sample 3, score the rest, keep the best consensus."""
    rng = np.random.default_rng(seed)
    n = len(src)
    if n < 4:
        return fit_affine(src, dst), np.ones(n, bool)
    best_in, best_M = None, None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        if np.linalg.matrix_rank(np.hstack([src[idx], np.ones((3, 1))])) < 3:
            continue
        M = fit_affine(src[idx], dst[idx])
        err = np.linalg.norm(apply_affine(M, src) - dst, axis=1)
        inl = err < tol
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best_M = inl, M
    if best_in.sum() >= 3:
        best_M = fit_affine(src[best_in], dst[best_in])   # refit on the consensus
    return best_M, best_in


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-o", "--out", default="warp_hand.lut")
    ap.add_argument("--hot-pct", type=float, default=99.0,
                    help="thermal percentile that counts as the warm object")
    ap.add_argument("--move-thresh", type=float, default=6.0)
    ap.add_argument("--tol", type=float, default=6.0, help="RANSAC inlier tolerance, thermal px")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    frames = []
    for d in args.dirs:
        frames += load(d)
    if len(frames) < 4:
        raise SystemExit("need at least 4 frames, found %d" % len(frames))
    print("loaded %d frame(s)" % len(frames), file=sys.stderr)

    pts = correspondences(frames, args.hot_pct, args.move_thresh)
    if len(pts) < 4:
        raise SystemExit(
            "only %d usable correspondence(s). The warm object must be visible in BOTH: "
            "keep it inside the thermal field (the centre ~2/3 of the visible frame) and "
            "keep the rest of the scene still." % len(pts))

    src = np.array([p[0] for p in pts], float)      # low-res visible grid
    dst = np.array([p[1] for p in pts], float)      # thermal pixels
    M, inl = ransac(src, dst, tol=args.tol)
    resid = np.linalg.norm(apply_affine(M, src) - dst, axis=1)

    print("\n%d/%d inliers, residual median %.2f / max %.2f thermal px"
          % (inl.sum(), len(pts), np.median(resid[inl]), resid[inl].max()), file=sys.stderr)
    for (p, _, name), r, k in zip(pts, resid, inl):
        print("  %-10s %6.2f px %s" % (name, r, "" if k else "OUTLIER"), file=sys.stderr)

    if inl.sum() < 4:
        raise SystemExit("consensus too small to trust - wave the object to more, "
                         "better-spread positions")

    # Validity gates. A low RANSAC residual is NOT evidence the fit is real: a
    # cluster of near-identical points fits anything. Three separate degenerate
    # fits on this data reported medians of 2.10, 0.65 and 0.44 px while the
    # thermal detections barely moved and, in one case, two frames with the same
    # visible position mapped to thermal points 20px apart - a contradiction no
    # transform can satisfy. These gates test the correspondences themselves.
    si, di = src[inl], dst[inl]

    dup = 0
    for i in range(len(si)):
        for j in range(i + 1, len(si)):
            if np.linalg.norm(si[i] - si[j]) < 3.0:      # same place in the visible frame
                if np.linalg.norm(di[i] - di[j]) > args.tol * 2:
                    dup += 1
    if dup:
        raise SystemExit(
            "%d pair(s) of frames put the object at the same visible position but "
            "different thermal positions. No transform can satisfy that - the "
            "detector is tracking something other than the object (a body, a warm "
            "background, a defective row). Fix the capture, do not trust this fit."
            % dup)

    for name, pts_, span in (("visible", si, (LOW_W, LOW_H)), ("thermal", di, (TH_W, TH_H))):
        frac = pts_.ptp(0) / np.array(span, float)
        if min(frac) < 0.25:
            # Say which axis and what to do about it. Every capture on this bench
            # so far has failed here, and the bare percentages did not tell you
            # whether to wave wider, wave slower, or hold still more - so the
            # session got repeated with the same mistake in it.
            thin = "horizontally" if frac[0] < frac[1] else "vertically"
            why = ("the hand was not the brightest thing the thermal detector "
                   "could find - a face or a radiator outshone it"
                   if name == "thermal" else
                   "the hand was not the biggest thing that MOVED - the visible "
                   "side keys off motion against the sequence median, so a "
                   "shifting body, a swaying arm or a moved camera takes over")
            raise SystemExit(
                "%s detections only span %.0f%%x%.0f%% of the frame, mostly too thin %s.\n"
                "The fit would be an extrapolation from a cluster, so it is refused.\n"
                "\n"
                "Most likely %s.\n"
                "\n"
                "A capture that works: 30-60 frames (~5s), bare hand, sleeve down,\n"
                "0.6-1.0m from the lens, held still for two or three frames at each\n"
                "of ~9 positions on a 3x3 grid that reaches the corners of the\n"
                "thermal field. Everything else in the scene stays put, including\n"
                "you - sit down and move only the arm. Nothing else warm in view.\n"
                "Then: ./handreg.py <dir> -o warp.lut --preview"
                % (name, frac[0] * 100, frac[1] * 100, thin, why))

    # both views must agree the object moved the same way
    for k, ax in ((0, "x"), (1, "y")):
        r = np.corrcoef(si[:, k], di[:, k])[0, 1]
        if r < 0.85:
            raise SystemExit(
                "%s positions correlate at only %.2f between the two cameras. They are "
                "not tracking the same object." % (ax, r))
    spread = src[inl].ptp(0)
    if spread[0] < LOW_W * 0.35 or spread[1] < LOW_H * 0.35:
        print("warning: the inlier positions only span %.0fx%.0f of the %dx%d grid. "
              "A fit from a small patch extrapolates badly - spread the object wider."
              % (spread[0], spread[1], LOW_W, LOW_H), file=sys.stderr)

    gy, gx = np.mgrid[0:LOW_H, 0:LOW_W].astype(float)
    grid = np.stack([gx.ravel(), gy.ravel()], -1)
    lut = calib.pack_lut(apply_affine(M, grid))
    lut.tofile(args.out)
    cov = 100.0 * np.mean(lut[..., 0] != calib.FUSION_INVALID)

    json.dump({"affine": M.tolist(), "inliers": int(inl.sum()), "points": len(pts),
               "residual_median_px": float(np.median(resid[inl])),
               "coverage_percent": cov,
               "method": "moving warm object, thermal hot blob vs visible motion blob",
               "caveat": "global affine only - no lens distortion, tied to the wave depth"},
              open(os.path.splitext(args.out)[0] + ".json", "w"), indent=2)
    print("wrote %s (%d B, %.1f%% thermal coverage)" % (args.out, lut.nbytes, cov),
          file=sys.stderr)

    if args.preview:
        from PIL import Image
        im = Image.new("RGB", (LOW_W * 4, LOW_H * 4), (20, 20, 20))
        from PIL import ImageDraw
        d = ImageDraw.Draw(im)
        for (p, q, _), k in zip(pts, inl):
            c = (90, 220, 120) if k else (220, 90, 90)
            d.ellipse([p[0]*4-4, p[1]*4-4, p[0]*4+4, p[1]*4+4], outline=c, width=2)
            pr = apply_affine(M, np.array([p]))[0]
            d.line([p[0]*4, p[1]*4, pr[0]*4*LOW_W/TH_W, pr[1]*4*LOW_H/TH_H], fill=c)
        im.save(os.path.splitext(args.out)[0] + "_points.png")
        print("wrote point map", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
