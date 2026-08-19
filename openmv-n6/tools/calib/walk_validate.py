#!/usr/bin/env python3
"""Radar->thermal validation on a walking-person session (plan of 2026-08-19).

For every recorded thermal frame: the warm-body center in the REGISTERED
thermal plane (the production warp LUT, so output/RGB pixel coordinates)
against the radar's projected azimuth under the frozen 2026-08-18 extrinsic.

    ./walk_validate.py ../captures/session_walk

Reports the horizontal error in degrees (the budget number: <= 2 deg wanted,
1.7 radar + 0.4 stereo) and the fraction of frames where the radar projection
falls inside the body's horizontal extent (>= 90% wanted).

Method choices, deliberate:
  - u only. Radar elevation is unusable on this rig (sigma ~12 deg, el reads
    ~3x true) and the calibration itself is a u-only REDUCED fit.
  - moving points only (|v| >= 0.1): the radar cannot see a static person and
    the room's static returns are furniture, not the target.
  - the radar u is interpolated in time between the two radar frames that
    bracket the thermal timestamp - at 27 deg/s of lateral walk, the ~50 ms
    nearest-frame error alone would eat a quarter of the 2 deg budget.
  - the body center comes from the registered thermal plane sampled through
    warp.lut, NOT from the raw thermal frame: whatever the LUT gets wrong is
    part of the chain being judged.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # openmv-n6/
sys.path.insert(0, ROOT)
from perception.project import RadarProjector          # noqa: E402

TH_W, TH_H = 160, 120
LOW_W, LOW_H = 160, 100
DECIM = 4
INVALID = 0xFFFF
F_PX = 525.0                                           # tape-fixed RGB focal

MIN_SPEED = 0.1        # m/s - below this the radar return is room clutter
RANGE_GATE = (1.5, 9.0)
MIN_BLOB_CELLS = 25    # low-res cells; a person at 8 m is ~280
MAX_BLOB_W_PX = 350    # a person at 2 m is ~160 px wide; wider = merged
                       # blobs (two people, a warm wall) - ambiguous, skip
WARM_MARGIN = 30       # codes above background median (~1.5 C at 0.051 C/code)
MAX_DT = 0.12          # s - thermal frame with no radar frame this close: skip
ASSOC_GATE_PX = 100    # nearest-cluster association gate (~11 deg, >5x the
                       # residual being measured; sensitivity checked 60-200)


def load_lut(path):
    lut = np.fromfile(path, np.uint16).reshape(LOW_H, LOW_W, 2)
    valid = lut[..., 0] != INVALID
    # nearest-neighbour source index per cell (Q8 -> px)
    su = np.clip(np.round(lut[..., 0] / 256.0), 0, TH_W - 1).astype(np.int32)
    sv = np.clip(np.round(lut[..., 1] / 256.0), 0, TH_H - 1).astype(np.int32)
    return su, sv, valid


def register(thermal, su, sv, valid):
    """Raw 160x120 thermal -> 160x100 registered plane (invalid cells = 0)."""
    reg = thermal[sv, su]
    reg[~valid] = 0
    return reg


def body_center(reg, valid):
    """Warm-blob centroid and horizontal extent, in OUTPUT pixels.

    None when no blob passes the size gate (person out of thermal frame)."""
    vals = reg[valid]
    if vals.size == 0:
        return None
    bg = np.median(vals)
    mask = valid & (reg >= bg + WARM_MARGIN)
    if not mask.any():
        return None
    # largest 4-connected component, plain BFS-free labeling via scipy-less
    # two-pass would be overkill: cv2 is already a project dependency.
    import cv2
    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=4)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < MIN_BLOB_CELLS:
        return None
    if stats[big, cv2.CC_STAT_WIDTH] * DECIM > MAX_BLOB_W_PX:
        return None
    cx = cent[big][0]                                   # low-res cells
    x0 = stats[big, cv2.CC_STAT_LEFT]
    w = stats[big, cv2.CC_STAT_WIDTH]
    off = (DECIM - 1) / 2.0                             # cell centre convention
    return {
        "u": cx * DECIM + off,
        "u_lo": x0 * DECIM + off,
        "u_hi": (x0 + w - 1) * DECIM + off,
        "cells": int(stats[big, cv2.CC_STAT_AREA]),
    }


def radar_clusters(points, proj):
    """Moving in-gate points -> clusters by projected-u linkage (30 px).

    [(u_snr_weighted, sum_snr, n, median_range)], u ascending. Clustering is
    what makes a walking-person session measurable at all: the room's moving
    returns (multipath ghosts, a second person near the desk - closer, so
    1/R^4 stronger) otherwise contaminate any per-frame average. Measured on
    session_walk 2026-08-19: the raw SNR-median gave 7.4 deg where the
    per-cluster association gives 1.8."""
    P = np.asarray(points, np.float64).reshape(-1, 6)
    r = np.linalg.norm(P[:, :3], axis=1)
    keep = (np.abs(P[:, 3]) >= MIN_SPEED) & (r >= RANGE_GATE[0]) & (r <= RANGE_GATE[1])
    if not keep.any():
        return []
    P, r = P[keep], r[keep]
    uv, in_front = proj.project(P[:, :3])
    ok = in_front & np.isfinite(uv[:, 0])
    if not ok.any():
        return []
    u, snr, r = uv[ok, 0], P[ok, 4], r[ok]
    order = np.argsort(u)
    u, snr, r = u[order], snr[order], r[order]
    cl, start = [], 0
    for i in range(1, len(u) + 1):
        if i == len(u) or u[i] - u[i - 1] > 30:
            uu, ss, rr = u[start:i], snr[start:i], r[start:i]
            cl.append((float(np.sum(uu * ss) / ss.sum()), float(ss.sum()),
                       len(uu), float(np.median(rr))))
            start = i
    return cl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--lut", default=os.path.join(ROOT, "calib-artifacts", "warp.lut"))
    ap.add_argument("--out", default=None, help="write per-frame records (jsonl)")
    args = ap.parse_args()

    su, sv, valid = load_lut(args.lut)
    proj = RadarProjector()

    frames = [json.loads(l) for l in open(os.path.join(args.session, "frames.jsonl"))]
    radar = [json.loads(l) for l in open(os.path.join(args.session, "radar.jsonl"))]
    rt = np.array([f["t_mono"] for f in radar])
    thermal = np.memmap(os.path.join(args.session, "thermal.bin"), np.uint8,
                        mode="r").reshape(-1, TH_H, TH_W)

    # clusters per RADAR frame once; each thermal frame associates against the
    # clusters of the two radar frames bracketing its timestamp
    CL = [radar_clusters(f["points"], proj) if f["points"] else [] for f in radar]

    recs, n_noblob, n_noradar, n_nogate = [], 0, 0, 0
    for fr in frames:
        i = fr["i"]
        if i >= len(thermal):
            break
        body = body_center(register(np.array(thermal[i]), su, sv, valid), valid)
        if body is None:
            n_noblob += 1
            continue
        t = fr["t_mono"]
        k = int(np.searchsorted(rt, t))
        cands = [c for j in (k - 1, k)
                 if 0 <= j < len(radar) and abs(rt[j] - t) <= MAX_DT
                 for c in CL[j]]
        if not cands:
            n_noradar += 1
            continue
        u_r, _snr, npts, rng = min(cands, key=lambda c: abs(c[0] - body["u"]))
        du = u_r - body["u"]
        if abs(du) > ASSOC_GATE_PX:
            # radar saw motion, none of it at the person: a detectability
            # miss (or the person stood while something else moved) - counted
            # apart because it is not an angular error of the calibration.
            n_nogate += 1
            continue
        recs.append({
            "i": i, "t": t, "body_u": round(body["u"], 1),
            "body_lo": body["u_lo"], "body_hi": body["u_hi"],
            "cells": body["cells"], "radar_u": round(u_r, 1),
            "du_px": round(du, 2),
            "deg": round(np.degrees(np.arctan2(du, F_PX)), 3),
            "range": round(rng, 2), "n_pts": npts,
            "inside": bool(body["u_lo"] <= u_r <= body["u_hi"]),
        })

    if not recs:
        print("no measurable frames (%d without blob, %d without radar, %d out of gate)"
              % (n_noblob, n_noradar, n_nogate))
        return 1

    deg = np.array([r["deg"] for r in recs])
    inside = np.mean([r["inside"] for r in recs]) * 100
    print("frames: %d measured, %d no warm blob, %d no moving radar return, "
          "%d radar motion elsewhere only (detectability, not angle)"
          % (len(recs), n_noblob, n_noradar, n_nogate))
    print("error |deg|: median %.2f  p95 %.2f  max %.2f    signed median %+.2f"
          % (np.median(np.abs(deg)), np.percentile(np.abs(deg), 95),
             np.abs(deg).max(), np.median(deg)))
    print("radar inside body extent: %.1f%%" % inside)
    for lo, hi in ((1.5, 3.0), (3.0, 5.0), (5.0, 9.0)):
        sel = [r for r in recs if lo <= r["range"] < hi]
        if sel:
            d = np.array([r["deg"] for r in sel])
            print("  range %.1f-%.1f m: n=%-4d median %+.2f deg  |med| %.2f  inside %.0f%%"
                  % (lo, hi, len(sel), np.median(d), np.median(np.abs(d)),
                     100 * np.mean([r["inside"] for r in sel])))
    for name, sel in (("left  (u<320)", [r for r in recs if r["body_u"] < 320]),
                      ("right (u>=320)", [r for r in recs if r["body_u"] >= 320])):
        if sel:
            d = np.array([r["deg"] for r in sel])
            print("  %s: n=%-4d median %+.2f deg  |med| %.2f  inside %.0f%%"
                  % (name, len(sel), np.median(d), np.median(np.abs(d)),
                     100 * np.mean([r["inside"] for r in sel])))
    # error vs body azimuth - the flank residual (documented 2026-08-18,
    # V05/acceptance note) shows up here as a slope, a plain yaw as an offset
    az = np.degrees(np.arctan2(np.array([r["body_u"] for r in recs]) - 320, F_PX))
    for lo in range(-25, 25, 5):
        m = (az >= lo) & (az < lo + 5)
        if m.sum() >= 15:
            print("  body az %+3d..%+3d deg: n=%-4d signed median %+.2f deg"
                  % (lo, lo + 5, m.sum(), np.median(deg[m])))

    verdict = np.median(np.abs(deg)) <= 2.0 and inside >= 90.0
    print("VERDICT: %s (targets: median <= 2 deg, inside >= 90%%)"
          % ("PASS" if verdict else "FAIL"))
    if args.out:
        with open(args.out, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print("per-frame records: %s" % args.out)
    return 0 if verdict else 2


if __name__ == "__main__":
    sys.exit(main())
