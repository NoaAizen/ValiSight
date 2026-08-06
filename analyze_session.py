"""
Analyze a recorded live-scan session (logs/session_*/ from live_radar_camera).

Prints a full report: radar activity, cluster statistics per label/material,
reflectivity distributions, detection statistics, and match rate. Also exports
clusters.csv + points.csv for spreadsheet/pandas analysis.

Usage:
    python analyze_session.py                 # latest session
    python analyze_session.py logs/session_20260713_221530
    python analyze_session.py --list          # show all sessions
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):        # Hebrew paths on cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))


def sessions():
    return sorted(glob.glob(os.path.join(HERE, "logs", "session_*")))


def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass                     # torn last line on hard exit
    return out


def hist(values, bins, unit=""):
    """Simple text histogram lines."""
    if not values:
        return ["  (no data)"]
    lines = []
    lo, hi = min(values), max(values)
    if lo == hi:
        return ["  all = %.1f%s (n=%d)" % (lo, unit, len(values))]
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(int((v - lo) / step), bins - 1)] += 1
    peak = max(counts)
    for i, c in enumerate(counts):
        bar = "#" * max(1, int(30 * c / peak)) if c else ""
        lines.append("  %6.1f-%6.1f%s %5d %s" %
                     (lo + i * step, lo + (i + 1) * step, unit, c, bar))
    return lines


def main():
    ap = argparse.ArgumentParser(description="Analyze a live-scan session")
    ap.add_argument("session", nargs="?", help="session dir (default: latest)")
    ap.add_argument("--list", action="store_true", help="list sessions")
    args = ap.parse_args()

    if args.list:
        for s in sessions():
            meta_p = os.path.join(s, "meta.json")
            note = ""
            if os.path.isfile(meta_p):
                with open(meta_p) as f:
                    m = json.load(f)
                dur = m.get("end_time", 0) - m.get("start_time", 0)
                note = "%.0fs, radar %s, fusion %s" % (
                    dur, m.get("radar_frames", "?"), m.get("fusion_frames", "?"))
            print("%s   %s" % (os.path.basename(s), note))
        return

    sdir = args.session or (sessions()[-1] if sessions() else None)
    if not sdir or not os.path.isdir(sdir):
        sys.exit("No session found. Run live_radar_camera.py first.")
    print("Session:", sdir)

    meta_p = os.path.join(sdir, "meta.json")
    if os.path.isfile(meta_p):
        with open(meta_p) as f:
            meta = json.load(f)
        print("Params :", {k: meta[k] for k in
                           ("cam", "hfov", "metal_db", "fabric_db") if k in meta})
        if "end_time" in meta:
            print("Length : %.1f s" % (meta["end_time"] - meta["start_time"]))

    radar = load_jsonl(os.path.join(sdir, "radar.jsonl"))
    fusion = load_jsonl(os.path.join(sdir, "fusion.jsonl"))

    # ---- radar activity -------------------------------------------------
    print("\n=== RADAR ===")
    print("frames: %d" % len(radar))
    if radar:
        counts = [len(r["points"]) for r in radar]
        dur = radar[-1]["t"] - radar[0]["t"] if len(radar) > 1 else 0
        print("rate  : %.1f Hz" % (len(radar) / dur if dur else 0))
        print("points/frame: min %d  avg %.1f  max %d  (empty frames: %d)" %
              (min(counts), sum(counts) / len(counts), max(counts),
               sum(1 for c in counts if c == 0)))
        ranges = [math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
                  for r in radar for p in r["points"]]
        print("range distribution (m):")
        print("\n".join(hist(ranges, 10, "m")))

    # ---- clusters --------------------------------------------------------
    print("\n=== CLUSTERS (from fusion log) ===")
    by_label, by_mat, refl_by_mat = {}, {}, {}
    for rec in fusion:
        for c in rec["clusters"]:
            by_label[c["label"]] = by_label.get(c["label"], 0) + 1
            m = c.get("material", "unknown")
            by_mat[m] = by_mat.get(m, 0) + 1
            if c.get("refl_db") is not None:
                refl_by_mat.setdefault(m, []).append(c["refl_db"])
    print("by motion class :", by_label or "(none)")
    print("by material     :", by_mat or "(none)")
    for m, vals in sorted(refl_by_mat.items()):
        vs = sorted(vals)
        print("reflectivity %-7s: n=%-5d median %.1f dB  (p10 %.1f / p90 %.1f)"
              % (m, len(vs), vs[len(vs) // 2], vs[len(vs) // 10],
                 vs[len(vs) * 9 // 10]))

    # ---- object tracks (signatures) --------------------------------------
    tracks = load_jsonl(os.path.join(sdir, "tracks.jsonl"))
    print("\n=== TRACKS (one row per physical object) ===")
    if not tracks:
        print("(none — old session, or nothing confirmed)")
    for tr in tracks:
        f = tr["features"]
        print("T%-3d %-10s %-7s %5.1fs  %.1fm az %+.0f  refl %s dB  "
              "hits %d  moved %.1fm%s" % (
                  tr["track_id"], tr["motion"], tr["material"],
                  f["persistence_s"], f["range_m"], f["az_deg"],
                  f["refl_median"], f["hits"], f["displacement_m"],
                  ("  [label: %s]" % tr["session_label"])
                  if tr.get("session_label") else ""))

    # ---- camera detections & matching ------------------------------------
    print("\n=== CAMERA / FUSION ===")
    det_counts, matched, unmatched = {}, 0, 0
    for rec in fusion:
        for i, d in enumerate(rec["detections"]):
            det_counts[d["label"]] = det_counts.get(d["label"], 0) + 1
            if i < len(rec["assigned"]) and rec["assigned"][i] is not None:
                matched += 1
            else:
                unmatched += 1
    print("fusion frames  :", len(fusion))
    print("detections     :", det_counts or "(none)")
    tot = matched + unmatched
    if tot:
        print("radar match    : %d/%d boxes (%.0f%%)" %
              (matched, tot, 100.0 * matched / tot))
    frames_dir = os.path.join(sdir, "frames")
    n_snaps = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print("snapshots      : %d in %s" % (n_snaps, frames_dir))

    # ---- CSV export -------------------------------------------------------
    cpath = os.path.join(sdir, "clusters.csv")
    with open(cpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "label", "material", "refl_db", "range_m",
                    "doppler_mps", "extent_m", "n_points", "x", "y", "z"])
        for rec in fusion:
            for c in rec["clusters"]:
                w.writerow([rec["t"], c["label"], c.get("material"),
                            c.get("refl_db"), c["range_m"], c["doppler_mps"],
                            c["extent_m"], c["n_points"]] + list(c["centroid"]))
    ppath = os.path.join(sdir, "points.csv")
    with open(ppath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "frame", "x", "y", "z", "doppler", "snr_db",
                    "noise_db"])
        for r in radar:
            for p in r["points"]:
                # old sessions logged 5-element points (no noise field)
                w.writerow([r["t"], r["frame"]] + list(p) +
                           [None] * (6 - len(p)))
    paths = [cpath, ppath]
    if tracks:
        from radar_tracker import SIGNATURE_FIELDS
        spath = os.path.join(sdir, "signatures.csv")
        n_dop = len(tracks[0]["doppler_hist"])
        with open(spath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "session_label", "motion", "material"] +
                       SIGNATURE_FIELDS + ["dop%d" % i for i in range(n_dop)])
            for tr in tracks:
                w.writerow([tr["track_id"], tr.get("session_label"),
                            tr["motion"], tr["material"]] +
                           tr["vector"] + tr["doppler_hist"])
        paths.append(spath)
    print("\nCSV exported: %s" % ", ".join(paths))


if __name__ == "__main__":
    main()
