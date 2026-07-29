"""Offline driver: score a recorded session for radar-odometry feasibility.

    python3 radar_replay.py [SESSION_DIR] [--v-max 1.001]
    python3 radar_replay.py --list

With no SESSION_DIR it takes the most recent one under data/recordings/.

This imports no serial and touches no hardware, so Stage 1 and Stage 2 are
developed against recordings rather than against the rig. The whole point of
storing raw ungated points is that this can be re-run with different thresholds,
a different defect map or a different estimator over data collected weeks ago.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chirp_geometry
import ego_velocity as ev
import radar_metrics
import radar_static as rs

RECORDINGS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "recordings")


def sessions():
    return sorted(glob.glob(os.path.join(RECORDINGS, "*", "")), reverse=True)


def load(session):
    path = os.path.join(session, "radar_frames.jsonl")
    if not os.path.exists(path):
        return None, "no radar_frames.jsonl (recorded before per-frame logging?)"
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "pts" in r:
                out.append(r)
    return out, None


def clock_report(frames, frame_s=None):
    """The radar's own counter versus host arrival, side by side.

    `frame_s` is the configured frame period in seconds, needed to turn
    cycles-per-frame into a tick rate. It used to be hardcoded to 0.1 -- which
    silently reports a 200 MHz counter as 400 MHz under odom_e_20hz.cfg (50 ms)
    and halves the jitter along with it. Pass it from the session's recorded
    config; with None it is estimated from the median host arrival interval,
    which is good to a fraction of a percent because the host is locked to the
    radar on average even when individual arrivals are quantised.

    Every key is always present, and every value may be None. A caller
    formatting a report must not have to know which of the two shapes it got.
    """
    empty = {"frames": len(frames), "frame_number_gaps": None,
             "cycles_per_frame_median": None, "implied_tick_hz": None,
             "radar_dt_sd_us": None, "host_dt_sd_us": None,
             "host_dt_median_ms": None, "host_dt_mean_ms": None,
             "frame_s": frame_s, "frame_s_source": "cfg" if frame_s else None,
             "batched_reads": sum(1 for f in frames if f.get("batch_n", 1) > 1)}
    if len(frames) < 3:
        return empty

    cyc = [f["cycles"] for f in frames]
    ta = [f["t_arr_mono"] for f in frames]
    fn = [f["frame"] for f in frames]
    # Jitter is only meaningful across CONSECUTIVE frames. A dropped frame makes
    # dt jump by a whole period, and averaging that in reports a clock as noisy
    # when what actually happened is a gap -- two different faults that need
    # two different fixes, so they are counted separately.
    step = [b - a == 1 for a, b in zip(fn, fn[1:])]
    dc = [(b - a) for (a, b), s in zip(zip(cyc, cyc[1:]), step) if s]
    dt = [(b - a) for (a, b), s in zip(zip(ta, ta[1:]), step) if s]
    gaps = sum(1 for s in step if not s)
    out = dict(empty, frame_number_gaps=gaps)
    # No two frame numbers consecutive: every frame was dropped. That is a real
    # session -- and precisely the one someone opens this tool to diagnose -- so
    # it reports the gap count rather than raising IndexError on an empty list.
    if not dc:
        return out

    def sd(v):
        m = sum(v) / len(v)
        return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5

    def med(v):
        return sorted(v)[len(v) // 2]

    dt_med = med(dt)
    # MEAN, not median, when falling back to the host clock. Host arrivals are
    # bimodal -- measured ~1/3 at 90 ms and ~2/3 at 105 ms -- so the median sits
    # on the larger mode (104.9 ms) and understates the tick rate by 5%. The mean
    # is 100.007 ms, i.e. the host is locked to the radar on average even though
    # no single arrival is. Still a fallback: deriving the period from the host
    # clock and then using it to certify the radar clock against the host clock
    # is circular, which is why the provenance is printed.
    dt_mean = sum(dt) / len(dt)
    period = frame_s if frame_s else dt_mean
    cyc_med = med(dc)
    rate = (cyc_med / period) if (cyc_med and period) else None
    out.update(cycles_per_frame_median=cyc_med,
               implied_tick_hz=rate,
               frame_s=period,
               frame_s_source=("cfg" if frame_s else "host mean (circular)"),
               radar_dt_sd_us=(sd([d / rate * 1e6 for d in dc]) if rate else None),
               host_dt_sd_us=sd([d * 1e6 for d in dt]),
               host_dt_median_ms=dt_med * 1e3,
               host_dt_mean_ms=dt_mean * 1e3)
    return out


def resolve_grid(session, frames, cfg_path=None, v_max=None):
    """Which Doppler axis to score this session on, and where it came from.

    Order: explicit --cfg, then the config the session recorded, then inference
    from the reported velocities, then the stock default. The provenance is
    returned alongside and MUST be printed -- scoring a session on the wrong
    grid is silent, and the only defence is saying out loud which one was used.
    """
    if cfg_path:
        g = chirp_geometry.from_cfg_text(open(cfg_path).read())
        if g:
            return (rs.DopplerGrid(g["bin_mps"], g["n_bins"],
                                   source=os.path.basename(cfg_path)),
                    "--cfg %s" % os.path.basename(cfg_path))

    meta_p = os.path.join(session, "meta.json")
    if os.path.exists(meta_p):
        try:
            rc = json.load(open(meta_p)).get("radar_cfg") or {}
        except ValueError:
            rc = {}
        text = rc.get("text") or "\n".join(rc.get("lines") or [])
        if text:
            g = chirp_geometry.from_cfg_text(text)
            if g:
                tag = rc.get("name", "?")
                if not rc.get("sent", True):
                    tag += " (declared, NOT sent by the recorder)"
                return (rs.DopplerGrid(g["bin_mps"], g["n_bins"], source=tag),
                        "session meta.json: %s" % tag)

    vals = [p[3] for f in frames for p in f.get("pts", [])]
    got = rs.infer_doppler_grid(vals)
    if got is not None:
        return got, ("INFERRED from %d Doppler samples -- unverified. If the "
                     "scene never reached the outer bins, n_bins is "
                     "underestimated and wrap detection turns over-eager."
                     % len(vals))

    return rs.STOCK_GRID, "STOCK DEFAULT -- this session did not record its cfg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?")
    ap.add_argument("--cfg", help="score against this .cfg's Doppler grid")
    ap.add_argument("--v-max", type=float, default=None,
                    help="override the unambiguous velocity (m/s)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for s in sessions():
            n = 0
            p = os.path.join(s, "radar_frames.jsonl")
            if os.path.exists(p):
                n = sum(1 for _ in open(p))
            print("%-60s %6d radar frames" % (s, n))
        return 0

    session = args.session or (sessions()[0] if sessions() else None)
    if not session:
        print("no sessions under %s" % RECORDINGS)
        return 1
    frames, err = load(session)
    if err:
        print("%s: %s" % (session, err))
        return 1

    print("session : %s" % session)
    meta_p = os.path.join(session, "meta.json")
    if os.path.exists(meta_p):
        m = json.load(open(meta_p))
        print("started : %s" % m.get("started_iso"))

    ck = clock_report(frames)

    def fmt(key, spec, scale=1.0):
        v = ck.get(key)
        return "n/a" if v is None else (spec % (v * scale))

    print("\n== clocks ==")
    print("  radar frames            : %d  (frame-number gaps: %s)"
          % (ck["frames"], fmt("frame_number_gaps", "%d")))
    print("  cycles per frame        : %s   (assumption-free)"
          % fmt("cycles_per_frame_median", "%d"))
    print("  implied radar tick rate : %s   (frame period %s from %s)"
          % (fmt("implied_tick_hz", "%.3f MHz", 1e-6),
             fmt("frame_s", "%.1f ms", 1e3), ck.get("frame_s_source") or "n/a"))
    print("  radar dt jitter         : %s sd   <- its own counter, dates the "
          "measurement" % fmt("radar_dt_sd_us", "%.1f us"))
    # The host sd is NOT a link-quality number. Measured on a real session it is
    # bimodal -- ~1/3 of arrivals at 90 ms and ~2/3 at 105 ms, summing to 300 ms
    # per 3 frames -- so it reads as huge scatter while the mean is locked to the
    # radar. Print the median beside it, or the sd alone invites "the link is
    # broken" when what is really there is a quantised read schedule.
    print("  host arrival dt         : %s sd, median %s   <- delivery, not "
          "measurement" % (fmt("host_dt_sd_us", "%.1f us"),
                           fmt("host_dt_median_ms", "%.2f ms")))
    print("  reads yielding >1 frame : %d" % ck["batched_reads"])

    grid, provenance = resolve_grid(session, frames, args.cfg, args.v_max)
    if args.v_max:
        grid = rs.DopplerGrid(2.0 * args.v_max / grid.n_bins, grid.n_bins,
                              source="--v-max %.4f" % args.v_max)
        provenance = "--v-max %.4f on %d bins" % (args.v_max, grid.n_bins)
    print("\n== doppler grid ==")
    print("  bin %.6f m/s   bins %d   v_max %.4f   fold %.4f"
          % (grid.bin_mps, grid.n_bins, grid.v_max, grid.fold))
    print("  source: %s" % provenance)

    per = radar_metrics.score_frames(frames, v_max=None, grid=grid)
    agg = radar_metrics.aggregate(per, v_max=grid.v_max)

    print("\n== yield ==")
    print("  raw points/frame        : median %.1f  p10 %.1f  mean %.2f"
          % (agg["n_raw"]["median"], agg["n_raw"]["p10"], agg["n_raw"]["mean"]))
    print("  frames solved           : %d/%d (%.0f%%)"
          % (agg["frames_solved"], agg["frames"], 100 * agg["solve_rate"]))
    print("  frames all-zero-Doppler : %.0f%%  (scene says the rig is still)"
          % (100 * agg["stationary_frac"]))

    rows, overall = radar_metrics.verdict(agg)
    print("\n== GO / NO-GO ==")
    print("  %-26s %-9s %-46s %s" % ("metric", "status", "value", "threshold"))
    for r in rows:
        print("  %-26s %-9s %-46s %s"
              % (r["metric"], r["status"], r["value"], r["note"]))
    print("\n  OVERALL: %s" % overall)
    if overall == "PARTIAL":
        print("  (UNTESTED metrics need a walked session with people moving;")
        print("   a static recording cannot exercise aliasing or rejection.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
