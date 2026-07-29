"""Stage-1 feasibility scorer: can this radar support Doppler ego-velocity?

Pure. Takes decoded radar frames, returns numbers and a verdict. No serial, no
files, no plotting -- `radar_replay.py` handles I/O so this stays testable.

The thresholds are not taste. Each one is the point past which the estimator
stops being able to tell a good answer from a bad one:

  n_static   Two points determine a 2-D velocity, so at 2 there is no residual
             and no way to know the solve was wrong. Three is the minimum with
             any self-check; five gives RANSAC something to work with.
  az_spread  Cross-track velocity is only observable through bearing diversity.
             With every return straight ahead the lateral component is
             unconstrained no matter how many points there are.
  gdop       Directly multiplies velocity error. 6 turns a 36 mm/s Doppler
             quantum into 216 mm/s.
  aliasing   The one that is catastrophic AND silent: past v_max the Doppler
             wraps and the trajectory reverses with no residual to show for it.
  resid_rms  Reported beside n_static ALWAYS. Lowering the CFAR threshold
             inflates the point count with noise; if count improves while
             residual worsens, nothing was gained and the pair says so.

A NOTE ON WHAT A STATIC-SCENE SESSION CAN AND CANNOT DECIDE. With the rig on a
desk, every return is world-fixed and zero-Doppler, so it measures point yield,
bearing spread, geometry and the zero-velocity bias directly -- metrics 1, 2, 3
and 5. It says nothing about aliasing or moving-object rejection, which need a
walked session with people in the room. `verdict()` reports which metrics were
actually exercised rather than passing the ones it never tested.
"""
import math

import ego_velocity as ev
import radar_static as rs

THRESHOLDS = {
    "n_static":   {"median_min": 5.0, "p10_min": 3.0},
    "az_spread":  {"median_min": 60.0, "p10_min": 30.0},
    "gdop":       {"median_max": 3.0, "p90_max": 6.0},
    "zero_bias":  {"max_mps": 0.005},          # 60 s still -> <0.30 m of drift
    "alias_rate": {"max_frac": 0.01},
    "inlier_frac": {"median_min": 0.6},
    "resid_rms":  {"max_mps": 0.053},          # 1.5 x the 35 mm/s quantum
                                               # (was 0.054 off a 36 mm/s quantum,
                                               #  which came from the old lambda)
    # How much of a session has to actually exercise the residual before its
    # median is worth reading. BOTH conditions, not either: a count floor alone
    # passes a session that is 200 frames of motion and 9000 of stillness on the
    # strength of the 200, and a fraction floor alone passes a 60-frame clip that
    # is all motion but too short to have a stable median. Below one frame in
    # five, the median is describing the still part of the recording.
    "exercised":  {"min_frames": 100, "min_frac": 0.20},
}


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round((len(s) - 1) * p))))
    return s[k]


def _summary(vals):
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "median": _pct(vals, 0.5), "p10": _pct(vals, 0.10),
            "p90": _pct(vals, 0.90), "mean": sum(vals) / len(vals)}


def score_frames(frames, v_max=ev.V_MAX_DEFAULT, grid=None):
    """frames: iterable of dicts with 'pts' (and optional 'snr'/'noise').

    `grid` is the session's radar_static.DopplerGrid; None means the stock
    config. It is passed to `condition` and rides on the points from there, so
    no other signature in the solve path has to know about it.

    Returns per-frame records plus aggregate metrics.
    """
    per = []
    prior = 0.0
    for fr in frames:
        pts = rs.condition(fr["pts"], fr.get("snr"), fr.get("noise"), grid=grid)
        diag = rs.frame_diagnostics(pts)
        sol = ev.solve(pts, prior_speed=prior, v_max=v_max)
        if sol.ok and sol.v:
            prior = sol.speed
        # The consensus solve() ACTUALLY used. This used to be recomputed here
        # with a per-point band 3*p.sigma(prior) against solve's global
        # max(3*max(sigma), MIN_INLIER_MPS) -- and with `prior` already advanced
        # to this frame -- so gdop and az_spread_static described a different
        # point set from the one inlier_frac and n_static counted. One row, two
        # inlier definitions. Now there is one.
        inl = [pts[k] for k in sol.in_idx]
        # A residual is only evidence where there was something to fit. Two
        # independent degeneracies make resid_rms exactly 0 while proving
        # nothing: no motion (b = 0, so v = 0 fits perfectly), and no redundancy
        # (2 inliers, 2 unknowns, an exact fit whatever the Doppler). Test the
        # INLIERS, not the raw points -- an outlier's Doppler never entered the
        # residual.
        informative = (sol.ok and sol.n_in >= 3
                       and any(pts[k].vr != 0.0 for k in sol.in_idx))
        per.append({
            "n_raw": len(pts),
            "n_static": sol.n_in if sol.ok else 0,
            "az_spread_all": diag["az_spread"],
            "az_spread_static": rs.azimuth_spread(inl),
            "gdop": rs.gdop(inl) if len(inl) >= 2 else None,
            "speed": sol.speed if sol.v else None,
            "resid_rms": sol.resid_rms,
            "resid_informative": informative,
            "aliased": sol.aliased,
            "alias_margin": sol.alias_margin,
            "inlier_frac": (sol.n_in / sol.n_tot) if sol.n_tot else 0.0,
            "ok": sol.ok,
            "flags": sol.flags,
            "stationary": ev.is_stationary(pts),
            "v": sol.v,
        })
    return per


def aggregate(per, v_max=ev.V_MAX_DEFAULT):
    ok = [r for r in per if r["ok"]]
    speeds = [r["speed"] for r in ok if r["speed"] is not None]
    gd = [r["gdop"] for r in ok if r["gdop"] is not None]
    # Frames whose residual and inlier fraction were actually put at risk. See
    # score_frames for the two degeneracies this excludes. Everything else in
    # this function is computed over `ok`; these two are not, and that is the
    # whole point -- a metric averaged over frames that cannot falsify it is
    # arithmetic wearing the costume of evidence.
    inform = [r for r in per if r["resid_informative"]]
    moving = [r for r in per if not r["stationary"]]
    out = {
        "frames": len(per),
        "frames_solved": len(ok),
        "solve_rate": (len(ok) / len(per)) if per else 0.0,
        "n_raw": _summary([r["n_raw"] for r in per]),
        "n_static": _summary([r["n_static"] for r in per]),
        "az_spread_static": _summary([r["az_spread_static"] for r in ok]),
        "gdop": _summary(gd),
        "resid_rms": _summary([r["resid_rms"] for r in inform
                               if r["resid_rms"] is not None]),
        "inlier_frac": _summary([r["inlier_frac"] for r in inform]),
        "resid_informative_n": len(inform),
        "resid_informative_frac": (len(inform) / len(per)) if per else 0.0,
        "moving_frames": len(moving),
        # EXACT, and denominated over ALL frames. Two changes from what this used
        # to be. It counted `speed > 0.85*v_max` on the REPORTED speed, which
        # folds back down past the limit -- a band detector that went silent
        # exactly where it mattered. And it divided by the solved frames, so
        # aliasing severe enough to destroy the consensus deleted itself from
        # the metric. Now it counts frames whose winning hypothesis needed a
        # whole fold added to a measured radial velocity, over every frame.
        "alias_rate": (sum(1 for r in per if r["aliased"]) / len(per)
                       if per else None),
        "alias_unresolved_frac": (sum(1 for r in per
                                      if "alias_unresolved" in r["flags"])
                                  / len(per) if per else 0.0),
        # The canary for aliasing that broke the solve entirely. It belongs
        # beside alias_rate, not buried: a rising no-consensus rate on a moving
        # session is what severe wrapping looks like from the outside.
        "no_consensus_frac": (sum(1 for r in per
                                  if "no_consensus" in r["flags"]) / len(per)
                              if per else 0.0),
        "stationary_frac": (sum(1 for r in per if r["stationary"]) / len(per)
                            if per else 0.0),
    }
    # Zero-velocity bias: only meaningful where the scene says we are not
    # moving. Reported as the mean speed the estimator claims while every
    # return is in the zero-Doppler bin -- ideally 0, and it integrates
    # straight into position error if it is not.
    # SIGNED, and per axis. The old form averaged math.hypot(vx, vy), which is
    # non-negative by construction: for zero-mean noise it returns the Rayleigh
    # mean sigma*sqrt(pi/2), and the verdict then read that as drift. Only a
    # signed, direction-stable error integrates into position; a magnitude does
    # not. Filtered on `ok` too -- it used to admit solves every other row
    # discards.
    still = [r["v"] for r in per
             if r["stationary"] and r["ok"] and r["v"] is not None]
    if still:
        bx = sum(v[0] for v in still) / len(still)
        by = sum(v[1] for v in still) / len(still)
        out["zero_bias_xy"] = (bx, by)
        out["zero_bias_mps"] = math.hypot(bx, by)   # of the MEAN, not mean of |v|
    else:
        out["zero_bias_xy"] = None
        out["zero_bias_mps"] = None
    out["zero_bias_n"] = len(still)
    return out


def verdict(agg):
    """Per-metric GO / NO-GO / UNTESTED, plus an overall call."""
    T = THRESHOLDS
    rows = []

    def add(name, value, passed, note=""):
        rows.append({"metric": name, "value": value,
                     "status": ("UNTESTED" if passed is None
                                else ("GO" if passed else "NO-GO")),
                     "note": note})

    ns = agg["n_static"]
    add("static points/frame",
        ("median %.1f  p10 %.1f" % (ns["median"], ns["p10"])) if ns.get("n") else "n/a",
        (ns["median"] >= T["n_static"]["median_min"]
         and ns["p10"] >= T["n_static"]["p10_min"]) if ns.get("n") else None,
        "need median>=5 p10>=3")

    az = agg["az_spread_static"]
    add("azimuth spread (deg)",
        "median %.1f  p10 %.1f" % (az.get("median", 0), az.get("p10", 0)) if az.get("n") else "n/a",
        (az["median"] >= T["az_spread"]["median_min"]
         and az["p10"] >= T["az_spread"]["p10_min"]) if az.get("n") else None,
        "need median>=60 p10>=30")

    gd = agg["gdop"]
    add("velocity GDOP",
        "median %.2f  p90 %.2f" % (gd.get("median", 0), gd.get("p90", 0)) if gd.get("n") else "n/a",
        (gd["median"] <= T["gdop"]["median_max"]
         and gd["p90"] <= T["gdop"]["p90_max"]) if gd.get("n") else None,
        "need median<=3 p90<=6")

    zb = agg["zero_bias_mps"]
    zxy = agg.get("zero_bias_xy")
    add("zero-velocity bias (m/s)",
        ("%.4f  (vx %+.4f vy %+.4f) over %d still frames"
         % (zb, zxy[0], zxy[1], agg["zero_bias_n"])) if zb is not None
        else "no still frames",
        (abs(zb) <= T["zero_bias"]["max_mps"]) if zb is not None else None,
        "need |bias|<=0.005 -> <0.30 m drift in 60 s")

    # A session with no motion cannot exercise aliasing, and one whose residuals
    # were never at risk cannot exercise the residual or the inlier fraction.
    # Reporting GO on either is the module's own stated anti-pattern arriving
    # from the other direction: not "the count improved while the residual got
    # worse", but "the residual is perfect because nothing was ever fitted".
    exercised = (agg["resid_informative_n"] >= T["exercised"]["min_frames"]
                 and agg["resid_informative_frac"] >= T["exercised"]["min_frac"])
    inform_note = ("%d frames (%.1f%%) exercise it"
                   % (agg["resid_informative_n"],
                      100 * agg["resid_informative_frac"]))
    moving_enough = agg["moving_frames"] >= T["exercised"]["min_frames"]

    ar = agg["alias_rate"]
    add("aliasing rate (unwrapped)",
        ("%.2f%%  unresolved %.2f%%  no-consensus %.2f%%"
         % (100 * ar, 100 * agg["alias_unresolved_frac"],
            100 * agg["no_consensus_frac"])) if ar is not None else "n/a",
        (ar <= T["alias_rate"]["max_frac"]) if (ar is not None and moving_enough)
        else None,
        "need <1%% of ALL frames; %d moving frames" % agg["moving_frames"])

    inf = agg["inlier_frac"]
    add("RANSAC inlier fraction",
        ("median %.2f  (%s)" % (inf["median"], inform_note)) if inf.get("n")
        else "n/a (%s)" % inform_note,
        (inf["median"] >= T["inlier_frac"]["median_min"])
        if (inf.get("n") and exercised) else None,
        "need median>=0.6 in a populated room")

    rr = agg["resid_rms"]
    add("inlier residual RMS (m/s)",
        ("median %.4f  (%s)" % (rr["median"], inform_note)) if rr.get("n")
        else "n/a (%s)" % inform_note,
        (rr["median"] <= T["resid_rms"]["max_mps"])
        if (rr.get("n") and exercised) else None,
        "need <=%.3f; read together with point count" % T["resid_rms"]["max_mps"])

    statuses = [r["status"] for r in rows]
    overall = ("NO-GO" if "NO-GO" in statuses
               else ("GO" if "UNTESTED" not in statuses else "PARTIAL"))
    return rows, overall
