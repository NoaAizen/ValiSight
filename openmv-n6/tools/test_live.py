#!/usr/bin/env python3
"""Exercise the live viewer against recorded frames, with no board attached.

    ./test_live.py ../captures/handwave3

live.py is the only thing that drives fusion.c from outside the C - through a
hand-written ctypes mirror of two structs that fusion_init() memsets through. A
mirror that has drifted from fusion.h is a heap overwrite, not a wrong-looking
picture, so it is worth having a check that does not need the board to run.

Everything here is offline: the Streamer is never started, frames come off disk,
and the HTTP handlers are driven directly. What it does not cover is the serial
bring-up, which needs hardware.
"""
import argparse
import glob
import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print("  %-52s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


class Args:
    """Stands in for the argparse namespace Pipeline expects.

    Add a field here whenever live.py grows a --flag that Pipeline reads, or this
    stub raises AttributeError before the first check runs.
    """
    gain, eps, radius, agc = 200, 200, 4, 0
    palette, warp = "ironbow", None
    emissivity, reflected = 1.0, 20.0
    # Off, matching fusion_default_cfg: the mirror check below compares layouts,
    # and the 768KB the visible filter allocates has nothing to do with that.
    y_knee, y_frames = 0, 8


def load_pair(d):
    for jp in sorted(glob.glob(os.path.join(d, "*.json"))):
        st = jp[:-5]
        rgb = st + "_rgb0.raw"
        if not os.path.exists(rgb):
            rgb = st + "_rgb.raw"
        th = st + "_thermal.raw"
        if os.path.exists(rgb) and os.path.exists(th):
            return open(rgb, "rb").read(), open(th, "rb").read(), json.load(open(jp))
    raise SystemExit("no frame pair in %s" % d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", nargs="?", default="../captures/handwave3")
    args = ap.parse_args()

    y, thermal, meta = load_pair(args.capture)
    print("frames from %s: %d B visible, %d B thermal, range %d..%dC"
          % (args.capture, len(y), len(thermal), meta["tmin"], meta["tmax"]))

    # 1. Constructing the Pipeline is itself the struct-layout check: it compares
    #    ctypes.sizeof against fusion_sizeof_state/_cfg and refuses to start on a
    #    mismatch. Reaching this line at all means the mirror still matches.
    pipe = live.Pipeline(Args())
    check("ctypes mirror matches fusion.h", True,
          "fusion_t %d B, fusion_cfg_t %d B" % (pipe.lib.fusion_sizeof_state(),
                                                pipe.lib.fusion_sizeof_cfg()))

    pipe.set_range(meta["tmin"], meta["tmax"])

    # 2. a temperature query before any frame must refuse, not invent a number
    check("no reading before a frame has been processed", pipe.temp_at(320, 200) is None)
    check("no frame statistics before a frame either", pipe.frame_stats() is None)

    rgb = pipe.process(y, thermal)
    check("a frame comes back the right shape", rgb.shape == (live.OUT_H, live.OUT_W, 3),
          str(rgb.shape))

    # 3. readings must land inside the sensor's own range, since that is the only
    #    thing the code -> temperature mapping is built from
    pts = [pipe.temp_at(x, yy) for x in range(40, live.OUT_W, 97)
           for yy in range(40, live.OUT_H, 71)]
    got = [p for p in pts if p]
    lo, hi = meta["tmin"], meta["tmax"]
    check("every reading lands inside the sensor range",
          got and all(lo - 0.5 <= p["c"] <= hi + 0.5 for p in got),
          "%d readings, %.1f..%.1f C" % (len(got), min(p["c"] for p in got),
                                         max(p["c"] for p in got)))

    st = pipe.frame_stats()
    check("frame statistics are consistent with the readings",
          st and st["min"] <= min(p["c"] for p in got) + 0.1
          and st["max"] >= max(p["c"] for p in got) - 0.1,
          "min %.1f max %.1f delta %.1f C" % (st["min"], st["max"], st["delta"]))

    # 4. the dead rows this sensor produces must be flagged, not quoted
    rebuilt = pipe.f.rows_rebuilt
    flagged = sum(1 for p in got if p["repaired"])
    check("rebuilt rows are reported and readings off them flagged",
          (rebuilt == 0) == (flagged == 0),
          "%d rows rebuilt, %d of %d readings flagged" % (rebuilt, flagged, len(got)))

    # 5. the emissivity control has to actually reach the C
    before = pipe.temp_at(320, 200)
    pipe.set_emissivity(0.5, 20.0)
    after = pipe.temp_at(320, 200)
    check("emissivity changes the corrected reading but not the raw one",
          before and after and abs(after["raw"] - before["raw"]) < 0.01
          and after["c"] > before["c"] + 0.5,
          "%.2f -> %.2f C (raw %.2f)" % (before["c"], after["c"], after["raw"]))
    pipe.set_emissivity(1.0, 20.0)

    # 6. palette switching must change the picture, and black-hot must invert the
    #    detail sign with it or the embossed texture fights the base tone
    base = pipe.process(y, thermal).copy()
    pipe.set_palette("white")
    white = pipe.process(y, thermal).copy()
    pipe.set_palette("black")
    black = pipe.process(y, thermal).copy()
    check("palette switch changes the output", np.abs(white.astype(int) - base).mean() > 5,
          "mean px change %.1f" % np.abs(white.astype(int) - base).mean())
    check("black-hot is roughly the inverse of white-hot",
          abs(white.mean() + black.mean() - 255) < 30,
          "means %.0f / %.0f" % (white.mean(), black.mean()))
    check("black-hot flips the detail sign with the ramp", pipe.f.cfg.detail_invert == 1)
    pipe.set_palette("ironbow")
    check("switching back clears the detail inversion", pipe.f.cfg.detail_invert == 0)

    # 7. the registration views. These exist because the fused image cannot show
    #    whether the warp is right - the guided filter sharpens edges into place
    #    either way - so each one has to genuinely differ from the fused frame.
    fused = pipe.process(y, thermal).copy()
    yarr = np.frombuffer(y, np.uint8).reshape(live.OUT_H, live.OUT_W)
    cover, treg = pipe.cover_grid(), pipe.treg_grid()

    check("coverage mask has the grid's shape",
          cover.shape == (pipe.f.cfg.low_h, pipe.f.cfg.low_w), str(cover.shape))
    check("registered thermal plane is not empty", treg.ptp() > 0,
          "range %d..%d" % (treg.min(), treg.max()))

    vis = live.compose("visible", fused, yarr)
    check("the visible view is exactly the source luma, in grey",
          np.array_equal(vis[..., 0], yarr) and np.array_equal(vis[..., 1], vis[..., 2]))

    on = live.compose("blink", fused, yarr, phase=True)
    off = live.compose("blink", fused, yarr, phase=False)
    check("blink alternates between fused and visible",
          np.array_equal(on, fused) and np.array_equal(off[..., 0], yarr))

    m0 = live.compose("mix", fused, yarr, mix=0)
    m100 = live.compose("mix", fused, yarr, mix=100)
    m50 = live.compose("mix", fused, yarr, mix=50)
    check("mix 0 and 100 are the two endpoints",
          np.abs(m0[..., 0].astype(int) - yarr).max() <= 1
          and np.abs(m100.astype(int) - fused).max() <= 1)
    check("mix 50 sits between them",
          m0.astype(int).mean() < m50.mean() < m100.astype(int).mean()
          or m0.astype(int).mean() > m50.mean() > m100.astype(int).mean(),
          "%.1f / %.1f / %.1f" % (m0.mean(), m50.mean(), m100.mean()))

    excl = pipe.repaired_grid()
    ed = live.compose("edges", fused, yarr, treg=treg, exclude=excl)
    painted = np.all(ed == (80, 255, 255), axis=2)
    check("the edges view draws thermal edges over the visible image",
          0.005 < painted.mean() < 0.15,
          "%.1f%% of the frame painted" % (100 * painted.mean()))

    # A rebuilt row is a ramp spliced into real data, and the splice has a slope
    # discontinuity at each end. That is a gradient, so the overlay draws it -
    # measured, the two busiest painted rows in the frame were splice boundaries
    # and a fifth of the drawn edges were this artefact. An overlay meant to prove
    # the warp is right must not invent its own lines.
    if pipe.f.rows_rebuilt:
        check("rebuilt rows are marked on the grid", 0 < excl.mean() < 0.5,
              "%.0f%% of grid cells" % (100 * excl.mean()))
        loose = live.thermal_edges(treg, live.OUT_W, live.OUT_H)
        tight = live.thermal_edges(treg, live.OUT_W, live.OUT_H, exclude=excl)
        check("excluding rebuilt rows removes drawn edges", tight.sum() < loose.sum(),
              "%.2f%% -> %.2f%% of the frame" % (100 * loose.mean(), 100 * tight.mean()))

        grown = cv2.dilate(excl.astype(np.uint8), np.ones((5, 5), np.uint8))
        banned = cv2.resize(grown, (live.OUT_W, live.OUT_H),
                            interpolation=cv2.INTER_NEAREST) != 0
        check("no edge is drawn anywhere a rebuilt row could reach",
              not (tight & banned).any(), "%d px inside the excluded region"
              % int((tight & banned).sum()))
    # and they must be the *thermal* camera's edges, not the visible one's
    check("those edges come from the thermal layer, not the luma",
          not np.array_equal(painted, live.thermal_edges(
              cv2.resize(yarr, (cover.shape[1], cover.shape[0])),
              live.OUT_W, live.OUT_H)))

    # a flat thermal frame has no edges to draw - it must paint nothing rather
    # than threshold noise into a full-frame mess
    flat = np.full_like(treg, 128)
    check("a featureless thermal frame draws no edges",
          not live.thermal_edges(flat, live.OUT_W, live.OUT_H).any())

    # The placeholder warp stretches the thermal frame over everything, so there
    # is no footprint edge to draw - and a line appearing anyway would be a lie
    # about where the thermal camera stops seeing.
    outlined = live.compose("fused", fused, yarr, cover=cover)
    band = np.all(outlined == (255, 210, 40), axis=2)
    check("full coverage draws no footprint line",
          cover.all() and not band.any(),
          "coverage %.0f%%, %d px drawn" % (100 * cover.mean(), band.sum()))

    # so the drawing itself is checked against a mask that does have an edge,
    # the shape a real calibrated warp produces
    partial = np.zeros_like(cover)
    partial[10:-10, 20:-20] = 1
    ring = live.coverage_outline(partial, live.OUT_W, live.OUT_H)
    check("a partial footprint is outlined as a boundary, not filled",
          0.001 < ring.mean() < 0.08 and not ring[live.OUT_H // 2, live.OUT_W // 2],
          "%.2f%% of the frame" % (100 * ring.mean()))
    check("compose never writes into the pipeline's own frame",
          np.array_equal(fused, pipe.process(y, thermal)))

    # 8. the health panel. Each check has to fire on its own condition and stay
    #    quiet otherwise - a panel that is always green, or always amber, tells
    #    you nothing and would be worse than not having one.
    pipe.set_palette("ironbow")
    pipe.set_emissivity(1.0, 20.0)
    pipe.process(y, thermal)
    # real wall-clock, because the HTTP handler calls health() with time.time()
    # and a synthetic epoch would make every request look like a stalled stream
    now = time.time()
    good = {"range": (lo, hi), "fps": 8.8, "last_frame_t": now - 0.1,
            "torn_window": [False] * 30, "rows_window": [pipe.f.rows_rebuilt] * 30}

    def levels(state, at=now):
        return {c["name"]: c["level"] for c in live.health(pipe, state, at)}

    base = levels(good)
    check("a healthy stream reports no failures", "fail" not in base.values(),
          " ".join("%s=%s" % kv for kv in base.items()))
    check("stream, tearing, coverage and range all report",
          {"stream", "tearing", "coverage", "range"} <= set(base),
          "got %s" % sorted(base))

    check("a stalled stream fails", levels(dict(good), now + 60.0)["stream"] == "fail")
    check("a reported stream error fails",
          levels(dict(good, error="boom"))["stream"] == "fail")
    check("a slow stream warns", levels(dict(good, fps=3.0))["stream"] == "warn")
    check("heavy tearing fails",
          levels(dict(good, torn_window=[True] * 30))["tearing"] == "fail")
    check("occasional tearing warns",
          levels(dict(good, torn_window=[True] * 3 + [False] * 27))["tearing"] == "warn")

    # the one that matters most: a dead-row count that moves frame to frame means
    # the detector is following the scene, which is the exact failure the
    # flat+lifted test was written to avoid
    check("a varying dead-row count fails",
          levels(dict(good, rows_window=[14, 14, 9, 14]))["dead rows"] == "fail")
    check("a steady dead-row count does not fail",
          levels(dict(good, rows_window=[14] * 30))["dead rows"] != "fail")

    check("a missing sensor range fails", levels(dict(good, range=(0, 0)))["range"] == "fail")
    check("a coarse range warns", levels(dict(good, range=(-10, 140)))["range"] == "warn")
    check("a narrow range is fine", levels(dict(good, range=(20, 40)))["range"] == "ok")

    check("the placeholder warp warns about registration",
          base.get("registration") == "warn")

    pipe.tune(agc_permille=20)
    check("scene AGC warns that tone is scene-relative", levels(good).get("agc") == "warn")
    pipe.tune(agc_permille=0)
    check("...and stops warning when it is off", "agc" not in levels(good))

    pipe.set_emissivity(0.9, 20.0)
    check("a non-unity emissivity is surfaced", levels(good).get("emissivity") == "warn")
    pipe.set_emissivity(1.0, 20.0)

    # clipping is judged after the repair, or the dead rows would be counted as a
    # saturated scene and the panel would cry wolf on every frame
    prep = pipe.prep_frame()
    pinned = float(((prep == 0) | (prep == 255)).mean())
    check("clipping is measured on the repaired frame", pinned < 0.05 and "clipping" in base,
          "%.2f%% pinned, level %s" % (100 * pinned, base.get("clipping")))

    # Built rather than taken from the capture: whether this unit's dead rows have
    # clipped at 255 yet depends on the frame's own level, and the first frame of
    # a sequence usually has them at ~238. Saturating them explicitly makes the
    # point deterministic - 14 rows is 11% of the frame, which on the raw data
    # would trip the clipping check every frame and train the operator to ignore
    # the panel.
    clipped_raw = np.frombuffer(thermal, np.uint8).reshape(meta["th_h"], meta["th_w"]).copy()
    clipped_raw[[6, 12, 28, 35, 36, 55, 56, 57, 58, 59, 60, 61, 62, 63]] = 255
    pipe.process(y, clipped_raw.tobytes())
    after_repair = float(((pipe.prep_frame() == 0) | (pipe.prep_frame() == 255)).mean())
    before = float(((clipped_raw == 0) | (clipped_raw == 255)).mean())
    check("the repair is what keeps the clipping check from crying wolf",
          before > 0.05 and after_repair < 0.01,
          "raw %.1f%% -> repaired %.2f%%" % (100 * before, 100 * after_repair))
    pipe.process(y, thermal)

    # 9. the HTTP surface the browser actually talks to
    srv = ThreadingHTTPServer(("127.0.0.1", 0), live.make_handler(good, pipe))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = "http://127.0.0.1:%d" % srv.server_address[1]

    def get(path):
        with urlopen(base_url + path, timeout=5) as r:
            return r.status, r.read().decode()

    try:
        code, body = get("/")
        check("/ serves the page", code == 200 and "<img" in body)

        code, body = get("/temp?x=320&y=200")
        d = json.loads(body)
        check("/temp returns a reading", code == 200 and d["valid"] and lo - 1 <= d["c"] <= hi + 1,
              "%.2f C" % d["c"] if d["valid"] else "no coverage")

        code, body = get("/temp?x=99999&y=99999")
        check("/temp refuses an out-of-frame pixel", json.loads(body)["valid"] is False)

        code, body = get("/stats")
        d = json.loads(body)
        check("/stats returns the frame band",
              d["valid"] and d["delta"] >= 0 and d["max"] >= d["min"],
              "min %.1f max %.1f" % (d["min"], d["max"]))

        get("/set?palette=white&gain=300")
        check("/set reaches the pipeline",
              pipe.palette_name == "white" and pipe.f.cfg.detail_gain == 300,
              "palette %s, gain %d" % (pipe.palette_name, pipe.f.cfg.detail_gain))

        get("/set?emissivity=0.3&reflected=25")
        check("/set carries emissivity and reflected temperature",
              abs(pipe.eps - 0.3) < 1e-6 and abs(pipe.refl - 25.0) < 1e-6,
              "eps %.2f refl %.1f" % (pipe.eps, pipe.refl))

        # out-of-range emissivity must be clamped, not passed through: below 0.05
        # the inversion divides by almost nothing
        get("/set?emissivity=0.001")
        check("/set clamps an unusable emissivity", pipe.eps >= 0.05, "eps %.3f" % pipe.eps)

        get("/set?view=edges&mix=25&outline=1")
        check("/set carries the view controls",
              pipe.view == "edges" and pipe.mix == 25 and pipe.outline is True,
              "view %s, mix %d, outline %s" % (pipe.view, pipe.mix, pipe.outline))

        get("/set?view=nonsense")
        check("/set ignores an unknown view rather than breaking the stream",
              pipe.view == "edges")

        get("/set?view=fused&outline=0")

        code, body = get("/health")
        h = json.loads(body)
        check("/health returns the checks and an overall level",
              code == 200 and h["worst"] in ("ok", "warn", "fail")
              and len(h["checks"]) >= 5
              and all({"name", "level", "text"} <= set(c) for c in h["checks"]),
              "worst=%s, %d checks" % (h["worst"], len(h["checks"])))
        check("the overall level is the worst of the individual ones",
              h["worst"] == ("fail" if any(c["level"] == "fail" for c in h["checks"])
                             else "warn" if any(c["level"] == "warn" for c in h["checks"])
                             else "ok"))

        code, body = get("/stat")
        check("/stat reports the range and the warp state", code == 200 and "range" in body,
              body.strip())
    finally:
        srv.shutdown()

    print()
    if FAILS:
        print("FAILED: %s" % ", ".join(FAILS))
        return 1
    print("all viewer checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
