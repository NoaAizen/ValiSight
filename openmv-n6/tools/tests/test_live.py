#!/usr/bin/env python3
"""Exercise the live viewer against recorded frames, with no board attached.

    ./test_live.py                  # defaults to captures/handwave3

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
import shutil
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import live      # noqa: E402
import detect    # noqa: E402
import recorder  # noqa: E402

# Script-relative so the test passes from any CWD - the old "../captures"
# default silently depended on being run from tools/.
HANDWAVE3 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "captures", "handwave3")

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


def board_clock():
    """The cadence/skew panel, off a dict, plus the header that feeds it.

    Worth testing offline because every input is board-side and the failure is
    quiet: a wrong median here does not break the picture, it just misreports how
    far apart in time the two planes being fused were looked at.
    """
    import ast

    print("\nboard clock")
    src = (live.SETUP_CODE.replace("__TMIN__", "-10").replace("__TMAX__", "140")
           .replace("__AUTORANGE__", "True").replace("__Q__", "80"))
    try:
        ast.parse(src)
        ok = "#F %d %d %d %d %d" in src and "ticks_diff" in src
    except SyntaxError as e:
        ok = False
        src = str(e)
    check("the board code parses and emits both clocks in the header", ok)

    now = time.time()
    st = {"dt_window": [113, 114, 114, 115, 114, 1824, 114, 113],
          "skew_window": [11, 12, 12, 13, 12], "fps": 8.7,
          "last_ffc_t": now - 42, "ffcs": 3}
    t = live.timing(st, now)
    check("the FFC gap is kept out of the cadence median",
          t["thermal_ms"]["median"] == 114 and t["thermal_ms"]["n"] == 7
          and t["thermal_ms"]["max"] == 115,
          "median %d ms over %d frames" % (t["thermal_ms"]["median"], t["thermal_ms"]["n"]))
    check("cadence is reported as the sensor's own rate, not the link's",
          abs(t["thermal_fps"] - 8.77) < 0.01 and t["host_fps"] == 8.7,
          "sensor %.2f fps, link %.2f fps" % (t["thermal_fps"], t["host_fps"]))
    check("the skew is expressed as a share of one thermal period",
          abs(t["skew_frac"] - 12 / 114.0) < 1e-3, "%.1f%%" % (100 * t["skew_frac"]))
    check("a clock with no data reports nothing rather than zero",
          live.timing({}, now)["thermal_ms"] is None)

    # the health panel's reading of the same numbers
    pipe = live.Pipeline(Args())
    pipe.process(*load_pair(HANDWAVE3)[:2])
    names = lambda s: {c["name"]: c for c in live.health(pipe, s, now)}  # noqa: E731

    h = names(st)
    check("a healthy cadence and pairing read green",
          h["cadence"]["level"] == "ok" and h["pairing"]["level"] == "ok",
          "%s | %s" % (h["cadence"]["text"], h["pairing"]["text"]))
    check("a fresh FFC is surfaced, an old one is not",
          "ffc" not in h and "ffc" in names(dict(st, last_ffc_t=now - 1)))

    h = names(dict(st, skew_window=[40] * 8))
    check("a visible frame that lags most of a period warns",
          h["pairing"]["level"] == "warn", h["pairing"]["text"])
    h = names(dict(st, dt_window=[300] * 8))
    check("a thermal cadence far off 114 ms warns",
          h["cadence"]["level"] == "warn", h["cadence"]["text"])

    # the header the reader parses, including the resync case that made the
    # trailing fields optional in the first place
    def parse(line):
        f = line.split()
        return tuple(int(f[i]) if len(f) > i else None for i in (4, 5, 6))
    check("the header carries both clocks and the stall count",
          parse("#F 11024 19200 0 114 12 0") == (114, 12, 0))
    check("a truncated header does not raise, it reports no timing",
          parse("#F 11024 19200 0") == (None, None, None))

    # --- protecting the sensor from host load. The Lepton's one unrecoverable
    # failure is the collector firing while it is up, and the one way host load
    # reaches it is the board sitting in out.write() instead of snapshot().
    check("the write-stall limit keeps the sensor inside its tested envelope",
          "if stalls > 2:" in src and str(live.LEPTON_SAFE_GAP_MS) == "2500",
          "3 timeouts = 1500ms, envelope %dms" % live.LEPTON_SAFE_GAP_MS)

    h = names(dict(st, stalls=4))
    check("write stalls are reported as a threat to the sensor, not a slow link",
          h["thermal load"]["level"] == "warn", h["thermal load"]["text"])
    h = names(dict(st, starved=2, last_starve_ms=3100))
    check("a gap past the tested envelope fails outright",
          h["thermal load"]["level"] == "fail", h["thermal load"]["text"])
    check("a clean run says so rather than staying silent",
          names(st)["thermal load"]["level"] == "ok",
          names(st)["thermal load"]["text"])

    # the heap watch tightens as the margin closes, because near the floor the
    # measured drain rate is exactly what should not be trusted
    s = live.Streamer.__new__(live.Streamer)
    rate = lambda free: sum(  # noqa: E731
        live.Streamer._want_heap(type("S", (), {"state": {"heap_free": free}})(), b)
        for b in range(30))
    check("the heap is read every batch once the margin is thin",
          rate(live.HEAP_FLOOR) == 30 and rate(3 * live.HEAP_FLOOR) == 10
          and rate(20 * live.HEAP_FLOOR) == 3,
          "%d/30 near the floor, %d/30 mid, %d/30 with room"
          % (rate(live.HEAP_FLOOR), rate(3 * live.HEAP_FLOOR), rate(20 * live.HEAP_FLOOR)))
    check("a session with no reading yet takes one immediately",
          live.Streamer._want_heap(type("S", (), {"state": {}})(), 7) is True)


def host_soc():
    """The host panel, off synthetic readings.

    The thresholds are the whole content of this module - the sysfs reads either
    work or return None - and they encode a judgment that is easy to get wrong in
    the direction that matters: a busy host must not raise the same alarm as a
    starved sensor, or an operator learns to discount both. So the point of these
    checks is mostly that CPU and heat cannot reach 'fail', and memory can.
    """
    import soc as hostsoc

    print("\nhost soc")
    r = hostsoc.Soc().read()
    check("a reading comes back with the fields the panel draws",
          {"cpu_pct", "mem_avail_mb", "t_max", "ncpu"} <= set(r),
          "%d cores, %s MB free, %s C" % (r["ncpu"], r["mem_avail_mb"], r["t_max"]))

    def levels(**kw):
        base = {"cpu_pct": 10.0, "self_pct": 5.0, "ncpu": 6, "load1": 1.0,
                "mem_avail_mb": 4000, "mem_total_mb": 8000, "t_max": 45.0,
                "t_crit": 100.0, "power_w": None, "power": {}}
        base.update(kw)
        return {c["name"]: c["level"] for c in hostsoc.checks(base)}

    check("an idle host says so", levels() == {"host cpu": "ok", "host memory": "ok",
                                               "host thermal": "ok"})
    check("a saturated host warns but never fails - the board's clock owns that verdict",
          levels(cpu_pct=97.0)["host cpu"] == "warn")
    check("a host near its thermal trip warns rather than failing, for the same reason",
          levels(t_max=93.0)["host thermal"] == "warn"
          and levels(t_max=99.5)["host thermal"] == "warn"
          and levels(t_max=80.0)["host thermal"] == "ok")
    # The exception, and the reason for it: an OOM kill during a recording is not
    # a risk to the numbers, it is a session that ends without a finalised mp4.
    check("memory about to run out does fail", levels(mem_avail_mb=180)["host memory"] == "fail")
    check("a part with no critical trip still reports its temperature",
          levels(t_crit=None)["host thermal"] == "ok")
    check("a host with no power rail simply omits the row",
          "host power" not in levels() and "host power" in levels(power_w=7.2, power={}))


def failure_reporting():
    """The board's own exception must survive the port dying while it is read.

    Both halves of this were real losses. _dump printed the first 32 bytes of the
    residual buffer, and a traceback puts the one line worth having - the
    exception type - last; and _traceback let a read error propagate, so
    SerialException replaced the board fault it was in the middle of reporting.
    The two together turned `RuntimeError: Frame buffer overflow` into `device
    disconnected or multiple access on port?`.
    """
    import serial

    print("\nfailure reporting")
    buf = bytearray(b'  File "<stdin>", line 1, in <module>\r\n'
                    b'RuntimeError: Frame buffer overflow, try reducing the frame size\r\n')
    dump = live.Streamer._dump(buf)
    check("the buffer dump keeps the line that names the fault",
          "Frame buffer overflow" in dump and "line 1" in dump)

    class Dying(live.Streamer):
        def __init__(self, chunks):
            self.buf, self.chunks = bytearray(), list(chunks)
            self.last_line, self.last_line_t = b"", 0.0

        def _fill(self, timeout=10.0):
            if not self.chunks:
                raise serial.SerialException(
                    "device reports readiness to read but returned no data")
            self.buf += self.chunks.pop(0)
            return True

    got = Dying([b'  File "<stdin>", line 1, in <module>\r\n',
                 b'RuntimeError: Frame buffer overflow\r\n'])._traceback(
                     "\x04Traceback (most recent call last):")
    check("a port that dies mid-traceback still reports the board's exception",
          "Frame buffer overflow" in got and "port died" in got, got[-90:])

    got = Dying([b'OSError: 5\r\n\x04\x04>leftover'])._traceback(
        "\x04Traceback (most recent call last):")
    check("a traceback that arrives whole is not marked truncated",
          "OSError: 5" in got and "port died" not in got, got)

    d = live.Latest()
    d.put(1)
    d.put(2)
    check("the handoff slot keeps the newest frame and counts the drop",
          d.get(0.01) == 2 and d.dropped == 1)
    check("an empty slot returns None rather than blocking forever",
          d.get(0.01) is None)


def threading_and_detection(y, thermal):
    """Renderer + Detector driven off recorded frames, with no board attached.

    Worth having offline because the interesting failure is not the detector
    getting a box wrong - it is a box carrying a temperature that came from the
    wrong pixels. With the placeholder warp that is the *expected* state, so the
    check is that the flag saying so is present, not that it is absent.
    """
    print("\nrender thread and detection")
    pipe = live.Pipeline(Args())
    state, work = {}, live.Latest()

    try:
        det = detect.Detector(size=320)
    except FileNotFoundError as e:
        check("detector model is present", False, str(e))
        det = None

    r = live.Renderer(pipe, state, work, det)
    r.start()
    try:
        for _ in range(6):
            work.put((cv2.imencode(".jpg", np.frombuffer(y, np.uint8).reshape(
                live.OUT_H, live.OUT_W), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(),
                thermal))
            time.sleep(0.25)

        check("the render thread produces a frame without the reader touching fusion",
              state.get("frame") is not None and state.get("rendered", 0) > 0,
              "%d rendered, %d dropped" % (state.get("rendered", 0),
                                           state.get("dropped", 0)))
        check("rows_rebuilt is recorded by the thread that actually ran the fusion",
              len(state.get("rows_window") or []) > 0)

        if det:
            dets = state.get("detections")
            check("detection ran on the render pipeline", dets is not None,
                  "%d box(es) in %.0f ms" % (len(dets or []), state.get("detect_ms", 0)))
            withtemp = [d for d in (dets or []) if "max_c" in d]
            check("every box that overlaps the thermal footprint carries a temperature",
                  all(("max_c" in d) or d.get("no_thermal") for d in (dets or [])),
                  "%d of %d with a reading" % (len(withtemp), len(dets or [])))
            lo, hi = pipe.f.tmin_mc / 1000.0, pipe.f.tmax_mc / 1000.0
            check("box temperatures land inside the sensor range",
                  all(lo - 0.5 <= d["max_c"] <= hi + 0.5 for d in withtemp),
                  "%d readings" % len(withtemp))
            check("an unregistered warp is carried alongside every box",
                  pipe.warped is False)
    finally:
        r.stop.set()

    # The visible temporal filter: live.py set cfg.y_temporal_knee and then never
    # called fusion_y_temporal(), so --y-knee reserved 256KB and filtered nothing.
    a = Args()
    a.y_knee, a.y_frames = 4, 8
    p2 = live.Pipeline(a)
    out = [p2.process(y, thermal) for _ in range(3)]
    check("the visible temporal filter is actually applied now",
          p2.y_knee == 4 and out[-1].shape == (live.OUT_H, live.OUT_W, 3)
          and out[-1].any(), "knee %d" % p2.y_knee)
    p2.temporal_reset()
    check("temporal_reset is reachable and does not fault", True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", nargs="?", default=HANDWAVE3)
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
        # An error status is a result here, not an exception: /pose answers 409
        # when nothing is recording, and that response is exactly what the
        # operator's page has to render.
        try:
            with urlopen(base_url + path, timeout=5) as r:
                return r.status, r.read().decode()
        except HTTPError as e:
            return e.code, e.read().decode()

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

        code, body = get("/timing")
        t = json.loads(body)
        check("/timing answers before the board has reported a clock",
              code == 200 and t["thermal_ms"] is None and t["expected_ms"] == 114.0,
              "expected %.1f ms" % t["expected_ms"])

        code, body = get("/detections")
        d = json.loads(body)
        check("/detections answers even with no detector running",
              code == 200 and d["detections"] == [] and d["warped"] is False,
              "warped=%s" % d["warped"])

        get("/set?boxes=0")
        check("/set can turn the overlay off", pipe.show_detections is False)
        get("/set?boxes=1")

        # /ui is the page's only poll, so a field it stops carrying is a panel
        # that silently goes blank rather than an error anyone sees.
        code, body = get("/ui")
        d = json.loads(body)
        check("/ui carries every panel the page draws",
              code == 200 and {"worst", "checks", "timing", "stats", "detections",
                               "cfg", "heap_free", "coverage"} <= set(d),
              "keys: %s" % ", ".join(sorted(d)))
        check("/ui agrees with /health on the overall level",
              d["worst"] == json.loads(get("/health")[1])["worst"],
              d["worst"])

        # The controls are initialised from here rather than from the HTML. The
        # old page hardcoded its slider positions, so --gain 220 drew a handle at
        # 200 over a pipeline running at 220 and the first touch of it moved the
        # pipeline to wherever the handle happened to be.
        get("/set?gain=333&mix=17&palette=gray")
        cfg = json.loads(get("/ui")[1])["cfg"]
        check("/ui reports the live control state, not the page's defaults",
              cfg["gain"] == 333 and cfg["mix"] == 17 and cfg["palette"] == "gray"
              and cfg["warped"] is False,
              "gain %d, mix %d, palette %s" % (cfg["gain"], cfg["mix"], cfg["palette"]))
        check("a session with no radar reports none rather than dead knobs",
              cfg["radar"] is None)

        # --- /pose: the station stamp (V3 plan section 5d)
        # The recorder has carried pose_id since it was written but nothing
        # could set it, so every session so far went to disk unlabelled.
        check("a viewer that is not recording reports no pose at all",
              json.loads(get("/ui")[1])["pose"] is None)
        code, body = get("/pose?id=N04")
        check("stamping without a recording fails loudly rather than silently",
              code == 409 and "not recording" in json.loads(body)["error"],
              "HTTP %d" % code)

        tmp = tempfile.mkdtemp(prefix="posetest-")
        try:
            good["video"] = recorder.SessionRecorder(tmp, fps=30)
            p = json.loads(get("/ui")[1])["pose"]
            check("a recording with no stamp yet reads as unlabelled, not absent",
                  p is not None and p["id"] is None and p["stamped"] == 0)

            code, body = get("/pose?id=N04")
            check("/pose stamps the station and echoes it back",
                  code == 200 and json.loads(body)["pose"] == "N04")
            check("the recorder is what actually changed",
                  good["video"].pose_id == "N04")
            check("/ui reports the stamp the operator's phone is waiting on",
                  json.loads(get("/ui")[1])["pose"]["id"] == "N04")

            # The id must reach the row, not just meta.json: 'which frames are
            # pose N04' is the question the solver asks, and it asks it of
            # frames.jsonl.
            good["video"].write(np.zeros((8, 8, 3), np.uint8))
            get("/pose?id=N05")
            good["video"].write(np.zeros((8, 8, 3), np.uint8))
            get("/pose?clear=1")
            check("clearing leaves the frames unlabelled again",
                  good["video"].pose_id is None)
            good["video"].write(np.zeros((8, 8, 3), np.uint8))
            good["video"].close()

            rows = [json.loads(l) for l in
                    open(os.path.join(tmp, "frames.jsonl"))]
            check("every frame carries the station that was live when it was written",
                  [r.get("pose_id") for r in rows] == ["N04", "N05", None],
                  " ".join(str(r.get("pose_id")) for r in rows))
            marks = json.load(open(os.path.join(tmp, "meta.json")))["poses"]
            check("meta.json records where each station started",
                  [(m["pose_id"], m["first_frame"]) for m in marks]
                  == [("N04", 0), ("N05", 1), (None, 2)],
                  str([(m["pose_id"], m["first_frame"]) for m in marks]))
        finally:
            good.pop("video", None)
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        srv.shutdown()

    board_clock()
    host_soc()
    failure_reporting()
    threading_and_detection(y, thermal)

    print()
    if FAILS:
        print("FAILED: %s" % ", ".join(FAILS))
        return 1
    print("all viewer checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
