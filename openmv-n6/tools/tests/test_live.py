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
        det = detect.make_detector("gpu", model="yolov10n",
                                   classes=["person"])
        check("TensorRT detector model is present", True, det.model_name)
    except Exception as e:
        check("TensorRT detector model is present", False, str(e))
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
        r.join(timeout=2.0)
        if det is not None and hasattr(det, "close"):
            det.close()

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


class _FakeStudent:
    """An engine that records whether it was asked to run.

    The point of most of these checks is what did NOT happen: a channel that is
    switched off must not reach the GPU at all, and no assertion about the drawn
    picture can tell that apart from a channel that ran and found nobody.
    """

    def __init__(self, dets):
        self.dets, self.calls = dets, 0

    def push(self, frame, t_ms):          # thermal student
        self.calls += 1
        return [dict(d) for d in self.dets]

    def __call__(self, points):           # radar student
        self.calls += 1
        return [dict(d) for d in self.dets]


class _FakeLut:
    """thermal 160x120 -> visible 640x480, the 4x the placeholder warp implies."""

    def box(self, x, y, w, h):
        return (4 * x, 4 * y, 4 * w, 4 * h)


class _FakeRadar:
    def __init__(self):
        self.frames, self.dropped_bytes, self.error = 1, 0, None

    def get(self):
        return {"frame_number": 1,
                "points": [{"x": 0.1, "y": 3.0, "z": 0.0, "v": 0.8,
                            "snr": 20.0, "noise": 5.0}]}


def ai_channels(thermal):
    """The three student channels: pairing, and what each switch actually stops.

    Offline and engine-free. The fusion rule is a geometric claim - pair on u
    only, inside the gate D3 measured - and it is worth a test that does not
    need a GPU, a radar or a person to walk in front of the rig.
    """
    print("\nai channels")

    T = [{"conf": 0.8, "vis": (100, 50, 40, 90)},
         {"conf": 0.6, "vis": (300, 50, 40, 90)}]
    R = [{"conf": 0.5, "x": 110, "y": 40, "w": 44, "h": 100},   # du = 12
         {"conf": 0.9, "x": 500, "y": 40, "w": 40, "h": 100}]   # du = 200
    f = live.Renderer._fuse([dict(d) for d in T], [dict(d) for d in R])
    check("only the pair inside the horizontal gate is fused",
          len(f) == 1 and f[0]["du"] <= live.Renderer.FUSION_DU_PX,
          "%d pair(s), du %s" % (len(f), [x["du"] for x in f]))
    check("the fused box keeps the thermal extent, not the radar's guess",
          (f[0]["x"], f[0]["y"], f[0]["w"], f[0]["h"]) == T[0]["vis"])
    check("agreement scores above either channel on its own",
          f[0]["conf"] > max(T[0]["conf"], R[0]["conf"])
          and f[0]["conf_thermal"] == 0.8 and f[0]["conf_radar"] == 0.5,
          "%.2f from %.2f and %.2f" % (f[0]["conf"], f[0]["conf_thermal"],
                                       f[0]["conf_radar"]))

    # Two thermal boxes over one radar return is the crossing-walkers case: the
    # nearer one takes it, and the other stays an unconfirmed thermal box rather
    # than borrowing the same evidence twice.
    f = live.Renderer._fuse(
        [{"conf": 0.7, "vis": (100, 50, 40, 90)},
         {"conf": 0.7, "vis": (130, 50, 40, 90)}],
        [{"conf": 0.7, "x": 105, "y": 40, "w": 40, "h": 100}])
    check("one radar return cannot confirm two thermal boxes",
          len(f) == 1 and f[0]["x"] == 100, "%d pair(s)" % len(f))

    f = live.Renderer._fuse([{"conf": 0.9, "vis": None}],
                            [{"conf": 0.9, "x": 0, "y": 0, "w": 40, "h": 100}])
    check("a thermal box that never reached the visible plane is not fused",
          f == [])

    # --- the clutter controls. A student emits eight slots with no NMS of its
    # own, and a person standing still fills two or three of them.
    dup = live.Renderer._dedup([
        {"conf": 0.9, "x": 100, "y": 50, "w": 40, "h": 90},
        {"conf": 0.7, "x": 103, "y": 52, "w": 40, "h": 90},   # the same person
        {"conf": 0.6, "x": 300, "y": 50, "w": 40, "h": 90}])  # someone else
    check("a box that claims a region a stronger box already has is dropped",
          len(dup) == 2 and dup[0]["conf"] == 0.9 and dup[1]["x"] == 300,
          "%d of 3 kept" % len(dup))

    # --- what each switch stops. The engines are fakes; the wiring is real.
    pipe = live.Pipeline(Args())
    state = {}
    th = _FakeStudent([{"x": 10, "y": 10, "w": 10, "h": 20, "conf": 0.8}])
    rd = _FakeStudent([{"x": 40, "y": 40, "w": 40, "h": 80, "conf": 0.7}])
    students = {"thermal": th, "radar": rd, "th2vis": _FakeLut(),
                "c_per_lsb": 60 / 255.0, "tmin": 0.0}
    r = live.Renderer(pipe, state, live.Latest(), None, radar=_FakeRadar(),
                      students=students)
    blank = np.zeros((live.OUT_H, live.OUT_W, 3), np.uint8)

    def draw():
        rgb = blank.copy()
        r._run_students(thermal, rgb)
        return rgb

    def has(rgb, col):
        return bool((rgb == np.array(col, np.uint8)).all(axis=2).any())

    pipe.ai_thermal = pipe.ai_radar = pipe.ai_fusion = False
    rgb = draw()
    check("a channel switched off does not reach its engine",
          th.calls == 0 and rd.calls == 0 and not rgb.any(),
          "%d thermal, %d radar inferences" % (th.calls, rd.calls))

    pipe.ai_fusion = True
    rgb = draw()
    check("fusion runs both engines even with both channels hidden",
          th.calls == 1 and rd.calls == 1)
    check("only the agreement is drawn when only fusion is on",
          has(rgb, live.Renderer.STUDENT_FU_COL)
          and not has(rgb, live.Renderer.STUDENT_TH_COL)
          and not has(rgb, live.Renderer.STUDENT_RD_COL))
    check("the fused box is reported alongside its components",
          len(state["student_fused"]) == 1
          and len(state["student_thermal"]) == 1
          and len(state["student_radar"]) == 1)
    check("both components are marked as spoken for by the pair",
          state["student_thermal"][0].get("fused") is True
          and state["student_radar"][0].get("fused") is True)

    pipe.ai_fusion, pipe.ai_thermal = False, True
    rgb = draw()
    check("the thermal channel alone leaves the radar engine idle",
          rd.calls == 1 and th.calls == 2, "%d radar inferences" % rd.calls)
    check("the thermal channel draws its own boxes and no agreement",
          has(rgb, live.Renderer.STUDENT_TH_COL)
          and not has(rgb, live.Renderer.STUDENT_FU_COL)
          and state["student_fused"] == [])

    # The floor hides boxes; it must never hide the fact that it hid them.
    th.dets = [{"x": 10, "y": 10, "w": 10, "h": 20, "conf": 0.81},
               {"x": 60, "y": 10, "w": 10, "h": 20, "conf": 0.55}]
    pipe.ai_thermal, pipe.ai_fusion, pipe.ai_radar = True, False, False
    pipe.ai_conf_thermal = 0.5
    draw()
    check("both boxes are drawn while the floor is at the engine threshold",
          len(state["student_thermal"]) == 2)
    pipe.ai_conf_thermal = 0.7
    draw()
    check("raising the floor hides the weaker box",
          len(state["student_thermal"]) == 1
          and state["student_thermal"][0]["conf"] == 0.81)
    check("what was hidden is still counted, so a quiet scene and a high "
          "slider do not look alike",
          state["student_seen"]["thermal"] == 2)
    pipe.ai_conf_thermal = 0.5
    th.dets = [{"x": 10, "y": 10, "w": 10, "h": 20, "conf": 0.8}]

    # An engine that has not seen enough frames to build its temporal chain
    # answers confidently about cold structure; those frames are dropped.
    th.primed = False
    draw()
    check("boxes from an unprimed temporal chain are dropped, not drawn",
          state["student_thermal"] == [] and state["student_seen"]["thermal"] == 1,
          "found %d, shown %d" % (state["student_seen"]["thermal"],
                                  len(state["student_thermal"])))
    th.primed = True
    draw()
    check("once the chain is primed the boxes come back",
          len(state["student_thermal"]) == 1)

    # --- the person outline. The locator and the shape come from different
    # sensors, so the failure that matters is drawing a shape when there is no
    # way to know where the thermal frame maps to.
    check("no warp LUT means no outline callback at all, not one that says no",
          r._person_outline(thermal, blank.copy()) is None)

    drew = []

    def fake_outline(d, col):
        drew.append(d["cls"])
        return True

    img = np.zeros((live.OUT_H, live.OUT_W, 3), np.uint8)
    detect.annotate(img, [{"cls": "person", "conf": 0.9, "x": 100, "y": 40,
                           "w": 60, "h": 200, "body_heat": True}],
                    True, outline=fake_outline)
    # The label is still drawn, so the frame is not blank - what must be gone
    # is the rectangle: no full-height vertical run of the person colour.
    col = np.array(detect.PERSON_COL, np.uint8)
    side = (img[40:240, 100] == col).all(axis=1).mean()
    check("a drawn shape replaces the box rather than being drawn over it",
          drew == ["person"] and side < 0.5, "%.0f%% of the left edge painted"
          % (100 * side))

    lut_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "calib-artifacts", "warp.lut")
    if os.path.isfile(lut_path):
        import trt_students
        t2v = trt_students.ThermalToVisible(lut_path)
        hot = np.full((120, 160), 40, np.uint8)
        hot[40:80, 60:100] = 200                  # one warm rectangle
        # Queried through a box wider than the warm patch: a box whose every
        # thermal sample is the person has no shape to find that is not the
        # box itself, and shape_in() says so by returning None.
        box = t2v.box(50, 30, 60, 60)
        m = t2v.shape_in(hot, *box)
        ys, xs = np.nonzero(m)
        check("the outline is the warm shape inside the box, in visible pixels",
              m is not None and m.shape == (live.OUT_H, live.OUT_W)
              and box[0] - 4 <= xs.min() and xs.max() <= box[0] + box[2] + 4
              and box[1] - 4 <= ys.min() and ys.max() <= box[1] + box[3] + 4
              and (xs.max() - xs.min()) < box[2],
              "box %s -> shape x %d..%d y %d..%d"
              % (box, xs.min(), xs.max(), ys.min(), ys.max()))
        check("a box with nothing to split reports no shape rather than a blob",
              t2v.shape_in(np.full((120, 160), 40, np.uint8), *box) is None)
    else:
        check("warp.lut is present for the outline check", False, lut_path)

    # No warp LUT is the state every session before B2 was solved in: the boxes
    # are found, they simply have nowhere to go. They must still be reported.
    students["th2vis"] = None
    pipe.ai_fusion = True
    rgb = draw()
    check("without a warp LUT the thermal boxes are reported but not drawn",
          len(state["student_thermal"]) == 1
          and state["student_thermal"][0]["vis"] is None
          and state["student_fused"] == []
          and not has(rgb, live.Renderer.STUDENT_TH_COL))
    r.stop.set()


def lock_mode():
    """The lock: identity between frames, and a coast that says it is one.

    All offline and engine-free - the tracker is plain python - which is the
    point: whether a person survives the frames nobody found them in is a
    question about this logic, not about the GPU.
    """
    print("\nlock on people")
    import tracker as tracking

    def walk(t0, n, x0, step, src="det", tk=None, dt=0.114):
        tk = tk or tracking.Tracker()
        out = []
        for i in range(n):
            out = tk.update([{"x": x0 + i * step, "y": 100, "w": 60, "h": 180,
                              "conf": 0.9, "src": src}], t0 + i * dt)
        return tk, out

    tk, tracks = walk(1000.0, 6, 100, 25)
    check("a person walking across the frame stays one track",
          len(tracks) == 1 and tracks[0].id == 1 and tracks[0].hits == 6,
          "%d track(s), %d hits" % (len(tracks), tracks[0].hits if tracks else 0))
    check("the box follows them rather than sitting where they started",
          tracks[0].box[0] > 180, "x %d" % tracks[0].box[0])

    # One observation is not a lock: a single frame of a false positive on a
    # warm doorway is exactly what MIN_HITS exists to swallow.
    tk = tracking.Tracker()
    once = tk.update([{"x": 10, "y": 10, "w": 40, "h": 90, "conf": 0.9,
                       "src": "det"}], 2000.0)
    check("one sighting is not yet a lock", once == [] and len(tk.all()) == 1)

    # The core of it: the detector drops out and the person is still held.
    tk, _ = walk(3000.0, 4, 100, 20)
    now = 3000.0 + 4 * 0.114
    kept = []
    for i in range(6):                     # six frames with nothing at all
        now += 0.114
        kept = tk.update([], now)
    check("a person nobody sees this frame is still locked, and still P1",
          len(kept) == 1 and kept[0].id == 1 and kept[0].coasting_for(now) > 0.5,
          "coasting %.2fs" % (kept[0].coasting_for(now) if kept else -1))
    check("the coast is carried on their velocity, not frozen in place",
          kept[0].box[0] > 160, "x %d" % kept[0].box[0])

    now += tracking.MAX_COAST_S
    check("a coast that outlives its budget is dropped, not kept forever",
          tk.update([], now) == [] and tk.all() == [])

    # ...and the lock survives on another sensor when the detector is the one
    # that dropped out, which is the whole reason three of them feed this.
    tk, _ = walk(4000.0, 3, 100, 20)
    now = 4000.0 + 3 * 0.114
    held = tk.update([{"x": 160, "y": 100, "w": 60, "h": 180, "conf": 0.7,
                       "src": "thermal"}], now + 0.114)
    check("the thermal channel alone can hold a lock the detector lost",
          len(held) == 1 and held[0].id == 1
          and held[0].held_by == {"thermal"} and held[0].coasting_for(now) < 0.2,
          "held by %s" % sorted(held[0].held_by) if held else "lost")

    # Cross-sensor merge: three sensors, one person, one track.
    tk = tracking.Tracker()
    obs = [{"x": 100, "y": 50, "w": 60, "h": 180, "conf": 0.9, "src": "det",
            "max_c": 32.1},
           {"x": 104, "y": 54, "w": 58, "h": 176, "conf": 0.8, "src": "thermal"},
           {"x": 96, "y": 40, "w": 70, "h": 200, "conf": 0.7, "src": "radar",
            "radar_m": 4.2}]
    for i in range(2):
        merged = tk.update([dict(o) for o in obs], 5000.0 + i * 0.114)
    check("one person seen by three sensors is one track, not three",
          len(merged) == 1 and merged[0].held_by == {"det", "thermal", "radar"},
          "%d track(s), held by %s" % (len(merged), sorted(merged[0].held_by)))
    check("the lock carries the range and temperature the sensors brought",
          merged[0].radar_m == 4.2 and merged[0].max_c == 32.1)
    check("the detector's geometry wins the merge, not the radar's guess",
          abs(merged[0].box[2] - 60) <= 2, "w %d" % merged[0].box[2])

    # --- the warm door. Measured in the lobby this rig sits in: the lit glass
    # door reads 31.7 C mean and a person reads 30.9 C, so the discriminator
    # cannot be temperature. It is that a door has never moved.
    tk = tracking.Tracker()
    now = 6000.0
    for i in range(40):
        tk.update([{"x": 300, "y": 120, "w": 90, "h": 200, "conf": 0.95,
                    "src": "thermal"},
                   {"x": 298 + (i % 3), "y": 118, "w": 94, "h": 204,
                    "conf": 0.9, "src": "radar"}], now + i * 0.114)
    end = now + 40 * 0.114
    check("a warm thing that never moves is not drawn as a person, however "
          "confident the students are",
          tk.confirmed(end) == [] and len(tk.suppressed(end)) == 1,
          "%d drawn, %d suppressed" % (len(tk.confirmed(end)),
                                       len(tk.suppressed(end))))
    check("it is counted rather than discarded - warm and motionless is not "
          "nobody",
          tk.suppressed(end)[0].as_dict(end)["moved_px"] < tracking.STATIC_PX)

    # ...and the same evidence, once it moves, is a person - which is what
    # keeps this safe in the dark, where the visible detector sees nothing.
    tk = tracking.Tracker()
    for i in range(40):
        tk.update([{"x": 100 + i * 12, "y": 120, "w": 90, "h": 200,
                    "conf": 0.9, "src": "thermal"}], now + i * 0.114)
    check("a walker the detector never saw is drawn once they have moved",
          len(tk.confirmed(end)) == 1 and tk.suppressed(end) == [],
          "moved %.0f px" % tk.all()[0].moved)

    # The detector is trusted on sight: it knows a door from a person, and a
    # person standing still must not need to walk to be believed.
    tk = tracking.Tracker()
    for i in range(6):
        tk.update([{"x": 300, "y": 120, "w": 90, "h": 200, "conf": 0.95,
                    "src": "det"}], now + i * 0.114)
    check("a motionless person the detector sees is drawn immediately",
          len(tk.confirmed(now + 0.7)) == 1)

    # Two boxes from the SAME sensor are two people and must stay two.
    two = tracking.merge([
        {"x": 100, "y": 50, "w": 60, "h": 180, "conf": 0.9, "src": "det"},
        {"x": 118, "y": 50, "w": 60, "h": 180, "conf": 0.8, "src": "det"}])
    check("a sensor reporting two boxes is never overruled into one",
          len(two) == 2, "%d group(s)" % len(two))


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

    operator = live.compose("operator", fused, yarr, cover=cover, mix=60,
                            outline=False)
    check("operator view preserves visible structure without becoming grey",
          operator.shape == fused.shape
          and np.abs(operator.astype(int) - fused.astype(int)).mean() > 2
          and np.abs(operator.astype(int) - vis.astype(int)).mean() > 2
          and (np.max(operator, axis=2).astype(int)
               - np.min(operator, axis=2).astype(int)).mean() > 1,
          "distance fused %.1f, visible %.1f"
          % (np.abs(operator.astype(int) - fused.astype(int)).mean(),
             np.abs(operator.astype(int) - vis.astype(int)).mean()))

    partial_for_operator = np.zeros_like(cover)
    partial_for_operator[10:-10, 20:-20] = 1
    soft = live.compose("operator", fused, yarr, cover=partial_for_operator,
                        mix=60, outline=False)
    check("operator view is visible outside the thermal footprint",
          np.abs(soft[0, 0].astype(int) - vis[0, 0].astype(int)).max() <= 1,
          "corner delta %d"
          % np.abs(soft[0, 0].astype(int) - vis[0, 0].astype(int)).max())
    edge_x = int(20 * live.OUT_W / cover.shape[1])
    check("operator footprint has a soft edge instead of a hard seam",
          np.abs(soft[live.OUT_H // 2, edge_x].astype(int)
                 - vis[live.OUT_H // 2, edge_x].astype(int)).mean()
          < np.abs(operator[live.OUT_H // 2, edge_x].astype(int)
                   - vis[live.OUT_H // 2, edge_x].astype(int)).mean(),
          "edge is partially blended")

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

        get("/set?view=operator")
        check("/set accepts the operator view", pipe.view == "operator", pipe.view)

        get("/set?view=nonsense")
        check("/set ignores an unknown view rather than breaking the stream",
              pipe.view == "operator")

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

        # The AI channels are three switches and not a mode: any combination has
        # to be reachable, including fusion with both components hidden.
        get("/set?ai_thermal=0&ai_radar=1&ai_fusion=0")
        check("/set switches the AI channels independently",
              pipe.ai_thermal is False and pipe.ai_radar is True
              and pipe.ai_fusion is False,
              "T %s R %s TR %s" % (pipe.ai_thermal, pipe.ai_radar, pipe.ai_fusion))
        code, body = get("/ai")
        d = json.loads(body)
        check("/ai answers with no students loaded rather than 404",
              code == 200 and d["available"] is False
              and set(d["channels"]) == {"thermal", "radar", "fusion"},
              "available=%s" % d["available"])
        check("a session with no students offers no AI card",
              json.loads(get("/ui")[1])["cfg"]["ai"] is None)

        # Loaded students with neither a radar engine nor a LUT: the switches
        # exist, and the page has to say they cannot draw rather than showing a
        # zero that reads as an empty scene.
        good["students"] = {"thermal": object(), "radar": None, "th2vis": None}
        try:
            aicfg = json.loads(get("/ui")[1])["cfg"]["ai"]
            check("the AI card appears once the students are loaded",
                  aicfg is not None and aicfg["radar"]["on"] is True)
            check("a channel that cannot draw reads as unready, not as empty",
                  aicfg["fusion"]["ready"] is False
                  and aicfg["radar"]["ready"] is False)
            get("/set?ai_fusion=1")
            names = {c["name"]: c for c in json.loads(get("/health")[1])["checks"]}
            check("health says why a switched-on fusion channel draws nothing",
                  names.get("ai", {}).get("level") == "warn"
                  and "cannot pair" in names.get("ai", {}).get("text", ""),
                  names.get("ai", {}).get("text", "no ai check at all"))
        finally:
            good.pop("students", None)
        get("/set?lock=0")
        check("/set can switch the lock off", pipe.ai_lock is False)
        get("/set?lock=1")
        check("/ui carries the tracks as their own list, not as detections",
              "tracks" in json.loads(get("/ui")[1])
              and json.loads(get("/ui")[1])["cfg"]["lock"] is True)

        get("/set?conf_thermal=0.8&conf_radar=0.75")
        check("/set carries a per-channel confidence floor",
              abs(pipe.ai_conf_thermal - 0.8) < 1e-6
              and abs(pipe.ai_conf_radar - 0.75) < 1e-6,
              "T %.2f R %.2f" % (pipe.ai_conf_thermal, pipe.ai_conf_radar))
        get("/set?conf_thermal=0.1")
        check("a floor below the engine's own threshold is clamped, not obeyed",
              pipe.ai_conf_thermal == pipe.ai_conf_floor,
              "%.2f (engine floor %.2f)" % (pipe.ai_conf_thermal,
                                            pipe.ai_conf_floor))
        get("/set?ai_thermal=1&ai_radar=1&ai_fusion=1")

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
    ai_channels(thermal)
    lock_mode()

    print()
    if FAILS:
        print("FAILED: %s" % ", ".join(FAILS))
        return 1
    print("all viewer checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
