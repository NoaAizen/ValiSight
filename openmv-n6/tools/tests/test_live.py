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
import thermal_io  # noqa: E402
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
    check("the thermal temporal defaults reach the C pipeline",
          pipe.f.cfg.temporal_noise_mc == 148
          and pipe.f.cfg.temporal_frames == 8)
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
          and rate(20 * live.HEAP_FLOOR) == 1,
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

    # A clean raw-REPL footer before the advertised payload length is not a
    # dead board.  The reader must keep the prompt and abandon only this batch.
    s = live.Streamer.__new__(live.Streamer)
    s.buf = bytearray(b"x" * 501 + live._BATCH_FOOTERS[0])
    try:
        s._exact(19200, "thermal")
        dropped = None
    except live._DroppedBatch as e:
        dropped = e
    check("a board footer inside a short payload drops one batch immediately",
          dropped is not None and dropped.received == 501
          and s.buf == bytearray(b"\x04\x04>"),
          str(dropped) if dropped else "no exception")


def threading_and_detection(y, thermal):
    """Renderer + Detector driven off recorded frames, with no board attached.

    Worth having offline because the interesting failure is not the detector
    getting a box wrong - it is a box carrying a temperature that came from the
    wrong pixels. With the placeholder warp that is the *expected* state, so the
    check is that the flag saying so is present, not that it is absent.
    """
    print("\nrender thread and detection")
    check("a scene-sized landscape detector box cannot vouch for a person",
          live.Renderer._reject_person_geometry(
              {"cls": "person", "x": 8, "y": 2, "w": 613, "h": 392})
          is not None)
    check("a close portrait person remains admissible",
          live.Renderer._reject_person_geometry(
              {"cls": "person", "x": 180, "y": 0, "w": 260, "h": 400})
          is None)
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

    class CountingDetector:
        def __init__(self):
            self.calls, self.ms = 0, 0.0

        def __call__(self, _frame):
            self.calls += 1
            return [{"cls": "person", "conf": 0.99,
                     "x": 0, "y": 0, "w": live.OUT_W, "h": live.OUT_H}]

    # Darkness is not visual evidence. One full-frame false positive here used
    # to set `ever det` on a track permanently, after which thermal/radar clutter
    # inherited detector trust and filled the AI view with giant person boxes.
    blind_det = CountingDetector()
    blind_state, blind_work = {}, live.Latest()
    blind_renderer = live.Renderer(live.Pipeline(Args()), blind_state,
                                   blind_work, blind_det)
    blind_renderer.start()
    try:
        dark = np.zeros((live.OUT_H, live.OUT_W), np.uint8)
        dark_jpg = cv2.imencode(".jpg", dark)[1].tobytes()
        for _ in range(3):
            blind_work.put((dark_jpg, thermal))
            time.sleep(0.08)
        deadline = time.time() + 1.0
        while blind_state.get("detect_t") is None and time.time() < deadline:
            time.sleep(0.02)
        check("a blind visible frame never grants detector trust",
              blind_det.calls == 0
              and blind_state.get("detect_suppressed") == "blind"
              and blind_state.get("detections") == [],
              "%d detector call(s), suppressed=%s" % (
                  blind_det.calls, blind_state.get("detect_suppressed")))
    finally:
        blind_renderer.stop.set()
        blind_renderer.join(timeout=2.0)

    class CountingRadar:
        def __init__(self):
            self.calls, self.frames, self.dropped_bytes, self.error = 0, 1, 0, None

        def get(self):
            self.calls += 1
            return {"frame_number": self.calls,
                    "points": [{"x": 3.0, "y": 0.0, "z": 0.0, "v": 0.2,
                                "snr": 20.0, "noise": 5.0}]}

    radar = CountingRadar()
    p3, s3, w3 = live.Pipeline(Args()), {}, live.Latest()
    rr = live.Renderer(p3, s3, w3, radar=radar,
                       radar_proj=live.radar_overlay.Bootstrap(
                           live.OUT_W, live.OUT_H))
    rr.start()
    try:
        jpg = cv2.imencode(".jpg", np.frombuffer(y, np.uint8).reshape(
            live.OUT_H, live.OUT_W))[1].tobytes()
        w3.put((jpg, thermal))
        deadline = time.time() + 2
        while s3.get("rendered", 0) < 1 and time.time() < deadline:
            time.sleep(0.02)
        check("one displayed frame samples radar exactly once for every consumer",
              s3.get("rendered") == 1 and radar.calls == 1,
              "%d render(s), %d radar get(s)" % (s3.get("rendered", 0), radar.calls))
        check("map state and overlay carry that same radar frame number",
              s3.get("radar_frame") == 1 and s3.get("radar_drawn") == 1)
    finally:
        rr.stop.set()
        rr.join(timeout=2.0)

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

    def shape_in(self, thermal, x, y, w, h):
        m = np.zeros((live.OUT_H, live.OUT_W), dtype=bool)
        # A tall warm component inside the existing person box.
        x0, x1 = max(0, x + w // 4), min(live.OUT_W, x + 3 * w // 4)
        y0, y1 = max(0, y + h // 8), min(live.OUT_H, y + 7 * h // 8)
        m[y0:y1, x0:x1] = True
        return m


class _HumanLut(_FakeLut):
    """A connected warm head, tapered torso and two legs inside the box."""

    def shape_in(self, thermal, x, y, w, h):
        m = np.zeros((live.OUT_H, live.OUT_W), dtype=np.uint8)
        cx = int(x + w / 2)
        r = max(4, min(w // 5, h // 10))
        cv2.circle(m, (cx, int(y + 0.13 * h)), r, 1, -1)
        cv2.rectangle(m, (cx - max(2, r // 3), int(y + 0.13 * h)),
                      (cx + max(2, r // 3), int(y + 0.25 * h)), 1, -1)
        cv2.fillConvexPoly(m, np.array([
            [int(x + 0.18 * w), int(y + 0.23 * h)],
            [int(x + 0.82 * w), int(y + 0.23 * h)],
            [int(x + 0.68 * w), int(y + 0.64 * h)],
            [int(x + 0.32 * w), int(y + 0.64 * h)]], np.int32), 1)
        leg = max(3, int(0.16 * w))
        cv2.rectangle(m, (int(x + 0.32 * w), int(y + 0.60 * h)),
                      (int(x + 0.32 * w) + leg, int(y + 0.96 * h)), 1, -1)
        cv2.rectangle(m, (int(x + 0.68 * w) - leg, int(y + 0.60 * h)),
                      (int(x + 0.68 * w), int(y + 0.96 * h)), 1, -1)
        return m.astype(bool)


class _FakeRadar:
    def __init__(self):
        self.frames, self.dropped_bytes, self.error = 1, 0, None

    def get(self):
        return {"frame_number": 1,
                "points": [{"x": 0.1, "y": 3.0, "z": 0.0, "v": 0.8,
                            "snr": 20.0, "noise": 5.0}]}


class _FakeRadarProj:
    """Place every synthetic moving return at the paired boxes' horizontal u."""

    def project(self, points):
        return [(60.0, 80.0, True) for _ in points]


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
    check("agreement confidence is limited by the weaker correlated student",
          f[0]["conf"] == min(T[0]["conf"], R[0]["conf"])
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
                      radar_proj=_FakeRadarProj(), students=students)
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
    lock_obs = r._lock_observations(thermal)
    check("fusion-only lock exposes thermal only as hold-only evidence",
          len(lock_obs) == 2
          and any(d["src"] == "fusion" and not d.get("hold_only")
                  for d in lock_obs)
          and any(d["src"] == "thermal" and d.get("hold_only")
                  for d in lock_obs)
          and all(d["src"] != "radar" for d in lock_obs),
          "sources %s" % [(d["src"], d.get("hold_only", False))
                           for d in lock_obs])

    # Selecting the thermal product is an explicit request to SEE the thermal
    # result, but it is not permission to call every warm box a person. The
    # already-running component is visible while the human-shape gate still
    # owns whether it may establish a lock.
    pipe.channel = "thermal"
    rgb = draw()
    thermal_obs = r._lock_observations(thermal)
    check("the thermal product does not bypass the human-shape lock gate",
          any(d["src"] == "thermal" and d.get("hold_only")
              for d in thermal_obs))
    check("the thermal product draws the detected body with fusion-only enabled",
          has(rgb, live.Renderer.STUDENT_TH_COL))
    pipe.channel = "fusion"

    # Cold start with somebody already seated: there is no displacement and no
    # Doppler to earn the ordinary thermal track. A strong thermal student plus
    # an independently human-shaped CURRENT warm component may vouch at rest.
    # A filled warm rectangle (the doorway failure this gate protects against)
    # must still remain suppressed.
    static_det = {"conf": 0.99, "x": 10, "y": 10, "w": 10, "h": 20,
                  "vis": (180, 50, 160, 300)}
    state["student_thermal"], state["student_fused"] = [static_det], []
    r.th2vis = _HumanLut()
    r.lock = live.tracking.Tracker()
    early_static = []
    for i in range(live.tracking.STATIC_SHAPE_HITS):
        seated = r.lock.update(r._lock_observations(thermal), 900.0 + i * 0.114)
        if i < live.tracking.STATIC_SHAPE_HITS - 1:
            early_static.extend(seated)
    check("one thermal-shape flicker cannot vouch for a static person",
          early_static == [])
    check("a human thermal shape locks a person already seated at startup",
          len(seated) == 1 and seated[0].static_vouched
          and seated[0].moved == 0 and seated[0].held_by == {"thermal"},
          "tracks %s" % [t.as_dict(900.5) for t in seated])
    check("the thermal cold-start verdict is explained in the AI result",
          static_det.get("human_shape") is True
          and static_det.get("shape_metrics", {}).get("reason")
          == "human thermal shape")

    doorway = dict(static_det)
    state["student_thermal"] = [doorway]
    r.th2vis = _FakeLut()
    r.lock = live.tracking.Tracker()
    for i in range(3):
        rejected_static = r.lock.update(
            r._lock_observations(thermal), 910.0 + i * 0.114)
    check("a filled warm rectangle cannot cold-start a person lock",
          rejected_static == [] and r.lock.all() == []
          and doorway.get("human_shape") is False,
          doorway.get("shape_metrics", {}).get("reason", "no verdict"))

    # If both students drop while a proven person stands, the current thermal
    # shape inside that existing track becomes hold-only evidence. It is not a
    # detector and therefore cannot start a track of its own.
    r.lock = live.tracking.Tracker()
    for i in range(8):
        r.lock.update([{"x": 100 + 12 * i, "y": 80, "w": 80, "h": 240,
                        "conf": 0.9, "src": "fusion", "radar_m": 2.0}],
                      1000.0 + i * 0.114)
    r.th2vis = _FakeLut()
    state["student_thermal"] = []
    state["student_fused"] = []
    held_obs = r._lock_observations(thermal)
    check("a current thermal body shape can hold a proven standing person",
          len(held_obs) == 1 and held_obs[0]["src"] == "thermal"
          and held_obs[0]["hold_only"] is True,
          "observations %s" % held_obs)
    r.th2vis = None

    pipe.ai_fusion, pipe.ai_thermal = False, True
    before_th, before_rd = th.calls, rd.calls
    rgb = draw()
    check("the thermal channel alone leaves the radar engine idle",
          rd.calls == before_rd and th.calls == before_th + 1,
          "%d radar inferences" % rd.calls)
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
    pipe.ai_silhouette = True
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

        # The per-box stretch that lets 16-bit words through Otsu must not move
        # what an 8-bit session already did. Otsu maximises between-class
        # variance, which is invariant under an affine rescale, so the cut has
        # to land between the same two values however the codes are shifted or
        # spread - and if it did not, every silhouette drawn since the students
        # shipped would quietly change shape.
        same = True
        for mul, add in ((1, 0), (2, 5), (4, 30)):
            h2 = np.clip(hot.astype(np.int32) * mul + add, 0, 255).astype(np.uint8)
            m2 = t2v.shape_in(h2, *box)
            same &= m2 is not None and np.array_equal(m2, m)
        check("rescaling the codes does not move the split", same)

        # The case the words exist for, and it is contrast against NOISE
        # rather than contrast alone. A person 0.8 C from the wall behind them
        # is 3.4 codes at the 0:60 window - which survives quantisation on a
        # clean frame, so a noiseless test proves nothing. The sensor's measured
        # per-pixel temporal noise is ~0.18 C, i.e. 0.77 of a code: on the 8-bit
        # plane the signal is a few codes wide and so is the noise, and Otsu is
        # choosing a cut inside a histogram with almost no shape. In the words
        # the same scene is 80 counts of signal against 18 of noise.
        #
        # Scored over repeated noise draws by how much of the recovered shape
        # lands on the patch that is really there.
        rng = np.random.RandomState(7)
        window = thermal_io.window_ck({"tmin": 0, "tmax": 60})
        truth = np.zeros((120, 160), bool)
        truth[40:80, 60:100] = True
        won = {16: 0.0, 8: 0.0}
        for _ in range(12):
            w16 = (30000 + rng.normal(0, 18, (120, 160))).astype(np.uint16)
            w16[truth] += 80
            for bits, plane in ((16, w16), (8, thermal_io.codes8(w16, window))):
                m = t2v.shape_in(plane, *box)
                if m is None:
                    continue
                t = np.repeat(np.repeat(truth, 4, 0), 4, 1)[:m.shape[0], :m.shape[1]]
                union = float((m | t).sum())
                won[bits] += float((m & t).sum()) / union if union else 0.0
        iou16, iou8 = won[16] / 12, won[8] / 12
        check("the words recover the warm shape through the sensor's own noise",
              iou16 > 0.5, "IoU %.2f" % iou16)
        # And the honest result, which is not the one this was wired for: at the
        # 0:60 window the words are no BETTER. Counting histogram levels made
        # 8-bit look starved - 8 against 79 - but levels are the wrong figure of
        # merit. What a segmenter feels is added noise, and uniform quantisation
        # contributes step/sqrt(12): 0.068 C against the sensor's own 0.18 C, so
        # 8 bits inflate the total by 7% and Otsu cannot tell the difference. It
        # would take a window wider than 159 C for quantisation to reach the
        # sensor's noise, and nothing here uses one. The wiring stays because it
        # costs nothing and is right when the window IS wide - not because it
        # sharpens this rig's outline, which is bounded by geometry instead: one
        # thermal pixel is 4x3.3 visible pixels and no precision changes that.
        check("and at this window they are not worse, which is all that is claimed",
              iou16 >= iou8 - 0.02,
              "16-bit IoU %.2f vs 8-bit %.2f - quantisation adds 0.068 C to a "
              "0.18 C noise floor" % (iou16, iou8))
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

    class ViewTrack:
        def __init__(self, box, coast):
            self.box, self.coast = box, coast

        def coasting_for(self, _now):
            return self.coast

    measured = ViewTrack((100, 50, 200, 300), 0.0)
    replaced_coast = ViewTrack((110, 55, 190, 290), 0.7)
    separate_coast = ViewTrack((400, 60, 80, 200), 0.7)
    shown = live.Renderer._display_tracks(
        [measured, replaced_coast, separate_coast], 0.0)
    check("a measured track visually replaces an overlapping coast",
          measured in shown and replaced_coast not in shown
          and separate_coast in shown)

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

    # In normal fusion-only deployment a stationary person loses Doppler, so
    # fusion stops even though thermal still measures their body. Hidden
    # thermal evidence may hold a person already proven by motion, but must not
    # create or promote the warm doorway this rule exists to reject.
    tk_hold = tracking.Tracker()
    warm = {"x": 100, "y": 100, "w": 60, "h": 180, "conf": 0.95,
            "src": "thermal", "hold_only": True}
    for i in range(20):
        tk_hold.update([dict(warm)], 3500.0 + i * 0.114)
    check("hold-only thermal cannot create a person",
          tk_hold.all() == [] and tk_hold.confirmed(3503.0) == [])

    tk_hold = tracking.Tracker()
    for i in range(8):
        tk_hold.update([{"x": 100 + i * 12, "y": 100, "w": 60, "h": 180,
                         "conf": 0.9, "src": "fusion", "radar_m": 2.0}],
                       3600.0 + i * 0.114)
    tid = tk_hold.confirmed(3601.0)[0].id
    for i in range(30):
        held = tk_hold.update([dict(warm, x=184)], 3601.0 + i * 0.114)
    check("a person proven in motion stays while standing on thermal",
          len(held) == 1 and held[0].id == tid
          and held[0].held_by == {"thermal"},
          "track %s held by %s" % (held[0].id if held else "lost",
                                    sorted(held[0].held_by) if held else []))

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

    # v2's live false positive: almost the whole frame, with enough edge jitter
    # over time to pass the ordinary motion gate.  Size makes Thermal alone
    # insufficient; a physically-gated fusion observation may still vouch for a
    # genuinely close person.
    tk = tracking.Tracker()
    now = 6100.0
    for i in range(45):
        tk.update([{"x": 20 + 3 * i, "y": 5, "w": 370, "h": 392,
                    "conf": 0.999, "src": "thermal"}], now + i * 0.114)
    end = now + 45 * 0.114
    check("a scene-sized thermal box cannot jitter its way into becoming a person",
          tk.confirmed(end) == [] and len(tk.suppressed(end)) == 1,
          "moved %.0f px" % tk.suppressed(end)[0].moved)

    # ...and the same evidence, once it moves, is a person - which is what
    # keeps this safe in the dark, where the visible detector sees nothing.
    tk = tracking.Tracker()
    for i in range(40):
        tk.update([{"x": 100 + i * 12, "y": 120, "w": 90, "h": 200,
                    "conf": 0.9, "src": "thermal"}], now + i * 0.114)
    check("a walker the detector never saw is drawn once they have moved",
          len(tk.confirmed(end)) == 1 and tk.suppressed(end) == [],
          "moved %.0f px" % tk.all()[0].moved)

    # --- the two phantoms from the lobby screenshot, by their labels.
    #
    # "P38 R": a radar-only track on a wall return that has wandered a long way.
    # It passes the displacement test and always did; what stops it now is that
    # radar's word alone does not vouch.
    tk = tracking.Tracker()
    for i in range(40):
        tk.update([{"x": 60 + i * 12, "y": 90, "w": 90, "h": 300,
                    "conf": 0.9, "src": "radar"}], now + i * 0.114)
    end = now + 40 * 0.114
    check("a radar-only track is not a person however far it has wandered",
          not tk.confirmed(end) and len(tk.all()) == 1,
          "moved %.0f px, still not drawn" % tk.all()[0].moved)

    # ...and the same track, once the thermal student also finds it, is drawn.
    # Radar is not distrusted, it is just not sufficient on its own.
    tk = tracking.Tracker()
    for i in range(40):
        box = {"x": 60 + i * 12, "y": 90, "w": 90, "h": 300, "conf": 0.9}
        tk.update([dict(box, src="radar"), dict(box, src="thermal")],
                  now + i * 0.114)
    check("the same evidence with a warm body behind it is",
          len(tk.confirmed(end)) == 1)

    # "P33 coast 4.8m": a box ~300 px tall quoting 4.8 m is claiming a person
    # 2.7 m tall. A specular return off the glass wall arrives at roughly twice
    # the distance to the glass, which is exactly how it gets there.
    tk = tracking.Tracker()
    for i in range(40):
        tk.update([{"x": 60 + i * 12, "y": 40, "w": 120, "h": 300,
                    "conf": 0.9, "src": "thermal", "radar_m": 4.8}],
                  now + i * 0.114)
    t = tk.all()[0]
    check("a box whose range makes it 2.7 m tall is not drawn as a person",
          not tk.confirmed(end),
          "implied height %.2f m at %.1f m" % (t.implied_height_m, t.radar_m))

    # Detector trust must not bypass the same physical contradiction. The
    # detector can be right that a person exists while range association picks
    # a different radar return; drawing both geometries produced duplicate
    # full-frame people in the live AI view.
    tk = tracking.Tracker()
    for i in range(2):
        tk.update([{"x": 60, "y": 40, "w": 120, "h": 300,
                    "conf": 0.99, "src": "det", "radar_m": 4.8}],
                  now + i * 0.114)
    check("detector trust cannot override an impossible associated height",
          not tk.confirmed(now + 0.228))

    # The ceiling only. A person at 2 m is taller than the frame, so their box
    # is clipped and UNDER-states their height - a floor would fire on exactly
    # the close targets this rig exists for.
    tk = tracking.Tracker()
    for i in range(40):
        tk.update([{"x": 60 + i * 12, "y": 0, "w": 120, "h": 400,
                    "conf": 0.9, "src": "thermal", "radar_m": 2.0}],
                  now + i * 0.114)
    t = tk.all()[0]
    check("a clipped box on a close person is not vetoed for being short",
          len(tk.confirmed(end)) == 1,
          "implied %.2f m - clipping can only shrink a box, never grow it"
          % t.implied_height_m)

    # --- the rig turns. Every box in the picture moves; none of it is news
    # about anybody. Without the ego term a slow sweep past the same warm door
    # accumulates displacement and vouches for it as a person - the vouched
    # rule inverted by a pan.
    PAN = 9.0                       # px per frame, ~80 px/s: an unhurried sweep
    tk_blind, tk_told = tracking.Tracker(), tracking.Tracker()
    now = 7000.0
    for i in range(40):
        t = now + i * 0.114
        door = {"x": 300 + i * PAN, "y": 120, "w": 90, "h": 200,
                "conf": 0.95, "src": "thermal"}
        tk_blind.update([dict(door)], t)
        tk_told.update([dict(door)], t, ego=(PAN, 0.0))
    end = now + 40 * 0.114
    check("a rig sweeping past a warm door used to invent a person out of it",
          len(tk_blind.confirmed(end)) == 1,
          "%d drawn" % len(tk_blind.confirmed(end)))
    check("told what the rig did, the door is still a door",
          tk_told.confirmed(end) == [] and len(tk_told.suppressed(end)) == 1,
          "%d drawn, moved %.1f px" % (len(tk_told.confirmed(end)),
              tk_told.suppressed(end)[0].as_dict(end)["moved_px"]
              if tk_told.suppressed(end) else -1))

    # ...and a real walker seen from a turning rig is still one person, with
    # the pan taken out of their velocity rather than added to it.
    tk = tracking.Tracker()
    for i in range(12):
        # 12 px of walking on top of 9 px of pan: what the camera sees is 21.
        tk.update([{"x": 100 + i * 21.0, "y": 120, "w": 60, "h": 180,
                    "conf": 0.9, "src": "det"}], now + i * 0.114, ego=(PAN, 0.0))
    held = tk.confirmed(now + 12 * 0.114)
    check("a walker seen from a turning rig stays one track",
          len(held) == 1 and held[0].hits == 12,
          "%d track(s)" % len(held))
    check("their velocity is their own walking, not the sweep",
          held and abs(held[0].vx - 12.0 / 0.114) < 0.35 * (12.0 / 0.114),
          "vx %.0f px/s, walking is %.0f" % (held[0].vx if held else 0,
                                             12.0 / 0.114))


    # The detector is trusted on sight: it knows a door from a person, and a
    # person standing still must not need to walk to be believed.
    tk = tracking.Tracker()
    for i in range(6):
        tk.update([{"x": 300, "y": 120, "w": 90, "h": 200, "conf": 0.95,
                    "src": "det"}], now + i * 0.114)
    check("a motionless person the detector sees is drawn immediately",
          len(tk.confirmed(now + 0.7)) == 1)

    # Live false positive measured on the fixed standing lamps: YOLO repeats a
    # plausible person box at 0.35..0.75, and its thermal peak is also inside
    # the body band.  Confidence below the instant-vouch floor therefore keeps
    # the observation, but makes it earn person status through real motion.
    tk = tracking.Tracker()
    for i in range(20):
        tk.update([{"x": 255 + (i % 2), "y": 180, "w": 42, "h": 121,
                    "conf": 0.75, "src": "det_weak", "max_c": 33.7}],
                  now + i * 0.114)
    check("a weak detector box fixed on a lamp is not a person",
          tk.confirmed(now + 2.2) == [] and len(tk.suppressed(now + 2.2)) == 1)

    tk = tracking.Tracker()
    for i in range(8):
        moving = tk.update(
            [{"x": 100 + 8 * i, "y": 120, "w": 50, "h": 160,
              "conf": 0.55, "src": "det_weak", "max_c": 33.0}],
            now + i * 0.114)
    check("a weak visible detection becomes a person once it walks",
          len(moving) == 1 and "det_weak" in moving[0].ever,
          "moved %.1f px" % (moving[0].moved if moving else -1))

    # Two boxes from the SAME sensor are two people and must stay two.
    two = tracking.merge([
        {"x": 100, "y": 50, "w": 60, "h": 180, "conf": 0.9, "src": "det"},
        {"x": 118, "y": 50, "w": 60, "h": 180, "conf": 0.8, "src": "det"}])
    check("a sensor reporting two boxes is never overruled into one",
          len(two) == 2, "%d group(s)" % len(two))


def ego_motion(d):
    """Where the whole picture went - measured on real frames, not mocked.

    The interesting number is not the accuracy, which phase correlation gives
    away for free. It is the failure: this reports the DOMINANT translation, so
    a big enough mover is a lie waiting to happen, and what saves the tracker is
    that the lie is small and announces itself in the response.
    """
    print("\nego motion")
    import egomotion

    fs = sorted(glob.glob(os.path.join(d, "*_rgb0.raw")))
    if len(fs) < 3:
        check("frames to measure ego motion on", False, "only %d in %s" % (len(fs), d))
        return
    ys = [np.frombuffer(open(f, "rb").read(), np.uint8).reshape(400, 640) for f in fs]

    g = egomotion.GlobalShift()
    check("the first frame of a stream has nothing to be a shift from",
          g.measure(ys[0]) is None)

    worst = 0.0
    for y in ys[1:]:
        e = g.measure(y)
        if e is not None:
            worst = max(worst, abs(e[0]), abs(e[1]))
    # captures/handwave3 is a bolted rig with a hand waving through it: the
    # true answer is zero, and a hand is not allowed to make it anything else.
    check("a bolted rig reads as a bolted rig, hand or no hand",
          worst < 1.0, "worst %.2f px" % worst)

    def shifted(a, dx, dy=0):
        return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]),
                              (640, 400), borderMode=cv2.BORDER_REFLECT)

    errs = []
    for px in (2, 10, 40):
        g2 = egomotion.GlobalShift()
        g2.measure(ys[0])
        e = g2.measure(shifted(ys[0], px))
        errs.append(1e3 if e is None else abs(e[0] - px))
    check("a known sweep comes back as itself", max(errs) < 0.5,
          "worst error %.2f px" % max(errs))

    # A mover over half the frame, rig still. It must not hand back the
    # mover's own displacement as if the camera had moved.
    a, b = ys[0].copy(), ys[0].copy()
    w = 320
    blob = cv2.GaussianBlur(np.full((400, w), 200, np.uint8), (31, 31), 0)
    a[:, 50:50 + w] = blob
    b[:, 75:75 + w] = blob
    g3 = egomotion.GlobalShift()
    g3.measure(a)
    e = g3.measure(b)
    check("a mover filling half the frame does not become a camera pan",
          e is None, "reported %s, response %.2f"
          % ("nothing" if e is None else "%+.1f px" % e[0], g3.response))
    check("and the frame it refused is counted, not silently dropped",
          g3.as_dict()["rejected"] == 1)


def raw16_channel(thermal, meta):
    """The 16-bit radiometric channel, and the promise that it changes nothing.

    --raw16 swaps what travels on the wire: the board sends the Lepton's own
    words instead of the 8-bit plane it derives from them, and this host rebuilds
    that plane. Everything downstream - the warp, fusion.c's thresholds, the
    students, the detector - was tuned against the board's version, so the whole
    design rests on the rebuild being identical rather than merely similar. That
    is what is checked here, and it can be checked without a board: a recorded
    8-bit frame says which code each pixel got, so the words that produced it are
    known to within one code bin, and running them back through the conversion
    must return the same codes.

    The interesting failure is off-by-one at the bin edges, which no eyeball on a
    picture would ever catch and which would move every threshold downstream by a
    code.
    """
    print("\n16-bit thermal channel")
    tmin, tmax = meta["tmin"], meta["tmax"]
    window = thermal_io.window_ck({"tmin": tmin, "tmax": tmax})
    lo, hi = window
    step = (hi - lo) / 255.0
    codes = np.frombuffer(thermal, np.uint8)

    check("the window is the driver's own centi-kelvin conversion",
          lo == round((tmin + 273.15) * 100) and hi == round((tmax + 273.15) * 100),
          "%d..%d cK over %d..%dC" % (lo, hi, tmin, tmax))
    check("a 16-bit code is far finer than the 8-bit plane it replaces",
          abs(step / 100.0 - (tmax - tmin) / 255.0) < 1e-6,
          "0.01C vs %.3fC per code" % (step / 100.0))

    # The word at the centre of each code's bin. The conversion rounds rather
    # than truncates - floor(v/step + 0.5) - so code c owns v/step in
    # [c-0.5, c+0.5) and its centre is c*step, not (c+0.5)*step. Getting that
    # backwards puts every representative word on a bin boundary, which is
    # exactly where a half-code disagreement hides.
    words = np.round(lo + codes.astype(np.float64) * step).astype(np.uint16)
    check("every recorded code survives the round trip through 16-bit",
          np.array_equal(thermal_io.codes8(words, window), codes),
          "%d pixels, %d distinct codes" % (codes.size, len(np.unique(codes))))

    # The boundaries themselves, which is where a rounding rule that disagreed
    # with the firmware's would show up and nowhere else.
    c = np.arange(255)
    just_under = np.floor(lo + (c + 0.5) * step - 1e-9).astype(np.uint16)
    just_over = np.ceil(lo + (c + 0.5) * step + 1e-9).astype(np.uint16)
    check("a word just under a bin boundary stays on the lower code",
          np.array_equal(thermal_io.codes8(just_under, window),
                         c.astype(np.uint8)))
    check("a word just over it moves to the upper code, by exactly one",
          np.array_equal(thermal_io.codes8(just_over, window),
                         (c + 1).astype(np.uint8)))

    # Outside the window there is nothing to preserve - that clipping is the
    # loss --raw16 exists to record around, so it must still be reproduced.
    outside = np.array([0, lo - 1, hi + 1, 65535], np.uint16)
    check("words outside the window clip exactly as the firmware clips them",
          list(thermal_io.codes8(outside, window)) == [0, 0, 255, 255])

    # agc8() is what the worker actually calls; it must agree with the module.
    frame = words.reshape(live.TH_H, live.TH_W)
    check("the live path and the offline readers use one conversion",
          live.agc8(frame, (tmin, tmax)) == thermal_io.codes8(frame, window).tobytes())

    check("celsius is absolute, with no window in it",
          abs(float(thermal_io.celsius(np.array([27315], "<u2"))[0])) < 1e-3
          and abs(float(thermal_io.celsius(np.array([31015], "<u2"))[0]) - 37.0) < 1e-3)

    # A session declaring one thing and holding another must fail rather than
    # produce twice as many mangled frames, which is what the old readers did.
    check("a session whose dtype and frame size disagree is refused",
          _raises(lambda: thermal_io.dtype_of({"thermal_dtype": "uint16_le",
                                               "thermal_frame_bytes": 19200})))
    check("a session that declares nothing at all is refused, not guessed",
          _raises(lambda: thermal_io.dtype_of({"thermal_frame_bytes": 12345})))
    check("a legacy session with no declaration still reads as uint8",
          thermal_io.dtype_of({"thermal_frame_bytes": 19200}) == np.dtype("u1"))


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


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

    thermogram = pipe.thermal_image()
    check("the thermal product is a registered full-size colour frame",
          thermogram.shape == (live.OUT_H, live.OUT_W, 3)
          and thermogram.dtype == np.uint8)
    cover_full = cv2.resize(cover, (live.OUT_W, live.OUT_H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    smooth_codes = cv2.resize(treg, (live.OUT_W, live.OUT_H),
                              interpolation=cv2.INTER_LINEAR)
    palette = np.ctypeslib.as_array(pipe.palettes[pipe.palette_name]).reshape(256, 3)
    expected_thermal = palette[smooth_codes].copy()
    expected_thermal[~cover_full] = (24, 24, 24)
    check("the thermal product uses smooth interpolation before the palette",
          np.array_equal(thermogram, expected_thermal))
    check("outside the thermal footprint is no-reading grey, not false cold",
          not (~cover_full).any()
          or np.all(thermogram[~cover_full] == np.array([24, 24, 24], np.uint8)))
    inverted_y = (255 - np.frombuffer(y, np.uint8)).astype(np.uint8).tobytes()
    pipe.process(inverted_y, thermal)
    check("the pure thermal channel borrows no visible-camera detail",
          np.array_equal(thermogram, pipe.thermal_image()))
    pipe.process(y, thermal)
    check("the five product channels have one explicit, stable contract",
          live.CHANNELS == ("thermal", "rgb_radar", "thermal_radar", "fusion", "ai"))
    pipe.outline = False
    pipe.view = "fused"
    channel_fused = pipe.process(y, thermal).copy()
    products = {}
    for channel_name in live.CHANNELS:
        pipe.channel = channel_name
        products[channel_name] = live.compose_channel(pipe, channel_fused, yarr)
    check("thermal and thermal+radar share one clean thermogram base",
          np.array_equal(products["thermal"], products["thermal_radar"]))
    check("visible+radar and AI use the honest visible-luma base",
          np.array_equal(products["rgb_radar"], products["ai"])
          and np.array_equal(products["ai"][..., 0], yarr))
    check("the fusion product keeps the configured fusion base",
          np.array_equal(products["fusion"], channel_fused))
    check("only sensor-fusion products request raw radar drawing",
          [live.live_channels.get(c).raw_radar for c in live.CHANNELS]
          == [False, True, True, True, False]
          and [live.live_channels.get(c).ai_overlay for c in live.CHANNELS]
          == [True, False, True, True, True])
    pipe.channel = "fusion"

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

    pipe.channel = "thermal"
    check("the thermal-only channel is healthy without radar",
          levels(good)["channel"] == "ok")
    pipe.channel = "fusion"
    check("a radar product channel warns when radar is absent",
          levels(good)["channel"] == "warn")
    pipe.channel = "rgb_radar"
    rgb_channel = {c["name"]: c for c in live.health(
        pipe, dict(good, radar_proj=object()), now)}["channel"]
    check("the visible channel says that this build transports luma",
          rgb_channel["level"] == "warn" and "luma" in rgb_channel["text"])
    pipe.channel = "ai"
    check("the AI channel warns when no inference engine is available",
          levels(good)["channel"] == "warn")
    pipe.channel = "fusion"

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

    # the map layer (Yael's mapinit through perception.map_api). Absent without
    # --map, and when present the text must say WHICH fix: a failed stage is a
    # map-data problem, a boundary error is a deployment problem.
    check("no --map, no map check", "map" not in base)
    loc = {"latitude_deg": 31.7683, "longitude_deg": 35.2137}
    check("a map still initialising warns",
          levels(dict(good, map={"pending": True, "location": loc}))["map"] == "warn")
    okmap = {"ok": True, "location": loc, "geoid": {"undulation_m": 19.76},
             "ego_altitude_prior": {"orthometric_m": 778.2, "sigma_m": 5.4},
             "stages": [{"name": "geoid", "status": "ok"},
                        {"name": "priors", "status": "ok"},
                        {"name": "pose_init", "status": "skipped"}]}
    texts = {c["name"]: c for c in live.health(pipe, dict(good, map=okmap), now)}
    check("an initialised map reads green with its numbers",
          texts["map"]["level"] == "ok" and "19.76" in texts["map"]["text"],
          texts["map"]["text"])
    failed = dict(okmap, ok=False, stages=[{"name": "geoid", "status": "failed"},
                                           {"name": "priors", "status": "ok"}])
    texts = {c["name"]: c for c in live.health(pipe, dict(good, map=failed), now)}
    check("a failed stage fails and is named",
          texts["map"]["level"] == "fail" and "geoid" in texts["map"]["text"],
          texts["map"]["text"])
    texts = {c["name"]: c for c in live.health(
        pipe, dict(good, map={"ok": False, "deployment": True, "location": loc,
                              "error": "MapAPIUnavailable: no mapinit"}), now)}
    check("a missing package fails as a deployment problem",
          texts["map"]["level"] == "fail" and texts["map"]["text"].startswith("deployment:"),
          texts["map"]["text"])

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

        get("/set?channel=thermal_radar")
        check("/set switches product channel independently of calibration view",
              pipe.channel == "thermal_radar" and pipe.view == "operator")
        get("/set?channel=nonsense")
        check("/set ignores an unknown product channel",
              pipe.channel == "thermal_radar")
        get("/set?channel=fusion")

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

        check("the AI product defaults to silhouettes without debug evidence",
              pipe.ai_evidence is False and pipe.ai_silhouette is True)
        get("/set?evidence=1")
        check("/set can reveal student evidence without changing the engines",
              pipe.ai_evidence is True
              and pipe.ai_thermal and pipe.ai_radar and pipe.ai_fusion)

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
            check("/ui reports whether diagnostic student evidence is visible",
                  aicfg["evidence"] is True)
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
        get("/set?evidence=0")

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
        check("/ui publishes the product-channel contract from the backend",
              d["cfg"]["channel"] == "fusion"
              and [c["id"] for c in d["cfg"]["channels"]] == list(live.CHANNELS))

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
    ego_motion(HANDWAVE3)
    raw16_channel(thermal, meta)

    print()
    if FAILS:
        print("FAILED: %s" % ", ".join(FAILS))
        return 1
    print("all viewer checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
