#!/usr/bin/env python3
"""Live thermal/visible fusion: board streams, this host fuses, browser watches.

    ./live.py                       then open http://localhost:8088
    ./live.py --warp calib/warp.lut --gain 220

The board sends a hardware-JPEG of the visible frame (~11KB for 640x400, 4% of
raw) plus the thermal frame *uncompressed* - the thermal data is the measurement,
and lossy compression on radiometry is not a trade worth making for 17KB. That
puts a frame at ~30KB, which the FS-speed link carries faster than the Lepton
produces frames, so the stream is paced by the sensor rather than the pipe.

Fusion runs through libfusion.so - the same fusion.c that compiles into the
firmware - so what you see here is what the board will do once flashed.

Controls are live, no restart: http://localhost:8088/set?gain=250&eps=120&radius=5

The view selector is there to judge registration, which the fused image cannot
show you on its own - see the VIEWS comment below for why, and use blink/edges
during a calibration session rather than trusting how sharp the picture looks.
"""
import argparse
import ctypes
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
import serial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture  # noqa: E402  - reuse the bring-up that is known to survive
import detect   # noqa: E402
import radar_overlay  # noqa: E402

PORT = "/dev/ttyACM0"
LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host", "libfusion.so")

OUT_W, OUT_H = 640, 400
TH_W, TH_H = 160, 120

# Sent once. The raw REPL keeps globals between submissions, so the sensors are
# brought up a single time and the streaming batches below reuse them.
SETUP_CODE = capture._BRINGUP + r'''
out = sys.stdout.buffer
CHUNK = 4096
QUALITY = __Q__

# Last thermal arrival, on the board's own clock. The host cannot derive this:
# by the time a frame reaches it, the interval has been through a 4KB-chunked
# CDC write, a 500ms stall retry loop and the host's own scheduling, and the
# sensor cadence is buried under all three. ticks_ms() is a plain counter read,
# not a heap walk like gc.mem_free(), and small ints do not allocate - so unlike
# the heap reading this is safe to take every frame.
_t_prev = 0
# 500ms write timeouts hit on the previous frame. Nonzero means the host stopped
# draining the port, which is the only way host load reaches the thermal sensor:
# the board sits in out.write() instead of calling snapshot().
_stalled = 0


def stream(n, report=0):
    global _t_prev, _stalled
    # gc.mem_free() walks the whole 25MB heap and costs 227ms - measured on the
    # board 2026-08-09, and it is the same heap-sized cost as gc.collect(). Per
    # frame it took the stream from 8.8 to 3.2 fps. So the host asks for a reading
    # only every Nth batch, and it is taken here, before the frame loop, where a
    # stall is harmless: plain gaps of up to 2.5s between snapshots were measured
    # not to disturb the sensor (unlike a collect, which wedges it at any length).
    if report:
        sys.stdout.write("#HEAP %d\n" % gc.mem_free())
    for _ in range(n):
        t, was_torn = good_snapshot(lep)       # paces the loop at the thermal rate
        t_th = time.ticks_ms()
        j = rgb.snapshot().to_jpeg(quality=QUALITY)
        t_rgb = time.ticks_ms()
        tb = t.bytearray()
        # Two extra numbers, both differences taken on this clock so the host
        # never has to reason about ticks wrapping:
        #
        #   dt    since the previous thermal frame landed. The sensor's true
        #         cadence - 114ms, and 1824ms across an FFC.
        #   skew  from the thermal frame landing to the visible one being in
        #         hand. This is the pairing error the fusion inherits: the two
        #         planes it registers were not looked at at the same moment, and
        #         anything moving is displaced by whatever this says.
        #
        # skew is measured between *completions*, which is the honest thing to
        # report and not the whole story: the Lepton's frame is already ~one
        # VoSPI transfer old when snapshot() returns, while the PAG's is fresh.
        # So the true exposure gap is larger than this number, and this is its
        # lower bound.
        # A sixth field costs nothing. MicroPython's GC allocates in 16-byte
        # blocks, and both the 3-tuple and the 6-tuple, and both the short and
        # the long header string, land in the same block - so the whole clock is
        # free against the ~208 B/frame this loop already leaks. That mattered
        # enough to check: the leak is what drains the heap, an exhausted heap is
        # what fires the automatic collector, and a collect with the Lepton up
        # wedges it permanently.
        #
        # _stalled belongs to the PREVIOUS frame - this header is written before
        # this frame's payload, so its stall count does not exist yet. It is the
        # number that says whether the host starved the board, which is the one
        # form of host load that can reach the sensor.
        sys.stdout.write("#F %d %d %d %d %d %d\n" % (
            j.size(), len(tb), 1 if was_torn else 0,
            time.ticks_diff(t_th, _t_prev), time.ticks_diff(t_rgb, t_th), _stalled))
        _t_prev = t_th
        _stalled = 0
        for buf in (memoryview(j.bytearray()), memoryview(tb)):
            off = 0
            stalls = 0
            while off < len(buf):
                # out.write returns what the CDC actually took, and returns 0
                # when its 500ms no-progress timeout fires - the unsent tail is
                # then DISCARDED, never queued and never retried, with no
                # exception. Advancing by CHUNK regardless drops those bytes
                # while the header has already promised them, and the host reads
                # straight into the next frame. Measured on this board: a host
                # stall past 500ms costs exactly one CHUNK, every time.
                w = out.write(buf[off:off + CHUNK])
                if w:
                    off += w
                    stalls = 0
                    continue
                # Nothing moved for a full timeout. Retrying is right - that is
                # what recovers the frame - but not forever, and the limit is set
                # by the Lepton rather than by patience.
                #
                # Every millisecond in here is a millisecond snapshot() is not
                # being called. Measured 2026-08-09, plain gaps between snapshots
                # of 150/300/600/1000/1500/2500 ms were all harmless - 2500ms is
                # the largest gap this part is KNOWN to survive, and beyond it
                # there is no measurement, only hope. The old limit of 4 allowed
                # five 500ms timeouts, 2.5s in this loop alone, before the frame
                # loop and the host's round trip to submit the next batch were
                # added on top - so it could put the sensor outside the tested
                # envelope in exactly the situation where the host is already
                # struggling. Three timeouts is 1.5s, which leaves the rest of
                # the gap inside 2500ms.
                #
                # The trade is deliberate: giving up costs the tail of one frame
                # and a resync. Wedging the Lepton costs a full restart and is
                # unrecoverable any other way.
                stalls += 1
                _stalled += 1
                if stalls > 2:
                    return
    # The heap reading is the whole early-warning system. gc.collect() cannot be
    # called while the Lepton is up - it wedges the part permanently, measured
    # 2026-08-09 - so the automatic collector firing on an exhausted heap is a
    # guaranteed kill. This loop leaks ~208 B/frame, which is hours rather than
    # minutes, but hours is still finite. Reporting free memory lets the host tear
    # the bring-up down and stand it back up at a moment of its choosing, instead
    # of discovering the problem as a dead stream.
    sys.stdout.write("#BATCH\n")


sys.stdout.write("#READY %d %d %d %d %d %d\n" % (
    rgb.width(), rgb.height(), lep.width(), lep.height(), TMIN, TMAX))
'''

# Sent repeatedly. Deliberately bounded: an unbounded loop on the board keeps
# writing into a port nobody reads if this host dies, the CDC RX backs up, and
# the board wedges hard enough to need a physical replug. A batch self-terminates.
BATCH_CODE = "stream(%d, %d)\n"

# How often to ask the board what its heap looks like. The reading costs 227ms,
# so this is a rate, not a habit: every 10th batch is one reading per ~200 frames
# (~23s), which is ~1% of throughput to watch something that drains over hours.
HEAP_EVERY = 10


# ---------------------------------------------------------------- fusion via ctypes


# Both layouts must track fusion.h field for field. fusion_init() memsets
# sizeof(fusion_t) through this pointer, so a struct that is short by even one
# field is a heap overwrite here, not a wrong-looking image.
class Cfg(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in (
        "out_w", "out_h", "low_w", "low_h", "th_w", "th_h",
        "gf_radius", "gf_eps", "detail_radius", "detail_gain", "detail_invert",
        "agc_permille", "th_seg_rows", "badpix_thresh", "deadrow_flat", "deadrow_lift",
        "temporal_noise_mc", "temporal_frames", "y_temporal_knee", "y_temporal_frames",
        "show_uncovered", "out_rgb565")]


class Fusion(ctypes.Structure):
    _fields_ = [("cfg", Cfg)] + [(n, ctypes.c_void_p) for n in (
        "warp", "y_low", "t_prep", "t_reg", "cover",
        "a_q16", "b_q16", "s1", "s2", "blur")] + [
        ("agc_lo", ctypes.c_int32), ("agc_hi", ctypes.c_int32),
        ("agc_valid", ctypes.c_int),
        ("tmin_mc", ctypes.c_int32), ("tmax_mc", ctypes.c_int32),
        ("eps_q10", ctypes.c_int32), ("refl_mc", ctypes.c_int32),
        ("t_prev", ctypes.c_void_p), ("y_prev", ctypes.c_void_p),
        ("y_out", ctypes.c_void_p),
        ("t_prev_valid", ctypes.c_int), ("y_prev_valid", ctypes.c_int),
        ("temporal_moved", ctypes.c_int),
        ("row_bad", ctypes.c_void_p),
        ("rows_rebuilt", ctypes.c_int), ("have_frame", ctypes.c_int),
        ("palette", ctypes.c_void_p)]


class Temp(ctypes.Structure):
    _fields_ = [("milli_c", ctypes.c_int32), ("raw_milli_c", ctypes.c_int32),
                ("code_q8", ctypes.c_uint16),
                ("th_x", ctypes.c_int16), ("th_y", ctypes.c_int16),
                ("valid", ctypes.c_uint8), ("repaired", ctypes.c_uint8)]


class Region(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in ("min_milli_c", "max_milli_c", "mean_milli_c")] + \
               [(n, ctypes.c_int) for n in ("min_x", "min_y", "max_x", "max_y",
                                            "samples", "repaired")]


PALETTES = {"ironbow": "fusion_ironbow", "white": "fusion_whitehot",
            "black": "fusion_blackhot", "gray": "fusion_grayscale"}


class Pipeline:
    def __init__(self, args):
        self.lib = ctypes.CDLL(LIB)
        self.lib.fusion_init.argtypes = [ctypes.POINTER(Fusion), ctypes.POINTER(Cfg)]
        # y is c_void_p, not c_char_p, so that the pointer fusion_y_temporal()
        # returns can be passed straight through. A bytes object still converts.
        self.lib.fusion_process.argtypes = [ctypes.POINTER(Fusion),
                                            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.fusion_y_temporal.argtypes = [ctypes.POINTER(Fusion), ctypes.c_char_p]
        # c_void_p and never c_char_p: a c_char_p restype builds a Python bytes
        # by scanning to the first NUL, and a luma frame is full of them, so the
        # buffer handed on would be truncated at the first black pixel. That is
        # an out-of-bounds read inside fusion_process(), not a dim picture.
        self.lib.fusion_y_temporal.restype = ctypes.c_void_p
        self.lib.fusion_temporal_reset.argtypes = [ctypes.POINTER(Fusion)]
        self.lib.fusion_set_warp.argtypes = [ctypes.POINTER(Fusion), ctypes.c_char_p]
        self.lib.fusion_set_homography.argtypes = [ctypes.POINTER(Fusion),
                                                   ctypes.POINTER(ctypes.c_double)]
        self.lib.fusion_set_range.argtypes = [ctypes.POINTER(Fusion),
                                              ctypes.c_int32, ctypes.c_int32]
        self.lib.fusion_set_emissivity.argtypes = [ctypes.POINTER(Fusion),
                                                   ctypes.c_int32, ctypes.c_int32]
        self.lib.fusion_temp_at.argtypes = [ctypes.POINTER(Fusion), ctypes.c_int, ctypes.c_int,
                                            ctypes.POINTER(Temp)]
        self.lib.fusion_temp_region.argtypes = [ctypes.POINTER(Fusion)] + \
            [ctypes.c_int] * 4 + [ctypes.POINTER(Region)]
        self.lib.fusion_sizeof_state.restype = ctypes.c_size_t
        self.lib.fusion_sizeof_cfg.restype = ctypes.c_size_t

        # These two structs are hand-mirrored from fusion.h, and fusion_init()
        # memsets sizeof(fusion_t) through the pointer below. A mirror that has
        # drifted from the header is a heap overwrite, not a wrong-looking
        # picture, so it is worth refusing to start over.
        for name, mine, theirs in (("fusion_t", ctypes.sizeof(Fusion),
                                    self.lib.fusion_sizeof_state()),
                                   ("fusion_cfg_t", ctypes.sizeof(Cfg),
                                    self.lib.fusion_sizeof_cfg())):
            if mine != theirs:
                raise SystemExit(
                    "%s is %d bytes in libfusion.so but %d in live.py - the ctypes mirror "
                    "has drifted from fusion.h. Fix it before running; the layouts must "
                    "match field for field." % (name, theirs, mine))

        # fusion_process runs on the streamer thread while temperature queries
        # come off HTTP handler threads, and both touch t_prep and the warp
        # table. The queries are microseconds against a ~10ms frame, so a plain
        # lock costs nothing measurable and removes a torn read that would
        # otherwise surface as an occasional nonsense temperature.
        self.lock = threading.Lock()

        cfg = Cfg()
        self.lib.fusion_default_cfg(ctypes.byref(cfg))
        cfg.out_w, cfg.out_h = OUT_W, OUT_H
        cfg.low_w, cfg.low_h = OUT_W // 4, OUT_H // 4
        cfg.th_w, cfg.th_h = TH_W, TH_H
        cfg.detail_gain, cfg.gf_eps, cfg.gf_radius = args.gain, args.eps, args.radius
        cfg.agc_permille = args.agc
        cfg.detail_invert = 1 if args.palette == "black" else 0
        cfg.out_rgb565 = 0
        # Opt-in, and it must be set before fusion_init: the buffer it needs is
        # allocated there, so flipping the knee later cannot turn the filter on.
        cfg.y_temporal_knee = args.y_knee
        cfg.y_temporal_frames = args.y_frames
        self.y_knee = args.y_knee

        self.f = Fusion()
        if self.lib.fusion_init(ctypes.byref(self.f), ctypes.byref(cfg)) != 0:
            raise SystemExit("fusion_init failed")

        # The tables are only filled by fusion_init, so this has to come after it.
        # in_dll, not getattr: these are data symbols, and getattr on a CDLL
        # hands back a callable for the address instead of the array itself.
        self.palettes = {k: (ctypes.c_ubyte * (256 * 3)).in_dll(self.lib, v)
                         for k, v in PALETTES.items()}
        self.palette_name = args.palette
        self.set_palette(args.palette)

        self.set_emissivity(args.emissivity, args.reflected)

        # How the frame is presented. Display state, not pipeline state - the C
        # never sees it - but it lives here because this is the object the HTTP
        # handlers already reach and the streamer already holds.
        self.view = "fused"
        self.mix = 50
        self.outline = False
        self.blink_period = 0.5      # seconds per half-cycle; ~2 alternations/s
        self.show_detections = True
        self.show_radar = True
        # The whisker is the honest width of a radar detection's vertical
        # uncertainty. On by default precisely because the bare dot flatters the
        # sensor: elevation comes from a two-element aperture and is the least
        # trustworthy thing on the screen.
        self.radar_whisker = True

        if args.warp:
            with open(args.warp, "rb") as fp:
                lut = fp.read()
            want = 2 * 2 * cfg.low_w * cfg.low_h
            if len(lut) != want:
                raise SystemExit("warp LUT is %d bytes, expected %d" % (len(lut), want))
            self.lib.fusion_set_warp(ctypes.byref(self.f), lut)
            self.warped = True
        else:
            # placeholder: stretch the thermal frame over the whole output. Good
            # enough to see the pipeline run, useless for judging registration.
            H = (ctypes.c_double * 9)()
            H[0] = (TH_W - 1) / float(cfg.low_w - 1)
            H[4] = (TH_H - 1) / float(cfg.low_h - 1)
            H[8] = 1.0
            self.lib.fusion_set_homography(ctypes.byref(self.f), H)
            self.warped = False

        self.out = ctypes.create_string_buffer(OUT_W * OUT_H * 3)

    def tune(self, **kw):
        for k, v in kw.items():
            if hasattr(self.f.cfg, k):
                setattr(self.f.cfg, k, int(v))

    def set_palette(self, name):
        if name not in self.palettes:
            return
        # black-hot needs the detail layer flipped with it, or the embossed
        # texture reads as a positive laid over a negative
        self.palette_name = name
        self.f.cfg.detail_invert = 1 if name == "black" else 0
        self.lib.fusion_set_palette(ctypes.byref(self.f),
                                    ctypes.byref(self.palettes[name]))

    def set_range(self, tmin_c, tmax_c):
        """The sensor's own range, which auto-range re-picks every session. Every
        temperature below is scaled by it, so this has to follow the board."""
        self.lib.fusion_set_range(ctypes.byref(self.f),
                                  int(tmin_c * 1000), int(tmax_c * 1000))

    def set_emissivity(self, eps, refl_c):
        self.eps, self.refl = eps, refl_c
        self.lib.fusion_set_emissivity(ctypes.byref(self.f),
                                       int(round(eps * 1024)), int(refl_c * 1000))

    def process(self, y, thermal):
        with self.lock:
            if self.y_knee:
                # fusion.h's documented call site, which live.py never used:
                # setting cfg.y_temporal_knee only made fusion_init allocate the
                # buffer, so --y-knee reserved 256KB and filtered nothing. It
                # returns `y` unchanged when disabled, so the guard is only here
                # to skip a call, not to choose behaviour.
                y = self.lib.fusion_y_temporal(ctypes.byref(self.f), y)
            self.lib.fusion_process(ctypes.byref(self.f), y, thermal, self.out)
            return np.frombuffer(self.out, np.uint8,
                                 OUT_W * OUT_H * 3).reshape(OUT_H, OUT_W, 3).copy()

    def temporal_reset(self):
        """Drop both frame histories. For the caller that knows the previous
        frame has stopped being a statement about the same scene - a restart, a
        range change, an FFC."""
        with self.lock:
            self.lib.fusion_temporal_reset(ctypes.byref(self.f))

    def temp_at(self, x, y):
        t = Temp()
        with self.lock:
            rc = self.lib.fusion_temp_at(ctypes.byref(self.f), int(x), int(y), ctypes.byref(t))
        if rc != 0 or not t.valid:
            return None
        return {"c": t.milli_c / 1000.0, "raw": t.raw_milli_c / 1000.0,
                "tx": t.th_x, "ty": t.th_y, "repaired": bool(t.repaired)}

    def temp_region(self, x, y, w, h):
        """min/max/mean over a rectangle, plus where the peak sits.

        This is what turns a detection into a measurement: the box comes from the
        visible camera, and fusion_temp_region() maps it through the warp into the
        thermal frame. Which is also the catch - with the placeholder warp the
        rectangle lands wherever the stretch put it, so the number is real
        radiometry read from the wrong pixels. Callers must carry `warped`
        alongside anything they quote from here.
        """
        r = Region()
        with self.lock:
            rc = self.lib.fusion_temp_region(ctypes.byref(self.f), int(x), int(y),
                                             int(w), int(h), ctypes.byref(r))
        if rc != 0:
            return None
        return {"min": r.min_milli_c / 1000.0, "max": r.max_milli_c / 1000.0,
                "mean": r.mean_milli_c / 1000.0, "max_x": r.max_x, "max_y": r.max_y,
                "samples": r.samples, "repaired": r.repaired}

    def cover_grid(self):
        """The 0/1 thermal coverage mask on the low-res grid, straight out of the
        C. Read-only, and read under the lock like everything else that touches
        buffers fusion_process() is rewriting."""
        n = self.f.cfg.low_w * self.f.cfg.low_h
        with self.lock:
            buf = ctypes.string_at(self.f.cover, n)
        return np.frombuffer(buf, np.uint8).reshape(self.f.cfg.low_h, self.f.cfg.low_w)

    def treg_grid(self):
        """The registered thermal plane, low res. This is what the warp actually
        produced, before the guided filter borrows the visible camera's edges -
        which is the point: edges taken from t_reg belong to the thermal camera,
        so laying them over the visible image tests the registration rather than
        the sharpening."""
        n = self.f.cfg.low_w * self.f.cfg.low_h
        with self.lock:
            buf = ctypes.string_at(self.f.t_reg, n)
        return np.frombuffer(buf, np.uint8).reshape(self.f.cfg.low_h, self.f.cfg.low_w)

    def repaired_grid(self):
        """Low-res cells whose thermal sample came off a rebuilt row.

        The same test fusion_temp_at() applies per pixel, evaluated over the whole
        grid: walk the warp table and ask whether either row the bilinear stencil
        touches was reconstructed.

        Needed because a rebuilt row is a linear ramp spliced into real data, and
        the splice has a slope discontinuity at each end. That is a gradient, so
        the edge overlay draws it - measured on a real frame, the two busiest
        painted rows in the whole image were the boundaries of rebuilt blocks, and
        a fifth of the drawn edges were this artefact rather than the scene. An
        overlay meant to prove the warp is right must not invent its own lines.
        """
        lw, lh, th = self.f.cfg.low_w, self.f.cfg.low_h, self.f.cfg.th_h
        with self.lock:
            warp = np.frombuffer(ctypes.string_at(self.f.warp, 4 * lw * lh),
                                 np.uint16).reshape(lh, lw, 2)
            bad = np.frombuffer(ctypes.string_at(self.f.row_bad, th), np.uint8)

        if not bad.any():
            return np.zeros((lh, lw), bool)

        qy = warp[..., 1]
        valid = (warp[..., 0] != 0xFFFF) & (qy != 0xFFFF)
        y0 = np.clip(np.where(valid, qy, 0) >> 8, 0, th - 1)
        y1 = np.clip(y0 + 1, 0, th - 1)
        return valid & (bad[y0].astype(bool) | bad[y1].astype(bool))

    def prep_frame(self):
        """The thermal frame after deband, bad-pixel and dead-row repair, at the
        sensor's own 160x120. This is what the measurement path samples, so it is
        also what the health checks should judge - clipping counted on the raw
        frame would count the dead rows as saturated scene."""
        n = self.f.cfg.th_w * self.f.cfg.th_h
        with self.lock:
            buf = ctypes.string_at(self.f.t_prep, n)
        return np.frombuffer(buf, np.uint8).reshape(self.f.cfg.th_h, self.f.cfg.th_w)

    def frame_stats(self):
        r = Region()
        with self.lock:
            rc = self.lib.fusion_temp_region(ctypes.byref(self.f), 0, 0, OUT_W, OUT_H,
                                             ctypes.byref(r))
        if rc != 0:
            return None
        return {"min": r.min_milli_c / 1000.0, "max": r.max_milli_c / 1000.0,
                "mean": r.mean_milli_c / 1000.0,
                "delta": (r.max_milli_c - r.min_milli_c) / 1000.0,
                "max_x": r.max_x, "max_y": r.max_y,
                "min_x": r.min_x, "min_y": r.min_y}


# ---------------------------------------------------------------- registration views
#
# Judging registration from the fused picture alone is close to impossible, and
# that is not a UI complaint - it is the pipeline working as designed. The guided
# filter deliberately borrows the visible camera's edges to sharpen the thermal
# layer, so a *misregistered* frame still comes out with crisp edges in all the
# right places. It just colours the wrong side of them. The image looks fine and
# the measurement is wrong, which is the worst failure mode available.
#
# So these views break the two layers apart again:
#
#   blink   alternate visible and fused. Misalignment shows up as things jumping;
#           the eye is far better at spotting motion than at spotting offset.
#   mix     the same comparison, continuous, for judging how much offset there is
#   edges   thermal edges from t_reg drawn over the plain visible image. The
#           strongest test of the three: those edges belong to the thermal camera,
#           so if they land on the visible object's outline the warp is right.
#
# All host-side numpy. None of this belongs in fusion.c - that file compiles into
# firmware, and these are display questions asked while calibrating.

VIEWS = ("fused", "visible", "blink", "mix", "edges")


def thermal_edges(treg, w, h, pct=97.0, exclude=None):
    """Gradient magnitude of the registered thermal plane, thresholded.

    The threshold is a percentile rather than a constant because thermal contrast
    varies enormously between scenes: a fixed one either paints the whole frame on
    a high-contrast scene or nothing at all on a flat wall. The consequence is
    worth knowing when reading the view: a percentile always paints about
    (100-pct)% of the frame, so on a scene with no thermal structure what you see
    is noise, not edges. FLOOR below is the guard against that.

    `exclude` is a low-res mask of cells sampled from rebuilt rows. Those are
    dropped before the threshold is chosen as well as after, since otherwise the
    artefact's own gradients set the level for everything else.
    """
    # A gradient of this many codes per output pixel is roughly one thermal code
    # across one thermal pixel - below it there is no structure, only sensor
    # noise, and painting it would be inventing edges the camera did not see.
    FLOOR = 6.0

    t = cv2.resize(treg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    gx = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
    m = np.hypot(gx, gy)

    keep = np.ones((h, w), bool)
    if exclude is not None and exclude.any():
        # Dilate by two cells, not one. The artefact is at the *boundary* of a
        # rebuilt block and it spreads: the bilinear upsample to full resolution
        # reaches a cell either side, and the Sobel kernel another output pixel
        # beyond that. Measured with one cell, the two busiest painted rows in the
        # frame were still splice boundaries. Two cells costs coverage - about a
        # quarter of this sensor's frame is excluded - but a quarter of its
        # thermal rows really are reconstructed, so that is the honest number.
        grown = cv2.dilate(exclude.astype(np.uint8), np.ones((5, 5), np.uint8))
        keep = cv2.resize(grown, (w, h), interpolation=cv2.INTER_NEAREST) == 0

    vals = m[keep]
    if vals.size == 0:
        return np.zeros((h, w), bool)

    thr = max(float(np.percentile(vals, pct)), FLOOR)
    if float(vals.max()) < FLOOR:
        return np.zeros((h, w), bool)       # nothing in the thermal frame to draw
    return (m >= thr) & keep


def coverage_outline(cover, w, h):
    """Boundary of the thermal footprint: where the thermal camera stops seeing.

    Worth having on screen whatever the view. Without it, the plain-luma fallback
    outside the footprint reads as 'the thermal layer is grey here', when what it
    actually means is 'there is no thermal data here at all'.
    """
    c = cv2.resize(cover, (w, h), interpolation=cv2.INTER_NEAREST)
    k = np.ones((3, 3), np.uint8)
    return cv2.dilate(c, k) != cv2.erode(c, k)


def compose(view, fused, y, treg=None, cover=None, mix=50, phase=True, exclude=None):
    """Build the frame the browser sees. `fused` is never modified in place."""
    if view == "mix":
        a = max(0.0, min(1.0, mix / 100.0))
        out = (fused.astype(np.float32) * a
               + np.repeat(y[:, :, None], 3, 2).astype(np.float32) * (1.0 - a)).astype(np.uint8)
    elif view == "visible" or (view == "blink" and not phase):
        out = np.repeat(y[:, :, None], 3, 2)
    elif view == "edges":
        out = np.repeat(y[:, :, None], 3, 2)
        if treg is not None:
            out[thermal_edges(treg, out.shape[1], out.shape[0], exclude=exclude)] = (80, 255, 255)
    else:
        out = fused.copy()

    if cover is not None:
        out[coverage_outline(cover, out.shape[1], out.shape[0])] = (255, 210, 40)
    return out


# ---------------------------------------------------------------- health
#
# The offline suites prove the pipeline is correct on frames that sit still. None
# of them can tell you whether the thing in front of you right now is producing
# numbers worth writing down - that depends on the sensor's state, the range it
# auto-picked, whether the link is tearing, and whether a warp was ever loaded.
#
# So these are the checks that can only be made live, and the distinction the
# levels draw is deliberately about *trust in the reading*, not about tidiness:
#
#   fail  the numbers on screen are wrong or absent. Do not record anything.
#   warn  the numbers are usable but qualified, and the qualification changes how
#         they should be read - not a nag to be cleared.
#   ok    nothing known to be wrong.
#
# A green panel is not a claim that the measurement is accurate. It is a claim
# that none of the failures this code can see are happening.

STALE_S = 5.0           # no frame for this long and the stream is considered dead
# The sensor's own rate; the pipeline cannot beat it. Measured on the board
# 2026-08-06 over 5220 frames: the inter-frame interval is a three-valued delta
# function - 113 ms (326x), 114 ms (4742x), 115 ms (148x), and 1824 ms on the
# three FFCs. Median 114 ms = 8.772 fps, not the 8.82 the datasheet implies.
#
# The 70% threshold below is weaker than it looks, and the constant cannot fix
# that on its own: whether it can fire at all depends on the fps averaging
# window. Over that same run a 10-frame window dipped to 3.5 fps, a 30-frame
# window to 5.8, and a 100-frame window never below 7.6. So with a long window
# it never fires; with a short one it fires only on FFC gaps, which makes it an
# accidental FFC detector wearing a link-health label. Nothing else in ten
# minutes came within 25% of it.
LEPTON_FPS = 8.772


def health(pipe, state, now):
    """Returns a list of {name, level, text}. Pure enough to test off a dict."""
    out = []

    def add(name, level, text):
        out.append({"name": name, "level": level, "text": text})

    # --- the link
    err = state.get("error")
    last = state.get("last_frame_t")
    fps = state.get("fps", 0.0)
    restarting = state.get("restarting_since")
    if restarting is not None:
        # Recovering is not failing. A restart takes ~13s against a 5s stale
        # threshold, so without this branch every successful defence reports as a
        # dead stream - which is how a health panel gets ignored.
        add("stream", "warn", "restarting the board: %s (%.0fs)"
            % (state.get("last_restart", "fault"), now - restarting))
    elif err:
        add("stream", "fail", err)
    elif last is None:
        add("stream", "warn", "waiting for the first frame")
    elif now - last > STALE_S:
        add("stream", "fail", "no frame for %.0f s" % (now - last))
    elif fps < LEPTON_FPS * 0.7:
        add("stream", "warn", "%.1f fps, expected %.1f" % (fps, LEPTON_FPS))
    else:
        add("stream", "ok", "%.1f fps" % fps)

    # --- framing. A resync means bytes went missing on the wire and a frame was
    # dropped to get back in step. The stream survives it, which is the point,
    # but silently surviving is how this went unnoticed for so long - a link
    # that needs to resynchronise is not a healthy link, so say so.
    rs = state.get("resyncs", 0)
    if rs:
        add("framing", "warn", "%d resync%s: %s" % (
            rs, "" if rs == 1 else "s", state.get("last_resync", "")))
    else:
        add("framing", "ok", "in step")

    # --- the two sensors' timing, from the board's clock. Separate from the
    # "stream" check above on purpose: that one is about the link keeping up,
    # this one is about whether the pair being fused was looked at at the same
    # moment. A link can be perfect while the pairing is not.
    tm = timing(state, now)
    if tm["thermal_ms"]:
        med = tm["thermal_ms"]["median"]
        want = tm["expected_ms"]
        if abs(med - want) > 0.25 * want:
            add("cadence", "warn", "thermal frame every %d ms, expected %.0f" % (med, want))
        else:
            add("cadence", "ok", "thermal %d ms (%.2f fps)" % (med, tm["thermal_fps"]))
    if tm["skew_frac"] is not None:
        sm = tm["skew_ms"]["median"]
        if tm["skew_frac"] > SKEW_WARN:
            add("pairing", "warn", "visible frame lags thermal by %d ms (%.0f%% of a "
                "frame) - anything moving is displaced before the warp sees it"
                % (sm, 100 * tm["skew_frac"]))
        else:
            add("pairing", "ok", "visible +%d ms after thermal (%.0f%% of a frame)"
                % (sm, 100 * tm["skew_frac"]))
    # --- the sensor being starved by host load. This is the check the whole
    # board clock earns its keep on. Everything else about a slow host is a
    # frame rate complaint; this one is the part that cannot be undone, because
    # a wedged Lepton needs a fresh csi.CSI() object and no amount of retrying,
    # re-arming or soft re-init has ever recovered one.
    if tm["starved"]:
        add("thermal load", "fail",
            "%d thermal gap%s past %dms, worst %dms - the board was not being "
            "drained and the sensor went unserviced beyond anything measured safe"
            % (tm["starved"], "" if tm["starved"] == 1 else "s",
               tm["safe_gap_ms"], tm["last_starve_ms"] or 0))
    elif tm["stalls"]:
        add("thermal load", "warn",
            "%d write stall%s - the host stopped draining the port for 500ms at a "
            "time. Frames are lost and the sensor waits; reduce host work before "
            "this reaches %dms" % (tm["stalls"], "" if tm["stalls"] == 1 else "s",
                                   tm["safe_gap_ms"]))
    elif tm["thermal_ms"]:
        # The longest gap excluding FFCs. An FFC is a different mechanism and
        # not a hazard: the sensor stops delivering for 1824ms but snapshot()
        # BLOCKS, so the consumer stays parked in the driver rather than leaving
        # the part unserviced. Quoting it here would read as the sensor being
        # three quarters of the way to danger every three minutes.
        add("thermal load", "ok", "no write stalls, longest gap %dms of %dms"
            % (tm["thermal_ms"]["max"], tm["safe_gap_ms"]))

    if tm["ffc_ago_s"] is not None and tm["ffc_ago_s"] < 3.0:
        # The sensor needs time to settle after the shutter. Quoting through one
        # is the kind of error that survives into a report.
        add("ffc", "warn", "shutter closed %.1fs ago - let the sensor settle "
            "before quoting a reading" % tm["ffc_ago_s"])

    # --- short or unreadable JPEGs. The visible signature of bytes lost on the
    # wire: the board announced a length and the host got fewer usable bytes.
    # Framing can stay in step through this, so it does not always show up as a
    # resync - and a frame that silently vanishes makes the link look healthier
    # than it is. The known cause on this bench is another process opening the
    # port and toggling DTR, which makes the CDC discard what it has queued;
    # keep ModemManager off the device (ID_MM_DEVICE_IGNORE) before blaming the
    # firmware.
    bad = state.get("bad_jpeg", 0)
    if bad:
        add("jpeg", "warn", "%d frame%s arrived corrupt - bytes lost on the wire"
            % (bad, "" if bad == 1 else "s"))

    # --- the render thread. Dropping is by design when it falls behind, but the
    # viewer must say so: a page showing every third frame while quoting the
    # board's fps is describing a stream nobody is watching.
    dropped, rendered = state.get("dropped", 0), state.get("rendered", 0)
    if rendered:
        rate = dropped / float(dropped + rendered)
        if rate > 0.25:
            add("render", "fail", "%.0f%% of frames dropped before display" % (100 * rate))
        elif rate > 0.02:
            add("render", "warn", "%.0f%% dropped before display" % (100 * rate))
        else:
            add("render", "ok", "%d rendered" % rendered)

    # --- detection. Reported separately from the boxes themselves because the
    # thing worth knowing is whether the temperature beside a label means
    # anything, and without a calibrated warp it does not.
    dt = state.get("detect_t")
    if dt is not None:
        age = now - dt
        n_det = len(state.get("detections") or [])
        if state.get("detect_error"):
            add("detect", "fail", state["detect_error"])
        elif age > DETECT_STALE_S * 4:
            add("detect", "warn", "no detection for %.0fs" % age)
        elif not pipe.warped:
            add("detect", "warn", "%d box%s, %.0fms - temperatures are read through "
                "the placeholder warp and belong to the wrong pixels"
                % (n_det, "" if n_det == 1 else "es", state.get("detect_ms", 0)))
        else:
            add("detect", "ok", "%d box%s, %.0fms"
                % (n_det, "" if n_det == 1 else "es", state.get("detect_ms", 0)))

    # --- VoSPI tearing. Unlike the dead rows this really is random, and a torn
    #     frame is stale data in part of the image, not a marked defect.
    #
    #     Measured 2026-08-06: 0 torn frames in 5220 over ten minutes, so on this
    #     link tearing is rare rather than routine - read a green pill here as
    #     the expected state, not as a reassurance. One structural caveat: the
    #     detector evaluates the segment seams at rows 29/59/89, and the dead-row
    #     run 55..63 straddles the 59/60 seam. That seam is therefore measured
    #     across two stuck rows and cannot report a tear there at all.
    win = state.get("torn_window") or []
    if win:
        rate = sum(win) / float(len(win))
        if rate > 0.20:
            add("tearing", "fail", "%.0f%% of frames torn" % (100 * rate))
        elif rate > 0.02:
            add("tearing", "warn", "%.0f%% of frames torn" % (100 * rate))
        else:
            add("tearing", "ok", "%.0f%% torn" % (100 * rate))

    if not pipe.f.have_frame:
        return out

    # --- dead rows. On this unit the count is 0 or 14 and holds for a whole
    #     session, so a count that moves frame to frame is the detector following
    #     the scene instead of the defect - which is the failure the flat+lifted
    #     test was written to avoid, and worth catching live rather than in a
    #     capture review three weeks later.
    hist = state.get("rows_window") or []
    rows = pipe.f.rows_rebuilt
    total = pipe.f.cfg.th_h
    if len(set(hist)) > 1:
        add("dead rows", "fail", "count varies (%s) - the detector is tracking the scene"
            % "/".join(str(v) for v in sorted(set(hist))))
    elif rows == 0:
        add("dead rows", "ok", "none")
    else:
        add("dead rows", "warn", "%d of %d rows rebuilt - readings there are "
            "interpolated" % (rows, total))

    # --- board heap. The one check that predicts a failure instead of reporting
    # one. gc.collect() with the Lepton up wedges the part permanently, so the
    # automatic collector firing on an exhausted heap is fatal - and the heap
    # drains steadily because the streaming loop cannot be made to allocate
    # nothing. The supervisor restarts the bring-up before that happens; this
    # says how much room is left and whether it has had to.
    free = state.get("heap_free")
    restarts = state.get("restarts", 0)
    if free is None:
        add("board heap", "ok", "no reading yet" if restarts == 0 else
            "%d restart%s so far" % (restarts, "" if restarts == 1 else "s"))
    else:
        mb = free / (1 << 20)
        note = "%.1fMB free" % mb
        if restarts:
            note += ", %d restart%s (%s)" % (restarts, "" if restarts == 1 else "s",
                                             state.get("last_restart", "fault"))
        # The floor is where the supervisor acts, so approaching it is normal
        # operation and not worth a warning until it is close enough to be soon.
        add("board heap", "warn" if free < HEAP_FLOOR * 2 else "ok", note)

    # --- registration
    if not pipe.warped:
        add("registration", "warn", "placeholder warp - the thermal layer is "
                                    "stretched, not registered to the visible one")
    else:
        add("registration", "ok", "calibrated warp")

    cov = pipe.cover_grid().mean()
    if cov <= 0.0:
        add("coverage", "fail", "no thermal coverage anywhere")
    elif cov < 0.30:
        add("coverage", "warn", "%.0f%% of the frame has thermal data" % (100 * cov))
    else:
        add("coverage", "ok", "%.0f%% covered" % (100 * cov))

    # --- the sensor range, which is what every reading is scaled by
    lo, hi = state.get("range", (0, 0))
    if hi <= lo:
        add("range", "fail", "no sensor range reported - readings are meaningless")
    else:
        per_code = (hi - lo) / 255.0
        # Measured on this part 2026-08-06, 250-frame runs: NETD is 33 mK, not
        # the ~50 mK the datasheet implies, and it is the same in both gain
        # modes. So a code finer than ~0.033 C resolves noise; much coarser than
        # ~0.2 C and the auto-range has given away resolution it did not need to.
        #
        # NETD is the wrong figure for trusting an *absolute* reading, though.
        # Over the same runs the common-mode-removed temporal spread was
        # 126-148 mK - 4x worse. Two readings seconds apart are comparable to
        # ~0.15 C; NETD only bounds frame-to-frame differencing.
        lvl = "warn" if per_code > 0.2 else "ok"
        add("range", lvl, "%d..%d C, %.3f C/code%s" % (lo, hi, per_code,
            " - re-run auto-range for finer steps" if lvl == "warn" else ""))

    # --- clipping. Counted after the repair so the dead rows are not mistaken
    #     for a saturated scene. Pixels pinned at either end carry no temperature
    #     at all: they are 'at least this hot', which is not a measurement.
    t = pipe.prep_frame()
    clipped = float(((t == 0) | (t == 255)).mean())
    if clipped > 0.05:
        add("clipping", "fail", "%.1f%% of the thermal frame is pinned at the range "
            "ends - those pixels have no temperature" % (100 * clipped))
    elif clipped > 0.005:
        add("clipping", "warn", "%.1f%% pinned at the range ends" % (100 * clipped))
    else:
        add("clipping", "ok", "%.2f%% pinned" % (100 * clipped))

    # --- AGC. Does not touch the readings, which is exactly why it needs saying:
    #     the picture stops being an absolute temperature map while the hover
    #     numbers carry on being correct, and that mismatch is easy to misread.
    if pipe.f.cfg.agc_permille > 0:
        add("agc", "warn", "scene AGC on - tone is scene-relative, readings are not")

    if pipe.eps < 1.0:
        add("emissivity", "warn", "eps %.2f, reflected %.0f C - readings are corrected"
            % (pipe.eps, pipe.refl))

    return out


# ---------------------------------------------------------------- board stream


def _board_error(raw):
    """A real exception from the board, or a payload byte that happens to be 0x04?

    The raw REPL ends stdout with \\x04, so a line starting with one is how a
    traceback announces itself. But 0x04 occurs constantly inside JPEG and
    thermal data, and the moment framing slips, _line() starts handing back
    payload. Taking that at face value invents a board fault that never
    happened - and sends you debugging firmware that is working correctly.
    """
    if b"Traceback" in raw:
        return True
    if not raw.startswith(b"\x04"):
        return False
    body = raw[1:].strip()
    return bool(body) and all(c == 9 or 32 <= c < 127 for c in body)


class _PlannedRestart(Exception):
    """Not a failure: the supervisor is standing the board back up on purpose."""


# Free heap below this and the board gets restarted before the automatic collector
# can fire. The collector is the hazard, not the memory: a gc.collect() with the
# Lepton up wedges it permanently and no soft re-init recovers it - only a fresh
# csi.CSI() object does, which is precisely what a restart builds.
#
# 4MB against a measured ~208 B/frame is about 19000 frames, 36 minutes, of slack
# after the trigger. Deliberately enormous. The restart costs ~10s of bring-up and
# happens once every few hours, so there is nothing to be gained by cutting it
# fine and a dead stream to be lost by getting it wrong.
HEAP_FLOOR = 4 << 20

# The largest gap between snapshots this part is KNOWN to survive. Measured
# 2026-08-09: plain sleeps of 150/300/600/1000/1500/2500 ms between snapshots
# were all harmless, and a gc.collect() of any length wedges it. 2500ms is
# therefore the edge of the tested envelope, not a limit anyone has found - past
# it there is simply no measurement. The board's own thermal interval is the only
# way to see it, since a gap on this side is indistinguishable from a slow link.
LEPTON_SAFE_GAP_MS = 2500


class Latest:
    """One-slot handoff between threads: the newest item wins, older ones drop.

    A queue is wrong here in both of its usual forms. Unbounded, it grows without
    limit the moment the consumer falls behind, and every frame it eventually
    renders is already stale - a live viewer that is four seconds behind is not
    showing you the board. Bounded-and-blocking puts the consumer's latency
    straight back onto the producer, which is precisely the coupling this split
    exists to break: the serial reader must never wait for anything.

    So dropping is the correct behaviour, not a compromise. It is counted, though
    - a viewer quietly showing every third frame while reporting the board's fps
    would be lying about what you are looking at.
    """

    def __init__(self):
        self.cv = threading.Condition()
        self.item = None
        self.dropped = 0
        self.closed = False

    def put(self, item):
        with self.cv:
            if self.item is not None:
                self.dropped += 1
            self.item = item
            self.cv.notify()

    def get(self, timeout=0.5):
        with self.cv:
            if self.item is None:
                self.cv.wait(timeout)
            item, self.item = self.item, None
            return item

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()


class Renderer(threading.Thread):
    """Everything between a received frame and the JPEG the browser gets.

    Split off the reader so that host compute cannot back-pressure the board's
    CDC. It also gives the detector somewhere to live: at 68-99ms a detection is
    most of a 114ms frame period, and running it inline would have guaranteed the
    500ms write stall that costs a frame's tail.
    """

    def __init__(self, pipe, state, work, detector=None, radar=None,
                 radar_proj=None, video=None):
        super().__init__(daemon=True)
        self.pipe, self.state, self.work = pipe, state, work
        self.stop = threading.Event()
        # The radar overlay is drawn here rather than in compose() because it is
        # not part of the fused image: it is a separate sensor annotated ON TOP
        # of whichever view is showing, and it must never end up inside the
        # frame the temperature is read from.
        self.radar, self.radar_proj, self.video = radar, radar_proj, video
        # The detector gets its own thread and its own one-slot handoff for the
        # same reason this class exists: it is the slowest stage, and the frame
        # rate should be set by the sensor rather than by the network.
        self.det_in = Latest() if detector else None
        self.det = detector
        if detector:
            self.det_thread = threading.Thread(target=self._detect_loop, daemon=True)
            self.det_thread.start()

    def _detect_loop(self):
        while not self.stop.is_set():
            item = self.det_in.get()
            if item is None:
                continue
            y = item
            try:
                dets = self.det(y)
            except Exception as e:                  # a bad frame must not end detection
                self.state["detect_error"] = "%s: %s" % (type(e).__name__, e)
                continue
            # The temperature is the whole reason for the box. Taken here rather
            # than in the drawing code so that /detections and the overlay quote
            # the same number, and so a slow region query costs the detector's
            # thread rather than the frame rate.
            for d in dets:
                r = self.pipe.temp_region(d["x"], d["y"], d["w"], d["h"])
                if r is None or r["samples"] == 0:
                    d["no_thermal"] = True
                    continue
                d["max_c"], d["mean_c"] = round(r["max"], 1), round(r["mean"], 1)
                d["max_x"], d["max_y"] = r["max_x"], r["max_y"]
                d["repaired"] = bool(r["repaired"])
            self.state["detections"] = dets
            self.state["detect_ms"] = round(self.det.ms, 1)
            self.state["detect_t"] = time.time()

    def run(self):
        while not self.stop.is_set():
            item = self.work.get()
            if item is None:
                continue
            jpg, thermal = item

            y = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
            if y is None or y.shape != (OUT_H, OUT_W):
                # A short or corrupt JPEG is the visible signature of bytes lost
                # on the wire. Count it: without this the frame simply vanishes
                # and the link looks healthier than it is.
                self.state["bad_jpeg"] = self.state.get("bad_jpeg", 0) + 1
                continue

            p = self.pipe
            rgb = p.process(y.tobytes(), thermal)

            rw = self.state.setdefault("rows_window", [])
            rw.append(p.f.rows_rebuilt)
            del rw[:-60]

            if p.view != "fused" or p.outline:
                edges = p.view == "edges"
                rgb = compose(p.view, rgb, y,
                              treg=p.treg_grid() if edges else None,
                              exclude=p.repaired_grid() if edges else None,
                              cover=p.cover_grid() if p.outline else None,
                              mix=p.mix,
                              phase=int(time.time() / p.blink_period) % 2 == 0)

            # The recording feeds calibration picking, so it must not carry the
            # guessed radar overlay or detection boxes: a pixel clicked next to
            # a burned-in marker is a correspondence derived from the very
            # projection being solved for. Copy before anything is drawn;
            # radar_calib_web.py refuses frames that lack the 'clean' flag.
            clean = rgb.copy() if self.video is not None else None

            if self.det_in is not None:
                # The detector reads the plain visible luma, not the fused frame.
                # The COCO weights were trained on natural images; a false-colour
                # thermal composite is further from that than grey is, and the
                # boxes are wanted in visible-camera coordinates anyway - that is
                # the frame fusion_temp_region() maps through the warp.
                self.det_in.put(y)
                if p.show_detections:
                    dets = self.state.get("detections") or []
                    age = time.time() - self.state.get("detect_t", 0)
                    # Boxes outlive the frame they were found in by design - the
                    # detector runs slower than the stream. Past this they are a
                    # claim about a scene that may be gone, so drop them rather
                    # than draw a stale rectangle over a moved object.
                    if age < DETECT_STALE_S:
                        detect.annotate(rgb, dets, p.warped)

            if self.radar is not None and self.pipe.show_radar:
                fr = self.radar.get()
                if fr is not None:
                    d, off, al = radar_overlay.annotate(
                        rgb, fr["points"], self.radar_proj,
                        show_whisker=self.pipe.radar_whisker)
                    self.state["radar_drawn"] = d
                    self.state["radar_offscreen"] = off
                    self.state["radar_aliased"] = al
                    self.state["radar_frame"] = fr["frame_number"]
                self.state["radar_frames"] = self.radar.frames
                self.state["radar_dropped"] = self.radar.dropped_bytes
                if self.radar.error:
                    self.state["radar_error"] = self.radar.error

            if self.video is not None:
                # The radar frame number this picture was drawn against ties the
                # two recordings together even if a timestamp is ever doubted.
                self.video.write(clean, {
                    "radar_frame": self.state.get("radar_frame"),
                    "view": self.pipe.view,
                    "clean": True,
                })
                self.state["video_frames"] = self.video.frames
                if self.video.error:
                    self.state["video_error"] = self.video.error

            ok, enc = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                self.state["frame"] = enc.tobytes()
            self.state["rendered"] = self.state.get("rendered", 0) + 1
            self.state["dropped"] = self.work.dropped


# Boxes older than this are not drawn. The detector runs at ~10Hz against an
# 8.8Hz stream, so in normal operation a box is at most one frame old; this only
# fires when detection has actually fallen behind or died.
DETECT_STALE_S = 1.0


class Streamer(threading.Thread):
    """Reads framed board output as fast as the port will give it.

    Drains with in_waiting rather than a fixed read size. A fixed read blocks for
    the whole port timeout collecting bytes it may never get, which back-pressures
    the board's CDC; a blocked write there starves TinyUSB's tud_task (serviced
    from the MicroPython scheduler, not an ISR) and the board falls off the bus.
    That is the failure this project spent a long time chasing.
    """

    def __init__(self, port, pipeline, quality, state, work, batch=20):
        super().__init__(daemon=True)
        self.port, self.pipe, self.quality, self.state = port, pipeline, quality, state
        self.work = work
        self.batch = batch
        self.buf = bytearray()
        self.stop = threading.Event()
        # Failure forensics. A stall reports as a bare timeout, and the counters
        # in state are not enough to tell the two causes apart: the board going
        # quiet mid-write looks identical to this host losing count of the #F
        # headers and waiting for frames the board already finished sending.
        # What separates them is the residual buffer and what was owed at the
        # time, so keep both current.
        self.last_line, self.last_line_t = b"", 0.0
        self.headers, self.pending = 0, 0

    def _fill(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            n = self.s.in_waiting
            chunk = self.s.read(n if n else 1)
            if chunk:
                self.buf += chunk
                return True
        return False

    def _line(self, timeout=10.0):
        """One framed line. The timeout is a parameter because the two callers
        want very different limits: streaming lines arrive every ~115ms, while
        the one-off bring-up takes ~10s to reach #READY. The raw REPL acks a
        submission with OK *before* running it, so the ack tells you nothing
        about how long the code will take."""
        while True:
            i = self.buf.find(b"\n")
            if i >= 0:
                out = bytes(self.buf[:i])
                del self.buf[:i + 1]
                self.last_line, self.last_line_t = out, time.time()
                return out
            if not self._fill(timeout):
                raise TimeoutError("no line from board")

    def _exact(self, n):
        while len(self.buf) < n:
            if not self._fill():
                raise TimeoutError("short read %d/%d" % (len(self.buf), n))
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    @staticmethod
    def _dump(b, n=32, tail=96):
        """Head of the residual buffer as hex plus repr, then its tail as repr.

        Both encodings for the head, because what is left in there is either text
        or binary and you do not know which in advance: b'\\x04\\x04>' is legible
        in repr and noise in hex, the tail of a half-read frame payload is the
        other way round.

        The tail is here because this function once threw away the diagnosis. A
        board exception arrived, the port died mid-traceback, and the 174 bytes
        still sitting in the buffer held the one line worth having - the exception
        type and its message, which a traceback prints LAST. Dumping only the head
        printed `File "<stdin>", line 1, in <module>`, the outer frame, which
        names nothing at all. The answer was already on this host and got
        truncated away, so the tail gets the generous allowance: a MicroPython
        exception line runs to ~70 characters before the prompt bytes.
        """
        head = bytes(b[:n])
        out = "%s %r" % (" ".join("%02x" % c for c in head), head)
        rest = b[n:]
        if rest:
            end = bytes(rest[-tail:])
            skipped = len(rest) - len(end)
            out += " ...%s%r" % ("[%d more]" % skipped if skipped else "", end)
        return out

    def run(self):
        """Supervisor. Streaming is a session, not a one-shot, and every failure
        mode found so far is cured by standing the bring-up back up:

          - a wedged Lepton needs a fresh csi.CSI() object, which only a new
            SETUP_CODE submission creates. Waiting, framesize() and a full soft
            re-init were all measured against a wedged part and all raised.
          - a lost stream, a resync storm, a board that fell off USB: same cure.
          - the planned heap restart above: same cure, taken early and on purpose.

        Previously this recorded the exception and let the thread end, so the
        first hiccup ended the session silently and the browser kept showing the
        last frame forever.
        """
        backoff = 1.0
        while not self.stop.is_set():
            planned = False
            try:
                self._run()
                return                          # asked to stop, cleanly
            except _PlannedRestart as e:
                planned = True
                self.state["last_restart"] = str(e)
            except Exception as e:
                self._record_failure(e)
                self.state["last_restart"] = type(e).__name__
            finally:
                # A restart outlasts STALE_S by a wide margin - ~3s draining the
                # port plus ~10s of bring-up - so without this the health panel
                # would call the stream dead every time the defence works. The
                # panel is only worth having if it distinguishes "recovering" from
                # "broken".
                self.state["restarting_since"] = time.time()
                self.release()

            if self.stop.is_set():
                return
            self.state["restarts"] = self.state.get("restarts", 0) + 1
            # A planned restart is not a fault and must not be rate-limited into
            # a stutter; an unplanned one backs off so a genuinely dead board is
            # not hammered at full speed.
            if planned:
                backoff = 1.0
            else:
                time.sleep(backoff)
                backoff = min(backoff * 2, 15.0)
            self._reset_for_restart()

    def _reset_for_restart(self):
        """Everything that must not survive into the next session."""
        self.buf = bytearray()
        self.pending, self.headers = 0, 0
        self.last_line, self.last_line_t = b"", 0.0
        self.s = None
        self.state["ready"] = False
        self.state.pop("heap_free", None)

    def _record_failure(self, e):
        # The counters first, then the evidence. pending is the one that decides:
        # pending > 0 with the buffer holding the raw-REPL end marker means the
        # batch finished and this host is owed frames that were already sent - a
        # counting bug here, not a board fault. A partial line with pending > 0
        # means the board stopped mid-write.
        since = ("%.1fs" % (time.time() - self.last_line_t)
                 if self.last_line_t else "never")
        self.state["error"] = (
            "%s: %s (frames=%d, headers=%d, pending=%d, batches=%d, buf=%d)"
            " buf[%s] last[%r] +%s" % (
                type(e).__name__, e, self.state.get("frames", 0),
                self.headers, self.pending, self.state.get("batches", 0),
                len(self.buf), self._dump(self.buf), self.last_line[:64],
                since))

    def _attention(self, settle=8.0):
        """Take control of a board that may be in the middle of a batch.

        The old sequence was two ctrl-Cs and a 0.3s sleep, which assumes the board
        is sitting at a prompt. It very often is not: a host killed mid-session
        (a timeout, a ctrl-C, a crash) never runs release(), so the board is left
        streaming 26KB frames into a port nobody drains. Its write blocks, it
        cannot reach the point where it would notice the interrupt, and the next
        session then waits 30s for a prompt that cannot come while the buffer
        fills with a previous run's payload. That is a restart failing to restart,
        which is the one thing a supervisor may not do.

        So: the same read-first-then-interrupt dance release() already documents.
        Drain whatever is in flight so the board's write can complete, keep
        interrupting, and only enter the raw REPL once the port has gone quiet.

        A ctrl-D soft reboot was tried here as well, to get a genuinely fresh
        interpreter rather than one still holding the previous session's CSI
        objects. It desynchronised the raw-REPL handshake - the next submission
        came back as `NameError: name 'c' isn't defined`, a fragment of its own
        source - so it is deliberately not done. The heap side of that problem is
        handled instead by the gc.collect() at the top of capture._BRINGUP, which
        is safe there because no CSI object exists yet.
        """
        deadline = time.time() + settle
        quiet = 0
        while time.time() < deadline:
            n = self.s.in_waiting
            if n:
                self.s.read(n)
                quiet = 0
            else:
                quiet += 1
                if quiet >= 3:          # ~0.3s with nothing in flight
                    break
            self.s.write(b"\r\x03\x03")
            time.sleep(0.1)

        self.s.reset_input_buffer()
        self.buf = bytearray()
        self.s.write(b"\x01")           # raw REPL
        time.sleep(0.3)
        self.s.read_all()

    def release(self):
        """Read first, then interrupt. The board can only act on a ctrl-C once its
        own pending writes have somewhere to go."""
        s = getattr(self, "s", None)
        if s is None:
            return
        try:
            deadline = time.time() + 3
            while time.time() < deadline:
                nn = s.in_waiting
                s.read(nn if nn else 1)
                try:
                    s.write(b"\x03")
                except Exception:
                    pass
            s.write(b"\x02")
        except Exception:
            pass
        finally:
            s.close()

    def _traceback(self, first):
        """The whole traceback, not just the line that tripped the check.

        The one line worth having is the last one - the exception type and its
        message - and it arrives several lines after the marker. Reporting only
        the first names a file and says nothing about what went wrong, which is
        how a board fault reads as a mystery.

        Reading it must never be allowed to fail the report. Whatever stalls the
        interpreter enough to raise on the board is also what starves TinyUSB's
        tud_task, so the CDC very often dies in the middle of these very bytes -
        that is the common case here, not an edge case. Letting the read error
        propagate substitutes it for the board fault, and the session is then
        reported as `SerialException: device disconnected or multiple access on
        port?`, which sends you to check the cable while the real exception is
        sitting unread in the buffer. Keep whatever arrived and mark it partial.
        """
        deadline = time.time() + 3
        cut = ""
        while b"\x04\x04>" not in self.buf and time.time() < deadline:
            try:
                if not self._fill(0.5):
                    cut = "<no more output>"
                    break
            except Exception as e:
                cut = "<port died mid-traceback: %s: %s>" % (type(e).__name__, e)
                break
        i = self.buf.find(b"\x04\x04>")
        end = (i + 3) if i >= 0 else len(self.buf)
        rest = bytes(self.buf[:i if i >= 0 else len(self.buf)])
        del self.buf[:end]
        out = [first.lstrip("\x04")]
        out += [ln.strip() for ln in rest.decode("utf-8", "replace").splitlines()]
        if cut and i < 0:
            out.append(cut)
        return " | ".join(ln for ln in out if ln)

    def _resync(self, why):
        """Abandon this batch; the loop will start a clean one.

        Reading on past a line that is not a header is what turns one lost chunk
        into a dead stream. Bytes go missing on the wire, _exact() over-reads
        into the next frame, and from then on _line() finds newlines inside
        payloads forever - pending never returns to 0, so no further batch is
        ever submitted and a healthy board looks like it went quiet. A batch
        self-terminates, so its prompt is always still coming; drop the count
        and let the top of the loop wait for it.
        """
        self.state["resyncs"] = self.state.get("resyncs", 0) + 1
        self.state["last_resync"] = why
        self.pending = 0

    def _range(self, lo, hi):
        """The board picks its own range by auto-ranging the scene, and it is the
        only thing that turns an 8-bit code back into a temperature. Push it into
        the pipeline the moment the board reports it, or every reading is scaled
        by the ratio of the assumed range to the real one - wrong by tens of
        degrees while looking entirely reasonable."""
        self.state["range"] = (lo, hi)
        self.pipe.set_range(lo, hi)

    def _await(self, token, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            i = self.buf.find(token)
            if i >= 0:
                del self.buf[:i + len(token)]
                return
            self._fill(0.5)
        raise TimeoutError("board never sent %r" % token)

    def _submit(self, code, timeout=45.0):
        """Hand one unit of work to the raw REPL and wait for its ack.

        Generous by default: the one-off bring-up runs the Lepton's 2s VoSPI
        settle plus a histogram pass in interpreted Python and measured 10.3s
        end to end. A 10s limit sat just under that and looked exactly like a
        dead board.
        """
        self.s.write(code.encode() + b"\x04")
        self._await(b"OK", timeout)

    def _want_heap(self, batch_no):
        """Should this batch carry a heap reading?

        The reading costs 227ms - gc.mem_free() walks the whole 25MB heap - so it
        cannot be taken every batch. But a fixed rate is the wrong shape for what
        it is watching: the danger is the automatic collector firing on an
        exhausted heap, which wedges the Lepton permanently, and how urgent that
        is depends entirely on how much room is left.

        So the rate follows the margin. Far from the floor, every 10th batch
        (~23s) is plenty against a drain measured in hours. Close to it, the
        assumption that the drain rate is the one that was measured is exactly
        what should not be relied on - a leak this code does not know about, or a
        scene that makes the loop allocate more, would be invisible until the
        collector had already fired. Near the floor the reading is worth its
        227ms every batch.
        """
        free = self.state.get("heap_free")
        if free is None:
            return True                      # first batch of a session: establish it
        if free < HEAP_FLOOR * 2:
            return True                      # inside 4MB of the trigger: watch closely
        if free < HEAP_FLOOR * 4:
            return batch_no % 3 == 0
        return batch_no % HEAP_EVERY == 0

    def _run(self):
        self.s = serial.Serial(self.port, 115200, timeout=0.2, write_timeout=10)
        setup = (SETUP_CODE
                 .replace("__TMIN__", "-10").replace("__TMAX__", "140")
                 .replace("__AUTORANGE__", "True").replace("__Q__", str(self.quality)))
        self._attention()
        self._submit(setup)

        # Drain the bring-up before batching starts. pending begins at 0, so
        # without this the loop would wait for the raw-REPL prompt while the
        # sensors are still coming up and time out on a board that is fine.
        while True:
            line = self._line(timeout=60.0).decode("utf-8", "replace").strip()
            if line.startswith("\x04") or "Traceback" in line:
                raise RuntimeError("board: " + line.lstrip("\x04"))
            if line.startswith("#READY"):
                p = line.split()
                self._range(int(p[5]), int(p[6]))
                self.state["ready"] = True
                # Frames are flowing again, so the error that got us here is
                # history. Leaving it set would pin the health panel red for the
                # rest of the session and train the user to ignore it.
                self.state.pop("error", None)
                self.state.pop("restarting_since", None)
                break

        self.pending = 0
        started = False
        t_prev, n = time.time(), 0      # n counts frames, for fps - do not reuse
        batch_no = 0
        while not self.stop.is_set():
            if self.pending == 0:
                # every submission ends with \x04\x04> - consume the prompt
                # before handing over the next one. That prompt is also the only
                # dependable sign that a batch finished: #BATCH is written just
                # ahead of it, so _await eats the line before the loop below can
                # ever see it - count the prompt, not the line. Match all three
                # bytes, not a bare '>': after a resync the buffer still holds
                # payload, and 0x3e is a perfectly ordinary byte inside a JPEG.
                self._await(b"\x04\x04>", 30.0)
                if started:
                    self.state["batches"] = self.state.get("batches", 0) + 1
                self._submit(
                    BATCH_CODE % (self.batch, 1 if self._want_heap(batch_no) else 0),
                    timeout=20.0)
                batch_no += 1
                started = True
                self.pending = self.batch
            raw = self._line()
            line = raw.decode("utf-8", "replace").strip()
            if _board_error(raw):
                raise RuntimeError("board: " + self._traceback(line))
            if line.startswith("#HEAP"):
                free = int(line.split()[1])
                self.state["heap_free"] = free
                if free < HEAP_FLOOR:
                    raise _PlannedRestart("heap down to %.1fMB" % (free / (1 << 20)))
                continue
            if line.startswith("#BATCH"):
                self.pending = 0
                continue
            if line.startswith("#READY"):
                p = line.split()
                self._range(int(p[5]), int(p[6]))
                self.state["ready"] = True
                self.pending = 0
                continue
            if not line.startswith("#F "):
                self._resync("not a header: %r" % raw[:24])
                continue

            f = line.split()
            jlen, tlen, torn = (int(v) for v in f[1:4])
            # The two board clocks are optional in the parse, not because any
            # firmware omits them - the host submits the code that writes them -
            # but because a resync leaves payload in the buffer and a line that
            # happens to start "#F " must not take the whole stream down on an
            # index error.
            dt_ms = int(f[4]) if len(f) > 4 else None
            skew_ms = int(f[5]) if len(f) > 5 else None
            stalled = int(f[6]) if len(f) > 6 else None
            # Bytes can go missing after the board has already counted them as
            # sent - a DTR toggle from anything else opening the port makes the
            # firmware discard what is queued, and no board-side check can see
            # it. So the lengths here are not trustworthy just because the board
            # meant well. A negative jlen would be worse than a wrong one:
            # _exact's guard is vacuous for it and del buf[:-n] throws the
            # buffer away.
            if not (0 < jlen <= OUT_W * OUT_H and tlen == TH_W * TH_H):
                self._resync("implausible header %r" % line[:40])
                continue
            self.headers += 1
            self.pending -= 1
            jpg = self._exact(jlen)
            thermal = self._exact(tlen)

            # Hand off and go straight back to the port. Nothing that decodes,
            # fuses, composes or encodes belongs on this thread: every
            # millisecond spent here is a millisecond the CDC is not being
            # drained, and the board's out.write() discards the tail of a frame
            # it has already announced once no progress is made for 500ms.
            # Measured inline cost was 18ms fused / 28ms in the edges view
            # against a 114ms frame, so this is headroom rather than a rescue -
            # but the detector is 68-99ms and would not have fitted at all.
            self.work.put((jpg, thermal))

            n += 1
            now = time.time()
            self.state["frames"] = self.state.get("frames", 0) + 1
            self.state["last_frame_t"] = now

            # Short rolling windows rather than totals: the panel is meant to
            # report the state of the board now, and a fault that cleared ten
            # minutes ago should stop being red.
            # Tearing is read off the board's own header, so it belongs to this
            # thread. rows_rebuilt is a property of the fusion pass and is
            # recorded by the worker, which is the only thread that knows when
            # one has actually run.
            tw = self.state.setdefault("torn_window", [])
            tw.append(bool(torn))
            del tw[:-60]

            # The board's own cadence, kept separate from the host's fps. They
            # answer different questions and this project has already been
            # confused by conflating them: "8.8 fps" from arrival times says the
            # link is keeping up, and says nothing about whether the sensors are.
            #
            # The first interval of every batch is discarded. _t_prev survives
            # between submissions - the raw REPL keeps globals - so the gap it
            # measures spans the host's round trip to submit the next batch, not
            # a thermal frame period. Same on the very first frame, where
            # _t_prev is still 0 and the difference is the whole uptime.
            if dt_ms is not None and self.pending < self.batch - 1 and 0 < dt_ms < 60000:
                dw = self.state.setdefault("dt_window", [])
                dw.append(dt_ms)
                del dw[:-120]
                # An FFC parks snapshot() in the driver for 1824ms (measured
                # 2026-08-06, three in ten minutes). It is not a fault and must
                # not be averaged in with the 114ms frames, or the cadence reads
                # as chronically slow for a minute after every shutter event.
                if dt_ms > 1000:
                    self.state["last_ffc_t"] = now
                    self.state["ffcs"] = self.state.get("ffcs", 0) + 1
            if skew_ms is not None and 0 <= skew_ms < 60000:
                sw = self.state.setdefault("skew_window", [])
                sw.append(skew_ms)
                del sw[:-120]

            # Host load reaching the sensor. Counted rather than averaged: one
            # stall is 500ms the Lepton went unserviced, and the part is only
            # measured safe out to a 2500ms gap. This is the number that says
            # whether adding work on this host - a detector, a browser, anything
            # that stops draining the port - has started to cost the thermal
            # side, and it is the only such signal the board can give.
            if stalled:
                self.state["stalls"] = self.state.get("stalls", 0) + stalled
                self.state["last_stall_t"] = now
            if dt_ms is not None and dt_ms > LEPTON_SAFE_GAP_MS:
                self.state["starved"] = self.state.get("starved", 0) + 1
                self.state["last_starve_ms"] = dt_ms

            if now - t_prev >= 1.0:
                self.state["fps"] = n / (now - t_prev)
                self.state["torn"] = torn
                n, t_prev = 0, now


# ---------------------------------------------------------------- timing
#
# What the two sensors are actually doing, on the board's clock rather than on
# arrival times. The distinction is the whole point: between a frame being
# grabbed and this host seeing it lie a 4KB-chunked CDC write, a 500ms stall
# retry and the host's scheduler, so arrival-time fps measures the link. It has
# been read as a sensor rate more than once in this project.
#
# Three numbers come out of it:
#
#   thermal   the Lepton's cadence. Should be 114ms; it is a three-valued delta
#             function (113/114/115ms) with 1824ms across an FFC.
#   skew      thermal frame in hand -> visible frame in hand. This is the pairing
#             error fusion inherits: the two planes it registers were not looked
#             at at the same instant, so anything moving is displaced by roughly
#             (object speed x skew) before the warp ever sees it.
#   ffc       when the shutter last closed. Readings either side of one are not
#             comparable, which is exactly the kind of thing that is invisible
#             three weeks later in a capture review.


def timing(state, now):
    """Board-side cadence and pairing skew. Pure, so it tests off a dict."""
    dw = [d for d in (state.get("dt_window") or []) if d <= 1000]   # FFC gaps out
    sw = state.get("skew_window") or []
    ffc = state.get("last_ffc_t")

    def stats(v):
        if not v:
            return None
        s = sorted(v)
        return {"median": s[len(s) // 2], "min": s[0], "max": s[-1], "n": len(s)}

    th, sk = stats(dw), stats(sw)
    out = {
        "thermal_ms": th,
        "thermal_fps": round(1000.0 / th["median"], 2) if th and th["median"] else None,
        "skew_ms": sk,
        # The share of one thermal period that the visible frame lags by. This is
        # the number to judge the pairing on - a skew of 12ms means nothing until
        # you know the period is 114ms.
        "skew_frac": round(sk["median"] / float(th["median"]), 3)
                     if th and sk and th["median"] else None,
        "host_fps": round(state.get("fps", 0.0), 2),
        "ffc_ago_s": round(now - ffc, 1) if ffc else None,
        "ffcs": state.get("ffcs", 0),
        "expected_ms": round(1000.0 / LEPTON_FPS, 1),
        # Host load reaching the sensor: 500ms write timeouts the board sat
        # through, and thermal gaps past the envelope the part is measured safe
        # in. Both are about protecting the Lepton, not about frame rate.
        "stalls": state.get("stalls", 0),
        "starved": state.get("starved", 0),
        "last_starve_ms": state.get("last_starve_ms"),
        "safe_gap_ms": LEPTON_SAFE_GAP_MS,
    }
    return out


# Past this share of a thermal period between the two grabs, the pair stops being
# simultaneous in any useful sense. 0.25 of 114ms is 28ms, which at a walking
# 1.4 m/s is 4cm of subject travel - already several thermal pixels at close
# range, and drawn as a registration error rather than as the timing error it is.
SKEW_WARN = 0.25


# ---------------------------------------------------------------- http


PAGE = b"""<!doctype html><meta charset=utf-8><title>thermal fusion - live</title>
<style>body{background:#111;color:#ddd;font:14px system-ui;margin:0;padding:16px}
#wrap{position:relative;display:inline-block}
img{max-width:100%;border-radius:6px;display:block}
.row{display:flex;gap:16px;align-items:center;margin:10px 0;flex-wrap:wrap}
label{display:flex;gap:6px;align-items:center}input[type=range]{width:120px}
button{background:#222;color:#ccc;border:1px solid #444;border-radius:4px;padding:4px 10px;
cursor:pointer}button.on{background:#385;color:#fff;border-color:#5a7}
#s{color:#8b8}
#read{position:absolute;padding:3px 7px;background:#000c;border:1px solid #666;
border-radius:4px;font:13px ui-monospace,monospace;pointer-events:none;display:none;
white-space:nowrap}
#read.warn{border-color:#c84;color:#fc9}
#band{font:13px ui-monospace,monospace;color:#bbb}
#band b{color:#fda;font-weight:600}
#clock{font:12px ui-monospace,monospace;color:#999;gap:14px}
#clock b{font-weight:600}
#clock .th{color:#f0a860}#clock .vis{color:#6cc8ff}#clock .warn{color:#f0c060}
#health{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start}
.pill{border:1px solid #444;border-radius:4px;padding:3px 9px;font-size:12px;
background:#1a1a1a;line-height:1.5}
.pill .n{font-weight:600;letter-spacing:.02em}
.pill .t{color:#999;margin-left:7px}
.pill.ok{border-color:#2f5d3f}.pill.ok .n{color:#7fd39b}
.pill.warn{border-color:#7a5a1e}.pill.warn .n{color:#f0c060}.pill.warn .t{color:#c8ae7c}
.pill.fail{border-color:#7d2f2f;background:#2a1616}.pill.fail .n{color:#ff8a8a}
.pill.fail .t{color:#e0a5a5}
.note{color:#987;font-size:12px;max-width:60em;line-height:1.5}</style>
<h3>thermal + visible fusion &mdash; live</h3>
<div id=wrap><img id=im src="/stream"><div id=read></div></div>
<div class=row>
<label>gain <input type=range id=gain min=0 max=512 value=200></label>
<label>eps <input type=range id=eps min=1 max=1000 value=200></label>
<label>radius <input type=range id=radius min=1 max=8 value=4></label>
<label>agc <input type=range id=agc min=0 max=100 value=0></label>
</div>
<div class=row>
<span>view</span><span id=view></span>
<label>mix <input type=range id=mix min=0 max=100 value=50></label>
<label><input type=checkbox id=outline> thermal footprint</label>
<label><input type=checkbox id=boxes checked> detections</label>
</div>
<div class=row id=dets></div>
<div class=row>
<span>palette</span><span id=pal></span>
<label>&epsilon; <input type=number id=emis min=0.05 max=1 step=0.01 value=1 style=width:5em></label>
<label>reflected &deg;C <input type=number id=refl step=1 value=20 style=width:5em></label>
</div>
<div class=row id=band></div>
<div class=row id=clock></div>
<div class=row id=health></div>
<div class=row><span id=s></span></div>
<p class=note>Hover for a reading. The number comes from the thermal frame behind
that pixel &mdash; before the AGC and before the guided filter, which borrows the
visible camera's edges to sharpen the picture and must not be quoted off.
Absolute accuracy is &plusmn;5&deg;C at best. The board runs the Lepton in
<b>high gain</b> since 2026-08-09; before that it was silently in low gain, whose
specified accuracy is the greater of &plusmn;10&deg;C or 10%, so any reading quoted
off a frame recorded earlier than that carries the looser number. The <b>delta</b> between two nearby points of the same
material in one frame is far better than either, and is what a finding should rest
on &mdash; provided neither point is clipped, neither sits on a rebuilt row, and no
shutter event separates them. Readings keep working in every view.</p>
<p class=note><b>Judging registration.</b> The fused view cannot tell you whether the
warp is right: the guided filter puts crisp edges in the right places even when the
thermal layer is offset, so a misregistered frame still looks sharp &mdash; it just
colours the wrong side of the edge. Use <b>blink</b> (the eye catches motion far
better than offset), <b>mix</b> to judge how far off it is, or <b>edges</b>, which
draws the thermal layer's own edges over the plain visible image: where the warp is
right they land on the object's outline. <b>thermal footprint</b> outlines where the
thermal camera stops seeing at all &mdash; outside it the grey is not a cold reading,
it is no reading.</p>
<p class=note><b>The health row</b> is what only a live stream can tell you: the
offline suites prove the pipeline is correct on frames that sit still, not that the
board in front of you is producing numbers worth writing down.
<span style="color:#f0c060">Amber</span> means the readings are usable but qualified
&mdash; the qualification changes how to read them, it is not a nag to clear.
<span style="color:#ff8a8a">Red</span> means do not record anything. All green is not
a claim that the measurement is accurate; it is a claim that none of the failures
this code can see are happening.</p>
<script>
const im = document.getElementById('im'), read = document.getElementById('read');
for (const k of ['gain','eps','radius','agc']) {
  const el = document.getElementById(k);
  el.oninput = () => fetch('/set?'+k+'='+el.value);
}
for (const k of ['emis','refl']) {
  document.getElementById(k).onchange = () => fetch('/set?emissivity='
    + document.getElementById('emis').value + '&reflected='
    + document.getElementById('refl').value);
}
function buttons(boxId, names, current, param) {
  const box = document.getElementById(boxId);
  for (const p of names) {
    const b = document.createElement('button');
    b.textContent = p; b.className = (p === current) ? 'on' : '';
    b.onclick = async () => { await fetch('/set?'+param+'='+p);
      for (const c of box.children) c.className = (c.textContent === p) ? 'on' : ''; };
    box.appendChild(b);
  }
}
buttons('pal', ['ironbow','white','black','gray'], 'ironbow', 'palette');
buttons('view', ['fused','visible','blink','mix','edges'], 'fused', 'view');
document.getElementById('mix').oninput = (e) => fetch('/set?mix='+e.target.value);
document.getElementById('outline').onchange =
  (e) => fetch('/set?outline='+(e.target.checked ? 1 : 0));
document.getElementById('boxes').onchange =
  (e) => fetch('/set?boxes='+(e.target.checked ? 1 : 0));

// The image is scaled to fit, so client coords have to be mapped back to the
// 640x400 the pipeline actually indexes - otherwise the reading is off by the
// zoom factor and silently wrong rather than obviously broken.
let pending = false, last = 0;
im.onmousemove = async (e) => {
  const r = im.getBoundingClientRect();
  const x = Math.round((e.clientX - r.left) * im.naturalWidth / r.width);
  const y = Math.round((e.clientY - r.top) * im.naturalHeight / r.height);
  read.style.left = (e.clientX - r.left + 14) + 'px';
  read.style.top = (e.clientY - r.top + 14) + 'px';
  if (pending || performance.now() - last < 80) return;   // ~12 Hz is plenty
  pending = true; last = performance.now();
  try {
    const d = await (await fetch('/temp?x='+x+'&y='+y)).json();
    read.style.display = 'block';
    read.className = d.repaired ? 'warn' : '';
    read.textContent = d.valid
      ? d.c.toFixed(1) + ' \\u00b0C' + (d.repaired ? '  rebuilt row' : '')
      : 'no thermal';
  } finally { pending = false; }
};
im.onmouseleave = () => { read.style.display = 'none'; };

setInterval(async () => {
  const h = await (await fetch('/health')).json();
  document.getElementById('health').innerHTML = h.checks.map(c =>
    '<span class="pill '+c.level+'"><span class=n>'+c.name+'</span>'
    + '<span class=t>'+c.text.replace(/&/g,'&amp;').replace(/</g,'&lt;')+'</span></span>'
  ).join('');

  document.getElementById('s').textContent = await (await fetch('/stat')).text();

  // The clock. One thermal period drawn to scale, with the moment the visible
  // frame was actually grabbed marked inside it - which is the question the two
  // sensors raise and the fused picture cannot answer. Both timestamps come off
  // the board's own clock; arrival times here have been through USB and measure
  // the link, not the sensors.
  const t = await (await fetch('/timing')).json();
  const c = document.getElementById('clock');
  if (!t.thermal_ms) { c.innerHTML = '<span>waiting for board timing</span>'; }
  else {
    const per = t.thermal_ms.median, W = 380, X = 8;
    const frac = t.skew_frac === null ? 0 : Math.min(1, t.skew_frac);
    const vx = X + W * frac;
    // Jitter drawn as a band, not a number: the Lepton's interval is a
    // three-valued delta function (113/114/115ms), so a wide band here is a real
    // anomaly rather than ordinary spread.
    const jl = X + W * Math.max(0, (t.thermal_ms.min - per) / per + 1) - W;
    const jw = W * (t.thermal_ms.max - t.thermal_ms.min) / per;
    const bad = t.skew_frac !== null && t.skew_frac > 0.25;
    c.innerHTML =
      '<svg width="'+(W+2*X)+'" height="40">'
      + '<rect x="'+X+'" y="14" width="'+W+'" height="10" fill="#1e1e1e" stroke="#3a3a3a"/>'
      + '<rect x="'+X+'" y="14" width="'+(vx-X)+'" height="10" fill="'
        + (bad?'#5a4418':'#1d3346')+'"/>'
      + '<rect x="'+(X+W-jw)+'" y="14" width="'+jw+'" height="10" fill="#2a2a2a"/>'
      + '<line x1="'+X+'" y1="8" x2="'+X+'" y2="30" stroke="#f0a860" stroke-width="2"/>'
      + '<line x1="'+vx+'" y1="8" x2="'+vx+'" y2="30" stroke="#6cc8ff" stroke-width="2"/>'
      + '<line x1="'+(X+W)+'" y1="8" x2="'+(X+W)+'" y2="30" stroke="#f0a860" stroke-width="2"'
        + ' stroke-dasharray="2 2"/>'
      + '<text x="'+(X+2)+'" y="38" fill="#f0a860" font-size="10">thermal</text>'
      + '<text x="'+(vx+3)+'" y="12" fill="#6cc8ff" font-size="10">visible</text>'
      + '<text x="'+(X+W)+'" y="38" fill="#777" font-size="10" text-anchor="end">'
        + per+' ms</text></svg>'
      + '<span class=th>thermal <b>'+per+' ms</b> = '+t.thermal_fps+' fps'
        + (t.thermal_ms.min!==t.thermal_ms.max
            ? ' <span style=color:#666>('+t.thermal_ms.min+'-'+t.thermal_ms.max+')</span>' : '')
        + '</span>'
      + '<span class="vis'+(bad?' warn':'')+'">visible <b>+'
        + (t.skew_ms ? t.skew_ms.median : '?')+' ms</b> ('
        + (t.skew_frac===null?'?':Math.round(100*t.skew_frac))+'% of a frame)</span>'
      + '<span>link <b>'+t.host_fps+' fps</b></span>'
      + '<span>'+(t.ffc_ago_s===null ? 'no FFC seen yet'
          : 'FFC <b>'+t.ffc_ago_s+'s</b> ago ('+t.ffcs+')')+'</span>'
      // Load reaching the sensor. Always on screen, including when it is zero:
      // this is the number that decides whether more host work is affordable,
      // and it is worth watching go up rather than discovering afterwards.
      + '<span'+(t.starved||t.stalls ? ' class=warn' : '')+'>sensor '
        + (t.starved ? '<b>STARVED '+t.starved+'x</b> (worst '+t.last_starve_ms+' ms)'
           : t.stalls ? '<b>'+t.stalls+'</b> write stall(s)'
           : 'unstarved')+'</span>';
  }

  // The detection list in text as well as boxes. A box you have to squint at is
  // not a reading you would write down, and the peak location matters as much as
  // the value - "this motor is warm" and "this motor's near bearing is warm" are
  // different findings.
  const dd = await (await fetch('/detections')).json();
  document.getElementById('dets').innerHTML = dd.detections.length
    ? dd.detections.map(d => '<span class="pill '+(dd.warped?'ok':'warn')+'">'
        + '<span class=n>'+d.cls+'</span><span class=t>'+Math.round(100*d.conf)+'%'
        + (d.max_c !== undefined
            ? '  '+d.max_c.toFixed(1)+'\\u00b0C peak @'+d.max_x+','+d.max_y
              + (dd.warped ? '' : '  UNREGISTERED')
              + (d.repaired ? '  rebuilt row' : '')
            : '  no thermal')
        + '</span></span>').join('')
    : '';
  const d = await (await fetch('/stats')).json();
  document.getElementById('band').innerHTML = d.valid
    ? 'frame  min <b>' + d.min.toFixed(1) + '</b>  max <b>' + d.max.toFixed(1)
      + '</b>  mean ' + d.mean.toFixed(1) + '  &Delta; <b>' + d.delta.toFixed(1) + ' \\u00b0C</b>'
    : 'no thermal coverage';
}, 1000);
</script>"""


def make_handler(state, pipe):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(PAGE)
            elif u.path == "/set":
                q = {k: v[0] for k, v in parse_qs(u.query).items()}

                if "palette" in q:
                    pipe.set_palette(q.pop("palette"))
                if q.get("view") in VIEWS:
                    pipe.view = q.pop("view")
                if "mix" in q:
                    pipe.mix = max(0, min(100, int(q.pop("mix"))))
                if "outline" in q:
                    pipe.outline = q.pop("outline") not in ("0", "false", "")
                if "boxes" in q:
                    pipe.show_detections = q.pop("boxes") not in ("0", "false", "")
                if "radar" in q:
                    pipe.show_radar = q.pop("radar") not in ("0", "false", "")
                if "whisker" in q:
                    pipe.radar_whisker = q.pop("whisker") not in ("0", "false", "")
                rp = state.get("radar_proj")
                if rp is not None:
                    for k in ("yaw", "pitch", "roll"):
                        if k in q:
                            setattr(rp, k, float(q.pop(k)))
                    for i, k in enumerate(("tx", "ty", "tz")):
                        if k in q:      # millimetres in the URL, metres inside
                            rp.t[i] = float(q.pop(k)) / 1000.0
                if "emissivity" in q or "reflected" in q:
                    eps = float(q.pop("emissivity", pipe.eps))
                    refl = float(q.pop("reflected", pipe.refl))
                    pipe.set_emissivity(min(1.0, max(0.05, eps)), refl)

                pipe.tune(**{("detail_gain" if k == "gain" else
                              "gf_eps" if k == "eps" else
                              "gf_radius" if k == "radius" else
                              "agc_permille" if k == "agc" else k): v for k, v in q.items()})
                self.send_response(204)
                self.end_headers()
            elif u.path == "/health":
                checks = health(pipe, state, time.time())
                worst = ("fail" if any(c["level"] == "fail" for c in checks) else
                         "warn" if any(c["level"] == "warn" for c in checks) else "ok")
                body = json.dumps({"worst": worst, "checks": checks})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            elif u.path in ("/temp", "/stats"):
                q = parse_qs(u.query)
                if u.path == "/temp":
                    d = pipe.temp_at(int(q.get("x", ["0"])[0]), int(q.get("y", ["0"])[0]))
                else:
                    d = pipe.frame_stats()
                body = json.dumps({"valid": False} if d is None else dict(d, valid=True))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            elif u.path == "/timing":
                body = json.dumps(timing(state, time.time()))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            elif u.path == "/detections":
                # `warped` rides along with every response. A consumer that logs
                # these numbers has no other way to know whether the box and the
                # temperature refer to the same place in the world.
                body = json.dumps({
                    "warped": pipe.warped,
                    "age_s": round(time.time() - state.get("detect_t", 0), 2)
                             if state.get("detect_t") else None,
                    "ms": state.get("detect_ms"),
                    "detections": state.get("detections") or []})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            elif u.path == "/stat":
                lo, hi = state.get("range", (0, 0))
                msg = "%.1f fps | range %d..%dC (%.3f C/code) | %s | eps %.2f, refl %.0fC%s" % (
                    state.get("fps", 0.0), lo, hi, (hi - lo) / 255.0,
                    "calibrated warp" if pipe.warped else "PLACEHOLDER warp - not registered",
                    pipe.eps, pipe.refl,
                    "  | ERROR: " + state["error"] if state.get("error") else "")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(msg.encode())
            elif u.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    last = None
                    while True:
                        f = state.get("frame")
                        if f is not None and f is not last:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             b"Content-Length: %d\r\n\r\n" % len(f))
                            self.wfile.write(f)
                            self.wfile.write(b"\r\n")
                            last = f
                        else:
                            time.sleep(0.02)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(404)
                self.end_headers()
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--port", default=PORT)
    ap.add_argument("--http", type=int, default=8088)
    ap.add_argument("--warp", help="calibrated warp LUT; without it the thermal layer "
                                   "is merely stretched and is NOT registered")
    ap.add_argument("--gain", type=int, default=200)
    ap.add_argument("--eps", type=int, default=200)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--palette", default="ironbow",
                    choices=["ironbow", "white", "black", "gray"],
                    help="white/black are the mono ramps; black also flips the detail sign")
    ap.add_argument("--emissivity", type=float, default=1.0, metavar="E",
                    help="surface emissivity 0.05..1.0. 1.0 is what the sensor assumes; "
                         "bright metal is ~0.1 and reads tens of degrees cold without this")
    ap.add_argument("--reflected", type=float, default=20.0, metavar="C",
                    help="temperature the surface reflects, usually ambient (default 20)")
    ap.add_argument("--agc", type=int, default=0, metavar="PERMILLE",
                    help="scene AGC, per-mille clipped each end (20 = 2%%). "
                         "Makes tone scene-relative rather than absolute")
    # The visible temporal filter. fusion.c leaves it off because it costs 768KB
    # (fusion.c:993-994) that only pays back in the dark - but
    # dark is the case this project exists for. In an unlit cabinet the detail
    # layer amplifies read noise exactly as hard as it amplifies edges, so a
    # noisy high-pass makes the fused image worse than none at all. Motion-adaptive
    # IIR: 8 frames buys ~3.9x for one buffer, where an 8-frame boxcar buys 2.8x
    # for eight - and a host-side burst average of 8 measured only 1.86x on this
    # board (2026-08-09), because every blend step re-quantises to 8 bits.
    ap.add_argument("--y-knee", type=int, default=0, metavar="CODES",
                    help="visible temporal filter knee in 8-bit luma codes; 0 = off. "
                         "Try 3-6 for a dark scene. Costs 768KB when enabled "
                         "(512KB of Q8 state + a 256KB output plane)")
    ap.add_argument("--y-frames", type=int, default=8, metavar="N",
                    help="equivalent frames averaged by the visible filter (default 8)")
    # Raised from 50 on 2026-08-10, measured on captures/handwave3. The visible
    # frame is the guided filter's *guide*, so the codec's artefacts are not
    # cosmetic here - they are fed into the one layer whose job is to be
    # amplified. Energy sitting exactly on the JPEG 8x8 grid, against the
    # off-grid energy of the same frame (scene detail has no reason to prefer a
    # period of 8, so anything above 1.0 is the codec):
    #
    #     q30  2.57x   q50  1.97x   q65  1.74x
    #     q80  1.52x   q90  1.32x   q95  1.16x
    #
    # At q30 the fused frame carries 116% of the reference's high-frequency
    # energy - more detail than the uncompressed original, which is the pipeline
    # sharpening blocking artefacts into scene texture. That is the failure mode
    # to avoid, and q50 was closer to it than it looked.
    #
    # The cost is link time, and this link has a hard edge: the board's
    # out.write() discards the tail of a frame after 500ms of no progress. Frame
    # rate at 8.772 fps including the uncompressed 18.75KB thermal plane:
    # q50 256KB/s, q80 325KB/s, q90 420KB/s. FS CDC measures out around
    # 700-900KB/s, so q80 sits near 40% duty and q90 near 55%. q80 buys most of
    # the improvement for the smaller share of the pipe.
    ap.add_argument("--quality", type=int, default=80, help="board-side JPEG quality")
    ap.add_argument("--detect", default="off", metavar="WHAT",
                    help="object detection: 'off', 'all' for COCO-80, or a comma-separated "
                         "class list such as 'person,cat'. Runs yolov4-tiny on the host CPU "
                         "and attaches a temperature to every box")
    ap.add_argument("--detect-size", type=int, default=416, choices=[320, 416],
                    help="detector input size. 416 is 68ms and 320 is 46ms on this host, "
                         "against a 114ms frame period (default 416)")
    ap.add_argument("--detect-conf", type=float, default=0.35,
                    help="detection confidence threshold (default 0.35)")
    # Radar overlay. Off unless a port is given, because live.py must keep
    # working on a rig that has no radar attached.
    ap.add_argument("--radar", nargs="?", const=radar_overlay.DATA_PORT, default=None,
                    metavar="PORT",
                    help="overlay IWR1843 detections from this DATA port "
                         "(default %s when the flag is given bare)" % radar_overlay.DATA_PORT)
    ap.add_argument("--radar-record", metavar="DIR",
                    help="record the session: radar.bin, radar.jsonl and session.mp4. "
                         "This is what an offline calibration is solved against")
    ap.add_argument("--radar-hfov", type=float, default=70.0, metavar="DEG",
                    help="assumed horizontal FOV used to guess the focal length when no "
                         "solved intrinsics exist (default 70). DESIGN.md:239 records this "
                         "as UNVERIFIED and the measured triple implies 63.8")
    ap.add_argument("--radar-calib", metavar="JSON",
                    help="solved intrinsics/extrinsics to project with, instead of the guess")
    ap.add_argument("--view", default="fused",
                    choices=["fused", "visible", "blink", "mix", "edges"],
                    help="view to start in, and therefore what gets recorded. "
                         "For picking a pixel off the recording, see the note in "
                         "radar_correspond.py: the thermal layer is NOT registered "
                         "until a warp LUT exists, so a pixel taken from the thermal "
                         "content carries that unknown offset into the extrinsic")
    ap.add_argument("--seconds", type=int, default=0, help="exit after N seconds (for tests)")
    args = ap.parse_args()

    if not os.path.exists(args.port):
        # The board does not always come back on the node it left. Anything that
        # re-enumerates it - a replug, or the USB reset that follows a wedge -
        # can hand it ttyACM1 while the stale ttyACM0 is still being released,
        # and then the default looks exactly like a board that is not there.
        # Measured 2026-08-09: a healthy board sat on ttyACM1 while this printed
        # "replug the board", which is advice that would not have helped.
        found = sorted(glob.glob("/dev/ttyACM*"))
        if args.port == PORT and found:
            print("%s is gone; using %s instead" % (args.port, found[0]), file=sys.stderr)
            args.port = found[0]
        else:
            raise SystemExit("%s is not present%s - replug the board"
                             % (args.port, "" if not found else
                                " (found %s, pass --port)" % ", ".join(found)))
    if not os.path.exists(LIB):
        raise SystemExit("%s missing - run 'make libfusion.so' in host/" % LIB)

    pipe = Pipeline(args)
    pipe.view = args.view
    state = {}

    detector = None
    if args.detect != "off":
        classes = None if args.detect == "all" else [
            c.strip() for c in args.detect.split(",") if c.strip()]
        if classes:
            unknown = [c for c in classes if c not in detect.COCO]
            if unknown:
                raise SystemExit("not COCO classes: %s\navailable: %s"
                                 % (", ".join(unknown), " ".join(detect.COCO)))
        try:
            detector = detect.Detector(size=args.detect_size, conf=args.detect_conf,
                                       classes=classes)
        except FileNotFoundError as e:
            raise SystemExit("detector model missing: %s" % e)

    work = Latest()
    radar = radar_proj = video = None
    if args.radar:
        radar = radar_overlay.RadarReader(args.radar, record_dir=args.radar_record)
        radar.start()
        radar_proj = radar_overlay.Bootstrap(OUT_W, OUT_H, hfov_deg=args.radar_hfov,
                                             calib_path=args.radar_calib)
        state["radar_proj"] = radar_proj
        print("radar overlay: %s, intrinsics from %s, f=%.1f px"
              % (args.radar, radar_proj.source, radar_proj.f), file=sys.stderr)
        if not args.radar_calib:
            print("  the projection is a GUESS until radar_extrinsics is solved - "
                  "nudge it with /set?yaw=..&pitch=..&tz=..", file=sys.stderr)
        if args.radar_record:
            video = radar_overlay.SessionVideo(
                os.path.join(args.radar_record, "session.mp4"))
            print("recording to %s/" % args.radar_record, file=sys.stderr)

    render = Renderer(pipe, state, work, detector, radar=radar,
                      radar_proj=radar_proj, video=video)
    render.start()
    stream = Streamer(args.port, pipe, args.quality, state, work)
    stream.start()

    srv = ThreadingHTTPServer(("0.0.0.0", args.http), make_handler(state, pipe))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("live fusion on http://localhost:%d  (ctrl-c to stop)" % args.http, file=sys.stderr)
    if not args.warp:
        print("NOTE: no --warp, thermal layer is stretched not registered", file=sys.stderr)
    if detector:
        print("detecting %s at %d px" % (args.detect, args.detect_size), file=sys.stderr)

    # How long the stream may stay down before this gives up on it. The
    # supervisor in Streamer.run() exists to survive a fault - a wedged Lepton, a
    # resync storm, a board that fell off USB - and a restart costs ~3s of
    # draining plus ~10s of bring-up.
    #
    # This loop used to `return 1` the instant state["error"] appeared, which
    # meant the supervisor was never once allowed to finish: the process was gone
    # a fraction of a second into a recovery designed to take thirteen. Every
    # fault therefore read as fatal, including the ones the code already knew how
    # to repair. The error is only cleared on #READY, so polling for it is
    # polling for "a restart is in progress".
    DEAD_S = 90.0

    t0 = time.time()
    down_since, last_err = None, None
    try:
        while True:
            time.sleep(0.2)
            err = state.get("error")
            if err and err != last_err:
                print("stream fault (recovering): %s" % err, file=sys.stderr)
                last_err = err
            if err or state.get("restarting_since"):
                down_since = down_since or time.time()
                if time.time() - down_since > DEAD_S:
                    print("stream down for %.0fs across %d restart(s) - giving up: %s"
                          % (time.time() - down_since, state.get("restarts", 0), err),
                          file=sys.stderr)
                    return 1
            elif down_since is not None:
                print("stream recovered after %.0fs (%d restart(s))"
                      % (time.time() - down_since, state.get("restarts", 0)),
                      file=sys.stderr)
                down_since, last_err = None, None
            if args.seconds and time.time() - t0 > args.seconds:
                tm = timing(state, time.time())
                print("fps %.1f, frames %d, rendered %d, dropped %d, bad jpeg %d, "
                      "batches %d, resyncs %d, restarts %d, frames flowing: %s" % (
                          state.get("fps", 0.0), state.get("frames", 0),
                          state.get("rendered", 0), state.get("dropped", 0),
                          state.get("bad_jpeg", 0), state.get("batches", 0),
                          state.get("resyncs", 0), state.get("restarts", 0),
                          state.get("frame") is not None), file=sys.stderr)
                # Board clock separately: the fps above is arrival times and says
                # whether the link kept up, not what the sensors did.
                print("board clock: thermal %s, visible +%s ms, %d FFC(s)" % (
                    "%d ms (%.2f fps)" % (tm["thermal_ms"]["median"], tm["thermal_fps"])
                    if tm["thermal_ms"] else "not reported",
                    tm["skew_ms"]["median"] if tm["skew_ms"] else "?",
                    tm["ffcs"]), file=sys.stderr)
                return 0
    except KeyboardInterrupt:
        return 0
    finally:
        stream.stop.set()
        render.stop.set()
        srv.shutdown()


if __name__ == "__main__":
    sys.exit(main())
