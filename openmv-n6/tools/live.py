#!/usr/bin/env python3
"""Live thermal/visible fusion: board streams, this host fuses, browser watches.

    ./live.py                       then open http://localhost:8088
    ./live.py --warp calib/warp.lut --gain 220
    ./live.py --radar               live + IWR1843 overlay (USB DATA port)
    ./live.py --radar --channel thermal_radar
    ./live.py --radar --channel ai  final detections/tracks without raw radar clutter
    ./live.py --record              live + recording to captures/live-<stamp>/
    ./live.py --radar --students    live + the three AI channels (see below)
    ./live.py --radar --record DIR  a full calibration session: session.mp4,
                                    frames.jsonl, thermal.bin, radar.bin+jsonl

The board sends a hardware-JPEG of the visible frame (~11KB for 640x400, 4% of
raw) plus the thermal frame *uncompressed* - the thermal data is the measurement,
and lossy compression on radiometry is not a trade worth making for 17KB. That
puts a frame at ~30KB, which the FS-speed link carries faster than the Lepton
produces frames, so the stream is paced by the sensor rather than the pipe.

Fusion runs through libfusion.so - the same fusion.c that compiles into the
firmware - so what you see here is what the board will do once flashed.

Controls are live, no restart: http://localhost:8088/set?gain=250&eps=120&radius=5

During a calibration session, name each station before holding it - from the
page's pose bar, or /pose?id=N04 - and every frame written from then on carries
that pose_id. /pose?clear=1 between stations, /pose to read back what has been
stamped. Frames written with no station are labelled UNLABELLED on the page
rather than passing quietly; recovering the split afterwards from timestamps is
what the V3 plan section 5d exists to stop.

The channel selector is the product surface: thermal, visible+radar,
thermal+radar, three-sensor fusion, and AI. The separate view selector is there
to judge registration inside the fusion channel - see the VIEWS comment below
for why, and use blink/edges during a calibration session rather than trusting
how sharp the picture looks. The visible source is currently PAG7936 luma; the
rgb_radar API id is reserved for true colour once RGB565 passes its link test.

--students brings up three AI channels, switched on and off live from the page
(keys t, r, c) or over /set?ai_thermal=0&ai_radar=1&ai_fusion=1, and read back
on /ai. They feed fusion and tracking but their intermediate boxes are hidden
by default; final locks and silhouettes are drawn on the thermal-bearing
products and the AI product. Key e (or /set?evidence=1) reveals intermediate
evidence for model debugging:

    thermal   orange - the thermal student, drawn on the visible plane through
              the warp LUT. Without a LUT its boxes are still reported, on the
              thermal plane, but cannot be placed on the picture.
    radar     cyan   - the radar student, straight onto the visible plane.
    fusion    white  - the two of them agreeing: one person, seen by both. The
              candidate pair is made on u only, inside 180 px, because live
              motion can put the radar box behind the thermal box while radar elevation
              comes off a two-element aperture and says almost nothing about
              which box a return belongs to. A fused box carries the range.

`lock` (key l, on by default) is what makes a person survive a frame nobody
found them in. A person seen twice becomes a track with an id: associated
frame to frame by IoU with a centre-distance fallback, smoothed alpha-beta,
carried on their own velocity through the gaps, and dropped 1.5 s after the
last measurement. All three sensors feed one tracker and what describes one
person is merged before association, so a lock the detector loses in glare can
be held by the thermal student - and the letters on the label (D T R F) say
which sensors are holding it right now. A coasting track is amber and dashed
and says so: it is where somebody probably is, not where anybody saw them.

A candidate the DETECTOR has never confirmed must move before it is drawn as a
person. Measured in the lobby this rig sits in: a lit glass door reads 31.7 C
mean / 34.1 C p90 and a person reads 30.9 / 34.1 - the same temperature to
within the sensor's noise, so no radiometric test can separate them, and the
students and the radar will agree with each other about a door all day. What a
door has never done is move. Those candidates are counted as `static` on the
card and in /health rather than dropped quietly, because something warm and
motionless is not nobody. A person the detector sees is drawn at once, standing
still or not.

With `person outline` on (key s, opt-in), a detection is drawn as the warm
shape the thermal frame holds inside it rather than as a rectangle - the
detector's green person boxes included, which is the pairing worth having: the
COCO detector is the locator this rig trusts, and the thermal frame is the only
sensor here that knows the shape. Fusion draws its ring around that shape. The
split inside a box is Otsu, not a fixed body-heat threshold, and where no shape
can be found (no LUT, no thermal coverage, nothing warm in the box) the box
comes back.

Each channel has a confidence slider on the same card. It hides boxes rather
than restarting the engines, so the card counts `shown of found` and a quiet
scene never looks like a slider parked too high; below 0.75 a box is drawn as
four corner ticks rather than a rectangle, because a candidate and a detection
should not look alike. Overlapping and nested duplicates - the students emit
eight slots with no NMS of their own - are dropped before any of that.

They are three switches rather than a mode selector because agreement is only
readable beside its components: two channels that fire on everything agree on
everything too. Switching a channel off stops its engine - except when fusion
needs it, which still runs it and simply does not draw it.
"""
import argparse
import ctypes
import glob
import math
import json
import os
import signal
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
import radar_overlay
import tracker as tracking  # noqa: E402
import egomotion  # noqa: E402
import recorder  # noqa: E402
import thermal_io  # noqa: E402
import soc as hostsoc  # noqa: E402
import live_channels  # noqa: E402

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


def write_text(s):
    """Write one control record completely on the SAME stream as payloads.

    Mixing ``sys.stdout.write`` for headers with ``sys.stdout.buffer.write``
    for images gives the text wrapper permission to buffer a header while the
    binary payload passes it.  The host then sees JPEG/thermal bytes where #F
    must be.  Encoding the tiny record once and writing it through ``out`` keeps
    control and payload in one ordered CDC queue.
    """
    global _stalled
    b = s.encode()
    mv = memoryview(b)
    off = 0
    stalls = 0
    while off < len(b):
        w = out.write(mv[off:])
        if w:
            off += w
            stalls = 0
            continue
        stalls += 1
        _stalled += 1
        if stalls > 2:
            return False
    return True


def stream(n, report=0):
    global _t_prev, _stalled
    # gc.mem_free() walks the whole 25MB heap and costs 227ms - measured on the
    # board 2026-08-09, and it is the same heap-sized cost as gc.collect(). Per
    # frame it took the stream from 8.8 to 3.2 fps. So the host asks for a reading
    # only every Nth batch, and it is taken here, before the frame loop, where a
    # stall is harmless: plain gaps of up to 2.5s between snapshots were measured
    # not to disturb the sensor (unlike a collect, which wedges it at any length).
    if report:
        free = gc.mem_free()
        # mem_free() monopolises the VM long enough that TinyUSB cannot service
        # its CDC queue.  Yield once before putting the next header into that
        # queue; the 50ms is inside the Lepton's measured-safe service gap and
        # lets USB recover instead of dropping its first 512-byte packet.
        time.sleep_ms(50)
        if not write_text("#HEAP %d\n" % free):
            return
    for _ in range(n):
        t, was_torn = good_snapshot(lep)       # paces the loop at the thermal rate
        t_th = time.ticks_ms()
        j = rgb.snapshot().to_jpeg(quality=QUALITY)
        t_rgb = time.ticks_ms()
        tb = t.bytearray()
        if _raw is not None:
            # Replaces the 8-bit plane rather than joining it. Both would be
            # 56.25KB of thermal per frame, 493KB/s at 8.772fps, and with the
            # JPEG on top that is 654KB/s against a link measured at 700-900 -
            # inside the 500ms write timeout's reach, and the tail it discards
            # is silent. The 8-bit frame is a lossy function of this one, so the
            # host reconstructs it for free instead of paying to be told twice.
            #
            # t is still snapshotted and still used: good_snapshot() reads its
            # segment boundaries to reject torn frames, which is a check on the
            # VoSPI transport and is unaffected by which plane travels.
            lep.ioctl(csi.IOCTL_LEPTON_GET_RAW, _raw)
            tb = _raw
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
        if not write_text("#F %d %d %d %d %d %d\n" % (
                j.size(), len(tb), 1 if was_torn else 0,
                time.ticks_diff(t_th, _t_prev), time.ticks_diff(t_rgb, t_th),
                _stalled)):
            return
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
    write_text("#BATCH\n")


write_text("#READY %d %d %d %d %d %d\n" % (
    rgb.width(), rgb.height(), lep.width(), lep.height(), TMIN, TMAX))
'''

# Sent repeatedly. Deliberately bounded: an unbounded loop on the board keeps
# writing into a port nobody reads if this host dies, the CDC RX backs up, and
# the board wedges hard enough to need a physical replug. A batch self-terminates.
BATCH_CODE = "stream(%d, %d)\n"

# How often to ask the board what its heap looks like while there is ample room.
# gc.mem_free() walks the complete 25MB heap and blocks the MicroPython/TinyUSB
# scheduler for 227ms.  At every 10th batch that pause landed every ~23s and the
# live hardware lost 512-byte CDC packets at the same cadence.  With five frames
# per batch, every 400th batch is ~3.8 minutes: still almost ten observations
# inside the deliberately huge
# 36-minute 4MB safety margin.  _want_heap() tightens to every third/every batch
# as that margin closes, where memory safety correctly wins over frame cadence.
HEAP_EVERY = 400


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
        # fusion.c has measured, motion-adaptive thermal smoothing enabled by
        # default. Expose its two real controls here so the live product can
        # trade settling time for a quieter picture without changing raw
        # recordings or inventing a second host-side filter.
        cfg.temporal_noise_mc = getattr(args, "thermal_noise_mc",
                                        cfg.temporal_noise_mc)
        cfg.temporal_frames = getattr(args, "thermal_frames",
                                      cfg.temporal_frames)
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
        # Product channel, separate from the registration/calibration view
        # above.  `view` chooses how a fused base is inspected; `channel`
        # chooses which sensors and overlays are presented at all.
        self.channel = getattr(args, "channel", "fusion")
        self.mix = 60
        self.outline = False
        self.blink_period = 0.5      # seconds per half-cycle; ~2 alternations/s
        self.show_detections = True
        self.show_radar = True
        # The whisker is the honest width of a radar detection's vertical
        # uncertainty. On by default precisely because the bare dot flatters the
        # sensor: elevation comes from a two-element aperture and is the least
        # trustworthy thing on the screen.
        self.radar_whisker = True
        # The three AI channels. Switchable live rather than picked at launch:
        # which of them is worth believing is a question about the scene in
        # front of the rig - light, motion, clutter - and the operator answering
        # it must not have to restart the viewer and cut the recording in two.
        # Independent switches and not a mode selector, because `fusion` is the
        # AGREEMENT between the other two: seeing it beside its components is
        # the only way to tell a real agreement from two channels that are each
        # firing on everything.
        self.ai_thermal = True
        self.ai_radar = True
        self.ai_fusion = True
        # Raw student boxes are diagnostic evidence, not the product result.
        # Keep the engines enabled for fusion/tracking while presenting a clean
        # AI channel by default.  The operator can reveal these layers when
        # debugging a model without turning inference off and losing the lock.
        self.ai_evidence = False
        # Per-channel confidence floors, applied to what is DRAWN rather than
        # inside the engines. Two reasons: the engine's own threshold cannot be
        # lowered again without a restart, and a box that was found and then
        # hidden has to stay countable - the card says "3 of 7", because a
        # filter that silently eats detections is worse than the clutter it
        # removes. Set from --student-conf at startup; the page moves them.
        self.ai_conf_thermal = 0.5
        self.ai_conf_radar = 0.5
        # Draw the person rather than a rectangle around the person: the
        # thermal channel knows the shape, and a box is a claim about a
        # bounding rectangle that nothing in the scene has. Falls back to the
        # rectangle wherever the shape cannot be found - no warp LUT, no
        # thermal coverage, or a box with nothing warm in it.
        self.ai_silhouette = True
        # Lock mode. A detector answers one frame at a time and is right to;
        # an operator watching a person walk does not want that answer
        # re-litigated 8.7 times a second. With this on, a person who has been
        # seen twice is TRACKED: kept through the frames no sensor found them
        # in, carried on their own velocity, and dropped only after a bounded
        # coast - drawn as coasting the whole time it is not a measurement.
        self.ai_lock = True
        # The floor the engines themselves were built with: a slider below this
        # does nothing, so the page clamps to it rather than pretending.
        self.ai_conf_floor = 0.5

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
        # fusion_temp_region takes corners (x0,y0,x1,y1), not width/height -
        # see main.c's (0,0,out_w,out_h) call, where the two happen to coincide.
        with self.lock:
            rc = self.lib.fusion_temp_region(ctypes.byref(self.f), int(x), int(y),
                                             int(x + w), int(y + h), ctypes.byref(r))
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

    def thermal_image(self):
        """Pure registered thermogram on the 640x400 visible coordinate plane.

        Unlike the normal fused output this contains no detail borrowed from
        the visible camera.  That distinction is the contract of the thermal
        display channel: an edge in this image came from the thermal plane.
        Pixels outside the calibrated footprint are neutral dark grey rather
        than a made-up cold temperature.
        """
        treg = self.treg_grid()
        cover = self.cover_grid()
        # The registered grid is only 160x100. Nearest-neighbour expansion made
        # each sample a visible 4x4 tile, so the pure thermal product looked
        # blocky even though the temporal filter upstream was doing its job.
        # Interpolate temperature codes before applying the palette: this keeps
        # the colour ramp physically ordered and smooths only the operator
        # display. The raw plane, measurements, student input and recordings
        # remain untouched.
        full = cv2.resize(treg, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(cover, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST).astype(bool)
        palette = np.ctypeslib.as_array(self.palettes[self.palette_name]).reshape(256, 3)
        out = palette[full].copy()
        out[~valid] = (24, 24, 24)
        return out

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

    def bad_rows(self):
        """Which thermal rows fusion.c condemned on the frame it last processed.

        Exposed so the 16-bit path can exclude them without re-deriving the
        test. Two independent answers to "is this row dead" would eventually
        disagree, and the one that disagreed would be the one nobody was
        looking at."""
        th = self.f.cfg.th_h
        with self.lock:
            buf = ctypes.string_at(self.f.row_bad, th)
        return np.frombuffer(buf, np.uint8).astype(bool)

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
#   operator preserve visible luminance while carrying thermal information mostly
#           in colour. Unlike fused, this is meant for a person watching the scene,
#           not for reading a temperature back from the rendered pixel.
#
# All host-side numpy. None of this belongs in fusion.c - that file compiles into
# firmware, and these are display questions asked while calibrating.

VIEWS = ("fused", "visible", "blink", "mix", "edges", "operator")
CHANNELS = live_channels.CHANNELS


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


def operator_fusion(fused, y, cover=None, mix=60):
    """Operator-friendly fusion: visible structure with thermal colour.

    The radiometric fused view intentionally lets temperature own the whole
    low-frequency image and adds only visible high-frequency detail. That is an
    honest thermogram, but it hides large visible structures behind an opaque
    false-colour sheet. This view keeps the visible luminance and borrows the
    thermal layer's colour, with two deliberately display-only adaptations:

      * flat/low-contrast visible areas trust thermal luminance more, while
        textured areas retain more of the camera image;
      * the thermal footprint is feathered into visible grey instead of ending
        at a hard rectangular seam.

    Temperatures still come from temp_at()/temp_region(), never from this image.
    mix is the nominal thermal weight and remains one live control shared with
    the simple mix view.
    """
    gray = np.repeat(y[:, :, None], 3, 2)
    base = max(0.0, min(1.0, mix / 100.0))

    # Split fused into luma and colour residual directly. This has the useful
    # property of LAB/YCbCr (visible brightness and thermal colour can be mixed
    # independently) without two expensive full-frame colour conversions.
    fused_luma = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY).astype(np.float32)
    yf = y.astype(np.float32)
    local_mean = cv2.blur(y, (17, 17), borderType=cv2.BORDER_REFLECT)
    local_detail = cv2.absdiff(y, local_mean).astype(np.float32)
    visible_confidence = np.clip((local_detail - 2.0) / 22.0, 0.0, 1.0)
    thermal_luma = np.clip(base + (0.5 - visible_confidence) * 0.30, 0.20, 0.90)

    target_luma = fused_luma * thermal_luma + yf * (1.0 - thermal_luma)
    chroma = 0.25 + 0.75 * base
    out = target_luma[..., None] + (
        fused.astype(np.float32) - fused_luma[..., None]) * chroma
    out = np.clip(out, 0, 255).astype(np.uint8)

    if cover is not None:
        # Roughly a 20 px transition at 640x400: wide enough not to look like a
        # calibration cut, narrow enough not to imply thermal coverage far past
        # the last measured cell. Blur on the 160x100 grid first: the result is
        # the same scale, with a quarter of each dimension to process.
        footprint = cv2.GaussianBlur(cover.astype(np.float32), (0, 0),
                                     sigmaX=2.0, sigmaY=2.0)
        footprint = cv2.resize(footprint, (out.shape[1], out.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        footprint = np.clip(footprint, 0.0, 1.0)[..., None]
        out = (out.astype(np.float32) * footprint
               + gray.astype(np.float32) * (1.0 - footprint)).astype(np.uint8)
    return out


def agc8(raw16, rng):
    """The board's 8-bit plane, rebuilt from the 16-bit words it was made of.

    This is what makes --raw16 additive: everything downstream - the warp LUT,
    fusion.c's deband and dead-row thresholds, the students, the detector - was
    tuned against the plane the board used to send, so this reproduces it rather
    than improving on it. thermal_io.codes8() is the firmware's own arithmetic,
    kept in one place so the live path and the offline readers cannot drift.
    """
    tmin, tmax = rng if rng else (-10, 140)
    window = thermal_io.window_ck({"tmin": tmin, "tmax": tmax})
    return thermal_io.codes8(raw16, window).tobytes()


def compose(view, fused, y, treg=None, cover=None, mix=50, phase=True,
            exclude=None, outline=True):
    """Build the frame the browser sees. `fused` is never modified in place."""
    if view == "operator":
        out = operator_fusion(fused, y, cover=cover, mix=mix)
    elif view == "mix":
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

    if cover is not None and outline:
        out[coverage_outline(cover, out.shape[1], out.shape[0])] = (255, 210, 40)
    return out


def compose_channel(pipe, fused, y, now=None):
    """Build the clean image base for the selected product channel.

    Sensor/AI overlays are intentionally absent.  Keeping this as a pure-ish
    boundary makes the five channel contracts testable without a board, radar
    or detector, and keeps calibration recording able to copy the clean frame
    before any annotation is burned into it.
    """
    channel = live_channels.get(pipe.channel)
    if channel.base == "thermal":
        out = pipe.thermal_image()
        if pipe.outline:
            out[coverage_outline(pipe.cover_grid(), OUT_W, OUT_H)] = (255, 210, 40)
        return out
    if channel.base == "visible":
        out = np.repeat(y[:, :, None], 3, 2)
        if pipe.outline:
            out[coverage_outline(pipe.cover_grid(), OUT_W, OUT_H)] = (255, 210, 40)
        return out

    out = fused
    if pipe.view != "fused" or pipe.outline:
        edges = pipe.view == "edges"
        needs_cover = pipe.outline or pipe.view == "operator"
        clock = time.time() if now is None else now
        out = compose(pipe.view, out, y,
                      treg=pipe.treg_grid() if edges else None,
                      exclude=pipe.repaired_grid() if edges else None,
                      cover=pipe.cover_grid() if needs_cover else None,
                      mix=pipe.mix,
                      phase=int(clock / pipe.blink_period) % 2 == 0,
                      outline=pipe.outline)
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
    link_recovering = state.get("link_recovering_since")
    if restarting is not None:
        # Recovering is not failing. A restart takes ~13s against a 5s stale
        # threshold, so without this branch every successful defence reports as a
        # dead stream - which is how a health panel gets ignored.
        add("stream", "warn", "restarting the board: %s (%.0fs)"
            % (state.get("last_restart", "fault"), now - restarting))
    elif link_recovering is not None:
        add("stream", "warn", "dropping one incomplete batch: %s"
            % state.get("link_recovering_reason", "USB short write"))
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
    last_rs = state.get("last_resync_t")
    if rs and last_rs is not None and now - last_rs < 60.0:
        add("framing", "warn", "%d resync%s: %s" % (
            rs, "" if rs == 1 else "s", state.get("last_resync", "")))
    else:
        add("framing", "ok", "in step%s" % (
            " (%d recovered total)" % rs if rs else ""))

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

    # --- the machine this is running on. Deliberately next to the check above:
    # they are the same subject seen from opposite ends. "thermal load" is the
    # damage, measured on the board's own clock and therefore already in the
    # past by the time it appears; these are the causes, and they move first.
    s = state.get("soc")
    if s is not None:
        out.extend(hostsoc.checks(s.read()))

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
    # --- scene light. Its own line, above the detector, because it is what
    # decides how to read the detector's line: a detector reporting nothing is
    # either a working detector in an empty scene or a blind one, and those are
    # the same sentence until this number is beside it.
    light = state.get("light")
    if light is not None:
        if light["state"] == "blind":
            add("light", "warn",
                "visible is BLIND - %.1f%% of the frame is lit. Measured at "
                "this light the detector recovers 1.6%% of the people in "
                "front of it; thermal and radar are the channels to read"
                % (100 * light["lit"]))
        elif light["state"] == "dim":
            add("light", "warn",
                "part-lit - %.0f%% of the frame is lit, and nobody has measured "
                "what this detector finds there. Its silence is NOT excused"
                % (100 * light["lit"]))
        else:
            add("light", "ok", "lit - %.0f%% of the frame, median luma %.0f"
                % (100 * light["lit"], light["p50"]))

    dt = state.get("detect_t")
    if dt is not None:
        age = now - dt
        n_det = len(state.get("detections") or [])
        blind = (light or {}).get("state") == "blind"
        if state.get("detect_error"):
            add("detect", "fail", state["detect_error"])
        elif age > DETECT_STALE_S * 4:
            add("detect", "warn", "no detection for %.0fs" % age)
        elif blind and not n_det:
            # The one case where finding nothing is not a result. Saying "0
            # boxes" here reads as "nobody is there", which is the claim this
            # rig has no light to make.
            add("detect", "warn", "silent, and explained: no light to see by")
        elif state.get("detections_rejected"):
            rej = state["detections_rejected"]
            add("detect", "warn", "%d usable box%s; rejected %d implausible "
                "person box%s (%s)" % (
                    n_det, "" if n_det == 1 else "es", len(rej),
                    "" if len(rej) == 1 else "es",
                    rej[0].get("rejected", "geometry")))
        elif not pipe.warped:
            add("detect", "warn", "%d box%s, %.0fms - temperatures are read through "
                "the placeholder warp and belong to the wrong pixels"
                % (n_det, "" if n_det == 1 else "es", state.get("detect_ms", 0)))
        else:
            add("detect", "ok", "%d box%s, %.0fms"
                % (n_det, "" if n_det == 1 else "es", state.get("detect_ms", 0)))

    # --- the radar link. Separate from the "N in / M out" the overlay prints
    # on the frame: once nothing arrives the overlay has nothing to draw, and a
    # silently absent sensor looks exactly like an empty scene.
    if state.get("radar_proj") is not None:
        if state.get("radar_error"):
            add("radar", "fail", state["radar_error"])
        elif not state.get("radar_frames"):
            add("radar", "warn", "no radar frame yet - was the config sent? "
                "(tools/send_radar_cfg.py)")
        else:
            drop = state.get("radar_dropped", 0)
            add("radar", "warn" if drop else "ok",
                "%d frames, %d drawn / %d offscreen%s"
                % (state["radar_frames"], state.get("radar_drawn", 0),
                   state.get("radar_offscreen", 0),
                   ", %d bytes dropped" % drop if drop else ""))

    # --- selected product channel. An unavailable sensor must not look like an
    # empty scene, and the current visible transport must not be advertised as
    # colour merely because the stable API id is rgb_radar.
    product = live_channels.get(pipe.channel)
    if product.raw_radar and state.get("radar_proj") is None:
        add("channel", "warn", "%s selected without a radar source"
            % product.label)
    elif pipe.channel == "rgb_radar":
        add("channel", "warn", "visible + radar is PAG7936 luma in this build; "
            "true RGB waits on the RGB565 bandwidth gate")
    # An AI overlay is optional decoration on the thermal-bearing products;
    # missing inference must not make a healthy thermal camera look broken.
    # The dedicated AI product, however, has no useful contract without an
    # inference source, so it advertises that absence explicitly.
    elif pipe.channel == "ai" and not (state.get("detector_available")
                                      or state.get("students") is not None):
        add("channel", "warn", "AI selected but neither detector nor student "
            "engines are available")
    else:
        add("channel", "ok", "%s" % product.label)

    # --- the AI channels. Not "is the model loaded" - the switches are live,
    # so the question worth answering is whether what is on the screen right now
    # is what the operator thinks they are looking at. A fusion channel that
    # cannot pair draws nothing, and nothing looks exactly like an empty scene.
    st = state.get("students")
    if st is not None:
        n_t = len(state.get("student_thermal") or [])
        n_r = len(state.get("student_radar") or [])
        n_f = len(state.get("student_fused") or [])
        on = [n for n, flag in (("thermal", pipe.ai_thermal),
                                ("radar", pipe.ai_radar),
                                ("fusion", pipe.ai_fusion)) if flag]
        no_lut = st.get("th2vis") is None
        no_radar = st.get("radar") is None
        if state.get("student_error"):
            add("ai", "fail", state["student_error"])
        elif not on:
            add("ai", "warn", "all three channels switched off - the students "
                "are loaded and nothing is running")
        elif pipe.ai_fusion and (no_lut or no_radar):
            add("ai", "warn", "fusion is on but cannot pair: %s"
                % ("no warp LUT, so thermal boxes never reach the visible plane"
                   if no_lut else "no radar student engine on this run"))
        elif pipe.ai_thermal and no_lut:
            add("ai", "warn", "thermal channel has no warp LUT - %d box%s found, "
                "none can be placed on the picture"
                % (n_t, "" if n_t == 1 else "es"))
        else:
            seen = state.get("student_seen") or {}
            rejected_f = state.get("student_fused_rejected") or []
            hidden = (max(0, seen.get("thermal", 0) - n_t)
                      + max(0, seen.get("radar", 0) - n_r))
            add("ai", "warn" if rejected_f and pipe.ai_fusion else "ok",
                "%s on - T %d / R %d / TR %d%s%s"
                % ("+".join(on), n_t, n_r, n_f,
                   "" if not hidden else
                   ", %d below the confidence floor or deduplicated" % hidden,
                   "" if not rejected_f else
                   ", %d TR rejected by raw-radar/geometry gate" % len(rejected_f)))

    # --- the lock. On its own line rather than inside the detector's: the
    # question it answers is not "did a sensor see somebody" but "is the viewer
    # still holding the person it had", and a lock held only by a coast is a
    # different claim from a lock being measured.
    if pipe.ai_lock:
        tr = state.get("tracks") or []
        static = state.get("tracks_static") or []
        coasting = [t for t in tr if t.get("coasting")]
        held = ", %d warm and motionless (not drawn)" % len(static) if static else ""
        if not tr:
            add("lock", "ok", "nothing locked" + held)
        elif len(coasting) == len(tr):
            add("lock", "warn", "%d track%s, all coasting - no sensor has "
                "confirmed them this cycle (worst %.1fs)"
                % (len(tr), "" if len(tr) == 1 else "s",
                   max(t["coasting"] for t in coasting)))
        else:
            add("lock", "ok", "%d locked%s%s"
                % (len(tr) - len(coasting),
                   ", %d coasting" % len(coasting) if coasting else "", held))

    # --- recording. A recorder that died mid-session must not be discovered at
    # the end of the campaign; the mp4 writer's failure mode is a 0-byte file.
    rec = state.get("recording")
    if rec:
        if state.get("video_error"):
            add("recording", "fail", state["video_error"])
        else:
            add("recording", "ok", "%d frames -> %s"
                % (state.get("video_frames", 0), os.path.basename(rec)))

    # --- the map layer (Yael's mapinit, via perception.map_api). Only present
    #     when --map was given. A failed stage is reported with its name because
    #     "map failed" has two unrelated fixes: install the EGM2008 grid, or
    #     fetch the priors for this operating area.
    m = state.get("map")
    if m is not None:
        if m.get("pending"):
            add("map", "warn", "initialising at %.4f,%.4f" % (
                m["location"]["latitude_deg"], m["location"]["longitude_deg"]))
        elif m.get("ok"):
            alt = m.get("ego_altitude_prior") or {}
            add("map", "ok", "geoid %.2f m, ground %.0f m, %s" % (
                (m.get("geoid") or {}).get("undulation_m", float("nan")),
                alt.get("orthometric_m", float("nan")),
                "+".join(s["name"] for s in m.get("stages", [])
                         if s["status"] == "ok") or "no stage"))
        elif m.get("error"):
            add("map", "fail", ("deployment: " if m.get("deployment") else "")
                + m["error"].splitlines()[0])
        else:
            failed = [s["name"] for s in m.get("stages", []) if s["status"] == "failed"]
            add("map", "fail", "stage%s failed: %s" % (
                "" if len(failed) == 1 else "s", ", ".join(failed) or "unknown"))

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


class _DroppedBatch(Exception):
    """The board ended a batch while a promised payload was still incomplete.

    This is intentionally distinct from a dead serial link.  ``stream()`` gives
    up after three CDC no-progress timeouts to keep the Lepton inside its tested
    service-gap envelope, then returns normally to the raw REPL.  Restarting the
    cameras for that clean return turns one lost frame into a 20 second outage.
    The prompt is proof that the interpreter and both sensors are still alive,
    so the reader can discard this batch and submit the next one immediately.
    """

    def __init__(self, kind, received, expected):
        self.kind = kind
        self.received = received
        self.expected = expected
        super().__init__("%s short read %d/%d" % (kind, received, expected))


# A bounded raw-REPL submission ends with the line written by stream(), followed
# by MicroPython's stdout terminator and prompt.  Newline translation differs
# between ports, so accept both forms.  Detection is restricted to an exact
# BUFFER SUFFIX while fewer than the promised bytes are present; the prompt
# cannot have legitimate payload after it, and the long suffix makes mistaking
# binary image data for control framing vanishingly unlikely.
_BATCH_FOOTERS = (b"#BATCH\r\n\x04\x04>", b"#BATCH\n\x04\x04>")


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

# A luma a pixel has to reach before it counts as lit at all. Everything below
# is the noise floor of an unlit frame: the three dark sessions below sit at a
# median of 12.6-13.0 with a p95 under 22.
LIGHT_FLOOR = 48

# How much of the frame has to be lit before the detector can be expected to
# work in it. Measured 2026-08-26 by running this rig's own yolov10n over
# sampled frames of seven recorded sessions - `lit` is the share of pixels at or
# above LIGHT_FLOOR:
#
#   session                 luma p50   lit share            frames with a person
#   radar3-dark-empty2          12.6   0.000%               0 of 1413  (empty)
#   radar3-dark1                13.0   0.000%               0 of  701  (unusable)
#   radar3-dark-negative1       13.0   0.250%  (max 4.58%)  5 of  625
#   static1                     85.0  73.880%  (min 73.08%)         99%
#   moving1                     87.5  75.158%  (min 71.77%)        100%
#   lobby1                      86.7  77.555%                      100%
#   multi5                      88.5  78.113%                      100%
#   radar3-negative1            90.3  92.485%                          -
#
# The third row is the measurement that matters, and the dark sessions were run
# frame by frame rather than sampled to get it right: 306 of that session's
# frames hold a person standing close enough to fill the thermal frame, and the
# visible detector recovers FIVE of them - 1.6%. Not literally zero, and not a
# channel either. "The detector saw nobody" and "the detector could not have
# seen anybody" are the same picture and opposite claims, and until this metric
# existed nothing in the viewer could tell them apart.
#
# Why a lit SHARE and not a percentile: a percentile cannot see a small bright
# region. A dark room with one person-sized patch of light in it (20x50 px at
# 15 m, 0.39% of a 640x400 frame) has a p95 of 13 and reads as pitch black -
# and that is the one case where excusing the detector's silence would hide a
# person standing in the only lit spot in the room.
#
# The two thresholds bound what was MEASURED, and the gap between them is
# deliberately left unclaimed:
LIGHT_BLIND_FRAC = 0.05      # 0-4.58% measured: 1.6% of people recovered
LIGHT_LIT_FRAC = 0.70        # 71.8-92.5% measured: a person found in 99-100%
# Between them nobody has measured anything on this rig, so a frame that lands
# there is reported as part-lit and its silence is NOT excused. Excusing an
# unmeasured case is how a real miss gets filed as darkness.


def scene_light(y):
    """How much light the visible frame has, and whether that explains silence.

    Deliberately measured on the plain luma, before anything is drawn: an
    overlay is bright, and a picture with boxes burned into it reads as better
    lit than the scene it came from.
    """
    lit = float((y >= LIGHT_FLOOR).mean())
    if lit < LIGHT_BLIND_FRAC:
        state = "blind"
    elif lit < LIGHT_LIT_FRAC:
        state = "dim"
    else:
        state = "lit"
    return {"lit": round(lit, 4), "p50": round(float(np.median(y)), 1),
            "state": state}


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

    # A close, clipped person may legitimately fill the height of the camera,
    # so neither height nor area alone is grounds for rejection.  A landscape
    # box covering most of the entire sensor is different: it is the detector
    # calling the scene/doorway a person.  That exact failure was observed as a
    # 613x392 box on this 640x400 stream, and once admitted it permanently gave
    # a thermal/radar clutter track visible-detector trust.
    MAX_LANDSCAPE_PERSON_AREA = 0.65
    MIN_LANDSCAPE_PERSON_ASPECT = 1.10
    # The visible model sometimes calls the fixed standing lamps in this room
    # people at 0.35..0.75.  Those boxes are semantically plausible and have a
    # body-temperature peak, so neither box geometry nor radiometry can reject
    # them.  Above this floor the detector may vouch immediately; below it the
    # observation is `det_weak` and must move or gain independent thermal
    # evidence before the tracker presents it as a person.
    DET_INSTANT_VOUCH_CONF = 0.80

    def __init__(self, pipe, state, work, detector=None, radar=None,
                 radar_proj=None, video=None, students=None, th2vis=None):
        super().__init__(daemon=True)
        self.pipe, self.state, self.work = pipe, state, work
        # This frame's 16-bit words, or None on an 8-bit session. Only the
        # silhouette reads it - see _split_plane().
        self._raw16 = None
        self.students = students
        # The thermal->visible mapper. Held here and not inside `students`
        # because the person outline is drawn for the DETECTOR's boxes too,
        # and the detector runs whether or not the students were loaded.
        self.th2vis = th2vis
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
        # One tracker for the person class across every sensor. Fed once per
        # rendered frame, including the frames nothing was found in - a miss is
        # information and the tracker has to be told about it.
        self.lock = tracking.Tracker()
        # What the rig itself did between frames. The tracker cannot tell a
        # person crossing the frame from the frame moving under a person, and
        # the difference decides whether a warm door gets drawn as somebody.
        self.ego = egomotion.GlobalShift()
        # The detector runs on its own thread at its own rate, so the same
        # detection list is visible for several rendered frames. Feeding it
        # more than once would inflate the hit count and, worse, hold a lock
        # open on evidence that arrived long ago.
        self._fed_detect_t = None
        if detector:
            self.det_thread = threading.Thread(target=self._detect_loop, daemon=True)
            self.det_thread.start()

    def _detect_loop(self):
        while not self.stop.is_set():
            item = self.det_in.get()
            if item is None:
                continue
            y, light = item
            if light["state"] == "blind":
                # A detection in a frame whose pixels are all at the camera's
                # dark-noise floor is not visual confirmation. Letting one such
                # false positive reach the tracker permanently sets `ever det`
                # and allows thermal/radar clutter to inherit detector trust.
                # Report the channel as unavailable for this frame instead.
                self.state["detections"] = []
                self.state["detect_ms"] = 0.0
                self.state["detect_suppressed"] = "blind"
                self.state["detect_t"] = time.time()
                continue
            try:
                dets = self.det(y)
            except Exception as e:                  # a bad frame must not end detection
                self.state["detect_error"] = "%s: %s" % (type(e).__name__, e)
                continue
            accepted, rejected = [], []
            for d in dets:
                reason = self._reject_person_geometry(d)
                if reason is None:
                    accepted.append(d)
                else:
                    rejected.append(dict(d, rejected=reason))
            dets = accepted
            self.state["detections_rejected"] = rejected
            # The temperature is the whole reason for the box. Taken here rather
            # than in the drawing code so that /detections and the overlay quote
            # the same number, and so a slow region query costs the detector's
            # thread rather than the frame rate.
            for d in dets:
                r = self.pipe.temp_region(d["x"], d["y"], d["w"], d["h"])
                if r is None or r["samples"] == 0:
                    d["no_thermal"] = True
                else:
                    d["max_c"], d["mean_c"] = round(r["max"], 1), round(r["mean"], 1)
                    d["max_x"], d["max_y"] = r["max_x"], r["max_y"]
                    d["repaired"] = bool(r["repaired"])
                if d["cls"] == "person":
                    # Body-heat cross-check: a person the thermal camera cannot
                    # confirm (no coverage, or nothing at skin temperature in
                    # the box) is drawn dashed with a '?' instead of a check.
                    mx = d.get("max_c")
                    d["body_heat"] = (mx is not None
                                      and detect.BODY_C[0] <= mx <= detect.BODY_C[1])
            self.state["detections"] = dets
            self.state["detect_ms"] = round(self.det.ms, 1)
            self.state.pop("detect_suppressed", None)
            self.state["detect_t"] = time.time()

    @classmethod
    def _reject_person_geometry(cls, d):
        """Reason an obviously scene-sized person box is unusable, else None.

        This is deliberately one-sided and narrow.  It does not impose a
        textbook standing-person aspect ratio: seated people, partial bodies
        and close targets are all in scope.  It only refuses a box that is both
        landscape and covers most of the complete image.
        """
        if d.get("cls") != "person":
            return None
        w, h = float(d.get("w", 0)), float(d.get("h", 0))
        if w <= 0 or h <= 0:
            return "empty person box"
        area = w * h / float(OUT_W * OUT_H)
        if (area >= cls.MAX_LANDSCAPE_PERSON_AREA
                and w / h >= cls.MIN_LANDSCAPE_PERSON_ASPECT):
            return "scene-sized landscape box (%.0f%%, %.2f:1)" % (
                100.0 * area, w / h)
        return None

    # Student overlay colours, RGB frame order (imencode flips to BGR later).
    STUDENT_TH_COL = (255, 150, 0)       # orange: thermal student
    STUDENT_RD_COL = (0, 210, 255)       # cyan: radar student
    STUDENT_FU_COL = (255, 255, 255)     # white: both channels agree
    # Candidate pairing is horizontal-only. D3's static measurements put the
    # disagreement at p90 37.5 px, but the 2026-08-26 V6 dark walking test
    # measured 60..193 px while the radar box lagged the moving thermal box.
    # A tight 50 px gate therefore discarded two real detections before the
    # physical checks below could examine them. This wider gate only proposes
    # a pair: attach_range() must still find a current non-static raw return in
    # the thermal support, and _fused_geometry_reason() rejects impossible
    # range/height. Elevation remains ungated because the two-element aperture
    # has ~12 deg sigma and cannot reliably identify a box vertically.
    FUSION_DU_PX = 180.0

    # Two ways of being the same object twice, because the students emit eight
    # slots per frame with no NMS of their own:
    #   IoU        near-identical rectangles a pixel or two apart.
    #   CONTAINED  the nested case - a box around a person and another around
    #              the whole doorway they are standing in. Their IoU can be as
    #              low as 0.3 while one sits entirely inside the other, so IoU
    #              alone leaves exactly the overlapping pair that makes the
    #              picture unreadable. Measured against the smaller box, so a
    #              big weak claim cannot survive by being big.
    DEDUP_IOU = 0.55
    DEDUP_CONTAINED = 0.8

    # Above this a student box is drawn as a rectangle; below it, as four
    # corner ticks. A candidate and a detection should not look alike, and on
    # this scene at 0.6 the students claim doorways and warm pillars - drawn
    # solid they compete with the detector's green person box for attention
    # they have not earned. The number is the drawn form only; what is a
    # detection at all is the confidence slider's business.
    STRONG_CONF = 0.75

    # How far outside the person the fusion ring is drawn, in visible pixels.
    # Far enough to read as a ring around them rather than a second outline.
    HALO_PX = 5
    _RADAR_UNSET = object()

    @staticmethod
    def _draw_shape(rgb, mask, col, thick, grow=0):
        """Outline a visible-plane mask, optionally grown into a ring."""
        m = mask.astype(np.uint8)
        if grow:
            k = 2 * grow + 1
            m = cv2.dilate(m, np.ones((k, k), np.uint8))
        cnts = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
        if not cnts:
            return False
        cv2.drawContours(rgb, cnts, -1, col, thick, cv2.LINE_AA)
        return True

    @staticmethod
    def _corners(rgb, x, y, w, h, col):
        """Four corner ticks instead of a rectangle: a weaker visual claim."""
        d = int(min(max(6, min(w, h) // 5), 18))
        x1, y1 = x + w, y + h
        for (cx, sx) in ((x, 1), (x1, -1)):
            for (cy, sy) in ((y, 1), (y1, -1)):
                cv2.line(rgb, (cx, cy), (cx + sx * d, cy), col, 1, cv2.LINE_AA)
                cv2.line(rgb, (cx, cy), (cx, cy + sy * d), col, 1, cv2.LINE_AA)

    @classmethod
    def _dedup(cls, dets):
        """Drop each box that claims a region a more confident box already has.

        Plain greedy NMS over one channel's own output, on whichever plane the
        boxes are already in - the caller runs it before the thermal boxes are
        mapped to the visible plane, so the comparison is always like for like.
        _boxes_to_dets already sorts by confidence, so the first box to claim a
        region is the strongest one.
        """
        kept = []
        for d in dets:
            x0, y0 = d["x"], d["y"]
            x1, y1 = x0 + d["w"], y0 + d["h"]
            for k in kept:
                kx0, ky0 = k["x"], k["y"]
                kx1, ky1 = kx0 + k["w"], ky0 + k["h"]
                iw = min(x1, kx1) - max(x0, kx0)
                ih = min(y1, ky1) - max(y0, ky0)
                if iw <= 0 or ih <= 0:
                    continue
                inter = float(iw * ih)
                union = d["w"] * d["h"] + k["w"] * k["h"] - inter
                smaller = float(min(d["w"] * d["h"], k["w"] * k["h"]))
                if ((union > 0 and inter / union >= cls.DEDUP_IOU)
                        or (smaller > 0
                            and inter / smaller >= cls.DEDUP_CONTAINED)):
                    break
            else:
                kept.append(d)
        return kept

    @classmethod
    def _fuse(cls, tdets, rdets):
        """Pair thermal-student and radar-student boxes that are one person.

        Greedy on |du| rather than Hungarian: each channel emits at most 8
        boxes, so the assignment is small enough that nearest-under-a-hard-gate
        gives the same answer, and it keeps scipy out of the render thread.

        The fused box keeps the THERMAL extent. The radar box's height is a
        guess from that same two-element aperture; what the radar channel
        contributes here is agreement and range, not geometry. The score is the
        weaker component, not noisy-OR: both students were trained from the same
        RGB teacher, so their errors are correlated and treating them as
        independent inflated two weak claims into one strong claim.
        """
        pairs = []
        for i, t in enumerate(tdets):
            tv = t.get("vis")
            if tv is None:
                continue              # not on the visible plane: nothing to pair
            tu = tv[0] + tv[2] / 2.0
            for j, r in enumerate(rdets):
                du = abs(tu - (r["x"] + r["w"] / 2.0))
                if du <= cls.FUSION_DU_PX:
                    pairs.append((du, i, j))
        pairs.sort()
        used_t, used_r, fused = set(), set(), []
        for du, i, j in pairs:
            if i in used_t or j in used_r:
                continue
            used_t.add(i)
            used_r.add(j)
            t, r = tdets[i], rdets[j]
            x, y, w, h = t["vis"]
            fused.append({"x": x, "y": y, "w": w, "h": h,
                          "conf": min(t["conf"], r["conf"]),
                          "conf_thermal": round(t["conf"], 3),
                          "conf_radar": round(r["conf"], 3),
                          "du": round(du, 1), "_ti": i, "_ri": j})
        return fused

    @staticmethod
    def _fused_geometry_reason(d):
        """Why this candidate is not physical Thermal+Radar agreement.

        A radar-student rectangle by itself is a learned image-plane guess.  A
        TR claim additionally needs a current non-static raw return inside its
        horizontal support; attach_range() supplies that and excludes static
        room clutter.  Once range exists, the thermal extent must not imply an
        object taller than the tracker accepts as a person.
        """
        r = d.get("radar_m")
        if r is None:
            return "no moving radar return supports the thermal box"
        implied_h = float(r) * float(d["h"]) / tracking.FOCAL_PX
        if implied_h > tracking.MAX_PERSON_M:
            return "range/box implies %.1fm height" % implied_h
        d["implied_h_m"] = round(implied_h, 2)
        return None

    def _run_students(self, thermal, rgb, radar_frame=_RADAR_UNSET, draw=True):
        """Run enabled student evidence, optionally drawing it on this channel.

        Runs inline in the render thread on purpose: both engines together
        are ~1.5 ms, two orders of magnitude under the frame period, and a
        third thread would buy nothing but a handoff to race.

        Which engines run is decided here and not at launch. A channel switched
        off does not run at all - "off" has to mean off, or the GPU work and the
        error surface stay whether or not anything is drawn - but a channel that
        is off while fusion is on still runs, because fusion is a statement
        about both of them.
        """
        s = self.students
        pl = self.pipe
        # On a thermal operator product the thermal student is the primary
        # visual answer, even when the deployment switches expose only the
        # stricter fusion result.  Fusion still owns which engines run; this
        # only lets the already-computed thermal result speak on the view whose
        # whole purpose is seeing heat.
        thermal_product = live_channels.get(pl.channel).base == "thermal"
        want_t = pl.ai_thermal or pl.ai_fusion
        want_r = pl.ai_radar or pl.ai_fusion
        # The sentinel preserves the small direct-call surface used by offline
        # tests and tools.  The live render path always passes its one sampled
        # frame explicitly, including None when the radar is stale.
        fr = (self.radar.get() if radar_frame is self._RADAR_UNSET
              and self.radar is not None else radar_frame)
        tdets, rdets, fused = [], [], []
        if want_t and thermal is not None and len(thermal) == 160 * 120:
            try:
                th = (np.frombuffer(thermal, np.uint8)
                      .astype(np.float32).reshape(120, 160)
                      * s["c_per_lsb"] + s["tmin"])
                tdets = s["thermal"].push(th, time.monotonic() * 1e3)
            except Exception as e:
                self.state["student_error"] = "thermal %s: %s" % (
                    type(e).__name__, e)
        seen_t = len(tdets)
        if tdets and not getattr(s["thermal"], "primed", True):
            # The first two frames of a session run with no temporal chain
            # behind them and answer confidently about cold structure - a
            # pillar, a doorway (measured on captures/test6). Two frames is
            # 230 ms of a session; they are dropped rather than drawn, and
            # seen_t above still counts them, so the card shows "0 of 2"
            # rather than a silent gap.
            tdets = []
        tdets = self._dedup([d for d in tdets
                             if d["conf"] >= pl.ai_conf_thermal])
        # The visible-plane box travels with the detection from here on: it is
        # what gets drawn, what the radar channel is paired against, and what
        # /ai reports. None means the box fell outside the thermal/visible
        # overlap - or that there is no warp LUT to map it with at all, which is
        # the same "cannot be placed" answer arrived at earlier.
        for d in tdets:
            d["vis"] = (s["th2vis"].box(d["x"], d["y"], d["w"], d["h"])
                        if s["th2vis"] is not None else None)
        if want_r and s["radar"] is not None and fr is not None:
            try:
                rdets = s["radar"](
                    [[p['x'], p['y'], p['z'], p['v'], p['snr'],
                      p['noise']] for p in fr['points']])
            except Exception as e:
                self.state["student_error"] = "radar %s: %s" % (
                    type(e).__name__, e)
        seen_r = len(rdets)
        rdets = self._dedup([d for d in rdets
                             if d["conf"] >= pl.ai_conf_radar])
        # One mask per thermal box, keyed by the visible box it maps to - which
        # is exactly the box the fused entry carries, so the fusion ring can
        # find its person without threading an index through _fuse().
        shapes = {}
        if (draw and pl.ai_silhouette and self.th2vis is not None
                and thermal is not None and len(thermal) == 160 * 120):
            plane = self._split_plane(thermal)
            for d in tdets:
                if d["vis"] is None:
                    continue
                m = self.th2vis.shape_in(plane, *d["vis"])
                if m is not None:
                    shapes[tuple(d["vis"])] = m

        if pl.ai_fusion:
            fused = self._fuse(tdets, rdets)
            if fused and fr is not None and self.radar_proj is not None:
                # Range is the one thing the radar channel knows that the
                # picture cannot show. Taken from the raw returns rather than
                # from the radar student, whose output is a box and carries no
                # distance at all.
                radar_overlay.attach_range(fused, fr["points"], self.radar_proj)
            accepted, rejected = [], []
            for d in fused:
                reason = self._fused_geometry_reason(d)
                if reason is not None:
                    rejected.append({k: v for k, v in d.items()
                                     if not k.startswith("_")}
                                    | {"rejected": reason})
                    continue
                tdets[d["_ti"]]["fused"] = True
                rdets[d["_ri"]]["fused"] = True
                d.pop("_ti", None)
                d.pop("_ri", None)
                accepted.append(d)
            fused = accepted
            self.state["student_fused_rejected"] = rejected
        else:
            self.state["student_fused_rejected"] = []
        self.state["student_thermal"] = tdets
        self.state["student_radar"] = rdets
        self.state["student_fused"] = fused
        # What the engines produced before the floor and the dedup, so the page
        # can say how much it is hiding and the operator can tell "quiet scene"
        # from "slider too high".
        self.state["student_seen"] = {"thermal": seen_t, "radar": seen_r}
        self.state["student_t"] = time.time()

        # A component that has been paired keeps its box and loses its label:
        # the white one is already quoting a number for that person, and three
        # labels stacked on one head is how a 40 px box at 15 m becomes
        # unreadable. Which channel contributed is still visible - that is what
        # the box colour is for - and an UNpaired box keeps its label, which is
        # the case where the number actually decides something.
        if draw and (pl.ai_thermal or (thermal_product and want_t)):
            for d in tdets:
                if d["vis"] is None:
                    continue          # no LUT, or outside the overlap
                x, y, w, h = d["vis"]
                m = shapes.get((x, y, w, h))
                strong = d["conf"] >= self.STRONG_CONF
                if m is None or not self._draw_shape(
                        rgb, m, self.STUDENT_TH_COL, 2 if strong else 1):
                    if strong:
                        cv2.rectangle(rgb, (x, y), (x + w, y + h),
                                      self.STUDENT_TH_COL, 2)
                    else:
                        self._corners(rgb, x, y, w, h, self.STUDENT_TH_COL)
                if not d.get("fused"):
                    cv2.putText(rgb, "T" + ("%.2f" % d["conf"]).lstrip("0"),
                                (x, max(11, y - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                self.STUDENT_TH_COL, 1, cv2.LINE_AA)
        if draw and pl.ai_radar:
            for d in rdets:
                x, y, w, h = d["x"], d["y"], d["w"], d["h"]
                if d["conf"] >= self.STRONG_CONF:
                    cv2.rectangle(rgb, (x, y), (x + w, y + h),
                                  self.STUDENT_RD_COL, 1)
                else:
                    self._corners(rgb, x, y, w, h, self.STUDENT_RD_COL)
                if not d.get("fused"):
                    cv2.putText(rgb, "R" + ("%.2f" % d["conf"]).lstrip("0"),
                                (x, min(OUT_H - 4, y + h + 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                self.STUDENT_RD_COL, 1, cv2.LINE_AA)
        # Drawn around the components rather than instead of them: a white box
        # with an orange one inside it says "the thermal channel found this and
        # the radar agreed", which is a different and more useful statement than
        # a single box in a third colour.
        for d in fused if draw else ():
            x, y = max(0, d["x"] - 3), max(0, d["y"] - 3)
            m = shapes.get((d["x"], d["y"], d["w"], d["h"]))
            # A ring at HALO_PX outside the person, so agreement reads as
            # something drawn AROUND them rather than a second outline on top
            # of the thermal channel's.
            if m is None or not self._draw_shape(rgb, m, self.STUDENT_FU_COL,
                                                 2, grow=self.HALO_PX):
                x2 = min(OUT_W - 1, d["x"] + d["w"] + 3)
                y2 = min(OUT_H - 1, d["y"] + d["h"] + 3)
                cv2.rectangle(rgb, (x, y), (x2, y2), self.STUDENT_FU_COL, 2)
            label = "TR %.2f" % d["conf"]
            if d.get("radar_m") is not None:
                label += "  %.1fm" % d["radar_m"]
            cv2.putText(rgb, label, (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        self.STUDENT_FU_COL, 1, cv2.LINE_AA)

    # Lock overlay colours, RGB frame order.
    LOCK_COL = (0, 230, 0)               # green, as the detector's person box
    LOCK_COAST_COL = (255, 190, 60)      # amber: predicted, not measured
    # Grey, and never green or amber: a suppressed candidate is not a person
    # the viewer is claiming. It is deliberately the dimmest thing drawn.
    STATIC_COL = (150, 150, 150)
    SRC_LETTER = {"det": "D", "det_weak": "d", "thermal": "T",
                  "radar": "R", "fusion": "F"}
    COAST_OVERLAP_IOU = 0.5
    # Once motion/fusion has vouched for a person, a current body-shaped warm
    # component may hold that same track while they stand. These gates apply to
    # the component inside the existing box; they never create a new track.
    THERMAL_HOLD_MIN_PIXELS = 120
    THERMAL_HOLD_MIN_BOX_FRAC = 0.04
    THERMAL_HOLD_MIN_HEIGHT_FRAC = 0.25
    # Cold-start exception for a person who was already sitting still when the
    # process began. The learned thermal detector supplies the semantic claim;
    # the current radiometric component must independently look like a body:
    # tall enough, connected down its height, not a filled rectangle, and with
    # a head/upper section narrower than the torso. A warm door fails the last
    # two checks even if the student gives it a confident box.
    THERMAL_SHAPE_CONF = 0.90
    THERMAL_STATIC_CONF = 0.98
    THERMAL_BODY_MIN_PIXELS = 400
    THERMAL_BODY_MIN_BOX_FRAC = 0.08
    THERMAL_BODY_MIN_HEIGHT_FRAC = 0.35
    THERMAL_BODY_MIN_ASPECT = 0.65
    THERMAL_BODY_MAX_FILL = 0.88
    THERMAL_BODY_MIN_ROW_COVERAGE = 0.70
    THERMAL_BODY_HEAD_TO_TORSO = 0.85

    @classmethod
    def _display_tracks(cls, tracks, now):
        """Hide a coast that is visually replaced by a current measurement.

        The tracker keeps both identities until its bounded coast expires. That
        state is useful for reassociation, but drawing a dashed old box over a
        measured box makes one nearby person look like two. Only presentation
        is filtered; non-overlapping coasts remain visible and every track stays
        in `/ui` and `/ai`.
        """
        measured = [t for t in tracks if t.coasting_for(now) <= 0.15]
        return [t for t in tracks
                if (t.coasting_for(now) <= 0.15
                    or not any(tracking.iou(t.box, m.box)
                               >= cls.COAST_OVERLAP_IOU for m in measured))]

    @classmethod
    def _thermal_human_shape(cls, mask, box):
        """Does this warm component have conservative seated-person geometry?

        This is not another object detector. It is an independent guard on a
        thermal student's already-high-confidence person result, used only to
        decide whether a motionless cold-start may be promoted to a lock.
        Returns the verdict and compact metrics so `/ai` can explain it.
        """
        if mask is None:
            return False, {"reason": "no warm component"}
        ys, xs = np.nonzero(mask)
        pixels = int(len(xs))
        if not pixels:
            return False, {"reason": "empty warm component", "pixels": 0}
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        bw, bh = x1 - x0, y1 - y0
        area = max(1, bw * bh)
        box_area = max(1, int(box[2]) * int(box[3]))
        fill = pixels / float(area)
        height_frac = bh / float(max(1, int(box[3])))
        box_frac = area / float(box_area)
        aspect = bh / float(max(1, bw))
        crop = mask[y0:y1, x0:x1]
        widths = np.count_nonzero(crop, axis=1).astype(np.float32)
        row_coverage = float(np.count_nonzero(widths)) / max(1, bh)
        cut = max(1, bh // 4)
        top = widths[:cut]
        torso = widths[cut:min(bh, 3 * cut)]
        top_w = float(np.percentile(top[top > 0], 75)) if np.any(top > 0) else 0.0
        torso_w = (float(np.percentile(torso[torso > 0], 75))
                   if np.any(torso > 0) else 0.0)
        head_ratio = top_w / max(1.0, torso_w)
        metrics = {
            "pixels": pixels,
            "box_frac": round(box_frac, 3),
            "height_frac": round(height_frac, 3),
            "aspect": round(aspect, 3),
            "fill": round(fill, 3),
            "row_coverage": round(row_coverage, 3),
            "head_torso": round(head_ratio, 3),
        }
        checks = (
            (pixels >= cls.THERMAL_BODY_MIN_PIXELS, "too few warm pixels"),
            (box_frac >= cls.THERMAL_BODY_MIN_BOX_FRAC, "component too small"),
            (height_frac >= cls.THERMAL_BODY_MIN_HEIGHT_FRAC, "component too short"),
            (aspect >= cls.THERMAL_BODY_MIN_ASPECT, "component too wide"),
            (fill <= cls.THERMAL_BODY_MAX_FILL, "filled rectangle"),
            (row_coverage >= cls.THERMAL_BODY_MIN_ROW_COVERAGE,
             "broken vertical component"),
            (torso_w > 0 and head_ratio <= cls.THERMAL_BODY_HEAD_TO_TORSO,
             "no head-to-torso taper"),
        )
        for ok, reason in checks:
            if not ok:
                return False, dict(metrics, reason=reason)
        return True, dict(metrics, reason="human thermal shape")

    def _lock_observations(self, thermal):
        """This cycle's person evidence, from every sensor that has any.

        The detector's list is only offered once per detection - it is produced
        on another thread and stays visible for several rendered frames - while
        the student channels are recomputed every frame and are offered every
        frame. The tracker merges what describes one person before it
        associates, so a person all three saw arrives as one observation
        carrying all three names.
        """
        obs = []
        det_t = self.state.get("detect_t")
        if (det_t is not None and det_t != self._fed_detect_t
                and time.time() - det_t < DETECT_STALE_S):
            self._fed_detect_t = det_t
            for d in self.state.get("detections") or []:
                if d.get("cls") != "person":
                    continue
                # A low-confidence COCO person is useful evidence, but not a
                # verdict: on this exact view the standing lamps repeatedly
                # score 0.35..0.75 and sit inside the body-heat band.  Keep the
                # observation so a real walker can earn the lock through
                # continuity, while tracker.py refuses to vouch for it at rest.
                conf = float(d.get("conf", 0.0))
                src = ("det" if conf >= self.DET_INSTANT_VOUCH_CONF
                       else "det_weak")
                obs.append({"x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"],
                            "conf": conf, "src": src,
                            "radar_m": d.get("radar_m"), "max_c": d.get("max_c")})
        # A hidden diagnostic channel must not blindly become person evidence
        # through the lock. In fusion-only operation ordinary thermal
        # components are hold-only: they may refresh a person but may not turn
        # a warm doorway into one.
        # There is one conservative cold-start exception: a >=0.98 thermal
        # person detection whose CURRENT warm component has independently
        # human geometry. That can establish a seated, motionless person even
        # before radar Doppler or visible light is available.
        plane = (self._split_plane(thermal)
                 if self.th2vis is not None and thermal is not None
                 and len(thermal) == TH_W * TH_H else None)
        if self.pipe.ai_thermal or self.pipe.ai_fusion:
            for d in self.state.get("student_thermal") or []:
                if d.get("vis"):
                    x, y, w, h = d["vis"]
                    mask = (self.th2vis.shape_in(plane, x, y, w, h)
                            if plane is not None else None)
                    shape_ok, shape_metrics = self._thermal_human_shape(
                        mask, (x, y, w, h))
                    human_shape = (d["conf"] >= self.THERMAL_SHAPE_CONF
                                   and shape_ok)
                    static_vouch = (d["conf"] >= self.THERMAL_STATIC_CONF
                                    and shape_ok)
                    d["human_shape"] = shape_ok
                    d["shape_metrics"] = shape_metrics
                    temp = self.pipe.temp_region(x, y, w, h)
                    if temp is not None and temp.get("samples", 0):
                        d["max_c"] = round(temp["max"], 1)
                        d["mean_c"] = round(temp["mean"], 1)
                    obs.append({"x": x, "y": y, "w": w, "h": h,
                                "conf": d["conf"], "src": "thermal",
                                "max_c": d.get("max_c"),
                                "human_shape": human_shape,
                                "shape_guarded": True,
                                "static_vouch": static_vouch,
                                # A thermal box is allowed to start only when
                                # its CURRENT radiometric component looks like
                                # a person. Selecting the thermal product still
                                # draws weaker evidence, but does not turn it
                                # into a green person lock.
                                "hold_only": not human_shape})
        if self.pipe.ai_radar:
            for d in self.state.get("student_radar") or []:
                obs.append({"x": d["x"], "y": d["y"], "w": d["w"],
                            "h": d["h"], "conf": d["conf"], "src": "radar"})
        if self.pipe.ai_fusion:
            for d in self.state.get("student_fused") or []:
                obs.append({"x": d["x"], "y": d["y"], "w": d["w"],
                            "h": d["h"], "conf": d["conf"], "src": "fusion",
                            "radar_m": d.get("radar_m")})
        # A stationary person can make both learned students disappear: radar
        # loses Doppler, while an occluder such as a laptop can break the
        # thermal student's learned full-body box. The radiometric image still
        # contains their warm head/arms/legs. Use that shape only to refresh a
        # track already vouched for by prior motion/fusion. tracker.py enforces
        # hold_only again, so a warm wall or empty-room blob cannot be born or
        # promoted here even if this shape test accepts it.
        if (self.pipe.ai_fusion and not (self.state.get("student_fused") or [])
                and not (self.state.get("student_thermal") or [])
                and self.th2vis is not None and thermal is not None
                and len(thermal) == TH_W * TH_H):
            plane = self._split_plane(thermal)
            for t in self.lock.all():
                if not t.vouched:
                    continue
                m = self.th2vis.shape_in(plane, *t.box)
                if m is None:
                    continue
                ys, xs = np.nonzero(m)
                if len(xs) < self.THERMAL_HOLD_MIN_PIXELS:
                    continue
                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
                bw, bh = x1 - x0, y1 - y0
                track_area = max(1, t.box[2] * t.box[3])
                if (bw * bh < self.THERMAL_HOLD_MIN_BOX_FRAC * track_area
                        or bh < self.THERMAL_HOLD_MIN_HEIGHT_FRAC * t.box[3]):
                    continue
                obs.append({"x": x0, "y": y0, "w": bw, "h": bh,
                            "conf": t.conf, "src": "thermal",
                            "hold_only": True})
        return obs

    def _draw_tracks(self, rgb, tracks, thermal, now):
        """Draw the locks. A coast never looks like a measurement."""
        outline = self._person_outline(thermal, rgb)
        for t in tracks:
            x, y, w, h = t.box
            coast = t.coasting_for(now)
            col = self.LOCK_COAST_COL if coast > 0.15 else self.LOCK_COL
            drawn = False
            if outline is not None and coast <= 0.15:
                # Only a measured lock gets the thermal shape: the silhouette
                # is read out of THIS frame, and drawing it around a predicted
                # box would dress a guess up as a reading.
                drawn = bool(outline({"cls": "person", "x": x, "y": y,
                                      "w": w, "h": h}, col))
            if not drawn:
                if coast > 0.15:
                    detect._dashed_rect(rgb, x, y, x + w, y + h, col)
                else:
                    cv2.rectangle(rgb, (x, y), (x + w, y + h), col, 2)
            label = "P%d" % t.id
            if coast > 0.15:
                label += " coast %.1fs" % coast
            else:
                held = "".join(self.SRC_LETTER.get(s, "?")
                               for s in sorted(t.held_by))
                if held:
                    label += " " + held
            if t.radar_m is not None:
                label += "  %.1fm" % t.radar_m
            if t.max_c is not None:
                label += "  %.1fC" % t.max_c
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.45, 1)
            ty = y - 6 if y - 6 - th > 0 else y + h + th + 6
            cv2.rectangle(rgb, (x, ty - th - 4), (x + tw + 6, ty + 3),
                          (0, 0, 0), -1)
            cv2.putText(rgb, label, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, col, 1, cv2.LINE_AA)

    def _draw_static(self, rgb, tracks, now):
        """Draw what the lock is holding back, as held back.

        These are candidates the visible detector never confirmed and which
        have never moved: a warm door, a radiator, a lit sign - and, in the
        dark, a person standing still. The tracker cannot tell those apart and
        neither can this, which is exactly why they are drawn differently
        rather than either claimed or hidden.

        Counted on the card and API even when hidden. This diagnostic overlay
        is shown only with `student evidence`: the normal AI product must not
        surround furniture and other rejected candidates with tracking marks.

        Corner ticks and not a rectangle, grey and not green, no thermal
        silhouette: every choice here is to keep it from reading as a person.
        The same corner-tick language the student channels already use below
        their confidence floor, and for the same reason - a candidate and a
        detection must not look alike.
        """
        for t in tracks:
            x, y, w, h = t.box
            arm = max(6, min(w, h) // 4)
            col = self.STATIC_COL
            for cx, sx in ((x, 1), (x + w, -1)):
                for cy, sy in ((y, 1), (y + h, -1)):
                    cv2.line(rgb, (cx, cy), (cx + sx * arm, cy), col, 1,
                             cv2.LINE_AA)
                    cv2.line(rgb, (cx, cy), (cx, cy + sy * arm), col, 1,
                             cv2.LINE_AA)
            # No id: it is not a track the viewer is claiming continuity for.
            # How long it has been there is the useful number - something warm
            # that has sat still for a minute is furniture, and something warm
            # that appeared eight seconds ago and has not moved is worth a look.
            label = "static %.0fs" % (now - t.born)
            if t.max_c is not None:
                label += "  %.1fC" % t.max_c
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.4, 1)
            ty = y - 5 if y - 5 - th > 0 else y + h + th + 5
            cv2.putText(rgb, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, col, 1, cv2.LINE_AA)

    def _split_plane(self, thermal):
        """The thermal plane the silhouette should split, at the best precision
        this session has.

        Otsu is separating a person from whatever is behind them, and that
        boundary is routinely a fraction of a degree - measured in this rig's
        lobby, a person reads 30.9 C and a lit glass door 31.7 C. On the 8-bit
        plane at the 0:60 window run_live.sh pins for the students, 0.8 C is 3.4
        codes; over 4000 person-sized boxes, one holding 1-2 C of spread hands
        Otsu a median of 8 distinct levels. Eight levels is not a histogram to
        split, it is quantisation noise with a threshold drawn through it.

        The same 0.8 C is 80 counts in the sensor's words, and the same boxes
        give 79 levels. So the shape splits the words whenever --raw16 is
        streaming.

        Deliberately the ONLY thing that moves to 16-bit. The student engines
        were trained on the 8-bit plane and have to keep seeing it, and this
        changes what is drawn, never what is measured or recorded.
        """
        if self._raw16 is not None:
            return self._raw16
        return np.frombuffer(thermal, np.uint8).reshape(TH_H, TH_W)

    def _person_outline(self, thermal, rgb):
        """An annotate() callback that draws a person's thermal shape.

        None when nothing could draw one - no LUT, no thermal frame, or the
        outline switched off - so annotate() falls straight back to its boxes
        rather than calling something that always says no.
        """
        if (not self.pipe.ai_silhouette or self.th2vis is None
                or thermal is None or len(thermal) != 160 * 120):
            return None
        plane = self._split_plane(thermal)

        def draw(d, col):
            if d["cls"] != "person":
                return False
            m = self.th2vis.shape_in(plane, d["x"], d["y"], d["w"], d["h"])
            return m is not None and self._draw_shape(rgb, m, col, 2)
        return draw

    def _raw_stats(self, raw16):
        """Scene temperature at the sensor's own precision, off the 16-bit words.

        Reported beside frame_stats() rather than instead of it, because the two
        answer different questions. That one asks "how hot is this pixel of the
        picture", sampling the registered, warped, 8-bit plane the fused image is
        built from, and it stays the right answer for a click on the image. This
        one asks "what is actually in front of the camera", on the thermal
        sensor's own grid, with no window, no clipping and no requantisation -
        0.01C per code against the 0.588C the wide-open 8-bit plane can express
        and the ~0.141C it manages after auto-ranging.

        The dead rows come out first or they own both ends of the answer. This
        unit returns 14 rows carrying no scene, pinned high - measured ~190
        codes above the frame median - so an unfiltered max is those rows every
        time and an unfiltered mean is 11.67% of the frame pulling upward.
        fusion.c has already decided which rows they are, on this very frame, so
        the verdict is borrowed rather than re-derived.
        """
        bad = self.pipe.bad_rows()
        kept = raw16[~bad] if bad.any() else raw16
        if kept.size == 0:
            return None
        v = thermal_io.celsius(kept)
        lo, hi = float(v.min()), float(v.max())
        return {"min": lo, "max": hi, "mean": float(v.mean()),
                "delta": hi - lo, "rows_dead": int(bad.sum())}

    def run(self):
        while not self.stop.is_set():
            item = self.work.get()
            if item is None:
                continue
            jpg, wire = item

            # One wire format, two meanings, told apart by length alone. 19200
            # bytes is the board's AGC'd picture and is used as-is. 38400 is the
            # Lepton's own radiometric output at 0.01C per code: the picture is
            # rebuilt from it so the pipeline sees exactly what it always saw,
            # and the words themselves go on to the recorder and the stats,
            # which are the two things that wanted a measurement.
            thermal, raw16 = wire, None
            if len(wire) == 2 * TH_W * TH_H:
                raw16 = np.frombuffer(wire, "<u2").reshape(TH_H, TH_W)
                thermal = agc8(raw16, self.state.get("range"))
            self._raw16 = raw16

            y = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
            if y is None or y.shape != (OUT_H, OUT_W):
                # A short or corrupt JPEG is the visible signature of bytes lost
                # on the wire. Count it: without this the frame simply vanishes
                # and the link looks healthier than it is.
                self.state["bad_jpeg"] = self.state.get("bad_jpeg", 0) + 1
                # The next good frame is not the successor of the last one, but
                # the shift across the gap is still the shift the tracker's own
                # dt covers, so the reference is kept rather than dropped.
                continue

            p = self.pipe
            fused = p.process(y.tobytes(), thermal)
            channel = live_channels.get(p.channel)

            # One radar answer per camera frame.  Every consumer below - the
            # radar student, detector range association, map state and raw
            # overlay - sees this exact object, so a single displayed frame can
            # no longer mix two adjacent radar frame numbers.
            radar_frame = self.radar.get() if self.radar is not None else None

            rw = self.state.setdefault("rows_window", [])
            rw.append(p.f.rows_rebuilt)
            del rw[:-60]

            # After process(), so fusion.c's row verdict describes this frame.
            if raw16 is not None:
                self.state["raw_stats"] = self._raw_stats(raw16)

            # PAG7936 is currently transported as luma.  compose_channel() is
            # the stable boundary where true RGB will enter after its board-side
            # bandwidth gate; fusion.c and the detector will continue to get y.
            rgb = compose_channel(p, fused, y)

            # The recording feeds calibration picking, so it must not carry the
            # guessed radar overlay or detection boxes: a pixel clicked next to
            # a burned-in marker is a correspondence derived from the very
            # projection being solved for. Copy before anything is drawn;
            # radar_calib_web.py refuses frames that lack the 'clean' flag.
            clean = rgb.copy() if self.video is not None else None

            # How much light the visible camera had this frame. Measured on the
            # plain luma before any overlay, every frame and not only under the
            # lock, because the question it answers - is the visible channel's
            # silence explained - is asked of the detector too.
            self.state["light"] = scene_light(y)

            if self.students is not None:
                self._run_students(
                    thermal, rgb, radar_frame=radar_frame,
                    # Raw student evidence normally stays behind the evidence
                    # switch. On a thermal product the thermal silhouette is
                    # the product itself, not debug clutter; _run_students()
                    # still keeps hidden radar components hidden.
                    draw=(channel.ai_overlay
                          and (p.ai_evidence or channel.base == "thermal")))

            if self.det_in is not None:
                # The detector reads the plain visible luma, not the fused frame.
                # The COCO weights were trained on natural images; a false-colour
                # thermal composite is further from that than grey is, and the
                # boxes are wanted in visible-camera coordinates anyway - that is
                # the frame fusion_temp_region() maps through the warp.
                self.det_in.put((y, self.state["light"]))
                if p.show_detections and channel.ai_overlay:
                    dets = self.state.get("detections") or []
                    age = time.time() - self.state.get("detect_t", 0)
                    # Boxes outlive the frame they were found in by design - the
                    # detector runs slower than the stream. Past this they are a
                    # claim about a scene that may be gone, so drop them rather
                    # than draw a stale rectangle over a moved object.
                    if age < DETECT_STALE_S:
                        if radar_frame is not None and self.radar_proj is not None:
                            radar_overlay.attach_range(dets, radar_frame["points"],
                                                       self.radar_proj)
                        # The detector says WHERE and the thermal frame says
                        # what shape is warm there. This is the pairing worth
                        # drawing: the detector is the locator this rig
                        # trusts, and a rectangle is a claim about a bounding
                        # box that nothing in the scene has.
                        if not p.ai_lock:
                            detect.annotate(
                                rgb, dets, p.warped,
                                outline=self._person_outline(thermal, rgb))
                        else:
                            # In lock mode the person boxes are drawn by the
                            # tracker below, with identity and coast state on
                            # them. Anything that is not a person still gets
                            # its box here.
                            detect.annotate(
                                rgb, [d for d in dets if d["cls"] != "person"],
                                p.warped)

            # The lock runs every frame, including the ones no sensor found
            # anybody in: that is the frame a track learns it was missed, and
            # skipping it would make a coast last as long as the scene is quiet.
            if p.ai_lock:
                now = time.time()
                # Measured on the plain luma, before anything was drawn on the
                # frame: a box annotated onto the picture moves with its target
                # and would correlate as if the camera had moved.
                ego = self.ego.measure(y)
                self.state["ego"] = self.ego.as_dict()
                tracks = self.lock.update(self._lock_observations(thermal), now,
                                          ego=ego)
                suppressed = self.lock.suppressed(now)
                if p.show_detections and channel.ai_overlay:
                    # Rejected/static candidates are diagnostics, not final AI
                    # output. When explicitly revealed, draw them first so a
                    # confirmed lock over the same pixels wins the picture.
                    if p.ai_evidence:
                        self._draw_static(rgb, suppressed, now)
                    self._draw_tracks(
                        rgb, self._display_tracks(tracks, now), thermal, now)
                self.state["tracks"] = [t.as_dict(now) for t in tracks]
                # Held back for never having moved and never having been seen
                # by the detector - a warm door, a radiator, a lit sign. Counted
                # and reported: "nothing there" and "something warm there that
                # has never moved" are different answers.
                self.state["tracks_static"] = [
                    t.as_dict(now) for t in suppressed]
            elif self.state.get("tracks"):
                self.state["tracks"] = []
                self.state["tracks_static"] = []

            if self.radar is not None:
                if radar_frame is not None:
                    fr = radar_frame
                    # Map and diagnostics consume radar whether or not this
                    # display channel draws the raw points.
                    self.state["radar_frame"] = fr["frame_number"]
                    self.state["radar_points"] = fr["points"]
                    self.state["radar_points_t"] = time.time()
                if (radar_frame is not None and channel.raw_radar
                        and self.pipe.show_radar):
                    fr = radar_frame
                    d, off, al = radar_overlay.annotate(
                        rgb, fr["points"], self.radar_proj,
                        show_whisker=self.pipe.radar_whisker)
                    self.state["radar_drawn"] = d
                    self.state["radar_offscreen"] = off
                    self.state["radar_aliased"] = al
                elif radar_frame is not None:
                    self.state["radar_drawn"] = 0
                    self.state["radar_offscreen"] = 0
                    self.state["radar_aliased"] = 0
                if (radar_frame is not None and channel.ai_overlay
                        and getattr(self, "radar_ai", False)):
                    # Optional legacy radar classifier belongs to the AI
                    # channel, not to the raw sensor channels.
                    self.state["radar_persons"] = radar_overlay.annotate_ai(
                        rgb, radar_frame["points"], self.radar_proj)
                self.state["radar_frames"] = self.radar.frames
                self.state["radar_dropped"] = self.radar.dropped_bytes
                if self.radar.error:
                    self.state["radar_error"] = self.radar.error

            if self.video is not None:
                extra = {"view": self.pipe.view, "channel": self.pipe.channel,
                         "clean": True}
                if self.radar is not None:
                    # The radar frame number this picture was drawn against ties
                    # the two recordings together even if a timestamp is doubted.
                    extra["radar_frame"] = self.state.get("radar_frame")
                # The raw thermal bytes go too: the mp4 is for looking, the
                # thermal stream is the measurement, and a temperature must
                # never be read back off an 8-bit lossy video.
                # The wire bytes, byte-for-byte as the board sent them, which
                # is what recorder.py's index promises and what lets it declare
                # thermal_dtype for the session. Writing the rebuilt 8-bit plane
                # here would record a picture and throw the measurement away -
                # the exact trade --raw16 exists to stop making.
                self.video.write(clean, thermal=wire, extra=extra)
                self.state["video_frames"] = self.video.frames
                if self.video.error:
                    self.state["video_error"] = self.video.error

            ok, enc = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                self.state["frame"] = enc.tobytes()
            self.state["rendered"] = self.state.get("rendered", 0) + 1
            self.state["dropped"] = self.work.dropped

        if self.video is not None:
            # mp4 is finalized on release(); a recording that is never closed
            # is a recording that may not open.
            self.video.close()


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

    def __init__(self, port, pipeline, quality, state, work, batch=5, raw16=False):
        super().__init__(daemon=True)
        self.port, self.pipe, self.quality, self.state = port, pipeline, quality, state
        # Ask the board for the Lepton's own 16-bit words instead of the 8-bit
        # plane it derives from them. See --raw16, and lepton_copy_raw() in the
        # firmware for what is being kept.
        self.raw16 = raw16
        # Range policy for the board's Lepton. None = auto-range (the default,
        # a percentile clip off one early frame); a (tmin, tmax) pair pins the
        # window instead. Pinning exists because auto-range samples ONCE, and
        # the sensor's output drifts for minutes after bring-up: measured here
        # 2026-08-16, a window chosen at start-up had 56.6% of the frame pinned
        # at its floor immediately and 100.0% pinned four minutes later, i.e.
        # the whole scene had fallen out of the bottom of it. A pinned wide
        # window is how you SEE that drift, and how a calibration session gets
        # a window that is still right at the end of it.
        self.fixed_range = None
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

    def _exact(self, n, kind="payload"):
        while len(self.buf) < n:
            for footer in _BATCH_FOOTERS:
                if self.buf.endswith(footer):
                    received = len(self.buf) - len(footer)
                    # Keep the raw-REPL prompt for the normal pending==0 path.
                    # It will consume it and submit the next batch immediately.
                    del self.buf[:received + len(footer) - 3]
                    raise _DroppedBatch(kind, received, n)
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
        # SETUP_CODE recreates the board clock and resets _t_prev to zero.  Its
        # first delta is therefore board uptime, not a thermal service gap.  Do
        # not let rolling values from the previous clock survive and make that
        # first sample look like a multi-hour Lepton starvation event.
        for key in ("dt_window", "skew_window", "torn_window", "starved",
                    "last_starve_ms", "stalls", "last_stall_t",
                    "board_session_frames"):
            self.state.pop(key, None)

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
        self.state["last_resync_t"] = time.time()
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

        So the rate follows the margin. Far from the floor, every 400th batch
        (~3.8min) is plenty against a drain measured in hours. Close to it, the
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
        # TIOCEXCL is the serial equivalent of owning this stream. A second
        # opener toggling DTR makes TinyUSB discard queued 512-byte packets; it
        # must fail to open instead of corrupting a live frame silently.
        self.s = serial.Serial(self.port, 115200, timeout=0.2,
                               write_timeout=10, exclusive=True)
        tmin, tmax = self.fixed_range if self.fixed_range else (-10, 140)
        setup = (SETUP_CODE
                 .replace("__TMIN__", str(tmin)).replace("__TMAX__", str(tmax))
                 .replace("__AUTORANGE__", str(self.fixed_range is None))
                 .replace("__RAW__", str(self.raw16))
                 .replace("__Q__", str(self.quality)))
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
                self.state["board_session_frames"] = 0
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
            # Two legal thermal sizes, and the header is what distinguishes
            # them: one byte per pixel is the board's AGC'd 8-bit plane, two is
            # the 16-bit radiometric one. Nothing else has to be negotiated -
            # the length already says which arrived.
            if not (0 < jlen <= OUT_W * OUT_H
                    and tlen in (TH_W * TH_H, 2 * TH_W * TH_H)):
                self._resync("implausible header %r" % line[:40])
                continue
            self.headers += 1
            self.pending -= 1
            try:
                jpg = self._exact(jlen, "jpeg")
                thermal = self._exact(tlen, "thermal")
            except _DroppedBatch as e:
                self.state["batch_drops"] = self.state.get("batch_drops", 0) + 1
                self.state["last_batch_drop_t"] = time.time()
                self.state["link_recovering_since"] = time.time()
                self.state["link_recovering_reason"] = str(e)
                self._resync("board ended batch during %s" % e)
                print("batch dropped (continuing without sensor restart): %s" % e,
                      file=sys.stderr)
                continue

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
            self.state["board_session_frames"] = (
                self.state.get("board_session_frames", 0) + 1)
            self.state["last_frame_t"] = now
            self.state.pop("link_recovering_since", None)
            self.state.pop("link_recovering_reason", None)

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
                self.state["stalls_total"] = self.state.get("stalls_total", 0) + stalled
                self.state["last_stall_t"] = now
            # frames > 1: the first interval of a session spans whatever came
            # before the viewer attached - board idle, bring-up, a previous
            # viewer's death - measured on the board clock, which survives
            # attach. That time was not this session's draining and one such
            # count would hold the panel red for the whole run (2026-08-23:
            # worst 21737302ms = six idle hours, on a link running at 8.77fps).
            if dt_ms is not None and dt_ms > LEPTON_SAFE_GAP_MS \
                    and self.state.get("board_session_frames", 0) > 1:
                self.state["starved"] = self.state.get("starved", 0) + 1
                self.state["starved_total"] = self.state.get("starved_total", 0) + 1
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


def worst_level(checks):
    return ("fail" if any(c["level"] == "fail" for c in checks) else
            "warn" if any(c["level"] == "warn" for c in checks) else "ok")


def ui_payload(pipe, state, now):
    """Everything the page redraws each second, in one response.

    The page used to poll five endpoints a second - /health, /timing, /stats,
    /detections, /stat. That is five handler threads a second competing with the
    MJPEG writer on a ThreadingHTTPServer for numbers that all come out of the
    same state dict, and it read them at five *different* instants: a health row
    taken before an FFC could sit on screen beside a temperature band taken after
    it, with nothing on the page to say so. One payload is one moment.

    The five endpoints stay. They are the debugging surface - `curl /timing` is
    worth having - and nothing else in the tree consumes them, so there is no
    compatibility argument either way; this is about what the browser does 86400
    times an hour.

    `cfg` is the part that is new rather than merely moved. The old page hardcoded
    its slider positions in the HTML, so `--gain 220` drew a slider sitting at 200
    over a pipeline running at 220, and the first touch of that slider silently
    moved the pipeline to wherever the handle happened to be.
    """
    checks = health(pipe, state, now)
    stats = pipe.frame_stats()
    lo, hi = state.get("range", (0, 0))
    rp = state.get("radar_proj")
    c = pipe.f.cfg
    stream_status = None
    if state.get("restarting_since") is not None:
        stream_status = {
            "kind": "restart",
            "title": "RECONNECTING SENSORS",
            "detail": "%s · %.0fs" % (
                state.get("last_restart", "stream fault"),
                now - state["restarting_since"]),
        }
    elif state.get("link_recovering_since") is not None:
        stream_status = {
            "kind": "drop",
            "title": "RECOVERING LINK",
            "detail": state.get("link_recovering_reason", "incomplete USB batch"),
        }
    elif state.get("last_frame_t") is not None and now - state["last_frame_t"] > 1.0:
        # The MJPEG connection deliberately keeps the last good frame visible.
        # Without this overlay that honest last frame looks like a live one.
        stream_status = {
            "kind": "stale",
            "title": "WAITING FOR A NEW FRAME",
            "detail": "last frame %.1fs ago" % (now - state["last_frame_t"]),
        }
    return {
        "worst": worst_level(checks),
        "checks": checks,
        "stream_status": stream_status,
        "timing": timing(state, now),
        "stats": dict(stats, valid=True) if stats else {"valid": False},
        # Only with --raw16, hence None rather than a default: the page hides
        # the row instead of showing an invented one. The two decimal places it
        # carries are real, which is the entire distinction from "stats" above -
        # that is this same scene after the sensor window has quantised it.
        "raw_stats": state.get("raw_stats"),
        "detections": {
            "warped": pipe.warped,
            "age_s": round(now - state["detect_t"], 2) if state.get("detect_t") else None,
            "ms": state.get("detect_ms"),
            "list": state.get("detections") or [],
            "rejected": state.get("detections_rejected") or [],
        },
        # Quantities that drift rather than jump, and which the page draws as a
        # 60s trace: a heap reading is not interesting, a heap reading that is
        # 0.6MB lower than a minute ago is the whole early-warning system.
        # The locks. Separate from `detections`: a detection is what a sensor
        # said about one frame, a track is what the viewer believes about a
        # person across frames, and conflating them is how a coasted box ends
        # up quoted as a measurement.
        "tracks": state.get("tracks") or [],
        "tracks_static": state.get("tracks_static") or [],
        # What the rig did between frames, and whether it could be read at all.
        # The lock silently assumes a still rig whenever this is refused, and
        # an assumption nobody can see is one nobody thinks to doubt.
        "ego": state.get("ego"),
        # Whether the visible camera had light. Beside the tracks rather than
        # buried in health, because it is the number that says how to read
        # every other channel on this frame.
        "light": state.get("light"),
        "heap_free": state.get("heap_free"),
        "restarts": state.get("restarts", 0),
        "batch_drops": state.get("batch_drops", 0),
        "coverage": round(float(pipe.cover_grid().mean()), 4) if pipe.f.have_frame else None,
        "recording": os.path.basename(state["recording"]) if state.get("recording") else None,
        # Which station the frames going to disk right now are labelled with.
        # The operator is at the target, metres from the host, holding a phone;
        # this is the only way to see that the stamp took before walking back.
        # `null` while recording means frames are being written with no pose_id,
        # which is the state a calibration session must never sit in unnoticed.
        "pose": None if state.get("video") is None else {
            "id": state["video"].pose_id,
            "stamped": len(state["video"].meta_poses()),
            "frames": state["video"].frames,
        },
        # Cached inside Soc for 0.4s, so health() above and this share one sample
        # rather than each taking a delta over a near-zero interval.
        "soc": state["soc"].read() if state.get("soc") else None,
        # Yael's map layer (perception.map_api contract). None without --map;
        # the page hides the card. Small and already JSON-safe, so sent whole.
        "map": state.get("map"),
        "map_fix": state.get("map_fix"),
        "map_prior": state.get("map_prior"),
        "nav": nav_horizon(state) if state.get("map_prior") is not None else None,
        "cfg": {
            "gain": c.detail_gain, "eps": c.gf_eps, "radius": c.gf_radius,
            "agc": c.agc_permille, "palette": pipe.palette_name,
            "channel": pipe.channel, "channels": live_channels.public_specs(),
            "view": pipe.view,
            "mix": pipe.mix, "outline": pipe.outline, "boxes": pipe.show_detections,
            "lock": pipe.ai_lock,
            "emissivity": round(pipe.eps, 3), "reflected": pipe.refl,
            "warped": pipe.warped, "range": [lo, hi],
            # None, not a zeroed dict: "no radar attached" and "radar attached
            # and pointing straight ahead" are different states, and the page
            # hides the whole card on the first rather than offering knobs that
            # move nothing.
            "radar": None if rp is None else {
                "on": pipe.show_radar, "whisker": pipe.radar_whisker,
                "yaw": round(rp.yaw, 2), "pitch": round(rp.pitch, 2),
                "roll": round(rp.roll, 2),
                "tx": round(rp.t[0] * 1000), "ty": round(rp.t[1] * 1000),
                "tz": round(rp.t[2] * 1000)},
            # None when --students was not given or its engines did not come
            # up: same argument as the radar block above, and the page hides
            # the card rather than offering switches that move nothing.
            # `ready` is separate from `on` because a channel can be switched on
            # and still be unable to draw - a thermal box with no warp LUT has
            # nowhere to go, and fusion needs both channels to exist at all.
            "ai": None if state.get("students") is None else {
                # `found` is what the engine produced this tick and `n` is what
                # survived the confidence floor and the dedup. Both, always: a
                # card that only showed `n` cannot tell a quiet scene from a
                # slider parked too high.
                "floor": round(pipe.ai_conf_floor, 2),
                "evidence": pipe.ai_evidence,
                "silhouette": pipe.ai_silhouette,
                "thermal": {"on": pipe.ai_thermal,
                            "n": len(state.get("student_thermal") or []),
                            "found": (state.get("student_seen") or {}).get("thermal", 0),
                            "conf": round(pipe.ai_conf_thermal, 2),
                            "ready": state["students"].get("th2vis") is not None},
                "radar": {"on": pipe.ai_radar,
                          "n": len(state.get("student_radar") or []),
                          "found": (state.get("student_seen") or {}).get("radar", 0),
                          "conf": round(pipe.ai_conf_radar, 2),
                          "ready": state["students"].get("radar") is not None},
                "fusion": {"on": pipe.ai_fusion,
                           "n": len(state.get("student_fused") or []),
                           "rejected": len(state.get("student_fused_rejected") or []),
                           "ready": (state["students"].get("th2vis") is not None
                                     and state["students"].get("radar") is not None)},
                "age_s": (round(now - state["student_t"], 2)
                          if state.get("student_t") else None),
                "error": state.get("student_error"),
            },
        },
    }


PAGE = b"""<!doctype html><meta charset=utf-8><title>thermal fusion - live</title>
<style>
/* Two colour systems, deliberately disjoint. THERMAL/VISIBLE/RADAR name the
   sensors and appear on data; OK/WARN/FAIL name trust and appear on status. The
   old page used amber for both, so "this line is about the thermal camera" and
   "this line is a warning" were the same colour. */
:root{
  --bg:#0c0e10;--panel:#141719;--panel2:#191d20;--line:#242a2f;
  --txt:#d8dee4;--dim:#79838d;--dimmer:#525b64;
  --thermal:#f0a860;--visible:#6cc8ff;--radar:#b98cff;--host:#8fa2b5;
  --ok:#7fd39b;--ok-bd:#2c5a3e;
  --warn:#f0c060;--warn-bd:#6d5220;--warn-bg:#211c11;
  --fail:#ff7d7d;--fail-bd:#7a2f2f;--fail-bg:#241414;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html,body{height:100%}
/* tabular-nums so a figure that changes every second does not shift the ones
   beside it - a number that jitters horizontally is a number nobody reads */
body{margin:0;background:var(--bg);color:var(--txt);font:13px/1.45 system-ui,-apple-system,
Segoe UI,sans-serif;font-variant-numeric:tabular-nums;display:flex;flex-direction:column}
b{font-weight:600}
.mono{font-family:var(--mono)}

/* --- trust bar: the only line you need before writing a number down */
#trust{display:flex;align-items:center;gap:18px;flex-wrap:wrap;padding:9px 16px;
border-bottom:1px solid var(--line);background:var(--panel)}
#trust.ok{box-shadow:inset 3px 0 0 var(--ok)}
#trust.warn{box-shadow:inset 3px 0 0 var(--warn);
background:linear-gradient(90deg,#1d1a12,var(--panel) 320px)}
#trust.fail{box-shadow:inset 3px 0 0 var(--fail);
background:linear-gradient(90deg,#231414,var(--panel) 320px)}
#verdict{display:flex;align-items:center;gap:9px;font-weight:650;letter-spacing:.06em;font-size:12px}
#verdict .dot{width:9px;height:9px;border-radius:50%}
#verdict small{font-weight:400;letter-spacing:0;color:var(--dim);font-size:11px}
.ok #verdict{color:var(--ok)}.ok #verdict .dot{background:var(--ok);box-shadow:0 0 8px #7fd39b60}
.warn #verdict{color:var(--warn)}.warn #verdict .dot{background:var(--warn);box-shadow:0 0 8px #f0c06060}
.fail #verdict{color:var(--fail)}.fail #verdict .dot{background:var(--fail);
box-shadow:0 0 10px #ff7d7d70;animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{50%{opacity:.45}}
#qual{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;font-family:var(--mono)}
#qual span{color:var(--dim)}
#qual span i{font-style:normal;color:var(--txt)}
#qual span.bad i{color:var(--warn)}
/* The pose bar is sized for the one hand that actually uses it: the operator is
   standing at the reflector holding a phone, not sitting at the host. Hence the
   44px touch targets and a current-station readout big enough to confirm at
   arm's length without zooming. */
#posebar{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
#posebar input{flex:1 1 140px;min-width:0;height:44px;padding:0 12px;
  font-family:var(--mono);font-size:16px;color:var(--txt);
  background:var(--panel2);border:1px solid var(--line);border-radius:6px}
#posebar button{height:44px;padding:0 18px;font-size:15px;font-family:inherit;
  color:var(--bg);background:var(--visible);border:0;border-radius:6px}
#posebar button.clear{background:var(--panel2);color:var(--dim);
  border:1px solid var(--line)}
#posenow{font-family:var(--mono);font-size:20px;font-weight:600;
  min-width:5ch;color:var(--ok)}
#posenow.none{color:var(--warn)}
#posen{font-size:12px;color:var(--dim);font-family:var(--mono)}
#brand{margin-left:auto;color:var(--dimmer);font-size:11px;letter-spacing:.08em}

/* --- main split */
#main{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:14px;padding:14px;
align-items:start}
body.field #main{grid-template-columns:minmax(0,1fr)}
body.field aside,body.field #notes{display:none}
#stage{display:flex;flex-direction:column;gap:12px;min-width:0}
/* Sized so the health panel stays above the fold on a laptop: the picture is
   what you look at, but the panel is what says whether to believe it, and a
   viewer that hides the panel below a scroll is back to being a webcam. 88vh
   is 55vh of height at this 1.6 aspect. */
#wrap{position:relative;line-height:0;align-self:start;border:1px solid var(--line);
border-radius:8px;overflow:hidden;background:#000;width:min(100%,1024px,88vh)}
#streamstate{position:absolute;inset:0;z-index:4;display:flex;flex-direction:column;
align-items:center;justify-content:center;gap:8px;background:#07090bba;line-height:1.35;
text-align:center;pointer-events:none;backdrop-filter:blur(2px)}
#streamstate[hidden]{display:none}
#streamtitle{font:650 15px/1.2 var(--mono);letter-spacing:.08em;color:var(--warn)}
#streamdetail{max-width:80%;font:12px/1.4 var(--mono);color:var(--txt)}
/* The picture and the map, side by side. The picture says what is out there;
   the map says where the rig was standing while it said it, and reading one
   against the other is the whole point of the map layer - it cannot be done
   with the map in the sidebar under four scrolls of numbers. The map keeps a
   fixed column rather than matching the picture's height: square at 55vh it
   would be wider than the picture it sits beside. It wraps under on a narrow
   window, where a 250px map is worse than no map. */
#viewrow{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;min-width:0}
#wrap{flex:0 1 auto}
#mapside{display:none;flex:0 1 320px;min-width:250px;flex-direction:column;gap:8px;
background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}
#mapside.on{display:flex}
#mapside h5{margin:0;font-size:10.5px;font-weight:650;letter-spacing:.11em;
color:var(--dim);text-transform:uppercase}
#mapside canvas{width:100%;aspect-ratio:1;background:#0a0c0e;
border:1px solid var(--line);border-radius:6px}
#mapside.nowalls .hintline{opacity:.4}
img{display:block;width:100%;height:auto}
#ovl{position:absolute;inset:0;pointer-events:none}
#read{position:absolute;padding:4px 8px;background:#000000d9;border:1px solid #5a636b;
border-radius:5px;font:12px/1.3 var(--mono);pointer-events:none;white-space:nowrap;
display:none;transform:translate(14px,14px)}
#read.warn{border-color:var(--warn-bd);color:var(--warn)}

/* the delta between two pinned points is what a finding rests on, so it gets a
   card rather than being something you compute in your head off two hovers */
#probes{position:absolute;left:10px;bottom:10px;display:flex;background:#000000cc;
border:1px solid var(--line);border-radius:6px;font-family:var(--mono);overflow:hidden}
#probes .p{padding:6px 11px;border-right:1px solid var(--line);line-height:1.25}
#probes em{font-style:normal;display:block;font-size:10px;color:var(--dim);letter-spacing:.08em}
#probes b{font-size:15px;font-weight:600}
#probes .d{padding:6px 13px;background:#ffffff08}
#probes .d b{font-size:17px;color:var(--thermal)}
#probes .hint{padding:6px 11px;color:var(--dim);font-size:11px;align-self:center}
#probes .flag{color:var(--warn);font-size:10px}

/* --- sidebar */
aside{display:flex;flex-direction:column;gap:10px;overflow:auto;min-height:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px}
.card>h4{margin:0;padding:8px 12px;font-size:10.5px;font-weight:650;letter-spacing:.11em;
color:var(--dim);text-transform:uppercase;border-bottom:1px solid var(--line);
display:flex;align-items:center;gap:8px}
.card>h4 .k{margin-left:auto;color:var(--dimmer);font-family:var(--mono);font-size:10px;
border:1px solid var(--line);border-radius:3px;padding:0 4px;text-transform:none;letter-spacing:0}
.card .body{padding:10px 12px;display:flex;flex-direction:column;gap:9px}
.seg{display:flex;background:var(--panel2);border:1px solid var(--line);border-radius:6px;
padding:2px;gap:2px}
.seg button{flex:1;background:none;border:0;color:var(--dim);font:inherit;font-size:11.5px;
padding:5px 2px;border-radius:4px;cursor:pointer;transition:background .12s,color .12s}
.seg button:hover{color:var(--txt);background:#ffffff0a}
.seg button.on{background:#2b3238;color:var(--txt);box-shadow:inset 0 0 0 1px #3a444c}
#channel{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}
#channel button{min-height:30px;line-height:1.2}
#channel button:last-child{grid-column:1/-1}
.seg.pal button.on{color:#0c0e10;font-weight:600}
.seg.pal button[data-v=ironbow].on{background:linear-gradient(90deg,#5a1a70,#e05020,#ffd23c)}
.seg.pal button[data-v=white].on{background:linear-gradient(90deg,#333,#fff)}
.seg.pal button[data-v=black].on{background:linear-gradient(90deg,#fff,#333);color:#fff}
.seg.pal button[data-v=gray].on{background:linear-gradient(90deg,#222,#bbb)}
.sl{display:grid;grid-template-columns:1fr auto;gap:2px 8px;align-items:center}
.sl label{font-size:11.5px;color:var(--dim)}
.sl output{font-family:var(--mono);font-size:11.5px}
.sl input[type=range]{grid-column:1/-1;width:100%;height:16px;-webkit-appearance:none;
background:none;margin:0}
input[type=range]::-webkit-slider-runnable-track{height:3px;background:#2c3338;border-radius:2px}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;
margin-top:-4.5px;border-radius:50%;background:var(--txt);border:0;cursor:pointer}
input[type=range]:hover::-webkit-slider-thumb{background:var(--visible)}
input[type=range]::-moz-range-track{height:3px;background:#2c3338;border-radius:2px}
input[type=range]::-moz-range-thumb{width:12px;height:12px;border:0;border-radius:50%;
background:var(--txt)}
.chk{display:flex;align-items:center;gap:8px;font-size:12px;cursor:pointer}
.chk input{accent-color:#4d7f9e;width:14px;height:14px}
.chk .sub{color:var(--dim);font-size:11px;margin-left:auto;font-family:var(--mono)}
.num{display:flex;align-items:center;gap:8px;font-size:11.5px;color:var(--dim)}
.num input{width:66px;background:var(--panel2);border:1px solid var(--line);border-radius:4px;
color:var(--txt);font:12px var(--mono);padding:3px 6px}
.num input:focus{outline:none;border-color:#3f5a6b}
.hintline{font-size:11px;color:var(--dimmer);line-height:1.45}
.nudge{display:grid;grid-template-columns:auto 1fr auto auto;gap:4px 7px;align-items:center;
font-size:11.5px}
.nudge label{color:var(--dim);font-family:var(--mono)}
.nudge output{font-family:var(--mono);text-align:right}
.nudge button{background:var(--panel2);border:1px solid var(--line);color:var(--dim);
width:22px;height:20px;border-radius:4px;cursor:pointer;font:12px var(--mono);line-height:1}
.nudge button:hover{color:var(--txt);border-color:#3c454c}

/* --- clock strip */
#clock,#socstrip{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:10px 14px;display:flex;align-items:flex-start;gap:14px 20px;flex-wrap:wrap}
.tag{font:10px/1.3 var(--mono);letter-spacing:.1em;color:var(--dimmer);border:1px solid var(--line);
border-radius:4px;padding:3px 6px;align-self:center;white-space:pre}
.btn{background:var(--panel2);color:inherit;border:1px solid var(--line);border-radius:6px;padding:4px 10px;font:inherit;cursor:pointer}
.btn:disabled{opacity:.5;cursor:wait}
.kvrow{display:grid;grid-template-columns:1fr 1fr;gap:6px 10px;margin-bottom:6px}
.kv{display:flex;flex-direction:column;gap:1px;font-family:var(--mono)}
.kv em{font-style:normal;font-size:10px;letter-spacing:.07em;color:var(--dim);text-transform:uppercase}
.kv b{font-size:14px;font-weight:600}
.kv .u{font-size:11px;color:var(--dim);font-weight:400}
.kv.th b{color:var(--thermal)}.kv.vis b{color:var(--visible)}.kv.host b{color:var(--host)}
.kv.bad b{color:var(--warn)}
.spark{display:block;margin-top:1px}

/* --- health, four groups */
#health{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:10px;
padding:0 14px 14px;align-items:start}
.grp{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.grp>summary{list-style:none;cursor:pointer;padding:9px 12px;display:flex;align-items:center;
gap:9px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:650}
.grp>summary::-webkit-details-marker{display:none}
.grp>summary .dot{width:8px;height:8px;border-radius:50%;flex:0 0 auto}
.grp>summary .cnt{margin-left:auto;font-family:var(--mono);font-size:11px;letter-spacing:0;
text-transform:none;color:var(--dimmer)}
.grp>summary .chev{color:var(--dimmer);transition:transform .15s}
.grp[open]>summary .chev{transform:rotate(90deg)}
.grp.ok>summary .dot{background:var(--ok)}
.grp.warn{border-color:var(--warn-bd)}
.grp.warn>summary{color:var(--warn)}.grp.warn>summary .dot{background:var(--warn)}
.grp.fail{border-color:var(--fail-bd);background:var(--fail-bg)}
.grp.fail>summary{color:var(--fail)}.grp.fail>summary .dot{background:var(--fail)}
.checks{padding:2px 10px 10px;display:flex;flex-direction:column}
.chk-row{display:grid;grid-template-columns:8px 94px 1fr;gap:8px;align-items:baseline;
padding:5px 2px;border-top:1px solid #ffffff08;font-size:11.5px}
.chk-row i{width:6px;height:6px;border-radius:50%;align-self:center;font-style:normal}
.chk-row .n{font-weight:600;font-size:11px}
.chk-row .t{color:var(--dim);font-family:var(--mono);font-size:11px;line-height:1.4}
.chk-row.ok i{background:var(--ok-bd)}.chk-row.ok .n{color:#9db4a6}
.chk-row.warn i{background:var(--warn)}.chk-row.warn .n{color:var(--warn)}.chk-row.warn .t{color:#cbb184}
.chk-row.fail i{background:var(--fail)}.chk-row.fail .n{color:var(--fail)}.chk-row.fail .t{color:#e5a8a8}

/* --- detections */
.detrow{display:flex;gap:8px;flex-wrap:wrap;min-height:0}
.det{display:flex;align-items:center;gap:9px;border:1px solid var(--line);background:var(--panel);
border-radius:6px;padding:4px 10px;font-size:11.5px}
.det.unreg{border-color:var(--warn-bd);background:var(--warn-bg)}
.det .cls{font-weight:650}
.det .m{font-family:var(--mono);color:var(--dim)}
.det .c{font-family:var(--mono);color:var(--thermal);font-weight:600}
.det .flag{font-family:var(--mono);font-size:10px;color:var(--warn);letter-spacing:.05em}

/* --- notes */
#notes{padding:0 14px 20px;max-width:78em;color:#8e8579;font-size:12px;line-height:1.6}
#notes details{border-top:1px solid var(--line);padding:9px 0}
#notes summary{cursor:pointer;color:var(--dim);font-size:11.5px}
#notes summary:hover{color:var(--txt)}
#notes p{margin:8px 0 0}
#notes b{color:#b6ab99}
kbd{font:11px var(--mono);border:1px solid var(--line);border-bottom-width:2px;border-radius:3px;
padding:1px 5px;color:var(--dim);background:var(--panel2)}
@media(max-width:1100px){#health{grid-template-columns:repeat(2,1fr)}
#main{grid-template-columns:minmax(0,1fr)}}
</style>

<div id=trust>
  <div id=verdict><span class=dot></span><span id=vtext>connecting</span>
    <small id=vsub></small></div>
  <div id=qual></div>
  <div id=posebar hidden>
    <span id=posenow class=none>&mdash;</span>
    <input id=poseid placeholder="station, e.g. N04" autocomplete=off
           autocapitalize=characters spellcheck=false>
    <button id=posego>stamp</button>
    <button id=poseclr class=clear>clear</button>
    <span id=posen></span>
  </div>
  <div id=brand>THERMAL + VISIBLE FUSION &middot; LIVE</div>
</div>

<div id=main>
  <div id=stage>
    <div id=viewrow>
      <div id=wrap>
        <img id=im src="/stream">
        <svg id=ovl viewBox="0 0 640 400" preserveAspectRatio=none></svg>
        <div id=streamstate hidden><div id=streamtitle></div><div id=streamdetail></div></div>
        <div id=read></div>
        <div id=probes></div>
      </div>

      <!-- Shown only when live.py was started with --map: without the layer the
           canvas has nothing to draw and an empty square beside the picture
           reads as a broken map rather than an absent one. -->
      <div id=mapside>
        <h5>map &middot; top-down</h5>
        <canvas id=topdown width=420 height=420></canvas>
        <div class=hintline>north up, 60 m: <b style=color:#9aa>map walls</b>,
          <b style=color:var(--warn)>returns at the prior</b>,
          <b style=color:#5cf>returns at the fix</b>. If the amber dots are not on
          grey lines, the prior is wrong; if the blue ones are not, the fix is.</div>
      </div>
    </div>
    <div class=detrow id=dets></div>
    <div id=clock>
      <div class=tag>BOARD
&amp; LINK</div>
      <svg id=clocksvg width=420 height=46></svg>
      <div class="kv th"><em>thermal</em><b id=k_th>&ndash;</b>
        <svg class=spark id=sp_th width=90 height=16></svg></div>
      <div class="kv vis" id=kv_sk><em>visible skew</em><b id=k_sk>&ndash;</b>
        <svg class=spark id=sp_sk width=90 height=16></svg></div>
      <div class=kv><em>link</em><b id=k_fps>&ndash;</b>
        <svg class=spark id=sp_fps width=90 height=16></svg></div>
      <div class=kv><em>board heap</em><b id=k_heap>&ndash;</b>
        <svg class=spark id=sp_heap width=90 height=16></svg></div>
      <div class=kv><em>coverage</em><b id=k_cov>&ndash;</b>
        <svg class=spark id=sp_cov width=90 height=16></svg></div>
      <div class=kv><em>last ffc</em><b id=k_ffc>&ndash;</b></div>
      <!-- Scene light. On the BOARD strip beside coverage rather than in the
           sidebar: it is a property of the frame that just arrived, and it is
           read together with what the detector said about that frame. -->
      <div class=kv id=kv_light><em>scene light</em><b id=k_light>&ndash;</b></div>
      <div class=kv id=kv_load><em>sensor load</em><b id=k_load>&ndash;</b></div>
      <div class=kv><em>frame band</em><b id=k_band>&ndash;</b></div>
      <!-- --raw16 only. Deliberately next to the frame band, because reading
           the two together is what shows the difference: same scene, one of
           them through the 8-bit window. -->
      <div class=kv id=kv_raw style=display:none><em>scene 16-bit</em>
        <b id=k_raw>&ndash;</b></div>
    </div>

    <!-- The machine the viewer runs on. A separate strip rather than more items
         on the clock: those are the board's numbers and these are this host's,
         and one row of figures that mixes the two is a row nobody can read. -->
    <div id=socstrip style=display:none>
      <div class=tag>HOST
SOC</div>
      <div class="kv host" id=kv_cpu><em>cpu</em><b id=k_cpu>&ndash;</b>
        <svg class=spark id=sp_cpu width=90 height=16></svg></div>
      <div class="kv host"><em>gpu</em><b id=k_gpu>&ndash;</b>
        <svg class=spark id=sp_gpu width=90 height=16></svg></div>
      <div class="kv host" id=kv_mem><em>memory</em><b id=k_mem>&ndash;</b>
        <svg class=spark id=sp_mem width=90 height=16></svg></div>
      <div class="kv host" id=kv_tj><em>soc temp</em><b id=k_tj>&ndash;</b>
        <svg class=spark id=sp_tj width=90 height=16></svg></div>
      <div class="kv host"><em>power</em><b id=k_pw>&ndash;</b>
        <svg class=spark id=sp_pw width=90 height=16></svg></div>
      <div class="kv host"><em>cpu clock</em><b id=k_clk>&ndash;</b></div>
    </div>
  </div>

  <aside>
    <div class=card>
      <h4>channel <span class=k>1-5</span></h4>
      <div class=body>
        <div class=seg id=channel></div>
        <div class=hintline>display channels choose the sensor product. The
          thermal/radar/fusion switches in the AI card are evidence layers,
          not additional display modes.</div>
        <label class=chk><input type=checkbox id=outline> thermal footprint
          <span class=sub id=s_cov></span></label>
        <label class=chk><input type=checkbox id=boxes checked> detection boxes
          <span class=sub id=s_det></span></label>
      </div>
    </div>

    <div class=card id=fusionviewcard>
      <h4>fusion / calibration view</h4>
      <div class=body>
        <div class=seg id=view></div>
        <div class=sl><label>thermal weight &mdash; mix/operator</label><output id=o_mix>60</output>
          <input type=range id=mix min=0 max=100 value=60></div>
      </div>
    </div>

    <div class=card>
      <h4>image</h4>
      <div class=body>
        <div class=sl><label>detail gain</label><output id=o_gain>200</output>
          <input type=range id=gain min=0 max=512 value=200></div>
        <div class=sl><label>guided-filter eps</label><output id=o_eps>200</output>
          <input type=range id=eps min=1 max=1000 value=200></div>
        <div class=sl><label>guided-filter radius</label><output id=o_radius>4</output>
          <input type=range id=radius min=1 max=8 value=4></div>
        <div class=sl><label>scene agc &permil; <span id=agcwarn></span></label>
          <output id=o_agc>0</output>
          <input type=range id=agc min=0 max=100 value=0></div>
      </div>
    </div>

    <div class=card>
      <h4>radiometry</h4>
      <div class=body>
        <div class="seg pal" id=pal></div>
        <div class=num><span>emissivity &epsilon;</span>
          <input type=number id=emis value=1 step=0.01 min=0.05 max=1></div>
        <div class=num><span>reflected &deg;C</span>
          <input type=number id=refl value=20 step=1></div>
        <div class=hintline>bright metal is &asymp;0.10 and reads tens of degrees cold
          at &epsilon;=1</div>
      </div>
    </div>

    <div class=card id=radarcard style=display:none>
      <h4>radar <span class=k>iwr1843</span></h4>
      <div class=body>
        <label class=chk><input type=checkbox id=radar checked> overlay
          <span class=sub id=s_radar></span></label>
        <label class=chk><input type=checkbox id=whisker checked> elevation whisker
          <span class=sub>2-element</span></label>
        <div class=nudge id=nudge></div>
        <div class=hintline>the whisker is the honest width of the elevation
          uncertainty &mdash; the bare dot flatters a two-element aperture</div>
      </div>
    </div>

    <div class=card id=mapcard style=display:none>
      <h4>map <span class=k>yael / mapinit</span></h4>
      <div class=body>
        <div class=kvrow>
          <div class=kv><em>geoid</em><b id=m_geoid>&ndash;</b></div>
          <div class=kv><em>ground</em><b id=m_ground>&ndash;</b></div>
          <div class=kv><em>surface</em><b id=m_surface>&ndash;</b></div>
          <div class=kv><em>ego alt &sigma;</em><b id=m_sigma>&ndash;</b></div>
        </div>
        <div class=hintline id=m_stages></div>
        <div class=hintline id=m_text></div>
        <div class=kvrow style="margin-top:8px">
          <div class=kv><em>fix</em><b id=m_fix>&ndash;</b></div>
          <div class=kv><em>heading</em><b id=m_head>&ndash;</b></div>
          <div class=kv><em>nav horizon</em><b id=m_nav>&ndash;</b></div>
          <div class=kv><em>ego speed</em><b id=m_speed>&ndash;</b></div>
        </div>
        <div class=hintline id=m_fixtext></div>
        <div class=hintline id=m_navtext></div>
        <div class=sl><label>heading prior</label><output id=o_heading>&ndash;</output>
          <input type=range id=heading min=0 max=359 value=0></div>
        <div style="display:flex;gap:8px;align-items:center;margin:6px 0">
          <button id=btn_fix class=btn>fix from walls</button>
          <span class=sub id=m_fixage></span>
        </div>
        <div class=hintline>orthometric heights against EGM2008 for the
          <b>--map</b> position. <i>calibration</i> is skipped without
          constraints; <i>pose_init</i> at startup is skipped because the
          radar has not reported yet &mdash; <b>fix from walls</b> runs
          it on the live returns, and the top-down beside the picture is
          where you check the answer. Full JSON on
          <a href=/map target=_blank>/map</a>.</div>
      </div>
    </div>

    <div class=card id=aicard>
      <h4>ai <span class=k>e t r c s l</span></h4>
      <div class=body>
        <label class=chk><input type=checkbox id=lock checked> lock on people
          <span class=sub id=s_lock></span></label>
        <label class=chk><input type=checkbox id=evidence> student evidence
          <span class=sub>debug overlays</span></label>
        <label class=chk><input type=checkbox id=silhouette>
          person outline <span class=sub id=s_ai_sil></span></label>
        <div id=aistudents style=display:none>
        <label class=chk><input type=checkbox id=ai_thermal checked> thermal
          <span class=sub id=s_ai_t></span></label>
        <label class=chk><input type=checkbox id=ai_radar checked> radar
          <span class=sub id=s_ai_r></span></label>
        <label class=chk><input type=checkbox id=ai_fusion checked> fusion
          <span class=sub id=s_ai_f></span></label>
        <div class=sl><label>thermal confidence</label><output id=o_conf_thermal>0.50</output>
          <input type=range id=conf_thermal min=50 max=99 value=50></div>
        <div class=sl><label>radar confidence</label><output id=o_conf_radar>0.50</output>
          <input type=range id=conf_radar min=50 max=99 value=50></div>
        <div class=hintline><b>student evidence</b> reveals the intermediate
          T/R/TR boxes and rejected static-candidate ticks for model debugging.
          It is hidden by default; inference, fusion and person lock keep running. The sliders hide candidates, they
          do not stop the engines: the count beside each channel reads
          <i>shown of found</i>, so a quiet scene and a slider parked too high
          never look the same. They cannot go below the threshold the engines
          were started with (&minus;&minus;student-conf).</div>
        <div class=hintline><b>lock</b> keeps a person between frames instead
          of deciding again every frame: seen twice, they get an id (P1) and
          are held through the frames no sensor found them in, for up to 1.5 s,
          carried on their own velocity. A held box is amber and dashed and
          says <i>coast 0.4s</i> &mdash; it is where somebody probably is, not
          where anybody saw them. The letters after the id are the sensors
          confirming it right now: D detector, T thermal, R radar, F fusion.
          A channel switched off still feeds the lock &mdash; hiding a channel
          hides its boxes, it does not make the viewer forget what it saw.</div>
        <div class=hintline>a candidate the <b>detector</b> has never seen has
          to MOVE before it is drawn as a person. In the lobby this rig sits in,
          a lit glass door reads 31.7&deg;C and a person reads 30.9&deg;C
          &mdash; the same temperature, so nothing radiometric can tell them
          apart, and a door has never moved. Those candidates are counted as
          <i>static</i> beside the lock rather than dropped silently: something
          warm and motionless is not nobody.</div>
        <div class=hintline>with <b>person outline</b> on, the thermal channel
          draws the warm shape it found instead of a rectangle, and fusion
          draws a ring around it &mdash; the 4 px staircase is the warp LUT's
          real resolution, not a rendering artefact. Where the shape cannot be
          found (no thermal coverage, nothing warm in the box) it falls back
          to a rectangle: a full rectangle above 0.75, four corner
          ticks below. orange = thermal student, cyan = radar
          student, white = the same person in both. The pair is made on the horizontal
          only (&plusmn;50 px): radar elevation comes from two elements and
          says almost nothing about which box a return belongs to. A fused box
          carries the range, which is the one thing the picture cannot show,
          and its components keep their colour but drop their own labels.</div>
        <div class=hintline>fusion keeps running the channel it needs even when
          that channel's own boxes are switched off &mdash; switching one off
          hides it, agreement still needs both.</div>
        </div>
      </div>
    </div>
  </aside>
</div>

<div id=health></div>

<div id=notes>
  <details open><summary>How to read this page</summary>
  <p><b>The trust bar</b> is the only thing to check before writing a number down.
  <span style="color:#f0c060">Amber</span> means the readings are usable but qualified
  &mdash; the qualification changes how to read them, it is not a nag to clear.
  <span style="color:#ff7d7d">Red</span> means do not record anything. All green is not a
  claim that the measurement is accurate; it is a claim that none of the failures this code
  can see are happening.</p></details>

  <details><summary>Hover readings, pinned probes, and the delta</summary>
  <p>Hover for a reading. The number comes from the thermal frame behind that pixel &mdash;
  before the AGC and before the guided filter, which borrows the visible camera's edges to
  sharpen the picture and must not be quoted off. Absolute accuracy is &plusmn;5&deg;C at
  best. The board runs the Lepton in <b>high gain</b> since 2026-08-09; before that it was
  silently in low gain, whose specified accuracy is the greater of &plusmn;10&deg;C or 10%,
  so any reading quoted off a frame recorded earlier than that carries the looser number.
  The <b>&Delta;</b> between two pinned points of the same material in one frame is far
  better than either, and is what a finding should rest on &mdash; provided neither point is
  clipped, neither sits on a rebuilt row, and no shutter event separates them. Click the
  image to pin a probe; readings keep working in every view.</p></details>

  <details><summary>Judging registration</summary>
  <p>The fused view cannot tell you whether the warp is right: the guided filter puts crisp
  edges in the right places even when the thermal layer is offset, so a misregistered frame
  still looks sharp &mdash; it just colours the wrong side of the edge. Use <b>blink</b> (the
  eye catches motion far better than offset), <b>mix</b> to judge how far off it is, or
  <b>edges</b>, which draws the thermal layer's own edges over the plain visible image: where
  the warp is right they land on the object's outline. <b>thermal footprint</b> outlines where
  the thermal camera stops seeing at all &mdash; outside it the grey is not a cold reading, it
  is no reading.</p></details>

  <details><summary>Operator view</summary>
  <p><b>operator</b> keeps visible luminance and uses the registered thermal layer
  mainly for colour. Its weight adapts to local visible contrast and the footprint
  fades into plain grey at the edge. It is easier to watch than <b>fused</b>, but its
  rendered colour is not a temperature scale; use the probe and detection labels
  for measurements.</p></details>

  <details><summary>The clock</summary>
  <p>One thermal period drawn to scale, with the moment the visible frame was actually grabbed
  marked inside it. Both timestamps come off the board's own clock; arrival times on this host
  have been through a 4KB-chunked CDC write, a 500ms stall retry and the host's scheduler, and
  measure the link rather than the sensors. The grey band at the right is the spread of the
  thermal interval, which on this part is a three-valued delta function (113/114/115 ms) &mdash;
  so a wide band there is a real anomaly, not ordinary jitter.</p></details>

  <details><summary>Keys</summary>
  <p><kbd>1</kbd>-<kbd>5</kbd> channel &middot; <kbd>f</kbd> field mode (hide everything but the
  stream and the health) &middot; <kbd>o</kbd> footprint &middot; <kbd>b</kbd> boxes &middot;
  <kbd>t</kbd> thermal ai &middot; <kbd>r</kbd> radar ai &middot; <kbd>c</kbd> fusion &middot;
  <kbd>s</kbd> person outline &middot; <kbd>l</kbd> lock &middot;
  <kbd>x</kbd> clear probes</p></details>
</div>

<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');
const set = (q) => fetch('/set?' + q);
$('heading').oninput = (e) => { $('o_heading').textContent = e.target.value + '\u00b0'; };
$('heading').onchange = (e) => set('heading=' + e.target.value);
$('btn_fix').onclick = async () => {
  $('btn_fix').disabled = true; $('m_fixage').textContent = 'solving\u2026';
  try { await fetch('/mapfix'); } catch (e) {}
  $('btn_fix').disabled = false; drawTopdown();
};

// The pose stamp. Deliberately NOT optimistic: the readout only changes after
// the server answers, because the whole point of the control is to tell an
// operator standing at the target whether the frames now going to disk carry
// the id. A bar that showed 'N04' on click would show it on a 409 too.
async function stamp(id) {
  const r = await fetch('/pose?' + (id === null ? 'clear=1' : 'id=' + encodeURIComponent(id)));
  const d = await r.json();
  if (!r.ok) { $('posenow').textContent = d.error || 'failed'; return; }
  $('poseid').value = '';
  $('poseid').blur();
  applyPose({id: d.pose, stamped: d.stamped, frames: d.first_frame});
}
function applyPose(p) {
  $('posebar').hidden = !p;
  if (!p) return;
  $('posenow').textContent = p.id || '\\u2014';
  $('posenow').className = p.id ? '' : 'none';
  $('posen').textContent = p.stamped + ' stamped \\u00b7 frame ' + p.frames;
}

// ------------------------------------------------------------------ controls
// Wired once, then initialised from /ui's cfg rather than from the values in the
// HTML: the page is not the source of truth for what the pipeline is doing, and
// --gain 220 used to draw a slider parked at 200 over a pipeline running at 220.
const SLIDERS = ['gain','eps','radius','agc','mix'];
for (const k of SLIDERS) {
  const el = $(k);
  el.oninput = () => { $('o_' + k).textContent = el.value; set(k + '=' + el.value); };
}
for (const k of ['emis','refl'])
  $(k).onchange = () => set('emissivity=' + $('emis').value + '&reflected=' + $('refl').value);
for (const [id, param] of [['outline','outline'],['boxes','boxes'],
                           ['radar','radar'],['whisker','whisker'],
                           ['ai_thermal','ai_thermal'],['ai_radar','ai_radar'],
                           ['ai_fusion','ai_fusion'],['silhouette','silhouette'],
                           ['evidence','evidence'],['lock','lock']])
  $(id).onchange = (e) => set(param + '=' + (e.target.checked ? 1 : 0));

// Sent as a fraction, drawn as one, but an <input type=range> only counts in
// integers - so the slider is in percent and nothing else in the page is.
for (const k of ['conf_thermal', 'conf_radar']) {
  const el = $(k);
  el.oninput = () => {
    $('o_' + k).textContent = (el.value / 100).toFixed(2);
    set(k + '=' + (el.value / 100));
  };
}

$('posego').onclick = () => { const v = $('poseid').value.trim(); if (v) stamp(v); };
$('poseclr').onclick = () => stamp(null);
$('poseid').onkeydown = (e) => { if (e.key === 'Enter') $('posego').click(); };

function seg(boxId, names, param, onpick) {
  const box = $(boxId);
  const items = names.map(n => typeof n === 'string' ? {id:n,label:n} : n);
  box.innerHTML = items.map(n => '<button data-v="' + n.id + '">' + n.label + '</button>').join('');
  box.onclick = (e) => {
    if (e.target.tagName !== 'BUTTON') return;
    pick(boxId, e.target.dataset.v);
    set(param + '=' + e.target.dataset.v);
    if (onpick) onpick(e.target.dataset.v);
  };
}
function pick(boxId, v) {
  for (const b of $(boxId).children) b.className = (b.dataset.v === v) ? 'on' : '';
}
function applyChannel(v) {
  $('fusionviewcard').style.display = v === 'fusion' ? '' : 'none';
}
let CHANNELS = [];
const VIEWS = ['fused','visible','blink','mix','edges','operator'];
seg('view', VIEWS, 'view');
seg('pal', ['ironbow','white','black','gray'], 'palette');

// The radar extrinsics. Nudges rather than free text: these are being adjusted
// against a live image, and the question being asked is always "is it better or
// worse than a moment ago", never "what if it were 7.3 degrees".
const NUDGE = [['yaw','&deg;',0.2],['pitch','&deg;',0.2],['roll','&deg;',0.2],
               ['tx',' mm',5],['ty',' mm',5],['tz',' mm',5]];
const rstate = {};
$('nudge').innerHTML = NUDGE.map(([k,u]) =>
  '<label>' + k + '</label><output id="o_' + k + '">-</output>'
  + '<button data-k="' + k + '" data-d="-1">-</button>'
  + '<button data-k="' + k + '" data-d="1">+</button>').join('');
$('nudge').onclick = (e) => {
  const k = e.target.dataset && e.target.dataset.k;
  if (!k) return;
  const step = NUDGE.find(n => n[0] === k)[2] * Number(e.target.dataset.d);
  rstate[k] = Math.round((rstate[k] + step) * 100) / 100;
  $('o_' + k).textContent = rstate[k] + (k[0] === 't' ? ' mm' : '\\u00b0');
  set(k + '=' + rstate[k]);
};

// ------------------------------------------------------------------ readings
// The image is scaled to fit, so client coords have to be mapped back to the
// 640x400 the pipeline actually indexes - otherwise the reading is off by the
// zoom factor and silently wrong rather than obviously broken.
const im = $('im'), read = $('read'), ovl = $('ovl');
function pixel(e) {
  const r = im.getBoundingClientRect();
  if (!im.naturalWidth) return [0, 0, 0, 0];   // no frame decoded yet
  return [Math.round((e.clientX - r.left) * im.naturalWidth / r.width),
          Math.round((e.clientY - r.top) * im.naturalHeight / r.height),
          e.clientX - r.left, e.clientY - r.top];
}
let pending = false, last = 0;
im.onmousemove = async (e) => {
  const [x, y, lx, ly] = pixel(e);
  read.style.left = lx + 'px'; read.style.top = ly + 'px';
  if (pending || performance.now() - last < 80) return;      // ~12 Hz is plenty
  pending = true; last = performance.now();
  try {
    const d = await (await fetch('/temp?x=' + x + '&y=' + y)).json();
    read.style.display = 'block';
    read.className = d.repaired ? 'warn' : '';
    read.textContent = d.valid
      ? d.c.toFixed(1) + ' \\u00b0C' + (d.repaired ? '  rebuilt row' : '')
      : 'no thermal';
  } finally { pending = false; }
};
im.onmouseleave = () => { read.style.display = 'none'; };

// Two pins at a time, A then B then back to A. Two is not a limitation: the
// delta is a statement about a PAIR of points on the same material, and a list
// of six probes invites averaging things that are not comparable.
let probes = [];
im.onclick = (e) => {
  const [x, y] = pixel(e);
  if (!im.naturalWidth) return;
  probes = probes.length >= 2 ? [{x, y}] : probes.concat([{x, y}]);
  refreshProbes();
};
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'x') { probes = []; refreshProbes(); }
  if (e.key === 'f') document.body.classList.toggle('field');
  if (e.key === 'o') { $('outline').click(); }
  if (e.key === 'b') { $('boxes').click(); }
  if (e.key === 'e') { $('evidence').click(); }
  if (e.key === 't') { $('ai_thermal').click(); }
  if (e.key === 'r') { $('ai_radar').click(); }
  if (e.key === 'c') { $('ai_fusion').click(); }
  if (e.key === 's') { $('silhouette').click(); }
  if (e.key === 'l') { $('lock').click(); }
  const i = Number(e.key) - 1;
  if (i >= 0 && i < CHANNELS.length) {
    pick('channel', CHANNELS[i].id); applyChannel(CHANNELS[i].id);
    set('channel=' + CHANNELS[i].id);
  }
});

async function refreshProbes() {
  for (const p of probes) {
    const d = await (await fetch('/temp?x=' + p.x + '&y=' + p.y)).json();
    p.c = d.valid ? d.c : null;
    p.repaired = !!d.repaired;
  }
  const NAME = ['A','B'], COL = ['#ffffff','#8fd0ff'];
  ovl.innerHTML = probes.map((p, i) =>
    '<g stroke="' + COL[i] + '" fill="none" stroke-width="1.4">'
    + '<circle cx="' + p.x + '" cy="' + p.y + '" r="9"/>'
    + '<line x1="' + (p.x - 13) + '" y1="' + p.y + '" x2="' + (p.x - 4) + '" y2="' + p.y + '"/>'
    + '<line x1="' + (p.x + 4) + '" y1="' + p.y + '" x2="' + (p.x + 13) + '" y2="' + p.y + '"/>'
    + '<line x1="' + p.x + '" y1="' + (p.y - 13) + '" x2="' + p.x + '" y2="' + (p.y - 4) + '"/>'
    + '<line x1="' + p.x + '" y1="' + (p.y + 4) + '" x2="' + p.x + '" y2="' + (p.y + 13) + '"/>'
    + '<text x="' + (p.x + 13) + '" y="' + (p.y + 4) + '" fill="' + COL[i] + '" stroke="none"'
    + ' font-family="ui-monospace,monospace" font-size="12">' + NAME[i]
    + (p.c === null ? '' : ' ' + p.c.toFixed(1)) + '</text></g>').join('');

  if (!probes.length) {
    $('probes').innerHTML = '<div class=hint>click the image to pin a probe</div>';
    return;
  }
  let html = probes.map((p, i) =>
    '<div class=p><em>PROBE ' + NAME[i] + '</em><b>'
    + (p.c === null ? '--' : p.c.toFixed(1)) + '</b>'
    + (p.repaired ? ' <span class=flag>rebuilt</span>' : '') + '</div>').join('');
  if (probes.length === 2 && probes[0].c !== null && probes[1].c !== null) {
    const dc = probes[0].c - probes[1].c;
    // The delta is only the trustworthy number while both ends are trustworthy.
    // A rebuilt row is a linear ramp spliced into real data, so a probe sitting
    // on one is not a measurement of the scene - say so beside the delta rather
    // than three paragraphs down the page.
    const dirty = probes.some(p => p.repaired);
    html += '<div class=d><em>' + '\\u0394' + ' A-B</em><b>' + dc.toFixed(1)
      + ' \\u00b0C</b>' + (dirty ? '<span class=flag>rebuilt row</span>' : '') + '</div>';
  }
  $('probes').innerHTML = html + '<div class=hint><kbd>x</kbd> clears</div>';
}
refreshProbes();

// ------------------------------------------------------------------ history
// fps, skew, heap, coverage all drift rather than jump, and a single reading
// hides drift entirely. The heap one is the point of the exercise: 12.4MB is not
// interesting, 12.4MB where a minute ago it was 13.0MB is the early warning.
const hist = {th: [], sk: [], fps: [], heap: [], cov: [],
              cpu: [], gpu: [], mem: [], tj: [], pw: []};
function push(k, v) {
  if (v === null || v === undefined || !isFinite(v)) return;
  hist[k].push(v);
  if (hist[k].length > 60) hist[k].shift();
}
function spark(id, key, colour, lo, hi) {
  const v = hist[key], W = 90, H = 16, el = $(id);
  if (v.length < 2) { el.innerHTML = ''; return; }
  const clamp = (t) => Math.max(0, Math.min(1, (t - lo) / (hi - lo)));
  const pts = v.map((t, i) => (i / (v.length - 1) * (W - 2) + 1).toFixed(1) + ','
                              + (H - 1 - clamp(t) * (H - 2)).toFixed(1));
  el.innerHTML = '<polyline points="' + pts.join(' ') + '" fill="none" stroke="' + colour
    + '" stroke-width="1.2" stroke-linejoin="round" opacity=".85"/>';
}

// ------------------------------------------------------------------ the clock
function drawClock(t) {
  const per = t.thermal_ms.median, X = 10, W = 330;
  const frac = t.skew_frac === null ? 0 : Math.min(1, t.skew_frac);
  const vx = X + W * frac, bad = t.skew_frac !== null && t.skew_frac > 0.25;
  const jw = W * (t.thermal_ms.max - t.thermal_ms.min) / per;
  $('clocksvg').innerHTML =
    '<rect x="' + X + '" y="16" width="' + W + '" height="12" fill="#181c1f" stroke="#2a3136" rx="2"/>'
    + '<rect x="' + X + '" y="16" width="' + (vx - X) + '" height="12" fill="'
      + (bad ? '#4a3714' : '#173044') + '" rx="2"/>'
    + '<rect x="' + (X + W - jw) + '" y="16" width="' + jw + '" height="12" fill="#2b3237"/>'
    + '<line x1="' + X + '" y1="9" x2="' + X + '" y2="35" stroke="#f0a860" stroke-width="2"/>'
    + '<line x1="' + vx + '" y1="9" x2="' + vx + '" y2="35" stroke="#6cc8ff" stroke-width="2"/>'
    + '<line x1="' + (X + W) + '" y1="9" x2="' + (X + W) + '" y2="35" stroke="#f0a860"'
      + ' stroke-width="2" stroke-dasharray="2 2"/>'
    + '<text x="' + (X + 3) + '" y="45" fill="#f0a860" font-size="9.5"'
      + ' font-family="ui-monospace,monospace">thermal grab</text>'
    + '<text x="' + (vx + 4) + '" y="13" fill="#6cc8ff" font-size="9.5"'
      + ' font-family="ui-monospace,monospace">visible</text>'
    + '<text x="' + (X + W) + '" y="45" fill="#6b747c" font-size="9.5" text-anchor="end"'
      + ' font-family="ui-monospace,monospace">' + per + ' ms period &#183; jitter '
      + t.thermal_ms.min + '-' + t.thermal_ms.max + '</text>';
}

// ------------------------------------------------------------------ health
// The same checks health() already produces, grouped by the question each group
// answers. Grouping is the whole point: thirteen flat pills read as thirteen
// equal facts, and "8.7 fps" is not equal to "the sensor went unserviced".
const GROUPS = [
  ['link', ['stream','framing','jpeg','render']],
  ['sensor', ['cadence','pairing','thermal load','ffc','tearing','dead rows','board heap']],
  ['measurement', ['registration','coverage','range','clipping','agc','emissivity']],
  ['perception', ['channel','detect','radar','ai','recording']],
  ['host', ['host cpu','host memory','host thermal','host power']],
  ['map', ['map']],
];
const RANK = {ok: 0, warn: 1, fail: 2}, LVL = ['ok','warn','fail'];
// Which groups the operator has opened by hand. Rebuilding the panel every
// second must not slam shut a group somebody is reading.
const opened = {};
$('health').onclick = (e) => {
  const d = e.target.closest('details');
  if (d) setTimeout(() => { opened[d.dataset.g] = d.open; }, 0);
};
function drawHealth(checks) {
  const by = {};
  for (const c of checks) by[c.name] = c;
  const seen = new Set();
  const groups = GROUPS.map(([name, names]) => {
    const cs = names.map(n => by[n]).filter(Boolean);
    cs.forEach(c => seen.add(c.name));
    return [name, cs];
  });
  // Anything health() grows later lands here rather than vanishing off the page.
  const rest = checks.filter(c => !seen.has(c.name));
  if (rest.length) groups.push(['other', rest]);

  $('health').innerHTML = groups.map(([name, cs]) => {
    if (!cs.length) return '<details class="grp ok" data-g="' + name + '">'
      + '<summary><span class=dot style=background:#2f363c></span>' + name
      + '<span class=cnt>nothing to report</span><span class=chev>&rsaquo;</span></summary></details>';
    const worst = LVL[Math.max.apply(null, cs.map(c => RANK[c.level]))];
    const bad = cs.filter(c => c.level !== 'ok').length;
    // Not-ok groups open themselves; ok groups stay shut unless opened by hand.
    const open = (opened[name] !== undefined) ? opened[name] : worst !== 'ok';
    return '<details class="grp ' + worst + '" data-g="' + name + '"' + (open ? ' open' : '') + '>'
      + '<summary><span class=dot></span>' + name + '<span class=cnt>'
      + (bad ? bad + ' of ' + cs.length : cs.length + ' ok')
      + '</span><span class=chev>&rsaquo;</span></summary><div class=checks>'
      + cs.map(c => '<div class="chk-row ' + c.level + '"><i></i><span class=n>' + c.name
          + '</span><span class=t>' + esc(c.text) + '</span></div>').join('')
      + '</div></details>';
  }).join('');
}

// ------------------------------------------------------------------ poll
const VERDICT = {
  ok: ['CLEAR', 'nothing this code can see is wrong'],
  warn: ['QUALIFIED', 'usable, but read them the way the amber says'],
  fail: ['DO NOT RECORD', 'the numbers on screen are wrong or absent'],
};
let inited = false;
function initControls(cfg) {
  CHANNELS = cfg.channels || [];
  seg('channel', CHANNELS, 'channel', applyChannel);
  for (const k of SLIDERS) { $(k).value = cfg[k]; $('o_' + k).textContent = cfg[k]; }
  $('emis').value = cfg.emissivity;
  $('refl').value = cfg.reflected;
  $('outline').checked = cfg.outline;
  $('boxes').checked = cfg.boxes;
  $('lock').checked = cfg.lock;
  pick('channel', cfg.channel);
  applyChannel(cfg.channel);
  pick('view', cfg.view);
  pick('pal', cfg.palette);
  if (cfg.ai) {
    $('aistudents').style.display = '';
    $('ai_thermal').checked = cfg.ai.thermal.on;
    $('ai_radar').checked = cfg.ai.radar.on;
    $('ai_fusion').checked = cfg.ai.fusion.on;
    $('evidence').checked = cfg.ai.evidence;
    $('silhouette').checked = cfg.ai.silhouette;
    for (const [k, c] of [['conf_thermal', cfg.ai.thermal],
                          ['conf_radar', cfg.ai.radar]]) {
      $(k).min = Math.round(100 * cfg.ai.floor);
      $(k).value = Math.round(100 * c.conf);
      $('o_' + k).textContent = c.conf.toFixed(2);
    }
  }
  if (cfg.radar) {
    $('radarcard').style.display = '';
    $('radar').checked = cfg.radar.on;
    $('whisker').checked = cfg.radar.whisker;
    for (const [k] of NUDGE) {
      rstate[k] = cfg.radar[k];
      $('o_' + k).textContent = rstate[k] + (k[0] === 't' ? ' mm' : '\\u00b0');
    }
  }
  inited = true;
}

function drawMap(m) {
  const card = $('mapcard'), side = $('mapside');
  if (!m) { card.style.display = 'none'; side.className = ''; return; }
  card.style.display = '';
  side.className = 'on';
  const num = (v, u, f) => (v === null || v === undefined) ? '\u2013'
      : v.toFixed(f) + '<span class=u> ' + u + '</span>';
  const loc = m.location ? m.location.latitude_deg.toFixed(5) + ', '
      + m.location.longitude_deg.toFixed(5) : '';
  if (m.pending) {
    $('m_stages').textContent = 'initialising at ' + loc + ' \u2026';
    $('m_text').textContent = '';
    return;
  }
  const g = m.geoid || {}, pr = m.priors || {}, alt = m.ego_altitude_prior || {};
  $('m_geoid').innerHTML = num(g.undulation_m, 'm', 2);
  $('m_ground').innerHTML = num(pr.ground_elevation_m, 'm', 1);
  $('m_surface').innerHTML = num(pr.surface_elevation_m, 'm', 1);
  $('m_sigma').innerHTML = num(alt.sigma_m, 'm', 1);
  const st = (m.stages || []).map(s => s.name + ': ' + s.status).join(' \u00b7 ');
  $('m_stages').innerHTML = (m.ok ? '<b>ok</b> ' : '<b style=color:var(--warn)>FAILED</b> ')
      + esc(loc) + (st ? ' \u2014 ' + esc(st) : '');
  $('m_text').textContent = m.error ? m.error
      : (pr.overture_path ? 'buildings: ' + pr.overture_path.split('/').pop() : '');
}
let topdownTimer = null;
function drawFix(d) {
  const prior = d.map_prior, fx = d.map_fix, nav = d.nav;
  if (!prior) return;
  if (document.activeElement !== $('heading') && prior.heading_deg !== null) {
    $('heading').value = Math.round(prior.heading_deg);
    $('o_heading').textContent = Math.round(prior.heading_deg) + '\u00b0';
  }
  const p = fx && fx.pose;
  if (!fx) { $('m_fix').textContent = '\u2013'; $('m_fixtext').textContent =
      prior.heading_deg === null ? 'set a heading prior, then fix from walls' : 'not solved yet'; }
  else if (!p) { $('m_fix').innerHTML = '<span style=color:var(--warn)>none</span>';
      $('m_fixtext').textContent = fx.error || ''; }
  else {
    $('m_fix').innerHTML = (p.accepted ? '<span style=color:#5cf>ok</span>'
        : '<span style=color:var(--warn)>rejected</span>')
        + '<span class=u> \u0394 ' + p.dx_m.toFixed(1) + ' E ' + p.dy_m.toFixed(1) + ' N</span>';
    $('m_head').innerHTML = p.heading_deg.toFixed(1) + '<span class=u>\u00b0 \u00b1' + p.sigma_heading_deg.toFixed(1) + '</span>';
    const bad = (fx.checks || []).filter(c => !c.passed).map(c => c.name.replace(/^walls\./, '') + ': ' + c.detail);
    $('m_fixtext').textContent = (p.accepted ? '' : bad.join(' \u00b7 ') + ' \u2014 ')
        + 'inliers ' + Math.round(100 * p.inlier_fraction) + '% of ' + p.n_returns
        + ', \u03c3 ' + p.sigma_horizontal_m.toFixed(1) + ' m, ' + fx.solve_s + ' s';
  }
  if (fx && fx.t) $('m_fixage').textContent = Math.round((Date.now() / 1000) - fx.t) + ' s ago';
  if (nav && !nav.error) {
    $('m_nav').innerHTML = nav.horizon_1m_s + '<span class=u> s to 1 m</span>';
    $('m_speed').innerHTML = nav.speed_mps.toFixed(2) + '<span class=u> m/s</span>';
    $('m_navtext').textContent = nav.regime + ', heading \u03c3 ' + nav.heading_sigma_deg
        + '\u00b0: ' + nav.error_60s_m + ' m after 60 s'
        + (nav.dominant ? ', mostly ' + nav.dominant : '')
        + (nav.rests_on_assumption ? ' (rests on an ASSUMED constant)' : '')
        + '; inertial-only would hold ' + nav.horizon_1m_inertial_s + ' s';
  } else if (nav) { $('m_navtext').textContent = nav.error; }
}
// An empty square beside the picture is indistinguishable from a broken one,
// and the reason used to be a 40-character slice in #667 at the top-left corner
// - the one place nobody looks, cut mid-module-path. Centred, wrapped, and in
// the warn colour, because a top-down that cannot draw is a finding.
function tdMessage(g, W, H, lead, detail) {
  g.textAlign = 'center';
  g.fillStyle = '#f0c060';
  g.font = '600 13px system-ui,-apple-system,sans-serif';
  g.fillText(lead, W / 2, H / 2 - 16);
  g.fillStyle = '#79838d';
  g.font = '11px ui-monospace,Menlo,Consolas,monospace';
  let line = '', y = H / 2 + 6;
  for (const w of String(detail || '').split(/\s+/).filter(Boolean)) {
    const t = line ? line + ' ' + w : w;
    if (line && g.measureText(t).width > W - 36) { g.fillText(line, W / 2, y); y += 15; line = w; }
    else line = t;
  }
  if (line) g.fillText(line, W / 2, y);
  g.textAlign = 'start';
}
async function drawTopdown() {
  let td;
  try { td = await (await fetch('/topdown')).json(); } catch (e) { return; }
  const c = $('topdown'), g = c.getContext('2d'), W = c.width, H = c.height;
  const side = $('mapside');
  // The legend promises walls and dots. While there are none it is a
  // description of a picture that is not on screen, so it goes quiet.
  const quiet = on => { if (side) side.classList.toggle('nowalls', !!on); };
  g.clearRect(0, 0, W, H);
  if (!td || td.error || !td.edges) {
    quiet(true);
    // Two different situations, and the fix differs: no --map is configuration,
    // anything else is the wall layer failing with the reason it gives.
    const cfg = td && /no map layer/.test(td.error || '');
    tdMessage(g, W, H, cfg ? 'no map layer' : 'top-down unavailable',
              (td && td.error) || 'no answer from /topdown');
    return;
  }
  quiet(!td.edges.length);
  const R = td.radius_m, k = (W / 2) / R;
  const X = x => W / 2 + x * k, Y = y => H / 2 - y * k;
  g.strokeStyle = '#1e2429'; g.lineWidth = 1;
  for (let r = 10; r <= R; r += 10) { g.beginPath(); g.arc(W/2, H/2, r * k, 0, 2 * Math.PI); g.stroke(); }
  g.strokeStyle = '#9aa'; g.lineWidth = 1.5;
  for (const [x0, y0, x1, y1] of td.edges) { g.beginPath(); g.moveTo(X(x0), Y(y0)); g.lineTo(X(x1), Y(y1)); g.stroke(); }
  const rig = (p, col) => {
    if (!p || p.heading_deg === null || p.heading_deg === undefined) return;
    const h = p.heading_deg * Math.PI / 180;
    g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
    g.moveTo(X(p.x), Y(p.y)); g.lineTo(X(p.x + 6 * Math.sin(h)), Y(p.y + 6 * Math.cos(h))); g.stroke();
    g.beginPath(); g.arc(X(p.x), Y(p.y), 3, 0, 2 * Math.PI); g.fillStyle = col; g.fill();
    if (p.sigma_m) { g.setLineDash([3, 3]); g.beginPath(); g.arc(X(p.x), Y(p.y), p.sigma_m * k, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]); }
  };
  const dots = (pts, col) => { g.fillStyle = col; for (const [x, y] of pts) { g.beginPath(); g.arc(X(x), Y(y), 2.5, 0, 2 * Math.PI); g.fill(); } };
  dots(td.returns_at_prior, '#e0a030'); rig(td.prior, '#e0a030');
  if (td.fix) { dots(td.returns_at_fix, '#5cf'); rig(td.fix, td.fix.accepted ? '#5cf' : '#c55'); }
  g.fillStyle = '#667'; g.font = '11px sans-serif';
  g.fillText('N', W / 2 - 4, 12); g.fillText(td.n_walls + ' returns', 8, H - 8);
  // Buildings loaded, none within the radius: the rings and the returns are
  // still worth drawing, but nothing here can confirm the prior.
  if (!td.edges.length) tdMessage(g, W, H, 'no map walls in ' + td.radius_m + ' m',
      'the priors loaded but hold no footprint within the radius, so the fix has nothing to match');
}
async function poll() {
  let d;
  try {
    d = await (await fetch('/ui')).json();
  } catch (err) {
    $('trust').className = 'fail';
    $('vtext').textContent = 'VIEWER DISCONNECTED';
    $('vsub').textContent = 'the page cannot reach live.py - is it still running?';
    return;
  }
  if (!inited) initControls(d.cfg);

  const [vt, vs] = VERDICT[d.worst];
  $('trust').className = d.worst;
  $('vtext').textContent = vt;
  $('vsub').textContent = vs;

  // The MJPEG endpoint keeps the last good frame on screen while the reader is
  // recovering.  That is preferable to a broken image icon, but only if the
  // operator can see that the pixels are stale rather than live.
  const ss = d.stream_status;
  $('streamstate').hidden = !ss;
  if (ss) {
    $('streamtitle').textContent = ss.title;
    $('streamdetail').textContent = ss.detail || '';
  }

  // The facts that qualify every number on screen, in one place. They used to be
  // spread across /stat, a health pill and the word UNREGISTERED on each box.
  const cfg = d.cfg, q = [];
  q.push(['warp', cfg.warped ? 'calibrated' : 'PLACEHOLDER', !cfg.warped]);
  q.push(['agc', cfg.agc ? cfg.agc + '\\u2030 - tone is scene-relative' : 'off', !!cfg.agc]);
  q.push(['\\u03b5', cfg.emissivity.toFixed(2) + ' / refl ' + Math.round(cfg.reflected)
          + ' \\u00b0C', cfg.emissivity < 1]);
  q.push(['range', cfg.range[1] > cfg.range[0]
          ? cfg.range[0] + '-' + cfg.range[1] + ' \\u00b0C' : 'none', cfg.range[1] <= cfg.range[0]]);
  if (d.recording) q.push(['rec', d.recording, false]);
  // Recording with no pose_id is flagged, not merely blank: unlabelled frames
  // are the failure this control exists to prevent, and it is silent otherwise.
  if (d.pose) q.push(['pose', d.pose.id || 'UNLABELLED', !d.pose.id]);
  $('qual').innerHTML = q.map(([k, v, bad]) =>
    '<span class="' + (bad ? 'bad' : '') + '">' + k + ' <i>' + esc(v) + '</i></span>').join('');

  // Skipped while the operator is mid-entry: the poll runs every second and
  // would otherwise wipe a half-typed station id under their thumb.
  if (document.activeElement !== $('poseid')) applyPose(d.pose);

  drawHealth(d.checks);
  drawMap(d.map);
  drawFix(d);
  if (d.map && !topdownTimer) { topdownTimer = setInterval(drawTopdown, 1000); drawTopdown(); }

  // --- the clock and the traces
  const t = d.timing;
  if (t.thermal_ms) {
    drawClock(t);
    push('th', t.thermal_ms.median);
    $('k_th').innerHTML = t.thermal_ms.median + '<span class=u> ms &#183; '
      + t.thermal_fps + ' fps</span>';
    if (t.skew_ms) {
      push('sk', t.skew_ms.median);
      $('kv_sk').className = 'kv vis' + (t.skew_frac > 0.25 ? ' bad' : '');
      $('k_sk').innerHTML = '+' + t.skew_ms.median + '<span class=u> ms &#183; '
        + Math.round(100 * t.skew_frac) + '% of a frame</span>';
    }
  }
  push('fps', t.host_fps);
  $('k_fps').innerHTML = t.host_fps.toFixed(1) + '<span class=u> fps</span>';
  $('k_ffc').innerHTML = t.ffc_ago_s === null ? 'none yet'
    : t.ffc_ago_s + '<span class=u> s ago &#183; ' + t.ffcs + '</span>';
  // Load reaching the sensor. On screen even when it is zero: this is the number
  // that decides whether more host work is affordable, and it is worth watching
  // go up rather than discovering afterwards.
  $('kv_load').className = 'kv' + (t.starved || t.stalls ? ' bad' : '');
  $('k_load').innerHTML = t.starved
    ? 'STARVED ' + t.starved + 'x<span class=u> worst ' + t.last_starve_ms + ' ms</span>'
    : t.stalls ? t.stalls + '<span class=u> write stall(s)</span>'
    : '<span style=color:#7fd39b>unstarved</span>';

  if (d.heap_free !== null && d.heap_free !== undefined) {
    const mb = d.heap_free / (1 << 20);
    push('heap', mb);
    $('k_heap').innerHTML = mb.toFixed(1) + '<span class=u> MB'
      + (d.restarts ? ' &#183; ' + d.restarts + ' restart(s)' : '') + '</span>';
  }
  if (d.coverage !== null && d.coverage !== undefined) {
    push('cov', 100 * d.coverage);
    $('k_cov').innerHTML = Math.round(100 * d.coverage) + '<span class=u> %</span>';
    $('s_cov').textContent = Math.round(100 * d.coverage) + '%';
  }
  if (d.light) {
    // "blind" is a warn and not a fail: the rig is working exactly as designed,
    // it simply has no light. What would be a fail is showing 0 boxes without
    // saying so.
    $('kv_light').className = 'kv' + (d.light.state === 'lit' ? '' : ' bad');
    $('k_light').innerHTML = (d.light.state === 'blind' ? 'BLIND'
        : d.light.state === 'dim' ? 'part-lit' : 'lit')
      + '<span class=u> &#183; ' + (100 * d.light.lit).toFixed(1) + '% lit</span>';
  }
  $('k_band').innerHTML = d.stats.valid
    ? d.stats.min.toFixed(1) + '<span class=u>..</span>' + d.stats.max.toFixed(1)
      + '<span class=u> \\u00b0C &#183; \\u0394 ' + d.stats.delta.toFixed(1) + '</span>'
    : '<span class=u>no thermal coverage</span>';
  if (d.raw_stats) {
    $('kv_raw').style.display = '';
    $('k_raw').innerHTML = d.raw_stats.min.toFixed(2)
      + '<span class=u>..</span>' + d.raw_stats.max.toFixed(2)
      + '<span class=u> \\u00b0C &#183; \\u0394 ' + d.raw_stats.delta.toFixed(2)
      + '</span>';
  } else {
    $('kv_raw').style.display = 'none';
  }
  $('agcwarn').innerHTML = cfg.agc
    ? '<span style=color:#f0c060>scene-relative</span>' : '';

  // --- the host. The same subject as the starve counter above, seen from the
  // other end: that one says the sensor went unserviced, this one says why, and
  // it is the half that moves first.
  const s = d.soc;
  if (s) {
    $('socstrip').style.display = '';
    if (s.cpu_pct !== null) {
      push('cpu', s.cpu_pct);
      $('kv_cpu').className = 'kv host' + (s.cpu_pct > 85 ? ' bad' : '');
      $('k_cpu').innerHTML = Math.round(s.cpu_pct) + '<span class=u>% of ' + s.ncpu
        + ' cores' + (s.self_pct === null ? ''
            : ' &#183; ' + Math.round(s.self_pct) + '% mine') + '</span>';
    }
    if (s.gpu_pct !== null) {
      push('gpu', s.gpu_pct);
      $('k_gpu').innerHTML = Math.round(s.gpu_pct) + '<span class=u>%</span>';
    }
    if (s.mem_total_mb) {
      push('mem', s.mem_avail_mb);
      $('kv_mem').className = 'kv host' + (s.mem_avail_mb < 600 ? ' bad' : '');
      $('k_mem').innerHTML = (s.mem_avail_mb / 1024).toFixed(1) + '<span class=u> GB free of '
        + (s.mem_total_mb / 1024).toFixed(1) + '</span>';
    }
    if (s.t_max !== null) {
      push('tj', s.t_max);
      // Headroom, not the raw temperature: 84 C is alarming on a part that trips
      // at 90 and unremarkable on one that trips at 105.
      const head = s.t_crit ? s.t_crit - s.t_max : null;
      $('kv_tj').className = 'kv host' + (head !== null && head < 8 ? ' bad' : '');
      $('k_tj').innerHTML = Math.round(s.t_max) + '<span class=u> &#176;C'
        + (head === null ? '' : ' &#183; ' + Math.round(head) + ' to trip') + '</span>';
    }
    if (s.power_w) {
      push('pw', s.power_w);
      $('k_pw').innerHTML = s.power_w.toFixed(1) + '<span class=u> W</span>';
    }
    if (s.freq_mhz)
      $('k_clk').innerHTML = s.freq_mhz + '<span class=u>/' + s.freq_max_mhz + ' MHz</span>';
    spark('sp_cpu', 'cpu', '#8fa2b5', 0, 100);
    spark('sp_gpu', 'gpu', '#8fa2b5', 0, 100);
    spark('sp_mem', 'mem', '#8fa2b5', 0, s.mem_total_mb || 1);
    spark('sp_tj', 'tj', '#8fa2b5', 20, s.t_crit || 100);
    spark('sp_pw', 'pw', '#8fa2b5', 0, 20);
  }

  spark('sp_th', 'th', '#f0a860', 108, 120);
  spark('sp_sk', 'sk', '#6cc8ff', 0, 40);
  spark('sp_fps', 'fps', '#7fd39b', 0, 9.5);
  spark('sp_heap', 'heap', '#7fd39b', 0, 26);
  spark('sp_cov', 'cov', '#9aa4ad', 0, 100);

  // --- detections. A box you have to squint at is not a reading you would write
  // down, and the peak location matters as much as the value: "this motor is
  // warm" and "this motor's near bearing is warm" are different findings.
  const dd = d.detections;
  $('s_det').textContent = dd.list.length + (dd.ms ? ' \\u00b7 ' + Math.round(dd.ms) + 'ms' : '');
  $('dets').innerHTML = dd.list.map(x =>
    '<div class="det' + (dd.warped ? '' : ' unreg') + '">'
    + '<span class=cls>' + esc(x.cls) + '</span>'
    + '<span class=m>' + Math.round(100 * x.conf) + '%</span>'
    + (x.max_c !== undefined
        ? '<span class=c>' + x.max_c.toFixed(1) + ' \\u00b0C</span>'
          + '<span class=m>peak @' + x.max_x + ',' + x.max_y + '</span>'
        : '<span class=m>no thermal in box</span>')
    + (x.body_heat === false ? '<span class=flag>NO BODY HEAT</span>' : '')
    + (x.repaired ? '<span class=flag>REBUILT ROW</span>' : '')
    + (dd.warped ? '' : '<span class=flag>UNREGISTERED</span>')
    + '</div>').join('');

  // --- the ai channels. The count is the point of the sub-label, but 'off'
  // and 'cannot draw' have to be told apart from 'nobody there': all three look
  // like a zero, and only one of them is news about the scene.
  if (cfg.ai) {
    const chan = (id, c) => {
      const el = $(id);
      const hidden = (c.found === undefined) ? 0 : c.found - c.n;
      el.textContent = !c.ready ? 'unavailable'
        : !c.on ? 'off'
        : hidden > 0 ? c.n + ' of ' + c.found
        : c.n + (c.n === 1 ? ' box' : ' boxes');
      el.style.color = !c.ready ? '#f0c060' : (c.on && c.n) ? '#7fd39b' : '';
    };
    chan('s_ai_t', cfg.ai.thermal);
    chan('s_ai_r', cfg.ai.radar);
    chan('s_ai_f', cfg.ai.fusion);
    $('s_ai_sil').textContent = cfg.ai.silhouette ? 'shape' : 'boxes';
  }

  // A coasting lock is not a lock that is working; it has to read differently.
  const held = d.tracks.filter(t => !t.coasting).length;
  const coast = d.tracks.length - held;
  const stat = (d.tracks_static || []).length;
  $('s_lock').textContent = !cfg.lock ? 'off'
    : !d.tracks.length && !stat ? 'nothing locked'
    : (d.tracks.length ? held + ' locked' : 'none')
      + (coast ? ' \u00b7 ' + coast + ' coasting' : '')
      + (stat ? ' \u00b7 ' + stat + ' static' : '');
  // What the rig is doing under the lock. Silent while it is still, because
  // that is the ordinary case; explicit when it is turning, and explicit when
  // the frame was too busy to read - then the lock is back to assuming still.
  const eg = d.ego;
  if (eg) {
    const mag = Math.sqrt(eg.du * eg.du + eg.dv * eg.dv);
    $('s_lock').textContent += eg.response < 0.7 ? ' \u00b7 rig unread'
      : mag >= 1.5 ? ' \u00b7 rig ' + mag.toFixed(0) + ' px/f' : '';
  }
  $('s_lock').style.color = !cfg.lock ? '' : coast && !held ? '#f0c060'
    : held ? '#7fd39b' : '';

  if (probes.length) refreshProbes();
}
poll();
setInterval(poll, 1000);
</script>
"""


def start_map_init(state, latlon, geoid_range=None, mapinit_dir=None):
    """Run Yael's map initialisation off the main thread, into state["map"].

    The result is Moshe's JSON contract (perception.map_api, schema 1.0), so
    what /map serves is exactly what the CLI prints.  Three shapes land in
    state["map"]:  {"pending": True} while it runs; the contract dict once it
    returns (ok may be false - that is a MAP finding, e.g. no EGM grid); and
    {"ok": False, "error": ...} when the boundary itself failed (package or a
    dependency missing) - a DEPLOYMENT finding, named so in the text.
    """
    try:
        lat, lon = (float(v) for v in latlon.split(","))
    except ValueError:
        raise SystemExit("--map wants LAT,LON in decimal degrees, got %r" % latlon)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from perception import map_api
    request = map_api.MapInitRequest(
        latitude=lat, longitude=lon,
        expected_geoid_range_m=tuple(geoid_range) if geoid_range else None)
    state["map"] = {"pending": True, "location": {"latitude_deg": lat,
                                                   "longitude_deg": lon}}

    def run():
        try:
            api = map_api.MapInitializationAPI(mapinit_dir=mapinit_dir)
            # fail_fast=False: the health line should name every failed stage,
            # not only the first, because the fixes differ (grid vs priors).
            result = api.initialize(request, fail_fast=False)
        except map_api.MapAPIError as e:
            result = {"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                      "deployment": True, "location": state["map"]["location"]}
        except Exception as e:  # never take the viewer down for the map
            result = {"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                      "location": state["map"]["location"]}
        state["map"] = result
        if result.get("ok"):
            alt = result.get("ego_altitude_prior") or {}
            print("map: initialised at %.5f,%.5f - geoid %.2f m, ground %.1f m "
                  "orthometric (sigma %.1f m), %s"
                  % (lat, lon, (result.get("geoid") or {}).get("undulation_m", float("nan")),
                     alt.get("orthometric_m", float("nan")), alt.get("sigma_m", float("nan")),
                     os.path.basename((result.get("priors") or {}).get("overture_path", "?"))),
                  file=sys.stderr)
        else:
            print("map: FAILED - %s" % (result.get("error") or result.get("summary", "")),
                  file=sys.stderr)

    threading.Thread(target=run, name="map-init", daemon=True).start()


def static_wall_returns(points, max_speed=0.25, min_range=0.5, max_range=40.0):
    """live.py's radar points (x fwd, y left, z, v, snr) -> (range, azimuth) walls.

    Azimuth positive to the RIGHT, as the rig reports it to the map layer:
    y is left in the radar frame, so azimuth = atan2(-y, x).
    """
    out = []
    for p in points or []:
        x, y = float(p["x"]), float(p["y"])
        rng = math.hypot(x, y)
        if not (min_range <= rng <= max_range):
            continue
        if abs(float(p.get("v") or 0.0)) > max_speed:
            continue
        out.append((rng, math.degrees(math.atan2(-y, x))))
    return out


def run_map_fix(state):
    """Solve position+heading from the current static returns, into state["map_fix"].

    Synchronous and a few seconds long (a 3-sigma grid at 5 m / 5 deg): the
    HTTP handler runs it on its own thread, so the stream never waits. The
    result is the map_api contract's "pose" block, or an error the operator
    can act on. It never touches state["map"], the startup layer.
    """
    prior = state.get("map_prior")
    if prior is None:
        return {"ok": False, "error": "no map layer; start live.py with --map LAT,LON"}
    if prior.get("heading_deg") is None:
        return {"ok": False, "error": "no heading prior; give --map-heading DEG or /set?heading=DEG"}
    pts = state.get("radar_points")
    age = time.time() - state.get("radar_points_t", 0) if pts is not None else None
    if pts is None or age > 2.0:
        return {"ok": False, "error": "no fresh radar frame (age %s s)" % (
            None if age is None else round(age, 1))}
    walls = static_wall_returns(pts)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from perception import map_api
    t0 = time.time()
    try:
        req = map_api.MapInitRequest(
            latitude=prior["latitude_deg"], longitude=prior["longitude_deg"],
            expected_geoid_range_m=state.get("map_geoid_range"),
            heading_prior_deg=prior["heading_deg"],
            sigma_position_m=prior["sigma_position_m"],
            sigma_heading_deg=prior["sigma_heading_deg"],
            wall_returns=tuple(walls) or None)
        if not req.has_pose_observations:
            return {"ok": False, "error": "no static returns in range 0.5-40 m (%d points)" % len(pts),
                    "walls": walls}
        res = map_api.MapInitializationAPI(mapinit_dir=state.get("mapinit_dir")).initialize(
            req, fail_fast=False)
    except map_api.MapAPIError as e:
        return {"ok": False, "error": "deployment: %s: %s" % (type(e).__name__, e)}
    except Exception as e:                        # noqa: BLE001
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    pose = res.get("pose")
    stage = next((s for s in res.get("stages", []) if s["name"] == "pose_init"), None)
    out = {"ok": bool(pose and pose.get("accepted")), "pose": pose,
           "prior": prior, "walls": walls, "n_walls": len(walls),
           "checks": (stage or {}).get("checks", []),
           "stage_status": (stage or {}).get("status"),
           "error": ((stage or {}).get("error") or {}).get("message"),
           "solve_s": round(time.time() - t0, 2), "t": time.time()}
    return out


def topdown_view(state, radius_m=60.0):
    """Footprint edges and radar returns in metres east/north of the prior.

    What the page draws so an operator can SEE whether the walls the radar
    reports sit on the map's walls - the check that no number replaces.
    Returns None without a map layer or before its priors loaded.
    """
    prior = state.get("map_prior")
    m = state.get("map") or {}
    pr = m.get("priors") or {}
    if prior is None or not pr.get("overture_path"):
        return None
    cache = state.get("_topdown_cache")
    if cache is None or cache["path"] != pr["overture_path"]:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from perception import map_api
        map_api.locate_mapinit(state.get("mapinit_dir"))
        from mapinit import BuildingLayer
        layer = BuildingLayer.from_geojson(pr["overture_path"])
        # The deployed mapinit checkout exports BuildingLayer but has no
        # mapinit.nav.walls module.  Top-down rendering only needs the public
        # footprint rings, so convert them locally instead of depending on a
        # private/nonexistent navigation helper. x is east, y is north.
        lat0, lon0 = prior["latitude_deg"], prior["longitude_deg"]
        lat_scale = 111_320.0
        lon_scale = lat_scale * math.cos(math.radians(lat0))
        segs = []
        for building in layer.near(lat0, lon0, radius_m * 2.0):
            ring = list(building.ring)
            for (lon_a, lat_a), (lon_b, lat_b) in zip(ring, ring[1:]):
                edge = [(lon_a - lon0) * lon_scale, (lat_a - lat0) * lat_scale,
                        (lon_b - lon0) * lon_scale, (lat_b - lat0) * lat_scale]
                if min(math.hypot(edge[0], edge[1]),
                       math.hypot(edge[2], edge[3])) <= radius_m * 1.5:
                    segs.append([round(v, 2) for v in edge])
        cache = {"path": pr["overture_path"], "edges": segs,
                 "scales": (lon_scale, lat_scale)}
        state["_topdown_cache"] = cache
    heading = prior.get("heading_deg")
    walls = static_wall_returns(state.get("radar_points"))
    fix = (state.get("map_fix") or {}).get("pose") or {}

    def place(h, dx, dy):
        if h is None:
            return []
        return [[round(dx + r * math.sin(math.radians(h + a)), 2),
                 round(dy + r * math.cos(math.radians(h + a)), 2)] for r, a in walls]

    lon_scale, lat_scale = cache["scales"]
    fx = fy = None
    if fix.get("latitude_deg") is not None:
        fx = (fix["longitude_deg"] - prior["longitude_deg"]) * lon_scale
        fy = (fix["latitude_deg"] - prior["latitude_deg"]) * lat_scale
    return {"radius_m": radius_m, "edges": cache["edges"],
            "prior": {"x": 0.0, "y": 0.0, "heading_deg": heading,
                      "sigma_m": prior["sigma_position_m"]},
            "returns_at_prior": place(heading, 0.0, 0.0),
            "fix": None if fx is None else {
                "x": round(fx, 2), "y": round(fy, 2), "heading_deg": fix.get("heading_deg"),
                "accepted": fix.get("accepted"), "ambiguous": fix.get("ambiguous"),
                "sigma_m": fix.get("sigma_horizontal_m")},
            "returns_at_fix": [] if fx is None else place(fix.get("heading_deg"), fx, fy),
            "n_walls": len(walls)}


def nav_horizon(state):
    """How long a fix survives, from the map package's DeadReckoner.

    Speed aiding comes from the radar: with the rig still, the static
    returns say speed 0 and the regime is speed-aided at zero speed, which
    is the honest reading for a parked rig. The IMU constants are the
    package's RIG_ERROR_MODEL (some measured, some assumed - the budget says
    which). Cheap: pure arithmetic, no I/O.
    """
    if state.get("map_prior") is None:
        return None
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from perception import map_api
        map_api.locate_mapinit(state.get("mapinit_dir"))
        from mapinit import DeadReckoner, SpeedAiding
    except Exception as e:                        # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, e)}
    pts = state.get("radar_points") or []
    statics = [p for p in pts if abs(float(p.get("v") or 0.0)) <= 0.25 and math.hypot(p["x"], p["y"]) >= 0.5]
    # Ego speed from the static returns' radial velocities (a still rig: 0).
    speed = 0.0
    if statics:
        speed = float(sum(abs(float(p.get("v") or 0.0)) for p in statics) / len(statics))
    fix = (state.get("map_fix") or {}).get("pose") or {}
    heading_sigma = fix.get("sigma_heading_deg") if fix.get("accepted") else         (state.get("map_prior") or {}).get("sigma_heading_deg") or 5.0
    aided = DeadReckoner(aiding=SpeedAiding(speed_mps=speed, sigma_speed_mps=0.05),
                         initial_heading_sigma_deg=float(heading_sigma))
    inertial = DeadReckoner(initial_heading_sigma_deg=float(heading_sigma))
    b = aided.budget_at(60.0)
    dom = b.dominant
    dom = dom() if callable(dom) else dom
    return {"regime": aided.regime, "speed_mps": round(speed, 2),
            "heading_sigma_deg": round(float(heading_sigma), 2),
            "horizon_1m_s": round(aided.horizon_for(1.0), 1),
            "horizon_1m_inertial_s": round(inertial.horizon_for(1.0), 1),
            "error_60s_m": round(b.total_m, 1),
            "dominant": None if dom is None else str(dom),
            "rests_on_assumption": bool(b.rests_on_assumption() if callable(b.rests_on_assumption)
                                        else b.rests_on_assumption)}


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
                if q.get("channel") in CHANNELS:
                    pipe.channel = q.pop("channel")
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
                # The AI channels. Accepted even when --students was not given:
                # the flags are display state like every other switch here, and
                # a 404 on a live control is harder to read than a switch that
                # holds a value nothing is currently drawing.
                if "lock" in q:
                    pipe.ai_lock = q.pop("lock") not in ("0", "false", "")
                if "silhouette" in q:
                    pipe.ai_silhouette = q.pop("silhouette") not in (
                        "0", "false", "")
                if "evidence" in q:
                    pipe.ai_evidence = q.pop("evidence") not in (
                        "0", "false", "")
                for k in ("ai_thermal", "ai_radar", "ai_fusion"):
                    if k in q:
                        setattr(pipe, k, q.pop(k) not in ("0", "false", ""))
                # Clamped at the bottom by the floor the engines were built
                # with: a slider below it would look like it was letting more
                # through while changing nothing at all.
                for k, attr in (("conf_thermal", "ai_conf_thermal"),
                                ("conf_radar", "ai_conf_radar")):
                    if k in q:
                        setattr(pipe, attr,
                                min(0.99, max(pipe.ai_conf_floor,
                                              float(q.pop(k)))))
                rp = state.get("radar_proj")
                if rp is not None:
                    for k in ("yaw", "pitch", "roll"):
                        if k in q:
                            setattr(rp, k, float(q.pop(k)))
                    for i, k in enumerate(("tx", "ty", "tz")):
                        if k in q:      # millimetres in the URL, metres inside
                            rp.t[i] = float(q.pop(k)) / 1000.0
                if "heading" in q and state.get("map_prior") is not None:
                    state["map_prior"]["heading_deg"] = float(q.pop("heading")) % 360.0
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
            elif u.path == "/pose":
                # Name the station the frames from here on belong to. The
                # recorder has carried pose_id since it was written, but nothing
                # could ever set it, so every session so far went to disk
                # unlabelled and "which rows are pose N04" was reconstructed
                # afterwards from timestamps and memory. V3 plan section 5d
                # requires the id on the row itself.
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                video = state.get("video")
                if video is None:
                    # Not a 404: the request is well-formed and the operator
                    # believes a session is being labelled. Failing loudly here
                    # is the difference between noticing now and discovering at
                    # solve time that 36 stations share one nameless heap.
                    self.send_response(409)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": "not recording; start live.py with --record",
                    }).encode())
                elif "id" in q or "clear" in q:
                    pid = None if "clear" in q else q["id"]
                    video.set_pose(pid)
                    body = json.dumps({"pose": video.pose_id,
                                       "first_frame": video.frames,
                                       "stamped": len(video.meta_poses())})
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body.encode())
                else:
                    body = json.dumps({"pose": video.pose_id,
                                       "frames": video.frames,
                                       "poses": video.meta_poses()})
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body.encode())
            elif u.path == "/ui":
                # The page's one poll. Everything else here is for curl.
                body = json.dumps(ui_payload(pipe, state, time.time()))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            elif u.path == "/mapfix":
                # Solve now, on this handler thread, and remember the answer.
                res = run_map_fix(state)
                state["map_fix"] = res
                self.send_response(200 if res.get("ok") else 409)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res, allow_nan=False, default=str).encode())
            elif u.path == "/topdown":
                try:
                    td = topdown_view(state)
                except Exception as e:            # noqa: BLE001
                    td = {"error": "%s: %s" % (type(e).__name__, e)}
                self.send_response(200 if td else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(json.dumps(
                        td or {"error": "no map layer or priors not loaded"}).encode())
                except (BrokenPipeError, ConnectionResetError):
                    # The first top-down build may outlive a browser navigation
                    # or readiness probe. The computed cache is still valid;
                    # a client going away is not a live-system exception.
                    pass
            elif u.path == "/map":
                # Yael's map layer as the schema-1.0 contract, or the reason
                # there is none. 404 only when live.py was started without
                # --map: then "no map" is configuration, not a failure.
                m = state.get("map")
                if m is None:
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": "no map layer; start live.py with --map LAT,LON",
                    }).encode())
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(m).encode())
            elif u.path == "/health":
                checks = health(pipe, state, time.time())
                body = json.dumps({"worst": worst_level(checks), "checks": checks})
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
            elif u.path in ("/ai", "/students"):
                # The curl-side view of the three channels, with the same
                # numbers the page draws. `vis` on a thermal box is what it maps
                # to on the visible plane - null means it could not be placed,
                # and a consumer that logs a thermal box without it is logging a
                # coordinate in the wrong plane.
                st = state.get("students")
                body = json.dumps({
                    "available": st is not None,
                    "age_s": (round(time.time() - state["student_t"], 2)
                              if state.get("student_t") else None),
                    "gate_du_px": Renderer.FUSION_DU_PX,
                    "dedup_iou": Renderer.DEDUP_IOU,
                    "conf": {"thermal": round(pipe.ai_conf_thermal, 2),
                             "radar": round(pipe.ai_conf_radar, 2),
                             "engine_floor": round(pipe.ai_conf_floor, 2)},
                    "found": state.get("student_seen") or {},
                    "lock": {"on": pipe.ai_lock,
                             "coast_s": tracking.MAX_COAST_S,
                             "min_hits": tracking.MIN_HITS,
                             "static_px": tracking.STATIC_PX,
                             "tracks": state.get("tracks") or [],
                             "suppressed": state.get("tracks_static") or []},
                    "error": state.get("student_error"),
                    "channels": {
                        "thermal": {"on": pipe.ai_thermal,
                                    "list": state.get("student_thermal") or []},
                        "radar": {"on": pipe.ai_radar,
                                  "list": state.get("student_radar") or []},
                        "fusion": {"on": pipe.ai_fusion,
                                   "list": state.get("student_fused") or [],
                                   "rejected": state.get("student_fused_rejected") or []},
                    }})
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


def session_meta(args):
    """Everything about a recording that cannot be recovered from the data.

    The radar cannot be asked which config it is running and the config is
    sent by a separate tool, so the one claim that matters - which chirp
    profile and which RX phase table produced these points - is copied from
    the stamp send_radar_cfg.py leaves behind. It is carried with its own
    timestamp beside the recording's, because the check a reader has to make
    is not "is there a stamp" but "was it stamped BEFORE this session, with no
    power cycle in between". Changing the phase table moves boresight and
    voids any extrinsic solved on older data (frame_conventions.txt), and that
    is invisible in the points themselves.
    """
    stamp, stamp_note = None, 'no stamp: send_radar_cfg.py has not run since ' \
                              'this checkout, so the radar config is UNKNOWN'
    try:
        import send_radar_cfg
        if os.path.exists(send_radar_cfg.STAMP_PATH):
            with open(send_radar_cfg.STAMP_PATH) as f:
                stamp = json.load(f)
            age = time.monotonic() - stamp.get('sent_at_mono', 0)
            stamp_note = ('sent %.0f s before this session started; valid only '
                          'if the radar was not power-cycled since' % age)
            if age < 0:
                stamp_note = ('STAMP IS FROM A PREVIOUS BOOT (negative age): '
                              'the monotonic clock restarted, so this cannot '
                              'be the config now on the sensor')
    except Exception as e:                     # provenance never breaks a run
        stamp_note = 'stamp unreadable: %s' % e

    # The thermal window is what turns the recorded uint8 back into degrees:
    # c_per_lsb = (tmax - tmin) / 255. Every session before 2026-08-19 omitted
    # it, which is why perception/out/gexport v1 carries c_per_lsb: null and
    # its thermal plane is intensity, not temperature. The default matches the
    # bring-up default in Streamer._setup (fixed_range or (-10, 140)); if that
    # pair ever moves, this one must move with it or the meta lies.
    if args.range:
        tmin_c, tmax_c = (int(v) for v in args.range.split(':'))
    else:
        tmin_c, tmax_c = -10, 140

    # --raw16 changes the unit and not merely the width, so everything that
    # describes how a count becomes a temperature has to move with it. Leaving
    # the uint8 block in place would declare 0.588C steps over a plane whose
    # steps are 0.01C, and a reader has no way to notice: both are plausible
    # numbers. This is the same failure that shipped gexport v1 with c_per_lsb
    # null and turned its thermal plane into intensity.
    if args.raw16:
        thermal_scale = {
            'thermal_dtype': 'uint16_le',
            # Absolute, straight off the sensor's TLinear output. Nothing to
            # undo: C = count / 100 - 273.15, whatever tmin/tmax happen to be.
            'thermal_encoding': 'radiometric_centikelvin',
            'c_per_lsb': 0.01,
            'thermal_counts_max': 65535,
        }
    else:
        thermal_scale = {
            'thermal_dtype': 'uint8',
            'thermal_encoding': 'linear_set_range',
            'c_per_lsb': (tmax_c - tmin_c) / 255.0,
            'thermal_counts_max': 255,
        }

    def _sha(path):
        # Calibration files are small (a LUT is ~1 MB); hashing at session
        # start is the only moment the file on disk is KNOWN to be the file
        # the session ran with.
        if not path or not os.path.exists(path):
            return None
        import hashlib
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    detector_engine = args.detect_engine
    if (args.detect != 'off' and args.detect_backend != 'cpu' and
            detector_engine is None):
        try:
            import trt_detect
            detector_engine = trt_detect.model_paths(args.detect_model)[1]
        except Exception:
            detector_engine = None

    # Which student engines this session ran, hashed for the same reason the
    # detector engine is: an engine is rebuilt in place whenever TensorRT or the
    # GPU moves under it, and "the v2 students" names a file, not a model.
    student_engines = {}
    if args.students:
        try:
            import trt_students
            for name in ('thermal_student.engine', 'radar_student.engine'):
                path = os.path.join(trt_students.ENGINE_DIR, name)
                student_engines[name] = {'path': os.path.abspath(path),
                                         'sha256': _sha(path)}
        except Exception:                      # provenance never breaks a run
            student_engines = {}

    return {
        'tool': 'live.py',
        'argv': sys.argv[1:],
        'board_port': args.port,
        'radar_port': args.radar,
        'channel': args.channel,
        'view': args.view,
        'warp_lut': args.warp,
        'warp_lut_sha256': _sha(args.warp),
        'radar_calib': args.radar_calib,
        'radar_calib_sha256': _sha(args.radar_calib),
        'detector': args.detect,
        'detector_model': (args.detect_model if args.detect != 'off' else None),
        'detector_engine': detector_engine,
        'detector_engine_sha256': _sha(detector_engine),
        'students': bool(args.students),
        'student_conf': args.student_conf if args.students else None,
        # What the session STARTED with. The channels are switchable live, so a
        # reader wanting to know what was on screen at frame N has to read the
        # picture, not this - but the engines and the threshold cannot change
        # under a running session, and those are what a label depends on.
        'ai_channels_at_start': args.ai_channels if args.students else None,
        'student_engines': student_engines or None,
        'radar_cfg_stamp': stamp,
        'radar_cfg_stamp_note': stamp_note,
        # Still recorded under --raw16, and still true: the window goes on
        # deciding the board's own 8-bit frame, which live.py reproduces for
        # the pipeline. It just stops being what turns a count into a degree.
        'tmin': tmin_c,
        'tmax': tmax_c,
        **thermal_scale,
        # The bring-up has run the Lepton in HIGH gain since 2026-08-09
        # (capture._BRINGUP, SET_MODE(True, False)); recorded so a future
        # low-gain session cannot be silently mixed in as the same scale.
        'lepton_gain': 'high',
    }


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
    ap.add_argument("--thermal-noise-mc", type=int, default=148, metavar="MC",
                    help="thermal temporal-filter noise estimate in milli-Celsius; "
                         "0 disables it (default 148, measured on this Lepton)")
    ap.add_argument("--thermal-frames", type=int, default=8, metavar="N",
                    help="equivalent frames averaged by the motion-adaptive thermal "
                         "filter; 1 disables smoothing (default 8)")
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
    ap.add_argument("--quality", type=int, default=70,
                    help="board-side JPEG quality (default 70 leaves CDC headroom; "
                         "80 was measured dropping 512-byte USB tails)")
    # Costs 18.75KB/frame more on the wire - 164KB/s at 8.772fps, taking a q80
    # session from ~325 to ~490KB/s against a link measured at 700-900. Sending
    # it is a swap, not an addition: the 8-bit plane is rebuilt on this side
    # from the same words, so nothing downstream sees a difference. Needs the
    # raw-passthrough firmware; an older build fails the bring-up with a clear
    # message rather than quietly streaming 8-bit.
    ap.add_argument("--raw16", action="store_true",
                    help="stream the Lepton's 16-bit radiometric frame (0.01C "
                         "per code) instead of the board's 8-bit plane, and "
                         "record it as the session's thermal.bin")
    ap.add_argument("--range", metavar="TMIN:TMAX",
                    help="pin the sensor range instead of auto-ranging, e.g. "
                         "--range 10:45. Auto-range picks off ONE frame at "
                         "bring-up and the sensor drifts for minutes after; pin "
                         "it when the session must outlast that (calibration, "
                         "long recordings) or to measure the drift itself")
    ap.add_argument("--detect", default="off", metavar="WHAT",
                    help="object detection: 'off', 'all' for COCO-80, or a comma-separated "
                         "class list such as 'person,cat'. Attaches a temperature to "
                         "every box")
    ap.add_argument("--detect-backend", default="auto", choices=["auto", "gpu", "cpu"],
                    help="'gpu' runs the selected TensorRT model; 'cpu' runs "
                         "yolov4-tiny with cv2.dnn. 'auto' prefers GPU")
    ap.add_argument("--detect-model", default="yolov10n",
                    choices=["yolov10n", "yolov8n", "yolo11n"],
                    help="TensorRT detector model (GPU only; default yolov10n)")
    ap.add_argument("--detect-engine", metavar="PATH",
                    help="override the registered TensorRT engine path")
    ap.add_argument("--detect-size", type=int, default=416, choices=[320, 416],
                    help="detector input size, CPU backend only. 416 is 68ms and 320 is "
                         "46ms on this host, against a 114ms frame period. The GPU "
                         "backend is fixed at 640 by its engine (default 416)")
    ap.add_argument("--detect-conf", type=float, default=0.35,
                    help="detection confidence threshold (default 0.35)")
    # Radar overlay. Off unless a port is given, because live.py must keep
    # working on a rig that has no radar attached.
    ap.add_argument("--radar", nargs="?", const=radar_overlay.DATA_PORT, default=None,
                    metavar="PORT",
                    help="overlay IWR1843 detections from this DATA port "
                         "(default %s when the flag is given bare)" % radar_overlay.DATA_PORT)
    ap.add_argument("--record", nargs="?", const="", default=None, metavar="DIR",
                    help="record the session: session.mp4 (clean of overlays), "
                         "frames.jsonl and thermal.bin - plus radar.bin and "
                         "radar.jsonl when --radar is on, which is what an offline "
                         "calibration is solved against. A bare --record picks "
                         "captures/live-<timestamp>")
    ap.add_argument("--radar-record", metavar="DIR",
                    help="deprecated alias for --record DIR")
    ap.add_argument("--radar-hfov", type=float, default=70.0, metavar="DEG",
                    help="assumed horizontal FOV used to guess the focal length when no "
                         "solved intrinsics exist (default 70). DESIGN.md:239 records this "
                         "as UNVERIFIED and the measured triple implies 63.8")
    ap.add_argument("--radar-ai", action="store_true",
                    help="draw the radar person/clutter classifier (perception/out/radar_ai/"
                         "cluster_model_v0.pkl) on the picture: green ring = PERSON")
    ap.add_argument("--radar-calib", metavar="JSON",
                    help="solved intrinsics/extrinsics to project with, instead of the guess")
    ap.add_argument("--channel", default="fusion", choices=list(CHANNELS),
                    help="operator display channel: thermal, rgb_radar (currently "
                         "visible luma), thermal_radar, fusion, or ai")
    ap.add_argument("--view", default="fused",
                    choices=list(VIEWS),
                    help="view to start in, and therefore what gets recorded. "
                         "For picking a pixel off the recording, see the note in "
                         "radar_correspond.py: the thermal layer is NOT registered "
                         "until a warp LUT exists, so a pixel taken from the thermal "
                         "content carries that unknown offset into the extrinsic")
    ap.add_argument("--students", action="store_true",
                    help="run the trained thermal+radar person students "
                         "(TensorRT engines from perception/out/gexport/v6/"
                         "models/) as extra detection channels: orange boxes "
                         "= thermal student, cyan = radar student, white = the "
                         "two of them agreeing. Pick which channels are on with "
                         "--ai-channels, or live from the page")
    ap.add_argument("--student-conf", type=float, default=0.5, metavar="C",
                    help="confidence threshold for both students")
    ap.add_argument("--students-dir", metavar="DIR",
                    help="load the student engines from this directory "
                         "instead of perception/out/gexport/v6/models. The "
                         "engines belong to the export that trained them; the "
                         "session meta records which ones actually loaded")
    ap.add_argument("--no-lock", action="store_true",
                    help="start with the person lock off, so every frame's "
                         "boxes stand on that frame alone. The lock is on by "
                         "default and switchable live (key l, /set?lock=0)")
    ap.add_argument("--ai-channels", default="fusion", metavar="LIST",
                    help="which AI channels start switched on: any of "
                         "thermal,radar,fusion (or 'none'). All three are "
                         "switchable live from the page and over "
                         "/set?ai_thermal=0&ai_radar=1&ai_fusion=1, so this only "
                         "picks what the first frame shows")
    ap.add_argument("--map", metavar="LAT,LON", default=None,
                    help="initialise Yael's map layer (mapinit) at this WGS84 "
                         "position: geoid undulation, ground/surface height, "
                         "building priors. Served on /map, graded in /health. "
                         "Runs in the background; the viewer never waits on it")
    ap.add_argument("--map-geoid", type=float, nargs=2, metavar=("LOW", "HIGH"),
                    default=None, help="assert the geoid undulation is in this "
                                       "range (m) - Jerusalem is 19..20.5")
    ap.add_argument("--map-heading", type=float, default=None, metavar="DEG",
                    help="compass heading prior for the map fix (0 = north); "
                         "changeable live with /set?heading=. Without it /mapfix "
                         "cannot run: heading has no other source on this rig")
    ap.add_argument("--map-sigma", type=float, nargs=2, default=(5.0, 5.0),
                    metavar=("M", "DEG"), help="how good the --map position and "
                    "--map-heading are believed to be (1 sigma). Be honest: the "
                    "fix searches 3 sigma and no further")
    ap.add_argument("--mapinit-dir", default=None, metavar="DIR",
                    help="directory holding Yael's mapinit/ package (default: "
                         "$VALISIGHT_MAPINIT, then a ValiSight_yael checkout "
                         "beside this repo)")
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

    # Set before anything imports trt_students: the module reads it at import
    # time, and a later assignment would silently load the default engines.
    if args.students_dir:
        os.environ["STUDENT_ENGINES"] = os.path.abspath(args.students_dir)

    pipe = Pipeline(args)
    pipe.channel = args.channel
    pipe.view = args.view

    # Validated here rather than shrugged off later: a typo in --ai-channels
    # would otherwise silently switch a channel off for a whole session, and a
    # channel that is off is indistinguishable on screen from a channel that is
    # on and finding nobody.
    want_ai = {c.strip() for c in args.ai_channels.split(",") if c.strip()} - {"none"}
    unknown_ai = want_ai - {"thermal", "radar", "fusion"}
    if unknown_ai:
        raise SystemExit("--ai-channels: not a channel: %s (thermal, radar, "
                         "fusion, none)" % ", ".join(sorted(unknown_ai)))
    pipe.ai_thermal = "thermal" in want_ai
    pipe.ai_radar = "radar" in want_ai
    pipe.ai_fusion = "fusion" in want_ai
    # The engines are built with --student-conf and cannot go below it later,
    # so that is where both sliders start and where they stop going down.
    pipe.ai_lock = not args.no_lock
    pipe.ai_conf_floor = args.student_conf
    pipe.ai_conf_thermal = pipe.ai_conf_radar = args.student_conf
    # One sampler for the process; it primes its own counters, so the first
    # /ui already carries a CPU figure rather than a null.
    state = {"soc": hostsoc.Soc(), "detector_available": False}

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
            detector = detect.make_detector(
                args.detect_backend, size=args.detect_size,
                conf=args.detect_conf, classes=classes,
                model=args.detect_model, engine=args.detect_engine)
        except (FileNotFoundError, RuntimeError) as e:
            # The stream, the radar and the thermal measurement do not need the
            # detector; losing all of them to a missing weights file or a GPU
            # that would not come up turned every detector fault into a dead
            # system (2026-08-23). Degrade loudly instead.
            print("detector unavailable, RUNNING WITHOUT DETECTION: %s" % e,
                  file=sys.stderr)
            detector = None
            args.detect = "off"
        if detector:
            # Session provenance must describe the backend that actually won.
            # In auto mode this may be the CPU fallback, not the requested engine.
            args.detect_backend = detector.backend
            args.detect_model = detector.model_name
            args.detect_engine = getattr(detector, "engine_path", None)
    state["detector_available"] = detector is not None

    work = Latest()
    radar = radar_proj = video = None

    record_dir = args.record if args.record is not None else args.radar_record
    if args.radar_record and args.record is None:
        print("--radar-record is now --record (recording no longer needs the "
              "radar); kept as an alias", file=sys.stderr)
    if record_dir == "":                            # bare --record
        record_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "captures",
            time.strftime("live-%Y%m%d-%H%M%S"))
    if record_dir:
        video = recorder.SessionRecorder(record_dir, meta=session_meta(args))
        state["recording"] = record_dir
        # Also in state, beside radar_proj: the recorder is owned by the
        # Renderer thread, but /pose is served on the HTTP thread and has no
        # other way to reach it. set_pose only appends to a list and rewrites
        # meta.json, so the cross-thread call is a write the capture path never
        # races on - it reads pose_id, and a str assignment is atomic.
        state["video"] = video
        print("recording to %s/" % record_dir, file=sys.stderr)

    if args.radar:
        radar = radar_overlay.RadarReader(args.radar, record_dir=record_dir)
        radar.start()
        radar_proj = radar_overlay.Bootstrap(OUT_W, OUT_H, hfov_deg=args.radar_hfov,
                                             calib_path=args.radar_calib)
        state["radar_proj"] = radar_proj
        print("radar overlay: %s, intrinsics from %s, f=%.1f px"
              % (args.radar, radar_proj.source, radar_proj.f), file=sys.stderr)
        if not args.radar_calib:
            print("  the projection is a GUESS until radar_extrinsics is solved - "
                  "nudge it with /set?yaw=..&pitch=..&tz=..", file=sys.stderr)

    # The thermal->visible mapper, loaded once for everything that needs it:
    # the person outline on the detector's boxes, the student channels, and
    # the fusion ring. Independent of --students on purpose - the outline is
    # worth having on a run with no students at all - and never fatal: a bad
    # LUT costs the outline, not the viewer.
    th2vis = None
    if args.warp:
        try:
            import trt_students
            th2vis = trt_students.ThermalToVisible(args.warp)
        except Exception as e:
            print("person outline unavailable (%s: %s) - boxes only"
                  % (type(e).__name__, e), file=sys.stderr)

    students = None
    if args.students:
        # The students are an OPT-IN extra: any failure here (missing engine,
        # GPU not up, bad LUT) must leave the plain pipeline running.
        try:
            import trt_students
            th_model = trt_students.ThermalStudentTrt(conf=args.student_conf)
            rd_model = (trt_students.RadarStudentTrt(conf=args.student_conf)
                        if args.radar else None)
            if not args.range:
                print("students: --range is not pinned, so the thermal input "
                      "is scene-relative instead of the Celsius the model was "
                      "trained on - expect degraded detections. Use the "
                      "training range (e.g. --range 0:60).", file=sys.stderr)
            if th2vis is None:
                print("students: no --warp LUT, thermal-student boxes cannot "
                      "be drawn on the visible frame (still served on "
                      "/state)", file=sys.stderr)
            # Same range->degrees mapping session_meta() documents: the
            # recorded uint8 is (tmax - tmin) / 255 per count above tmin.
            if args.range:
                s_tmin, s_tmax = (int(v) for v in args.range.split(':'))
            else:
                s_tmin, s_tmax = -10, 140
            students = {"thermal": th_model, "radar": rd_model,
                        "th2vis": th2vis,
                        "c_per_lsb": (s_tmax - s_tmin) / 255.0,
                        "tmin": float(s_tmin)}
            print("students: thermal%s engine(s) up, conf>=%.2f, channels on: %s"
                  % ("+radar" if rd_model else "", args.student_conf,
                     "+".join(sorted(want_ai)) or "none"),
                  file=sys.stderr)
            if pipe.ai_fusion and (rd_model is None or th2vis is None):
                print("students: the fusion channel cannot pair without %s - it "
                      "will draw nothing" % ("a radar student engine"
                                             if rd_model is None else "a warp LUT"),
                      file=sys.stderr)
        except Exception as e:
            print("students: DISABLED (%s: %s)" % (type(e).__name__, e),
                  file=sys.stderr)

    # The HTTP threads reach the students only through state: /ui hides the
    # card when this is None, and health() asks it whether a channel that is
    # switched on can actually draw.
    state["students"] = students

    if args.map:
        start_map_init(state, args.map, args.map_geoid, args.mapinit_dir)
        lat, lon = (float(v) for v in args.map.split(","))
        state["map_prior"] = {"latitude_deg": lat, "longitude_deg": lon,
                              "heading_deg": args.map_heading,
                              "sigma_position_m": float(args.map_sigma[0]),
                              "sigma_heading_deg": float(args.map_sigma[1])}
        state["map_geoid_range"] = tuple(args.map_geoid) if args.map_geoid else None
        state["mapinit_dir"] = args.mapinit_dir

    render = Renderer(pipe, state, work, detector, radar=radar,
                      radar_proj=radar_proj, video=video, students=students,
                      th2vis=th2vis)
    render.radar_ai = bool(args.radar_ai)
    render.start()
    stream = Streamer(args.port, pipe, args.quality, state, work, raw16=args.raw16)
    if args.range:
        lo, hi = (int(v) for v in args.range.split(":"))
        stream.fixed_range = (lo, hi)
        # Told, not inferred: the board reports its own range on #READY, but the
        # host must know NOW that the tone is absolute rather than scene-relative,
        # because temp_at() converts codes with it.
        pipe.set_range(lo, hi)
        print("range pinned to %d..%d C (auto-range off)" % (lo, hi), file=sys.stderr)
    stream.start()

    srv = ThreadingHTTPServer(("0.0.0.0", args.http), make_handler(state, pipe))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("live fusion on http://localhost:%d  (ctrl-c to stop)" % args.http, file=sys.stderr)
    if not args.warp:
        print("NOTE: no --warp, thermal layer is stretched not registered", file=sys.stderr)
    if detector:
        gpu = getattr(detector, "backend", "cpu") == "gpu"
        print("detecting %s at %d px (%s)"
              % (args.detect, detector.net_w if gpu else args.detect_size,
                 detector.model_name + "/TensorRT" if gpu else
                 detector.model_name + "/CPU"), file=sys.stderr)

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
    warned_no_radar = False
    # `run_live.sh` starts this process under nohup/setsid.  A background shell
    # may leave SIGINT ignored in that child, and Python deliberately preserves
    # an inherited SIG_IGN disposition.  The launcher used to wait, conclude
    # that live.py ignored it, and SIGKILL the process; SessionRecorder.close()
    # then never ran and the MP4 had no moov/index atom.  Explicit handlers make
    # both the launcher's SIGINT and an ordinary SIGTERM request the same clean
    # exit through the finally block below.
    stop_requested = threading.Event()

    def request_stop(_signum, _frame):
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stop_requested.wait(0.2):
            # A calibration recording with zero radar frames is a wasted
            # session that looks fine on screen (the camera side records
            # happily). The usual cause on this bench: a power cycle wiped the
            # IWR1843's config and nobody re-sent it. Say so on the console,
            # where the person who just typed the record command is looking.
            # Judged on the READER's own counter, not state["radar_frames"]:
            # that one is written by the renderer, which sits idle through the
            # ~15 s camera bring-up, and the first version of this check fired
            # a false alarm through exactly that window.
            if (radar is not None and record_dir and not warned_no_radar
                    and time.time() - t0 > 15 and radar.frames == 0):
                print("WARNING: recording with --radar but 0 radar frames "
                      "after 10s - the radar is probably unconfigured (a "
                      "power cycle wipes it). Run ./send_radar_cfg.py, then "
                      "restart this recording.", file=sys.stderr)
                warned_no_radar = True
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
                      "batches %d, batch drops %d, resyncs %d, restarts %d, "
                      "frames flowing: %s" % (
                          state.get("fps", 0.0), state.get("frames", 0),
                          state.get("rendered", 0), state.get("dropped", 0),
                          state.get("bad_jpeg", 0), state.get("batches", 0),
                          state.get("batch_drops", 0),
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
        if radar is not None:
            radar.stop.set()
        # The render thread owns the recorder, and the mp4 is only finalized by
        # its close() - so wait for the thread rather than letting the daemon
        # flag kill it mid-write.
        render.join(timeout=3.0)
        # Streamer.release() drains and interrupts the raw REPL in its own
        # finally block.  Give that cleanup a bounded chance to finish so a
        # replacement viewer does not inherit a board still writing payload.
        stream.join(timeout=5.0)
        if radar is not None:
            radar.join(timeout=2.0)
            radar.close()
        if video is not None and video.frames:
            print("recorded %d frames -> %s/" % (video.frames, record_dir),
                  file=sys.stderr)
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    sys.exit(main())
