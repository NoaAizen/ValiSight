"""ctypes ABI and synchronized access to the shared C fusion pipeline."""
import ctypes
import threading

import cv2
import numpy as np

from .constants import LIB, OUT_W, OUT_H, TH_W, TH_H
from .display import display_sharpen

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
                    "%s is %d bytes in libfusion.so but %d in viewer/native.py - the ctypes mirror "
                    "has drifted from fusion.h. Fix it before running; the layouts must "
                    "match field for field." % (name, theirs, mine))

        # fusion_process runs on the streamer thread while temperature queries
        # come off HTTP handler threads, and both touch t_prep and the warp
        # table. The queries are microseconds against a ~10ms frame, so a plain
        # lock costs nothing measurable and removes a torn read that would
        # otherwise surface as an occasional nonsense temperature.
        self.lock = threading.Lock()
        # Display enhancement only; never feed this back to detectors,
        # temperature measurements or the raw sensor recordings.
        self.visible_detail = 40
        self.thermal_detail = 35
        self.display_clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

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
        # Clean detection boxes are the operator default. Thermal boundaries
        # can include warm furniture and omit clothing; expose them only when
        # the operator explicitly enables the thermal-outline overlay.
        self.ai_silhouette = False
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
        codes = display_sharpen(treg, self.thermal_detail, valid=cover)
        interpolation = cv2.INTER_CUBIC if self.thermal_detail else cv2.INTER_LINEAR
        # Normalised interpolation prevents uncovered (zero) cells from making
        # an artificial cold fringe inside the thermal footprint.
        if self.thermal_detail:
            weight = cv2.resize(cover.astype(np.float32), (OUT_W, OUT_H),
                                interpolation=interpolation)
            full = cv2.resize(codes.astype(np.float32) * cover, (OUT_W, OUT_H),
                              interpolation=interpolation) / np.maximum(weight, 1e-6)
            values = codes[cover.astype(bool)]
            lo, hi = (float(values.min()), float(values.max())) if values.size else (0, 255)
            full = np.clip(np.rint(full), lo, hi).astype(np.uint8)
        else:
            full = cv2.resize(codes, (OUT_W, OUT_H), interpolation=interpolation)
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


