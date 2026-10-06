"""Frame handoff and rendering worker, independent of board serial reads."""
import threading
import time

import cv2
import numpy as np

import detect
import egomotion
import live_channels
import radar_overlay
import thermal_io
import tracker as tracking
from .constants import OUT_W, OUT_H, TH_W, TH_H, DETECT_STALE_S
from .display import agc8, compose_channel

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
    # box covering most of the entire sensor is ambiguous: it may be the
    # detector calling the scene/doorway a person. That failure was observed as a
    # 613x392 box on this 640x400 stream, and once admitted it permanently gave
    # a thermal/radar clutter track visible-detector trust. Edge-clipped close-up
    # candidates are retained below, but only as weak tracking evidence.
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
                    if self._close_clipped_person(d):
                        d = dict(d, geometry_warning="close clipped person candidate")
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
    def _close_clipped_person(cls, d):
        """Keep a partial near-camera person as evidence, not instant track trust.

        A torso can be landscape when both head and legs leave the frame.
        Require clipping at the vertical edges and one horizontal edge, with
        visible background remaining on the opposite side. The old almost-full
        doorway box has no such remaining side and still fails the guard.
        """
        if d.get("cls") != "person":
            return False
        x, y, w, h = (float(d.get(k, 0)) for k in ("x", "y", "w", "h"))
        if h <= 0 or w * h < cls.MAX_LANDSCAPE_PERSON_AREA * OUT_W * OUT_H:
            return False
        if w / h < cls.MIN_LANDSCAPE_PERSON_ASPECT:
            return False
        vertical_crop = y <= .05 * OUT_H and y + h >= .95 * OUT_H
        side_crop = ((x >= .08 * OUT_W and x + w >= .95 * OUT_W)
                     or (x <= .05 * OUT_W and x + w <= .92 * OUT_W))
        return vertical_crop and side_crop

    @classmethod
    def _detector_source(cls, d):
        if (float(d.get("conf", 0)) >= cls.DET_INSTANT_VOUCH_CONF
                and not cls._close_clipped_person(d)):
            return "det"
        return "det_weak"

    @classmethod
    def _reject_person_geometry(cls, d):
        """Reason an obviously scene-sized person box is unusable, else None.

        This is deliberately one-sided and narrow.  It does not impose a
        textbook standing-person aspect ratio: seated people, partial bodies
        and close targets are all in scope.  It only refuses a box that is both
        landscape and covers most of the complete image, except an edge-clipped
        near-camera candidate retained as weak tracking evidence.
        """
        if d.get("cls") != "person":
            return None
        w, h = float(d.get("w", 0)), float(d.get("h", 0))
        if w <= 0 or h <= 0:
            return "empty person box"
        area = w * h / float(OUT_W * OUT_H)
        if (area >= cls.MAX_LANDSCAPE_PERSON_AREA
                and w / h >= cls.MIN_LANDSCAPE_PERSON_ASPECT
                and not cls._close_clipped_person(d)):
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
    def _draw_shape(rgb, mask, col, thick, grow=0, box=None):
        """Draw a cleaned thermal boundary; refuse partial/fragmented masks.

        This is display-only: the measurement mask used by tracking and heat
        checks stays untouched. A thermal boundary is not person segmentation.
        """
        m = mask.astype(np.uint8).copy()
        if box is not None:
            x, y, w, h = map(int, box)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(m.shape[1], x + w), min(m.shape[0], y + h)
            if x1 - x0 < 12 or y1 - y0 < 20:
                return False
            roi = m[y0:y1, x0:x1].copy()
            m[:] = 0
            m[y0:y1, x0:x1] = roi
        # Remove narrow thermal-grid spurs before bridging small holes. Work
        # at display resolution so we never enlarge a coarse 4px staircase.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return False
        largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
        area = stats[largest, cv2.CC_STAT_AREA]
        if area < 40 or area < 0.8 * stats[1:, cv2.CC_STAT_AREA].sum():
            return False
        m = (labels == largest).astype(np.uint8)
        if box is not None:
            bw, bh = x1 - x0, y1 - y0
            sx, sy, sw, sh, _ = stats[largest]
            # A head/legs-only patch must not replace a full-person box.
            # Do not impose standing-person proportions: seated people count.
            if sh < 0.65 * bh or sw < 0.3 * bw or not 0.15 <= area / (bw * bh) <= 0.88:
                return False
            roi = m[y0:y1, x0:x1]
            borders = (roi[:2].mean(), roi[-2:].mean(),
                       roi[:, :2].mean(), roi[:, -2:].mean())
            if sum(v > 0.5 for v in borders) >= 3:
                return False
        m = (cv2.GaussianBlur(m.astype(np.float32), (7, 7), 1.4) >= 0.5).astype(np.uint8)
        if grow:
            k = 2 * grow + 1
            m = cv2.dilate(m, np.ones((k, k), np.uint8))
        cnts = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
        if not cnts:
            return False
        contour = max(cnts, key=cv2.contourArea)
        # Bounded simplification removes pixel-scale zigzags without inventing
        # anatomy or rounding a seated person into a standing template.
        contour = cv2.approxPolyDP(contour, 1.5, True)
        cv2.drawContours(rgb, [contour], -1, col, thick, cv2.LINE_AA)
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
        started = time.perf_counter()
        self.state["student_error"] = None
        decisions = {}
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
        decisions["thermal"] = ("off" if not want_t else
                                "waiting for valid thermal frame")
        decisions["radar"] = ("off" if not want_r else
                              "engine unavailable" if s["radar"] is None else
                              "waiting for fresh radar frame")
        if want_t and thermal is not None and len(thermal) == 160 * 120:
            try:
                th = (np.frombuffer(thermal, np.uint8)
                      .astype(np.float32).reshape(120, 160)
                      * s["c_per_lsb"] + s["tmin"])
                tdets = s["thermal"].push(th, time.monotonic() * 1e3)
                decisions["thermal"] = ("processed" if getattr(s["thermal"], "primed", True)
                                        else "warming up temporal history")
            except Exception as e:
                decisions["thermal"] = "inference failed"
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
                decisions["radar"] = "processed"
            except Exception as e:
                decisions["radar"] = "inference failed"
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
        self.state["student_diagnostics"] = {
            "cycle": self.state.get("student_diagnostics", {}).get("cycle", 0) + 1,
            "ms": round((time.perf_counter() - started) * 1000, 2),
            "decisions": decisions,
            "found": {"thermal": seen_t, "radar": seen_r},
            "thermal": tdets, "radar": rdets, "fusion": fused,
            "rejected": self.state["student_fused_rejected"],
            "error": self.state["student_error"],
            "t": self.state["student_t"],
        }

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
                        rgb, m, self.STUDENT_TH_COL, 2 if strong else 1,
                        box=(x, y, w, h)):
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
                                                 2, grow=self.HALO_PX,
                                                 box=(d["x"], d["y"], d["w"], d["h"])):
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
    # Static candidates use a darker green than a confirmed person. This keeps
    # the operator's requested green visual language while preserving an
    # immediate distinction from LOCK_COL's bright verified lock.
    STATIC_COL = (0, 150, 0)
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
    # Measured on smoke_dark_20260826-224846: the recurring smoke/lamp edge
    # peaked only 0.7 C above the frame background while the person was
    # 6.8..7.3 C above it.  Shape and confidence alone cannot separate those
    # cases (the V6 student called both a person at up to 1.00), so only a
    # radiometrically distinct component may create/promote a thermal lock.
    # This is a tracker evidence gate; the V6 engine and diagnostic drawing
    # remain untouched until a retrained model replaces them.
    THERMAL_LOCK_MIN_DELTA_C = 3.0
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
                src = self._detector_source(d)
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
        background_c = None
        if plane is not None and self.students is not None:
            scale = self.students.get("c_per_lsb")
            offset = self.students.get("tmin")
            if scale is not None and offset is not None:
                # Median is deliberately global and robust: a person-sized
                # component cannot pull it toward its own peak, unlike a frame
                # mean.  Use the student's uint8 input plane, whose scale is
                # part of the loaded engine contract, even when a raw16 plane
                # is available for drawing a finer silhouette.
                raw8 = np.frombuffer(thermal, np.uint8)
                background_c = (float(np.median(raw8)) * float(scale)
                                + float(offset))
        if self.pipe.ai_thermal or self.pipe.ai_fusion:
            for d in self.state.get("student_thermal") or []:
                if d.get("vis"):
                    x, y, w, h = d["vis"]
                    mask = (self.th2vis.shape_in(plane, x, y, w, h)
                            if plane is not None else None)
                    shape_ok, shape_metrics = self._thermal_human_shape(
                        mask, (x, y, w, h))
                    d["human_shape"] = shape_ok
                    d["shape_metrics"] = shape_metrics
                    temp = self.pipe.temp_region(x, y, w, h)
                    if temp is not None and temp.get("samples", 0):
                        d["max_c"] = round(temp["max"], 1)
                        d["mean_c"] = round(temp["mean"], 1)
                    delta_c = (float(temp["max"]) - background_c
                               if temp is not None
                               and temp.get("samples", 0)
                               and background_c is not None else None)
                    radiometric_body = (delta_c is not None
                                        and delta_c
                                        >= self.THERMAL_LOCK_MIN_DELTA_C)
                    human_shape = (d["conf"] >= self.THERMAL_SHAPE_CONF
                                   and shape_ok and radiometric_body)
                    static_vouch = (d["conf"] >= self.THERMAL_STATIC_CONF
                                    and shape_ok and radiometric_body)
                    d["background_c"] = (round(background_c, 1)
                                          if background_c is not None else None)
                    d["delta_c"] = (round(delta_c, 1)
                                    if delta_c is not None else None)
                    d["radiometric_body"] = radiometric_body
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
            return m is not None and self._draw_shape(
                rgb, m, col, 2, box=(d["x"], d["y"], d["w"], d["h"]))
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
