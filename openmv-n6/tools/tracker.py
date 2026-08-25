#!/usr/bin/env python3
"""Lock onto a person and keep the lock across frames and across sensors.

The detector answers one question per frame - "is there a person in THIS
picture" - and answers it independently every time. That is the right contract
for a detector and the wrong one for an operator: a person who is found, missed
for two frames because they turned side-on or walked behind a chair, and found
again, is one person the whole time, and a box that blinks off says the opposite.

So this keeps identity between frames. A track is born from detections, moved
by its own velocity when nothing arrives, and kept for a bounded time after the
last measurement - drawn, by whoever is drawing, as visibly coasting, because a
predicted box is a claim about where somebody probably is and must never look
like a claim about where they were seen.

Three things feed it: the COCO detector on the visible frame, the thermal
student, and the radar student. They fail at different times - the visible one
in glare and low light, the thermal one when a person is the same temperature as
what is behind them, the radar one when they stop moving - and any of them can
hold a lock the others lost. Which ones are holding it is carried on the track,
because "still locked" and "still locked BY THE RADAR ALONE" are different
statements about how much to trust the box.

No dependencies beyond the standard library: this runs inside the render thread,
between a frame arriving and the JPEG going out.
"""

# Association gates. IoU first because it is scale-aware; the centre-distance
# fallback catches the case IoU cannot - a fast walker at 8.7 fps whose boxes
# no longer overlap at all, which is exactly when losing the lock is most
# visible and least excusable.
IOU_GATE = 0.2
CENTRE_GATE = 0.7          # x mean box width

# How long a track survives with no measurement at all. 1.5 s is about thirteen
# frames on this stream: long enough to cross behind a chair, short enough that
# a coasted box cannot outlive the walk it belonged to.
MAX_COAST_S = 1.5

# Measurements needed before a track is drawn. One frame of a false positive on
# a warm doorway is exactly what this exists to swallow.
MIN_HITS = 2

# What a track has to do before it is drawn, when the visible detector has
# never confirmed it: MOVE. Measured on captures/test6, the lobby in the
# screenshots: the lit glass door reads 31.7 C mean / 34.1 C p90 and a person
# reads 30.9 / 34.1 - the same temperature to within the sensor's noise. No
# radiometric threshold can separate them and none is attempted here. What does
# separate them is that a door has never moved and never will, so a candidate
# the detector has not vouched for has to show displacement before it is
# treated as a person. This is D2's continuity lesson (occupancy cannot tell
# furniture from somebody's favourite pausing spot; continuity can) applied to
# the image plane instead of the radar grid.
#
# The threshold is generous at range and scale-aware up close: 20 px is under
# half a second of walking at 15 m, and a quarter of the box width is a fifth
# of a second at 2 m.
STATIC_PX = 20.0
STATIC_FRAC = 0.25

# Alpha-beta smoothing. Position is trusted (alpha high) because the boxes are
# already what the operator wants to see; velocity is not (beta low) because a
# jittery velocity estimate turns a coast into a box sliding off the person.
ALPHA = 0.7
BETA = 0.15


# Which sensor's geometry wins when several describe the same person. The
# detector first: it is the one measured against ground truth on this rig and
# the one the operator already reads as authoritative. Radar last: its box is a
# guess off a two-element aperture in elevation.
SOURCE_RANK = ("det", "fusion", "thermal", "radar")


def _as_set(src):
    return {src} if isinstance(src, str) else set(src)


def iou(a, b):
    """Intersection over union of two (x, y, w, h) boxes."""
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = float(iw * ih)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _centre_ok(a, b):
    """True when two boxes' centres are close relative to their width."""
    acx, acy = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
    bcx, bcy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
    reach = CENTRE_GATE * (a[2] + b[2]) / 2.0
    return abs(acx - bcx) <= reach and abs(acy - bcy) <= max(reach, 0.5 * a[3])


def merge(obs):
    """Collapse observations of the SAME person from DIFFERENT sensors.

    Without this, a person seen by the detector, the thermal student and the
    fusion channel arrives as three observations, the tracker gives one of them
    to the existing track and starts two new ones, and the picture grows the
    duplicates this whole layer exists to prevent.

    Two observations from the SAME sensor are never merged. A sensor reporting
    two boxes is the strongest evidence there is that there are two people,
    and it is not this function's place to overrule it.
    """
    order = sorted(range(len(obs)),
                   key=lambda i: (SOURCE_RANK.index(obs[i]["src"])
                                  if obs[i]["src"] in SOURCE_RANK else len(SOURCE_RANK),
                                  -float(obs[i].get("conf", 0.0))))
    groups = []
    for i in order:
        o = obs[i]
        box = (o["x"], o["y"], o["w"], o["h"])
        for g in groups:
            if o["src"] in g["srcs"]:
                continue
            gb = (g["x"], g["y"], g["w"], g["h"])
            if iou(box, gb) >= IOU_GATE or _centre_ok(gb, box):
                g["srcs"].add(o["src"])
                for k in ("radar_m", "max_c"):
                    if g.get(k) is None and o.get(k) is not None:
                        g[k] = o[k]
                g["conf"] = max(g["conf"], float(o.get("conf", 0.0)))
                break
        else:
            groups.append({"x": o["x"], "y": o["y"], "w": o["w"], "h": o["h"],
                           "conf": float(o.get("conf", 0.0)),
                           "srcs": {o["src"]},
                           "radar_m": o.get("radar_m"),
                           "max_c": o.get("max_c")})
    return [dict(g, src=g.pop("srcs")) for g in groups]


class Track:
    """One person, for as long as the evidence keeps arriving."""

    def __init__(self, tid, obs, now):
        self.id = tid
        self.x, self.y, self.w, self.h = (float(obs["x"]), float(obs["y"]),
                                          float(obs["w"]), float(obs["h"]))
        self.vx = self.vy = 0.0
        self.conf = float(obs.get("conf", 0.0))
        self.born = now
        self.last_seen = now
        self.last_move = now
        self.hits = 1
        self.sources = _as_set(obs["src"])   # what confirmed it this cycle
        self.held_by = set(self.sources)     # what confirmed it most recently
        self.ever = set(self.sources)        # everything that ever confirmed it
        # Birth centre and the furthest the smoothed centre has been from it.
        # Not the distance travelled: a jittering box on a doorway accumulates
        # a large path without ever going anywhere, and the question here is
        # whether this thing has GONE anywhere.
        self.cx0 = self.x + self.w / 2.0
        self.cy0 = self.y + self.h / 2.0
        self.moved = 0.0
        self.radar_m = obs.get("radar_m")
        self.max_c = obs.get("max_c")

    @property
    def box(self):
        return (int(round(self.x)), int(round(self.y)),
                int(round(self.w)), int(round(self.h)))

    def coasting_for(self, now):
        return now - self.last_seen

    @property
    def vouched(self):
        """Is this a person, or something warm that has never moved?

        The visible detector knows a door from a person and is trusted on
        sight. Nothing else is - not because the students are worse, but
        because what they measure (a warm shape, a radar return) is a property
        the door shares. Those have to earn it by moving.
        """
        return "det" in self.ever or self.moved >= max(STATIC_PX,
                                                       STATIC_FRAC * self.w)

    def predict(self, now):
        """Carry the box forward on its own velocity. Cheap and bounded."""
        dt = now - self.last_move
        self.last_move = now
        if dt <= 0 or dt > MAX_COAST_S:
            return
        self.x += self.vx * dt
        self.y += self.vy * dt

    def correct(self, obs, now):
        dt = max(1e-3, now - self.last_seen)
        mx, my = float(obs["x"]), float(obs["y"])
        rx, ry = mx - self.x, my - self.y
        self.x += ALPHA * rx
        self.y += ALPHA * ry
        self.vx += BETA * rx / dt
        self.vy += BETA * ry / dt
        self.w += ALPHA * (float(obs["w"]) - self.w)
        self.h += ALPHA * (float(obs["h"]) - self.h)
        self.conf = max(self.conf * 0.7, float(obs.get("conf", 0.0)))
        self.hits += 1
        self.last_seen = now
        self.ever |= _as_set(obs["src"])
        cx, cy = self.x + self.w / 2.0, self.y + self.h / 2.0
        self.moved = max(self.moved,
                         ((cx - self.cx0) ** 2 + (cy - self.cy0) ** 2) ** 0.5)
        # Range and temperature come from whichever observation carried them
        # and are not invented while coasting: they are cleared with the box
        # they belonged to, at the point the box stops being a measurement.
        if obs.get("radar_m") is not None:
            self.radar_m = obs["radar_m"]
        if obs.get("max_c") is not None:
            self.max_c = obs["max_c"]

    def as_dict(self, now):
        return {
            "id": self.id,
            "x": self.box[0], "y": self.box[1], "w": self.box[2], "h": self.box[3],
            "conf": round(self.conf, 3),
            "hits": self.hits,
            "age_s": round(now - self.born, 2),
            "coasting": round(self.coasting_for(now), 2) if self.coasting_for(now) > 0.05 else 0,
            "held_by": sorted(self.held_by),
            "ever": sorted(self.ever),
            "moved_px": round(self.moved, 1),
            "radar_m": self.radar_m,
            "max_c": self.max_c,
        }


class Tracker:
    """Greedy IoU tracker over one class - person - and several sensors.

    Greedy rather than Hungarian for the same reason the fusion pairing is:
    the frame holds a handful of people, the assignment is small enough that
    best-first under a hard gate gives the same answer, and this runs on the
    render thread where a scipy import would be the most expensive thing in it.
    """

    def __init__(self, max_coast_s=MAX_COAST_S, min_hits=MIN_HITS):
        self.max_coast_s = max_coast_s
        self.min_hits = min_hits
        self._tracks = []
        self._next_id = 1

    def update(self, obs, now):
        """Fold one cycle's observations in. Returns the confirmed tracks.

        `obs` is a list of {x, y, w, h, conf, src} - src naming the sensor, and
        optionally radar_m / max_c to carry along. An empty list is a normal
        call and not a no-op: it is how a track learns it was missed.
        """
        obs = merge(obs)
        for t in self._tracks:
            t.predict(now)
            t.sources = set()

        pairs = []
        for oi, o in enumerate(obs):
            ob = (o["x"], o["y"], o["w"], o["h"])
            for ti, t in enumerate(self._tracks):
                v = iou(ob, t.box)
                if v >= IOU_GATE:
                    pairs.append((-v, oi, ti))
                elif _centre_ok(t.box, ob):
                    # Ranked below every real overlap: a centre match is the
                    # fallback for a fast walker, not a competitor to one.
                    pairs.append((1.0, oi, ti))
        pairs.sort()

        used_o, used_t = set(), set()
        for _score, oi, ti in pairs:
            if oi in used_o or ti in used_t:
                continue
            used_o.add(oi)
            used_t.add(ti)
            t = self._tracks[ti]
            t.correct(obs[oi], now)
            t.sources |= _as_set(obs[oi]["src"])

        for oi, o in enumerate(obs):
            if oi not in used_o:
                t = Track(self._next_id, o, now)
                self._next_id += 1
                self._tracks.append(t)

        for t in self._tracks:
            if t.sources:
                t.held_by = set(t.sources)
        self._tracks = [t for t in self._tracks
                        if t.coasting_for(now) <= self.max_coast_s]
        return self.confirmed(now)

    def confirmed(self, now):
        """Tracks worth drawing: seen twice, inside the coast, and vouched for."""
        return [t for t in self._tracks
                if t.hits >= self.min_hits
                and t.coasting_for(now) <= self.max_coast_s
                and t.vouched]

    def suppressed(self, now):
        """Live tracks held back for not having moved and not being detected.

        Counted rather than discarded, and reported by the caller: a filter
        that silently eats candidates is worse than the clutter it removes, and
        this one is the difference between "nobody is there" and "something
        warm is there that has never moved".
        """
        return [t for t in self._tracks
                if t.hits >= self.min_hits
                and t.coasting_for(now) <= self.max_coast_s
                and not t.vouched]

    def all(self):
        return list(self._tracks)

    def reset(self):
        self._tracks = []
