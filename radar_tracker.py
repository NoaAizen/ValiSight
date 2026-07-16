"""
Persistent object tracking + signature accumulation over radar clusters.

NPU-style object characterization: instead of classifying each frame's
clusters in isolation, every physical object gets a Track that accumulates
evidence over seconds — position, motion statistics, micro-Doppler
distribution, reflectivity — and summarizes it as a fixed-order feature
vector (a "signature") ready for a learned classifier (RandomForest / tiny
MLP, int8-quantizable for the OpenMV N6 Neural-ART NPU).

Pure Python, no numpy — CPython and MicroPython compatible.

Usage:
    tracker = Tracker()
    each frame:
        clusters = classify_frame(points, include_points=True)
        tracks = tracker.update(clusters, t)     # confirmed tracks only
    at shutdown:
        dataset_rows = tracker.all_signatures()  # for tracks.jsonl / training
"""
import math

from radar_classify_n6 import STATIC_V
from radar_material import material, point_scores, peak_score

CONFIRM_HITS = 4        # updates before a track is trusted/displayed
MISS_TIMEOUT_S = 1.5    # drop a track not seen for this long
GATE_BASE_M = 0.9       # association gate radius (+ half cluster extent)
POS_ALPHA = 0.35        # centroid smoothing factor
MOVED_M = 0.5           # total displacement above this -> not static
REFL_KEEP = 256         # per-track reflectivity samples kept (rolling)
MAX_DEAD = 200          # finished-track signatures kept for logging

# person stickiness: a person who stops moving is still a person.
# Doppler goes to ~0 when someone stands still, so without memory the track
# degrades to "static" — exactly the failure seen in the dark, where the
# camera can't re-confirm. Evidence thresholds:
PED_STICKY_VOTES = 10   # per-frame 'pedestrian' votes needed to lock identity
PED_STICKY_FRAC = 0.15  # ...and as a fraction of hits (long static tracks
                        #    collect a few jitter votes — don't let them lock)
CAM_PERSON_VOTES = 3    # camera 'person' matches needed to lock identity
PERSON_TIMEOUT_S = 3.0  # person-locked tracks survive longer gaps — a still
                        # body returns few CFAR points, and losing the track
                        # means losing the identity right when it matters

# micro-Doppler histogram bin edges (m/s) -> len-1 bins
DOPPLER_EDGES = (-6.0, -3.0, -1.5, -0.5, -0.15, 0.15, 0.5, 1.5, 3.0, 6.0)

# fixed feature order for the ML vector (append-only — models depend on it)
SIGNATURE_FIELDS = [
    "range_m", "az_deg", "el_deg",
    "ext_x", "ext_y", "ext_z",
    "n_pts_mean",
    "refl_median", "refl_p90", "refl_std",
    "v_mean", "v_abs_mean", "v_std", "vspread_mean",
    "speed_est", "displacement_m", "persistence_s", "hits",
]


class _Stat(object):
    """Welford running mean/std."""
    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def std(self):
        return math.sqrt(self.m2 / self.n) if self.n > 1 else 0.0


def _refl_values(points):
    """Per-point range-normalized reflectivity scores (dB above the empirical
    baseline for that range — see radar_calibration)."""
    return point_scores(points)


class Track(object):
    _next_id = 1

    def __init__(self, cluster, t):
        self.id = Track._next_id
        Track._next_id += 1
        self.pos = list(cluster["centroid"])
        self.vel = [0.0, 0.0, 0.0]           # from centroid deltas (m/s)
        self.first_pos = tuple(self.pos)
        self.t_start = t
        self.t_last = t
        self.hits = 1
        # accumulators
        self.ext = [_Stat(), _Stat(), _Stat()]
        self.n_pts = _Stat()
        self.v = _Stat()                     # signed doppler, per point
        self.v_abs = _Stat()
        self.vspread = _Stat()               # per-cluster micro-Doppler span
        self.refl = []                       # rolling reservoir
        self.dop_hist = [0] * (len(DOPPLER_EDGES) - 1)
        self.motion_votes = {}
        self.cam_votes = {}                  # camera identity votes (fusion)
        self._absorb(cluster)

    # -- update ------------------------------------------------------------
    def predict(self, t):
        dt = t - self.t_last
        return (self.pos[0] + self.vel[0] * dt,
                self.pos[1] + self.vel[1] * dt,
                self.pos[2] + self.vel[2] * dt)

    def update(self, cluster, t):
        dt = max(t - self.t_last, 1e-3)
        cx, cy, cz = cluster["centroid"]
        for i, c in enumerate((cx, cy, cz)):
            v_inst = (c - self.pos[i]) / dt
            self.vel[i] += 0.2 * (v_inst - self.vel[i])
            self.pos[i] += POS_ALPHA * (c - self.pos[i])
        self.t_last = t
        self.hits += 1
        self._absorb(cluster)

    def _absorb(self, cluster):
        lbl = cluster.get("label", "unknown")
        self.motion_votes[lbl] = self.motion_votes.get(lbl, 0) + 1
        pts = cluster.get("points", ())
        self.n_pts.add(len(pts) or cluster.get("n_points", 0))
        if pts:
            for ax in range(3):
                vals = [p[ax] for p in pts]
                self.ext[ax].add(max(vals) - min(vals))
            vs = [p[3] for p in pts]
            for v in vs:
                self.v.add(v)
                self.v_abs.add(abs(v))
                for b in range(len(DOPPLER_EDGES) - 1):
                    if DOPPLER_EDGES[b] <= v < DOPPLER_EDGES[b + 1]:
                        self.dop_hist[b] += 1
                        break
            self.vspread.add(max(vs) - min(vs))
            self.refl.extend(_refl_values(pts))
            if len(self.refl) > REFL_KEEP:
                del self.refl[:len(self.refl) - REFL_KEEP]

    # -- derived properties --------------------------------------------------
    def displacement(self):
        dx = self.pos[0] - self.first_pos[0]
        dy = self.pos[1] - self.first_pos[1]
        dz = self.pos[2] - self.first_pos[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def person_evidence(self):
        """True once this track has been solidly identified as a person —
        either by the camera (YOLO 'person' matched to it) or by its own
        motion history (sustained pedestrian micro-Doppler)."""
        if self.cam_votes.get("person", 0) >= CAM_PERSON_VOTES:
            return True
        if self.material_class() == "metal":
            return False        # bodies aren't metal: a carried/nearby metal
                                # object moves like its carrier — motion alone
                                # must not make it a person
        ped = self.motion_votes.get("pedestrian", 0)
        return ped >= PED_STICKY_VOTES and ped >= PED_STICKY_FRAC * self.hits

    def motion_class(self):
        """Track-level motion label — steadier than per-frame votes."""
        # angular noise grows with range -> displacement threshold scales too
        moved_gate = MOVED_M + 0.08 * self.range_m()
        if self.v_abs.mean < STATIC_V and self.displacement() < moved_gate:
            # sticky person: standing still zeroes the Doppler, but a person
            # who stopped is still a person (critical in the dark, where the
            # camera cannot re-confirm)
            if self.person_evidence():
                return "pedestrian"
            return "static"
        moving = {k: v for k, v in self.motion_votes.items() if k != "static"}
        if moving:
            return max(moving, key=lambda k: moving[k])
        return "unknown"

    def refl_stats(self):
        """(median, p90, std) of reflectivity, or (None, None, None)."""
        if not self.refl:
            return None, None, None
        vs = sorted(self.refl)
        med = vs[len(vs) // 2]
        p90 = vs[min(len(vs) * 9 // 10, len(vs) - 1)]
        mean = sum(vs) / len(vs)
        var = sum((v - mean) ** 2 for v in vs) / len(vs)
        return med, p90, math.sqrt(var)

    def material_class(self):
        # With plenty of accumulated samples, judge by p90 instead of the
        # peak: a person brushing against / holding a metal object absorbs a
        # few glint points into the track, and a peak would flip the whole
        # person to "metal" — p90 stays with the majority. A true metal
        # object is glints throughout, so its p90 is high anyway.
        if len(self.refl) >= 20:
            _, p90, _ = self.refl_stats()
            return material(p90, range_m=self.range_m())
        # young/sparse track: peak still finds metal fastest
        return material(peak_score(self.refl), range_m=self.range_m())

    def range_m(self):
        x, y, z = self.pos
        return math.sqrt(x * x + y * y + z * z)

    def confirmed(self):
        return self.hits >= CONFIRM_HITS

    # -- exports ----------------------------------------------------------
    def signature(self):
        """Full characterization: named features + fixed-order ML vector."""
        med, p90, rstd = self.refl_stats()
        x, y, z = self.pos
        speed = math.sqrt(sum(v * v for v in self.vel))
        feats = {
            "range_m": round(self.range_m(), 2),
            "az_deg": round(math.degrees(math.atan2(-y, max(x, 0.01))), 1),
            "el_deg": round(math.degrees(
                math.atan2(z, max(math.sqrt(x * x + y * y), 0.01))), 1),
            "ext_x": round(self.ext[0].mean, 2),
            "ext_y": round(self.ext[1].mean, 2),
            "ext_z": round(self.ext[2].mean, 2),
            "n_pts_mean": round(self.n_pts.mean, 1),
            "refl_median": None if med is None else round(med, 1),
            "refl_p90": None if p90 is None else round(p90, 1),
            "refl_std": None if rstd is None else round(rstd, 1),
            "v_mean": round(self.v.mean, 2),
            "v_abs_mean": round(self.v_abs.mean, 2),
            "v_std": round(self.v.std(), 2),
            "vspread_mean": round(self.vspread.mean, 2),
            "speed_est": round(speed, 2),
            "displacement_m": round(self.displacement(), 2),
            "persistence_s": round(self.t_last - self.t_start, 2),
            "hits": self.hits,
        }
        n_dop = sum(self.dop_hist) or 1
        cam = max(self.cam_votes, key=lambda k: self.cam_votes[k]) \
            if self.cam_votes else None
        return {
            "track_id": self.id,
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_last, 3),
            "motion": self.motion_class(),
            "material": self.material_class(),
            "cam_label": cam,
            "features": feats,
            "vector": [0.0 if feats[f] is None else float(feats[f])
                       for f in SIGNATURE_FIELDS],
            "doppler_hist": [round(c / n_dop, 3) for c in self.dop_hist],
        }

    def as_cluster(self):
        """Cluster-compatible dict so display/matching code works on tracks."""
        med, _, _ = self.refl_stats()
        ext = math.sqrt(sum(s.mean ** 2 for s in self.ext))
        label = self.motion_class()
        # camera-confirmed identity beats the motion label on the display:
        # "person 1.7m" is what the user needs, even when the person is still
        if self.cam_votes.get("person", 0) >= CAM_PERSON_VOTES:
            label = "person"
        elif label == "pedestrian" and self.material_class() == "metal":
            # moving + metallic + never camera-confirmed as a person: this is
            # a carried/rolling metal object, not the person holding it
            label = "object"
        return {
            "track_id": self.id,
            "label": label,
            "range_m": round(self.range_m(), 2),
            "doppler_mps": round(self.v.mean, 2),
            "extent_m": round(ext, 2),
            "ext_xyz": tuple(round(s.mean, 2) for s in self.ext),
            "n_points": int(round(self.n_pts.mean)),
            "centroid": tuple(round(p, 2) for p in self.pos),
            "refl_db": None if med is None else round(med, 1),
            "material": self.material_class(),
        }


class Tracker(object):
    def __init__(self):
        self.tracks = []
        self.dead = []                       # signatures of finished tracks

    def update(self, clusters, t):
        """Associate clusters to tracks (greedy nearest with gating).
        Returns the confirmed tracks, freshest first."""
        # build all candidate (distance, track_idx, cluster_idx) pairs
        cands = []
        tr_mats = [tr.material_class() for tr in self.tracks]
        for ti, tr in enumerate(self.tracks):
            px, py, pz = tr.predict(t)
            for ci, c in enumerate(clusters):
                cx, cy, cz = c["centroid"]
                d = math.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)
                gate = GATE_BASE_M + 0.5 * c.get("extent_m", 0.0)
                if d > gate:
                    continue
                # a split metal glint and the body it was carved out of are
                # nearly co-located — material consistency keeps the metal
                # cluster feeding the metal track, not the person
                cmat, tmat = c.get("material"), tr_mats[ti]
                if ("metal" in (cmat, tmat) and cmat != tmat
                        and "unknown" not in (cmat, tmat)):
                    d += 0.4
                    if d > gate:
                        continue
                cands.append((d, ti, ci))
        cands.sort()
        used_t, used_c = set(), set()
        for d, ti, ci in cands:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            self.tracks[ti].update(clusters[ci], t)
        # unmatched clusters -> new tentative tracks
        for ci, c in enumerate(clusters):
            if ci not in used_c:
                self.tracks.append(Track(c, t))
        # expire stale tracks (keep their signature for the dataset)
        alive = []
        for tr in self.tracks:
            timeout = PERSON_TIMEOUT_S if tr.person_evidence() \
                else MISS_TIMEOUT_S
            if t - tr.t_last <= timeout:
                alive.append(tr)
            elif tr.confirmed():
                self.dead.append(tr.signature())
                if len(self.dead) > MAX_DEAD:
                    del self.dead[0]
        self.tracks = alive
        return [tr for tr in self.tracks if tr.confirmed()]

    def note_camera(self, track_id, label):
        """Register a camera identity vote (YOLO box matched to this track).
        Votes persist for the track's lifetime, so an identity earned in the
        light keeps the object named after the camera goes blind (darkness,
        occlusion by glare, etc.)."""
        for tr in self.tracks:
            if tr.id == track_id:
                tr.cam_votes[label] = tr.cam_votes.get(label, 0) + 1
                return

    def all_signatures(self):
        """Finished + live confirmed tracks — one row per physical object."""
        return self.dead + [tr.signature() for tr in self.tracks
                            if tr.confirmed()]
