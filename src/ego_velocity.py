"""Doppler ego-velocity: solve the sensor's own velocity from static returns.

Pure maths. No hardware, no serial, no numpy -- so it runs identically on a
recorded session and on the live stream, and its unit tests need neither.

THE SOLVE. For a world-fixed scatterer at unit bearing `u`, a sensor moving at
`v` sees radial velocity `v_r = -u . v`. Stack one row per point:

    A v = b        row_i = u_i        b_i = -v_r,i

and solve by weighted least squares. Two points suffice in 2-D provided their
bearings differ; a point straight ahead and a point at 40 degrees determine both
the along-track and cross-track components, while two points at the same bearing
determine only one and no amount of averaging fixes that.

WHY RANSAC IS NOT OPTIONAL. The model assumes the scatterer is world-fixed. A
walking person is not, and their returns satisfy a different equation. With
roughly six detections per frame, two or three from one person is a majority --
so a plain least-squares fit does not degrade gracefully, it reports the
person's motion as the sensor's, with a small residual and full confidence.
That failure is silent, and it is the one this module is built to survive.

Two hardening rules matter at this point density and are not decoration:

  * A minimal set must not draw both points from the same range bin. With CFAR
    peak grouping off, one wall yields several adjacent-bin detections; two of
    them have nearly identical bearings, so A is near-singular and the model
    fits any velocity at all.
  * A consensus whose bearings span less than MIN_CONSENSUS_AZ is rejected
    however many inliers it has. A tight cluster of agreeing points is the
    signature of one moving object, not of the world.

Neither rule catches everything. A person filling the field of view at close
range can still win. `EgoVel.flags` carries what the caller needs to distrust
the answer rather than pretending the problem is solved.
"""
import math

import radar_static
from radar_static import (DOPPLER_SIGMA_MPS, STOCK_GRID, range_bin_index)

MIN_CONSENSUS_AZ = 20.0        # degrees; below this a consensus is one object
MIN_INLIER_MPS = 0.08          # inlier band never tightens past this
RANSAC_ITERS = 30
# An unwrapped hypothesis must beat the best non-unwrapped one by this factor
# before it is believed. Measured over the 9752-frame static session, where the
# truth is v = 0 and every unwrap claim is a false positive:
#     margin 1.0 (lowest cost wins) -> 1.2% false alias
#     margin 0.5                    -> 0.2% false alias, 97-99% detection at 1.4-2.6 m/s
#     margin 0.25                   -> 0.2% false alias, 95% detection
# 0.5 is the knee. Below the fold the unwrapped branch wins on cost anyway, so
# this only ever gates the extraordinary claim.
ALIAS_COST_MARGIN = 0.5
# Absolute sanity ceiling on a candidate, replacing the old `v_max * 1.5`. That
# guard was the line that ENFORCED aliasing blindness: past the fold the correct
# hypothesis is by definition larger than v_max, so the guard deleted it. This
# one means "faster than anyone carries this rig".
V_CEILING_MPS = 4.0
# Stock chirp geometry. Defined FROM the grid so the number lives in one place;
# it used to be a literal 1.001, which was 2.8% high (lambda taken at the 77 GHz
# start frequency instead of the 79.21 GHz ADC-window centre -- see
# chirp_geometry). A session recorded under another config carries its own grid
# on its points and does not consult this.
V_MAX_DEFAULT = STOCK_GRID.v_max


class EgoVel:
    __slots__ = ("v", "cov", "n_in", "n_tot", "gdop", "resid_rms",
                 "az_span", "flags", "ok", "in_idx", "aliased", "alias_margin")

    def __init__(self, v=None, cov=None, n_in=0, n_tot=0, gdop=None,
                 resid_rms=None, az_span=0.0, flags=(), ok=False, in_idx=(),
                 aliased=False, alias_margin=None):
        # True when the winning hypothesis needed a whole fold added to a
        # measured radial velocity, i.e. the Doppler had wrapped and was
        # unwrapped. Exact, not inferred. `alias_margin` is
        # cost_unwrapped / cost_plain, so a human can see how close the call was.
        self.aliased, self.alias_margin = aliased, alias_margin
        self.v, self.cov = v, cov
        self.n_in, self.n_tot = n_in, n_tot
        self.gdop, self.resid_rms = gdop, resid_rms
        self.az_span, self.flags, self.ok = az_span, list(flags), ok
        # Indices into the caller's `pts` for the consensus this solve actually
        # used. Published because callers were reconstructing it with a
        # different inlier band than solve() used, so the geometry they reported
        # (gdop, azimuth spread) described a point set the estimator rejected.
        self.in_idx = list(in_idx)

    @property
    def speed(self):
        return math.hypot(*self.v[:2]) if self.v else 0.0

    def as_dict(self):
        return {"ok": self.ok,
                "v": [round(c, 4) for c in self.v] if self.v else None,
                "speed": round(self.speed, 4),
                "n_in": self.n_in, "n_tot": self.n_tot,
                "gdop": self.gdop,
                "resid_rms": round(self.resid_rms, 4) if self.resid_rms is not None else None,
                "az_span": round(self.az_span, 1),
                "flags": self.flags}


def _solve_wls(rows, targets, weights):
    """2-D weighted least squares by normal equations. Returns (v, cov) or None.

    Two unknowns and a handful of rows, so the normal equations are fine here --
    the conditioning worry is geometric (bearings too close together), and that
    shows up as a singular matrix which is checked for explicitly rather than
    being hidden by a more elaborate decomposition.
    """
    a11 = a12 = a22 = b1 = b2 = 0.0
    for (ux, uy), t, w in zip(rows, targets, weights):
        a11 += w * ux * ux
        a12 += w * ux * uy
        a22 += w * uy * uy
        b1 += w * ux * t
        b2 += w * uy * t
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        return None
    vx = (a22 * b1 - a12 * b2) / det
    vy = (a11 * b2 - a12 * b1) / det
    cov = ((a22 / det, -a12 / det), (-a12 / det, a11 / det))
    return (vx, vy), cov


def _residuals(pts, v):
    """Measured minus predicted radial velocity, per point."""
    return [p.vr - (-(p.u[0] * v[0] + p.u[1] * v[1])) for p in pts]


def cluster_sizes(pts, eps_m=0.6):
    """How many detections each point shares an object with.

    One physical object routinely produces several detections -- a person gives
    three at 0.6 degrees apart, a wall gives several adjacent range bins with
    CFAR peak grouping off. Those are correlated samples of ONE observation, and
    counting them as three independent votes is what lets a moving object
    outvote the static world.

    Measured on the failing case: a 3-point person 3.5 sigma from the true
    velocity beat the correct consensus on MSAC cost, because the correct answer
    had to pay the outlier penalty three times while the wrong one collected
    three near-zero residuals. Dividing each point's weight by its cluster size
    makes that trade honest -- the person then contributes one vote, not three.

    Single-link region growing, O(n^2), which is nothing at n<=15.
    """
    n = len(pts)
    lab = [-1] * n
    nxt = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        lab[i] = nxt
        stack = [i]
        while stack:
            k = stack.pop()
            for j in range(n):
                if lab[j] >= 0:
                    continue
                dx = pts[k].x - pts[j].x
                dy = pts[k].y - pts[j].y
                dz = pts[k].z - pts[j].z
                if dx * dx + dy * dy + dz * dz <= eps_m * eps_m:
                    lab[j] = nxt
                    stack.append(j)
        nxt += 1
    counts = [0] * nxt
    for l in lab:
        counts[l] += 1
    return [counts[l] for l in lab], lab


def solve(pts, prior_speed=0.0, v_max=None, iters=RANSAC_ITERS,
          grid=None, rng=None):
    """Estimate 2-D sensor velocity from conditioned points.

    prior_speed seeds the per-point sigma, whose bearing term scales with speed.
    Pass the previous frame's speed; being wrong only reweights slightly.

    `grid` is the Doppler axis. Resolution order: explicit argument, then the
    grid the points carry (set by radar_static.condition), then stock. `v_max`
    is kept for callers that only know that one number -- it builds a matching
    grid -- but a grid is the better thing to pass, because unwrapping needs the
    bin count as well as the limit.

    This function RESOLVES aliasing, it does not merely flag it. See the
    candidate enumeration below.
    """
    n = len(pts)
    flags = []
    if n < 2:
        return EgoVel(n_tot=n, flags=["too_few_points"])

    if grid is None:
        if v_max is not None:
            base = pts[0].grid
            grid = radar_static.DopplerGrid(
                2.0 * v_max / base.n_bins, base.n_bins, source="v_max override")
        else:
            grid = pts[0].grid

    sig = [p.sigma(prior_speed) for p in pts]
    # Weight by independent evidence, not by detection count: a point sharing an
    # object with k-1 others carries 1/k of a vote. Without this, three
    # detections off one walking person outvote four spread-out world returns.
    csize, clab = cluster_sizes(pts)
    w = [1.0 / (s * s * c) for s, c in zip(sig, csize)]
    band = max(3.0 * max(sig), MIN_INLIER_MPS)

    # Deterministic candidate enumeration rather than random sampling: with n
    # this small there are at most n*(n-1)/2 pairs (15 at n=6), so every minimal
    # set can be tried. That removes the seed as a source of run-to-run
    # variation, which matters for a replay that must be reproducible.
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if range_bin_index(pts[i]) == range_bin_index(pts[j]):
                continue                     # same scatterer, degenerate pair
            if abs(pts[i].az - pts[j].az) < 5.0:
                continue                     # bearings too close to determine v
            pairs.append((i, j))
    if not pairs:
        flags.append("no_independent_pair")
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs = pairs[:max(iters, 1)]

    # FOLD-AWARE CANDIDATE ENUMERATION.
    #
    # A wrapped return is not noise. It is wrong by EXACTLY one fold interval,
    # so aliasing is not merely detectable -- it is resolvable. For each minimal
    # pair, try the nine combinations of adding {-1, 0, +1} folds to the two
    # measured radial velocities, and score every candidate with residuals taken
    # MODULO the fold. Nine 2x2 normal-equation solves per pair is nothing at
    # n<=15 and the module stays pure.
    #
    # Why this is not luck: for a wrong velocity v' to explain the same folded
    # measurements as the true v, every point must satisfy u_i.(v - v') = 0 mod
    # fold SIMULTANEOUSLY. With an azimuth spread D the smallest such |v - v'| is
    # fold / (1 - cos D) -- 4.6 m/s at the 55 degrees this scene actually has. So
    # bearing diversity buys velocity unambiguity, exactly as it buys
    # observability of the cross-track component. The az_spread that
    # radar_metrics already reports IS the ambiguity margin.
    #
    # The unwrapped and unwrapped-not branches are tracked SEPARATELY and the
    # wrapped one is adopted only if it beats the other by ALIAS_COST_MARGIN.
    # Taking the lowest cost outright costs 1.2% false aliases on real static
    # data; the margin drops that to 0.2% at a few points of detection rate.
    fold = grid.fold
    best = {0: (None, None, None), 1: (None, None, None)}   # unwrapped? -> set,cost,v
    for i, j in pairs:
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                got = _solve_wls([pts[k].u[:2] for k in (i, j)],
                                 [-(pts[i].vr + a * fold),
                                  -(pts[j].vr + b * fold)],
                                 [w[k] for k in (i, j)])
                if got is None:
                    continue
                v_try = got[0]
                if math.hypot(*v_try) > V_CEILING_MPS:
                    continue                  # faster than anyone carries this rig
                res = [grid.wrap(r) for r in _residuals(pts, v_try)]
                # MSAC: inliers cost their squared residual, outliers a flat
                # penalty, so a marginally-better inlier set cannot win by
                # scraping in bad points. Each term is scaled by 1/cluster_size
                # so one object cannot buy the verdict by returning several
                # correlated detections.
                cost = sum(min(r * r, band * band) / c
                           for r, c in zip(res, csize))
                inl = [k for k, r in enumerate(res) if abs(r) <= band]
                if len({clab[k] for k in inl}) < 2:
                    continue                  # one object is not a consensus
                key = 1 if (a or b) else 0
                if best[key][1] is None or cost < best[key][1]:
                    best[key] = (inl, cost, v_try)

    plain_set, plain_cost, plain_v = best[0]
    un_set, un_cost, un_v = best[1]
    aliased = False
    margin_ratio = None
    if plain_cost is not None and un_cost is not None and plain_cost > 0:
        margin_ratio = un_cost / plain_cost
    if un_cost is not None and (plain_cost is None
                                or un_cost < ALIAS_COST_MARGIN * plain_cost):
        best_set, seed_v, aliased = un_set, un_v, True
    else:
        best_set, seed_v = plain_set, plain_v

    if best_set is None:
        return EgoVel(n_tot=n, flags=flags + ["no_consensus"])

    sel = [pts[k] for k in best_set]
    az = [p.az for p in sel]
    az_span = max(az) - min(az)

    # Refit on the consensus -- but against UNWRAPPED radial velocities. Past the
    # fold the stored p.vr is the folded value, so refitting on it would drag the
    # answer straight back down to the aliased velocity the seed just escaped.
    # Each inlier's true v_r is its prediction under the seed plus the wrapped
    # residual, which recovers vr + m*fold for whichever integer m applies.
    targets = []
    for p in sel:
        pred = -(p.u[0] * seed_v[0] + p.u[1] * seed_v[1])
        targets.append(-(pred + grid.wrap(p.vr - pred)))
    got = _solve_wls([p.u[:2] for p in sel], targets, [w[k] for k in best_set])
    if got is None:
        return EgoVel(n_tot=n, flags=flags + ["singular"])
    v, cov = got

    res_in = [grid.wrap(r) for r in _residuals(sel, v)]
    resid_rms = math.sqrt(sum(r * r for r in res_in) / len(res_in))

    if az_span < MIN_CONSENSUS_AZ:
        flags.append("narrow_consensus")      # probably one moving object
    if len(sel) < 3:
        flags.append("low_redundancy")        # no residual left to check with
    if aliased:
        # EXACT, not a proxy: the winning hypothesis needed a whole fold added
        # to a measured radial velocity. This replaces `near_aliasing`, which
        # tested the REPORTED speed against 0.85*v_max -- a band, not a
        # threshold. Measured, that flag fired on 100% of frames at 0.90 m/s
        # (where the answer was good to 0.06) and on 2% at 2.00 m/s (where it
        # was wrong by 2.00).
        flags.append("aliased")
        if az_span < MIN_CONSENSUS_AZ:
            # Narrow bearings are exactly the case where the unwrap is NOT
            # uniquely determined -- the ambiguity margin fold/(1-cos D) blows
            # up as D shrinks. Believing a 2 m/s answer here would be worse than
            # reporting the folded one.
            flags.append("alias_unresolved")

    ok = ("narrow_consensus" not in flags and "no_independent_pair" not in flags
          and "alias_unresolved" not in flags)
    from radar_static import gdop as _gdop
    return EgoVel(v=(v[0], v[1]), cov=cov, n_in=len(sel), n_tot=n,
                  gdop=_gdop(sel), resid_rms=resid_rms, az_span=az_span,
                  flags=flags, ok=ok, in_idx=best_set,
                  aliased=aliased, alias_margin=margin_ratio)


def is_stationary(pts, bin_mps=None):
    """True when every return says the platform is not moving.

    This is the zero-velocity detector worth trusting on a handheld rig. The
    IMU's own stationary flag needs |a| within 60 mg of 1 g, which holds on a
    desk but essentially never during a walk with arm swing -- it fired 96.6% of
    the time in a recorded session precisely because the rig was sitting still.
    A radar frame whose every return is in the zero-Doppler bin is a much
    stronger statement, and it survives hand tremor.

    `bin_mps` defaults to the grid the points themselves carry. It used to
    default to a literal 0.125, a second copy of a constant that lived in
    radar_static -- so correcting the constant there would have left this one
    silently on the old scale.

    CAVEAT this cannot see: zero Doppler means "not moving OR moving at exactly
    a multiple of the fold". A rig carried at 1.947 m/s on the stock grid reports
    every return at 0.000 and reads as perfectly still. That is not paranoia --
    it is the same wrap the aliasing work addresses, arriving in the one place
    that looks least like a velocity estimate.
    """
    if not pts:
        return False
    q = bin_mps if bin_mps else pts[0].grid.bin_mps
    return all(abs(p.vr) < q / 2.0 for p in pts)
