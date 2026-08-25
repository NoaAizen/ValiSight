#!/usr/bin/env python3
"""Position and heading from radar wall returns matched against the map.

This is the observation the whole map layer was built to consume: the radar
sees walls as static returns (range, azimuth); the map knows where walls are
(building footprints). Given a rough pose — GNSS, a manual pin, or dead
reckoning that has drifted — the pose that puts the returns *on* the
footprint edges is the map-relative fix.

**What is solved.** Three numbers: east/north offset from the prior and a
heading correction. Height is not solved here — the DEM and geoid stages own
it, and two-element radar elevation is far too coarse to argue with them.

**How.** A bounded grid search over (dx, dy, dyaw) around the prior, scoring
each candidate pose by how many returns land within a wall tolerance of a
footprint edge. The score is a sum of Gaussians in point-to-edge distance,
not a nearest-neighbour count, for the same reason ``HeadingMatcher`` votes
softly: a hard assignment needs the answer before it can be made. The grid
is bounded by the prior's own sigmas, so a look-alike street 40 m away
cannot win — and if the prior is worse than it claims, the fix is wrong in
a way the checks say (best pose on the search boundary) rather than silently.

**Ambiguity is reported, not resolved.** A single straight wall constrains
one axis and heading, and leaves motion along the wall free. The score then
has a ridge, not a peak, and the right answer is ``ambiguous=True`` with the
weak axis named — not the ridge's arbitrary midpoint quoted to a decimal.

**Frames, which break silently if wrong.**
- Rig / radar: x forward, y left, z up. Azimuth **positive is right** of
  boresight, as the rig reports it (``mapinit`` README, DRISHOT.md).
- Heading: compass, degrees clockwise from north.
- A return at (range r, azimuth a) with the rig at heading h therefore sits
  on the world bearing ``h + a``, i.e. east = r·sin(h+a), north = r·cos(h+a).
- Local frame: metres east/north of the prior position (equirectangular;
  fine for the ±50 m this works over).

**The map is static and the world is not** — a parked truck returns like a
wall and is on no map. It costs score, it does not break the solve, and the
inlier fraction says how much of what the radar saw the map could explain.

Numpy is required: a 25×25×31 grid against a few hundred edges is a few
million distance evaluations, which pure Python does in tens of seconds and
numpy in well under one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..check import Check

METRES_PER_DEG_LAT = 111_320.0

#: Point-to-edge distance tolerance, metres, one sigma. The IWR1843 range
#: bin at this chirp is ~4 cm but azimuth is ~1° at best, so at 10 m a return
#: is placed to ~20 cm across-range and rather worse along the wall's
#: direction for a glancing hit. Half a metre also absorbs facade depth
#: (balconies, recesses) that Overture footprints do not carry.
DEFAULT_WALL_SIGMA_M = 0.5

#: Radius around the prior within which footprint edges are considered.
#: Bounded by the radar's useful range on walls plus the position search.
DEFAULT_EDGE_RADIUS_M = 60.0

#: Below this many returns a fix is not attempted: three unknowns need more
#: than a handful of observations, and a lone corner reflector can be matched
#: to any edge anywhere.
MIN_RETURNS = 6

#: Fraction of the best score a rival peak must reach to count as a rival.
AMBIGUITY_RATIO = 0.85

#: Minimum separation for two grid cells to be different peaks rather than the
#: same peak's shoulder, metres and degrees.
PEAK_SEPARATION_M = 1.5
PEAK_SEPARATION_DEG = 4.0


class WallMatchError(RuntimeError):
    """Inputs that make a fix meaningless, as opposed to a fix that fails."""


@dataclass(frozen=True)
class WallReturn:
    """One static radar return, as ``radar_detections_all()`` reports it."""

    range_m: float
    azimuth_deg: float


@dataclass(frozen=True)
class PosePrior:
    """Where the rig is believed to be before the walls are consulted."""

    latitude: float
    longitude: float
    heading_deg: float
    sigma_position_m: float
    sigma_heading_deg: float
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not (self.sigma_position_m > 0 and self.sigma_heading_deg > 0):
            raise WallMatchError("prior sigmas must be positive")


@dataclass(frozen=True)
class WallFix:
    """The pose that best explains the returns, and how much to trust it."""

    latitude: float
    longitude: float
    heading_deg: float
    #: Offset from the prior, metres east/north, and degrees of heading
    dx_m: float
    dy_m: float
    dyaw_deg: float
    sigma_east_m: float
    sigma_north_m: float
    sigma_heading_deg: float
    #: Best score, as a fraction of the returns (1.0 = every return exactly on an edge)
    score: float
    #: Returns within 2 sigma of an edge at the fix, as a fraction
    inlier_fraction: float
    n_returns: int
    n_edges: int
    #: A second peak within AMBIGUITY_RATIO of the best, separated from it
    ambiguous: bool
    #: Which quantity the rival differs in: "east", "north", "heading"
    ambiguity_axis: Optional[str]
    #: The best cell touched the search boundary: the prior is worse than stated
    on_boundary: bool
    accepted: bool
    checks: List[Check] = field(default_factory=list)

    def __str__(self) -> str:
        status = "accepted" if self.accepted else "REJECTED"
        return (f"WallFix {status}: {self.latitude:.6f}, {self.longitude:.6f}, "
                f"heading {self.heading_deg:.1f}° (Δ {self.dx_m:+.1f} E {self.dy_m:+.1f} N "
                f"{self.dyaw_deg:+.1f}°), σ {self.sigma_east_m:.1f}/{self.sigma_north_m:.1f} m "
                f"{self.sigma_heading_deg:.1f}°, score {self.score:.2f}, "
                f"inliers {self.inlier_fraction:.0%} of {self.n_returns}"
                + (f", ambiguous along {self.ambiguity_axis}" if self.ambiguous else ""))


def local_scales(latitude: float) -> Tuple[float, float]:
    """Metres per degree of longitude and of latitude at this latitude."""
    return (METRES_PER_DEG_LAT * math.cos(math.radians(latitude)), METRES_PER_DEG_LAT)


def footprint_edges(buildings, prior: PosePrior, radius_m: float = DEFAULT_EDGE_RADIUS_M):
    """Footprint edges near the prior, as segments in local east/north metres.

    ``buildings`` is anything iterable of objects with a ``ring`` of
    (longitude, latitude) pairs — a ``BuildingLayer`` or a plain list.
    Returns an (N, 4) float array of (x0, y0, x1, y1).
    """
    import numpy as np

    lon_scale, lat_scale = local_scales(prior.latitude)
    segs = []
    r2 = radius_m * radius_m
    for b in buildings:
        ring = list(b.ring)
        if len(ring) < 2:
            continue
        pts = [((lon - prior.longitude) * lon_scale, (lat - prior.latitude) * lat_scale)
               for lon, lat in ring]
        if ring[0] != ring[-1]:
            pts.append(pts[0])
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            # keep an edge if either end, or its midpoint, is inside the radius
            if min(x0 * x0 + y0 * y0, x1 * x1 + y1 * y1,
                   ((x0 + x1) / 2) ** 2 + ((y0 + y1) / 2) ** 2) <= r2:
                segs.append((x0, y0, x1, y1))
    return np.asarray(segs, dtype=float).reshape(-1, 4)


def returns_to_local(returns: Sequence[WallReturn], heading_deg: float,
                     dx_m: float = 0.0, dy_m: float = 0.0):
    """Place returns in the local frame for a rig at (dx, dy) facing heading."""
    import numpy as np

    r = np.asarray([w.range_m for w in returns], dtype=float)
    b = np.radians(heading_deg + np.asarray([w.azimuth_deg for w in returns], dtype=float))
    return np.stack([dx_m + r * np.sin(b), dy_m + r * np.cos(b)], axis=1)


def _point_edge_distance(points, segs):
    """Distance from each point to its nearest segment. points (P,2), segs (S,4)."""
    import numpy as np

    p = points[:, None, :]                               # (P,1,2)
    a = segs[None, :, 0:2]                               # (1,S,2)
    d = segs[None, :, 2:4] - a                           # (1,S,2)
    len2 = np.maximum((d * d).sum(-1), 1e-12)            # (1,S)
    t = np.clip(((p - a) * d).sum(-1) / len2, 0.0, 1.0)  # (P,S)
    proj = a + t[..., None] * d                          # (P,S,2)
    return np.sqrt(((p - proj) ** 2).sum(-1)).min(axis=1)


class WallMatcher:
    """Grid search for the pose that puts radar wall returns on map edges."""

    def __init__(self, wall_sigma_m: float = DEFAULT_WALL_SIGMA_M,
                 edge_radius_m: float = DEFAULT_EDGE_RADIUS_M,
                 position_step_m: float = 0.25, heading_step_deg: float = 0.5,
                 search_sigmas: float = 3.0) -> None:
        self.wall_sigma_m = wall_sigma_m
        self.edge_radius_m = edge_radius_m
        self.position_step_m = position_step_m
        self.heading_step_deg = heading_step_deg
        self.search_sigmas = search_sigmas

    def match(self, returns: Sequence[WallReturn], buildings, prior: PosePrior) -> WallFix:
        import numpy as np

        checks: List[Check] = []
        n = len(returns)
        checks.append(Check.in_range("walls.returns", n, MIN_RETURNS, math.inf, unit="returns"))
        segs = footprint_edges(buildings, prior, self.edge_radius_m)
        checks.append(Check.in_range("walls.edges_nearby", len(segs), 1, math.inf, unit="edges"))
        if n < MIN_RETURNS or len(segs) == 0:
            return self._rejected(prior, n, len(segs), checks)

        r = np.asarray([w.range_m for w in returns], dtype=float)
        az = np.asarray([w.azimuth_deg for w in returns], dtype=float)

        # Coarse-to-fine. The coarse pass covers the prior's whole 3-sigma box
        # at a step the wall tolerance can still see (no finer than needed to
        # not step over a peak of width ~sigma); the fine pass resolves the
        # winner. Ambiguity is judged on the coarse pass, which is the one
        # that sees rival streets; sigmas on the fine one.
        half_p = self.search_sigmas * prior.sigma_position_m
        half_h = self.search_sigmas * prior.sigma_heading_deg
        coarse_p = max(self.position_step_m, min(self.wall_sigma_m, half_p / 8.0))
        coarse_h = max(self.heading_step_deg, min(2.0, half_h / 8.0))
        dxs = np.arange(-half_p, half_p + 1e-9, coarse_p)
        dyaws = np.arange(-half_h, half_h + 1e-9, coarse_h)
        weighted, gx, gy = self._score(r, az, segs, prior, dxs, dxs, dyaws)
        k, i, j = np.unravel_index(int(np.argmax(weighted)), weighted.shape)
        best = float(weighted[k, i, j])
        dx, dy, dyaw = float(dxs[i]), float(dxs[j]), float(dyaws[k])
        # "On the boundary" = within one coarse step of it: a peak that the box
        # only half-contains lands on the last or second-to-last cell.
        on_boundary = (abs(dx) >= half_p - coarse_p - 1e-9 or abs(dy) >= half_p - coarse_p - 1e-9
                       or abs(dyaw) >= half_h - coarse_h - 1e-9)

        coarse_best = weighted
        c_dxs, c_dyaws = dxs, dyaws

        # Fine pass around the coarse winner, clipped to the box.
        fx = np.clip(dx + np.arange(-2 * coarse_p, 2 * coarse_p + 1e-9, self.position_step_m), -half_p, half_p)
        fy = np.clip(dy + np.arange(-2 * coarse_p, 2 * coarse_p + 1e-9, self.position_step_m), -half_p, half_p)
        fh = np.clip(dyaw + np.arange(-2 * coarse_h, 2 * coarse_h + 1e-9, self.heading_step_deg), -half_h, half_h)
        fine, gx, gy = self._score(r, az, segs, prior, fx, fy, fh)
        k, i, j = np.unravel_index(int(np.argmax(fine)), fine.shape)
        best = float(fine[k, i, j])
        dx, dy, dyaw = float(fx[i]), float(fy[j]), float(fh[k])

        # Uncertainty from the peak's width: cells within half the best score,
        # treated as a likelihood, give a weighted spread per axis.
        w = np.where(fine >= 0.5 * best, fine, 0.0)
        wsum = w.sum()
        mx = (w * gx[None]).sum() / wsum
        my = (w * gy[None]).sum() / wsum
        mh = (w * fh[:, None, None]).sum() / wsum
        sig_e = math.sqrt(max((w * (gx[None] - mx) ** 2).sum() / wsum, (self.position_step_m / 2) ** 2))
        sig_n = math.sqrt(max((w * (gy[None] - my) ** 2).sum() / wsum, (self.position_step_m / 2) ** 2))
        sig_h = math.sqrt(max((w * (fh[:, None, None] - mh) ** 2).sum() / wsum, (self.heading_step_deg / 2) ** 2))
        raw_score = float(self._score(r, az, segs, prior, np.array([dx]), np.array([dy]),
                                      np.array([dyaw]), penalise=False)[0][0, 0, 0])

        # Ambiguity, two ways. (1) A rival: a coarse cell scoring within the
        # ratio of the best that lies beyond the fix's own 3-sigma along some
        # axis - a different answer, not this peak's shoulder. (2) A ridge:
        # the fix's sigma along an axis is a large fraction of the prior's,
        # meaning the walls seen did not narrow it (one straight wall, and
        # sliding along it changes nothing).
        ambiguous, axis = False, None
        sep_e = max(PEAK_SEPARATION_M, 3.0 * sig_e)
        sep_n = max(PEAK_SEPARATION_M, 3.0 * sig_n)
        sep_h = max(PEAK_SEPARATION_DEG, 3.0 * sig_h)
        for rk, ri, rj in np.argwhere(coarse_best >= AMBIGUITY_RATIO * float(coarse_best.max())):
            de, dn = abs(c_dxs[ri] - dx), abs(c_dxs[rj] - dy)
            dh = abs(c_dyaws[rk] - dyaw)
            if de >= sep_e or dn >= sep_n or dh >= sep_h:
                ambiguous = True
                axis = ("east" if de >= sep_e and de / sep_e >= dn / sep_n else
                        "north" if dn >= sep_n else "heading")
                break
        if not ambiguous:
            for name, sig, pri in (("east", sig_e, prior.sigma_position_m),
                                   ("north", sig_n, prior.sigma_position_m),
                                   ("heading", sig_h, prior.sigma_heading_deg)):
                if sig >= 0.5 * pri:
                    ambiguous, axis = True, name
                    break

        pts = returns_to_local(returns, prior.heading_deg + dyaw, dx, dy)
        dist = _point_edge_distance(pts, segs)
        inliers = float((dist <= 2.0 * self.wall_sigma_m).mean())
        score = raw_score / n

        lon_scale, lat_scale = local_scales(prior.latitude)
        checks.append(Check.in_range("walls.inlier_fraction", inliers, 0.5, 1.0, unit="fraction"))
        checks.append(Check.that("walls.unambiguous", not ambiguous,
                                 "one peak" if not ambiguous else
                                 f"rival peak along {axis}: the walls seen do not pin that axis"))
        checks.append(Check.that("walls.inside_search", not on_boundary,
                                 "best pose inside the prior's 3-sigma box" if not on_boundary else
                                 "best pose on the search boundary: the prior is worse than its sigma says"))
        accepted = all(c.passed for c in checks)
        return WallFix(
            latitude=prior.latitude + dy / lat_scale,
            longitude=prior.longitude + dx / lon_scale,
            heading_deg=(prior.heading_deg + dyaw) % 360.0,
            dx_m=dx, dy_m=dy, dyaw_deg=dyaw,
            sigma_east_m=sig_e, sigma_north_m=sig_n, sigma_heading_deg=sig_h,
            score=score, inlier_fraction=inliers, n_returns=n, n_edges=int(len(segs)),
            ambiguous=ambiguous, ambiguity_axis=axis, on_boundary=on_boundary,
            accepted=accepted, checks=checks,
        )

    def _score(self, r, az, segs, prior, dxs, dys, dyaws, penalise=True):
        """Score every (dyaw, dx, dy) cell; returns (scores, gx, gy)."""
        import numpy as np

        n = len(r)
        two_s2 = 2.0 * self.wall_sigma_m ** 2
        gx, gy = np.meshgrid(dxs, dys, indexing="ij")
        grid = np.stack([gx, gy], axis=-1).reshape(-1, 2)
        scores = np.empty((len(dyaws), len(dxs), len(dys)))
        for k, dyaw in enumerate(dyaws):
            b = np.radians(prior.heading_deg + dyaw + az)
            base = np.stack([r * np.sin(b), r * np.cos(b)], axis=1)          # (P,2)
            pts = (base[None, :, :] + grid[:, None, :]).reshape(-1, 2)      # (C*P,2)
            d = _point_edge_distance(pts, segs).reshape(len(grid), n)
            scores[k] = np.exp(-(d * d) / two_s2).sum(axis=1).reshape(len(dxs), len(dys))
        if penalise:
            # The prior as a soft penalty. Deliberately wide - sigma_prior * 3
            # - so a genuinely better match at 3 sigma still wins (x0.6), but
            # a look-alike at the box edge with the same raw score does not.
            sp, sh = 3.0 * prior.sigma_position_m, 3.0 * prior.sigma_heading_deg
            pen_p = np.exp(-(gx ** 2 + gy ** 2) / (2.0 * sp ** 2))
            pen_h = np.exp(-(dyaws ** 2) / (2.0 * sh ** 2))
            scores = scores * pen_p[None, :, :] * pen_h[:, None, None]
        return scores, gx, gy

    @staticmethod
    def _rejected(prior: PosePrior, n: int, n_edges: int, checks: List[Check]) -> WallFix:
        return WallFix(
            latitude=prior.latitude, longitude=prior.longitude, heading_deg=prior.heading_deg,
            dx_m=0.0, dy_m=0.0, dyaw_deg=0.0,
            sigma_east_m=prior.sigma_position_m, sigma_north_m=prior.sigma_position_m,
            sigma_heading_deg=prior.sigma_heading_deg,
            score=0.0, inlier_fraction=0.0, n_returns=n, n_edges=n_edges,
            ambiguous=False, ambiguity_axis=None, on_boundary=False,
            accepted=False, checks=checks,
        )


def static_returns(detections: Sequence[dict], max_speed_mps: float = 0.25,
                   min_range_m: float = 0.5, max_range_m: float = 40.0) -> List[WallReturn]:
    """Filter ``radar_detections_all()`` records down to usable wall returns.

    Uses ``is_static`` when the record carries it, else |velocity| under the
    threshold. Drops the near-field self-coupling returns (r < 0.5 m) and
    anything past the range where a wall still gives a usable return.
    """
    out = []
    for d in detections:
        rng = float(d.get("range_m", 0.0))
        if not (min_range_m <= rng <= max_range_m):
            continue
        static = d.get("is_static")
        if static is None:
            static = abs(float(d.get("velocity_mps", 0.0))) <= max_speed_mps
        if static:
            out.append(WallReturn(rng, float(d.get("azimuth_deg", 0.0))))
    return out
