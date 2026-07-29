"""Odometry-grade conditioning of radar detections. Pure maths, no hardware.

This is deliberately NOT `radar_gate`. That module is correct for its job --
feeding classification -- and `radar_classify_n6` depends on its behaviour, so
it is left alone. But its third gate drops any point with no neighbour within
1.2 m, and sparse world-fixed returns are exactly what has no neighbours. On a
real scene it measured **kept = 0 out of 39 raw**. Running odometry through it
would leave nothing to solve with.

What this module does instead: recover the geometry each point needs for the
Doppler ego-velocity model, and report honestly on whether a frame is solvable.

THE MODEL. A scatterer fixed to the world, seen from a sensor translating at
velocity `v`, returns a radial velocity that is just the projection of `-v` onto
the line of sight:

    v_r = -u . v          u = unit vector from sensor to the point

Two points at different bearings determine `v` in 2-D; three determine it in
3-D. Everything downstream is that equation plus the question of which points
are actually world-fixed.

WHAT LIMITS THE ANSWER. Not the Doppler quantum. The demo reports the peak
Doppler bin with no interpolation, so with 0.1217 m/s bins the velocity error per
point is 0.1217/sqrt(12) = 35 mm/s. But the bearing is uncertain too, and a
bearing error tilts the projection, contributing |v| * sigma_theta. At walking
speed and 2.5 degrees that is 61 mm/s -- larger. So bearing accuracy, not
Doppler resolution, sets the floor at walking speed.

Note 2.5 degrees is the ANGLE ESTIMATE error for one isolated detection, not the
14-degree beamwidth. Beamwidth governs whether two scatterers can be told apart
(and is why there are ~6 detections and not 60); the estimate for a single
resolved peak is far better than the beamwidth.
"""
import math

import chirp_geometry


class DopplerGrid:
    """The Doppler axis one chirp config produces: bin size and bin count.

    Carried on every Point so that nothing downstream has to be told separately
    which config a frame came from, and so the old failure -- a module-level
    0.125 repeated as a default argument in another module -- is structurally
    impossible.

    `fold` is the property that matters and the one the old code had no name
    for. The reported radial velocity is not `v_r`, it is `v_r` modulo `fold`,
    taken into (-fold/2, +fold/2]. Every wrap question is arithmetic on that.
    """

    __slots__ = ("bin_mps", "n_bins", "source")

    def __init__(self, bin_mps, n_bins, source=""):
        self.bin_mps = float(bin_mps)
        self.n_bins = int(n_bins)
        self.source = source

    @property
    def sigma(self):
        """Per-point velocity sd from the quantiser alone (uniform, no interp)."""
        return self.bin_mps / math.sqrt(12.0)

    @property
    def v_max(self):
        """Largest unambiguous |v_r|. Beyond it the axis folds, silently."""
        return self.n_bins * self.bin_mps / 2.0

    @property
    def fold(self):
        return self.n_bins * self.bin_mps

    def wrap(self, r):
        """Residual modulo one fold, into (-fold/2, +fold/2].

        A wrapped return is not noise -- it is wrong by EXACTLY one fold. That
        is what makes aliasing resolvable rather than merely detectable.
        """
        f = self.fold
        return r - f * math.floor(r / f + 0.5)

    def bin_index(self, vr):
        return int(round(vr / self.bin_mps))

    def is_outer_bin(self, vr):
        """True at the two bins either side of the fold boundary.

        Reported as a risk indicator only. It is NOT a speed detector: which
        bearing lands on the boundary depends on speed, so with ~6 sparse points
        occupancy is a fold-phase coin flip. Measured on real geometry it reads
        100% at 1.4 m/s forward and 0% at 2.6 m/s. Never gate on it.
        """
        k = self.bin_index(vr)
        return k >= self.n_bins // 2 - 1 or k <= -(self.n_bins // 2)

    def __repr__(self):
        return ("DopplerGrid(bin=%.6f, n=%d, v_max=%.4f, %s)"
                % (self.bin_mps, self.n_bins, self.v_max, self.source or "?"))


# The stock geometry, computed rather than copied so it cannot drift from the
# formula. See chirp_geometry for why lambda is taken at the ADC-window centre
# (79.21 GHz) and not at the 77 GHz start frequency -- the latter is what gave
# the 0.125 / 1.001 this module used to carry, both 2.8% high. Against 60144
# measured points the values below are right to 0.002%.
STOCK_GRID = DopplerGrid(chirp_geometry.STOCK["bin_mps"],
                         chirp_geometry.STOCK["n_bins"],
                         source="stock_iwr1843.cfg (derived)")

DOPPLER_BIN_MPS = STOCK_GRID.bin_mps            # 0.121713
DOPPLER_SIGMA_MPS = STOCK_GRID.sigma            # 35 mm/s, uniform quantiser
BEARING_SIGMA_RAD = math.radians(2.5)

MIN_RANGE_M = 0.25          # matches cfarFovCfg; below this is antenna coupling
MAX_RANGE_M = 9.0


class Point:
    """One detection with the geometry the ego-velocity model needs."""

    __slots__ = ("x", "y", "z", "vr", "rng", "az", "el", "u", "snr", "noise",
                 "grid")

    def __init__(self, x, y, z, vr, snr=None, noise=None, grid=None):
        self.x, self.y, self.z, self.vr = x, y, z, vr
        self.snr, self.noise = snr, noise
        # The Doppler axis this reading came off. Carried per point so a replay
        # can score a session under its own config without any signature in the
        # solve path changing.
        self.grid = grid or STOCK_GRID
        self.rng = math.sqrt(x * x + y * y + z * z)
        if self.rng > 1e-9:
            self.u = (x / self.rng, y / self.rng, z / self.rng)
        else:
            self.u = (0.0, 0.0, 0.0)
        # Project convention is x forward, y left, z up; azimuth positive to the
        # right, matching radar_gate._range_az so the two agree on sign.
        self.az = math.degrees(math.atan2(-y, x))
        horiz = math.sqrt(x * x + y * y)
        self.el = math.degrees(math.atan2(z, horiz)) if horiz > 1e-9 else 0.0

    def sigma(self, speed):
        """Velocity uncertainty for this point, given a current speed estimate.

        Combines the Doppler quantiser with the bearing error propagated through
        the projection. The bearing term scales with speed, which is why it
        dominates while moving and vanishes at rest.
        """
        return math.sqrt(self.grid.sigma ** 2 +
                         (abs(speed) * BEARING_SIGMA_RAD) ** 2)

    def as_tuple(self):
        return (self.x, self.y, self.z, self.vr)


def condition(points, snr=None, noise=None,
              min_range=MIN_RANGE_M, max_range=MAX_RANGE_M, grid=None):
    """Raw parser tuples -> [Point], keeping only range-plausible detections.

    Range gating only. No isolation gate, no SNR gate: a weak isolated return
    from a wall is signal here, not noise, and the ego-velocity residual is a
    far better outlier test than any per-point threshold could be.

    `grid` is the session's Doppler axis; None means the stock config. This is
    the ONLY place a caller has to supply it -- it rides on the points from here.
    """
    g = grid or STOCK_GRID
    out = []
    for i, p in enumerate(points):
        pt = Point(p[0], p[1], p[2], p[3],
                   snr[i] if snr and i < len(snr) else None,
                   noise[i] if noise and i < len(noise) else None,
                   grid=g)
        if min_range <= pt.rng <= max_range:
            out.append(pt)
    return out


def infer_doppler_grid(vr_values, source="inferred"):
    """Recover the Doppler axis from the reported velocities themselves.

    The demo reports peak bins with no interpolation, so every value is an exact
    integer multiple of one quantum. Take the smallest non-zero magnitude as a
    candidate quantum, check every value against it, and size the axis from the
    largest magnitude seen.

    Returns None when the values do not lie on a single grid -- better than a
    confident wrong answer.

    THE FAILURE MODE, which any caller must repeat to the user: if the session
    never populated the outer bins, n_bins comes out too SMALL, so `fold` is too
    small and wrap-detection becomes over-eager. It fails toward false alarms
    rather than toward silence, which is the safe direction, but it is a guess
    and must be labelled as one.
    """
    mags = sorted({abs(v) for v in vr_values if abs(v) > 1e-9})
    if not mags:
        return None
    q = mags[0]
    for m in mags:
        k = round(m / q)
        if k < 1 or abs(m - k * q) > 0.01 * q:
            return None                       # not a single uniform grid
    n_bins = 2
    while n_bins * q / 2.0 < mags[-1] - 1e-9:
        n_bins *= 2                           # Doppler FFT length is a power of 2
    return DopplerGrid(q, n_bins, source=source)


def azimuth_spread(pts):
    """Angular extent of the bearings, in degrees.

    Lateral velocity is only observable through the spread of bearings: with
    every point straight ahead, `u` is the same vector for all of them and the
    cross-track component is unconstrained no matter how many points there are.
    """
    if len(pts) < 2:
        return 0.0
    az = [p.az for p in pts]
    return max(az) - min(az)


def elevation_spread(pts):
    if len(pts) < 2:
        return 0.0
    el = [p.el for p in pts]
    return max(el) - min(el)


def _sym_inverse_2x2(m):
    a, b, c = m[0][0], m[0][1], m[1][1]
    det = a * c - b * b
    if abs(det) < 1e-12:
        return None
    return ((c / det, -b / det), (-b / det, a / det))


def gdop(pts, dims=2):
    """Geometric dilution of precision: how much geometry amplifies noise.

    sqrt(trace((A^T A)^-1)) for A whose rows are the unit vectors. A GDOP of 1
    means the geometry costs nothing; 6 means a 36 mm/s Doppler error becomes a
    216 mm/s velocity error. Returns None when the geometry is singular, which
    is the honest answer and must not be reported as a large number.
    """
    if len(pts) < dims:
        return None
    m = [[0.0] * dims for _ in range(dims)]
    for p in pts:
        u = p.u[:dims]
        for i in range(dims):
            for j in range(dims):
                m[i][j] += u[i] * u[j]
    if dims != 2:
        raise ValueError("gdop currently implements the 2-D case only")
    inv = _sym_inverse_2x2(m)
    if inv is None:
        return None
    tr = inv[0][0] + inv[1][1]
    return math.sqrt(tr) if tr > 0 else None


def range_bin_index(pt, bin_m=0.0436):
    """Which range bin a point fell in, for de-duplicating one scatterer.

    With CFAR peak grouping disabled a single wall returns several adjacent
    bins. Those are correlated samples of the same scatterer: they inflate the
    apparent point count without adding information, and a minimal RANSAC set
    drawn from two of them is near-singular.
    """
    return int(pt.rng / bin_m)


def frame_diagnostics(pts):
    """Everything needed to judge whether this frame could support a solve."""
    return {"n": len(pts),
            "az_spread": round(azimuth_spread(pts), 2),
            "el_spread": round(elevation_spread(pts), 2),
            "gdop": (round(gdop(pts), 3) if gdop(pts) is not None else None),
            "n_range_bins": len({range_bin_index(p) for p in pts}),
            "rng_min": round(min((p.rng for p in pts), default=0.0), 2),
            "rng_max": round(max((p.rng for p in pts), default=0.0), 2)}
