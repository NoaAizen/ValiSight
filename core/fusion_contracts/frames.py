"""Named reference-frame registry with a validated transform graph.

Every frame in the system has an explicit name; every transform declares its
source and destination. The rules (see the fusion spec and CLAUDE.md):

* An unregistered transform raises — it never silently falls back to identity.
* The transform graph is checked at load time: no cycles, and no two distinct
  paths between the same pair of frames. Two paths are a bug — that is exactly
  the shape of the two parallel fusion stacks the project is collapsing.
* The wgs84_ellipsoidal <-> egm2008_orthometric edge must go through
  ``geoid_is_live``. A height conversion that returns a value without proving
  the grid is live raises (CLAUDE.md trap #7).
"""
from collections import deque

from core.physics_invariants import geoid_is_live

# --- the frames that exist in the system -------------------------------------
RADAR_SPHERICAL = "radar_spherical"        # (phi, theta, rho) — IWR1843 output
RADAR_BODY = "radar_body"                  # cartesian, origin at array centre
THERMAL_PIXEL = "thermal_pixel"            # image plane, 160x120
THERMAL_CAMERA = "thermal_camera"          # cartesian, origin at optical centre
PLATFORM_BODY = "platform_body"            # the platform frame (vehicle/carrier)
ENU_LOCAL = "enu_local"                    # from copdem_prior.enu_grid()
WGS84_ELLIPSOIDAL = "wgs84_ellipsoidal"    # ellipsoidal height
EGM2008_ORTHOMETRIC = "egm2008_orthometric"  # orthometric height

KNOWN_FRAMES = frozenset({
    RADAR_SPHERICAL, RADAR_BODY, THERMAL_PIXEL, THERMAL_CAMERA,
    PLATFORM_BODY, ENU_LOCAL, WGS84_ELLIPSOIDAL, EGM2008_ORTHOMETRIC,
})


class FrameError(Exception):
    pass


class UnknownFrame(FrameError):
    pass


class UnregisteredTransform(FrameError):
    pass


class AmbiguousTransformGraph(FrameError):
    pass


class GeoidNotLive(FrameError):
    pass


class FrameGraph:
    """A registry of named frames and the directed transforms between them."""

    def __init__(self, frames=KNOWN_FRAMES):
        self._frames = set(frames)
        self._edges = {}          # (src, dst) -> callable(point) -> point
        self._adj = {}            # src -> [dst, ...]

    def add_frame(self, name):
        self._frames.add(name)

    def add_transform(self, src, dst, fn, inverse=None):
        for f in (src, dst):
            if f not in self._frames:
                raise UnknownFrame("frame %r is not registered" % (f,))
        self._edges[(src, dst)] = fn
        self._adj.setdefault(src, []).append(dst)
        if inverse is not None:
            self._edges[(dst, src)] = inverse
            self._adj.setdefault(dst, []).append(src)

    def validate(self):
        """No cycles and no two distinct paths between any frame pair.

        Collapses fwd/inverse to one undirected edge and checks the undirected
        graph is a forest via union-find. Adding an edge between two already
        connected frames means a second path exists -> raise.
        """
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        seen = set()
        for (s, d) in self._edges:
            pair = frozenset((s, d))
            if pair in seen:
                continue
            seen.add(pair)
            rs, rd = find(s), find(d)
            if rs == rd:
                raise AmbiguousTransformGraph(
                    "two distinct paths between %s and %s — that is the shape "
                    "of two parallel fusion stacks, not redundancy" % (s, d))
            parent[rs] = rd
        return True

    def _find_path(self, src, dst):
        q = deque([[src]])
        visited = {src}
        while q:
            path = q.popleft()
            node = path[-1]
            if node == dst:
                return path
            for nxt in self._adj.get(node, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(path + [nxt])
        return None

    def transform(self, point, src, dst):
        """Transform ``point`` from ``src`` to ``dst``.

        Same frame -> the point unchanged. No registered path -> raise
        (never a silent identity fallback).
        """
        for f in (src, dst):
            if f not in self._frames:
                raise UnknownFrame("frame %r is not registered" % (f,))
        if src == dst:
            return point
        path = self._find_path(src, dst)
        if path is None:
            raise UnregisteredTransform(
                "no registered transform %s -> %s; refusing to fall back to "
                "identity" % (src, dst))
        cur = point
        for a, b in zip(path, path[1:]):
            cur = self._edges[(a, b)](cur)
        return cur


# --- the geoid-guarded height edge -------------------------------------------

def make_geoid_transforms(separation_probe):
    """Build (ellipsoidal->orthometric, orthometric->ellipsoidal) transforms
    that refuse to run unless ``geoid_is_live(separation_probe)`` passes.

    Points are ``(lat, lon, height_m)``; only the height changes:
    orthometric = ellipsoidal - N,  ellipsoidal = orthometric + N.
    """
    def _guard():
        ok, why = geoid_is_live(separation_probe)
        if not ok:
            raise GeoidNotLive(
                "geoid height conversion blocked: %s" % why)

    def ell_to_ortho(point):
        _guard()
        lat, lon, h_ell = point
        n = separation_probe(lon, lat)
        return (lat, lon, h_ell - n)

    def ortho_to_ell(point):
        _guard()
        lat, lon, h_ortho = point
        n = separation_probe(lon, lat)
        return (lat, lon, h_ortho + n)

    return ell_to_ortho, ortho_to_ell


def register_geoid_edge(graph, separation_probe):
    """Register the guarded wgs84_ellipsoidal <-> egm2008_orthometric edge."""
    fwd, inv = make_geoid_transforms(separation_probe)
    graph.add_transform(WGS84_ELLIPSOIDAL, EGM2008_ORTHOMETRIC, fwd,
                        inverse=inv)
