"""
Radar return classification for IWR1843BOOST  —  OpenMV N6 (MicroPython) ready.

Pipeline (all lightweight, MCU-friendly):
  parsed points (x,y,z,doppler)  ->  cluster  ->  features  ->  classify
Classes: static (clutter/structure) / pedestrian / vehicle / unknown.
Rule-based; a trained model (Random Forest / XGBoost) is the upgrade path once
you can log features off-device. The clustering + feature core is pure Python
and CPython-testable.

Feed it the point list from iwr1843_uart.RadarReader:
    for fr in radar.feed(uart.read()):
        result = classify_frame(fr['points'])
"""
import math

# --- tunables ----------------------------------------------------------------
CLUSTER_EPS = 1.2       # metres: points within this join the same cluster
                        #   (large enough to keep a vehicle whole, small enough
                        #    to keep a nearby pedestrian separate)
CLUSTER_MIN_PTS = 2     # clusters smaller than this are dropped as noise
STATIC_V = 0.25         # |Doppler| below this ~ static  (raise if platform moves)
                        # CAVEAT: with the default 3-TX configs the unambiguous
                        # Doppler is only ~±0.65 m/s — a person walking head-on
                        # at 1.2 m/s ALIASES to ~-0.1 m/s and lands below this
                        # threshold. Doppler alone must not decide "static";
                        # the tracker's displacement logic is the safety net.
VEH_EXTENT = 1.5        # metres: bounding extent above this leans "vehicle"
                        # NOTE: extent is computed over a 0.5 s point window,
                        # so it convolves SIZE with MOTION (v*0.5 m of smear);
                        # partially, "fast" reads as "large"
VEH_MIN_PTS = 6         # vehicles return more points than pedestrians
PED_VSPREAD = 0.5       # micro-Doppler spread (limbs) hint for pedestrian


def cluster(points, eps=CLUSTER_EPS, min_pts=CLUSTER_MIN_PTS):
    """points: list of (x,y,z,v). Region-grow clustering by 3D distance."""
    n = len(points)
    used = [False] * n
    groups = []
    eps2 = eps * eps
    for i in range(n):
        if used[i]:
            continue
        stack = [i]
        used[i] = True
        grp = []
        while stack:
            j = stack.pop()
            grp.append(points[j])
            xj, yj, zj = points[j][0], points[j][1], points[j][2]
            for k in range(n):
                if used[k]:
                    continue
                dx = xj - points[k][0]
                dy = yj - points[k][1]
                dz = zj - points[k][2]
                if dx * dx + dy * dy + dz * dz <= eps2:
                    used[k] = True
                    stack.append(k)
        if len(grp) >= min_pts:
            groups.append(grp)
    return groups


def features(grp):
    """Compute the classification features for one cluster."""
    n = len(grp)
    xs = [p[0] for p in grp]
    ys = [p[1] for p in grp]
    zs = [p[2] for p in grp]
    vs = [p[3] for p in grp]
    cx, cy, cz = sum(xs) / n, sum(ys) / n, sum(zs) / n
    ext = math.sqrt((max(xs) - min(xs)) ** 2 +
                    (max(ys) - min(ys)) ** 2 +
                    (max(zs) - min(zs)) ** 2)
    vabs = sum(abs(v) for v in vs) / n
    return {
        "n": n,
        "range": math.sqrt(cx * cx + cy * cy + cz * cz),
        "extent": ext,
        "v_mean": sum(vs) / n,
        "v_abs": vabs,
        "v_spread": max(vs) - min(vs),   # micro-Doppler proxy
        "cx": cx, "cy": cy, "cz": cz,
    }


def classify(f):
    """Rule-based label from features. Order matters: static first."""
    if f["v_abs"] < STATIC_V:
        return "static"                              # not moving -> clutter/structure
    if f["extent"] > VEH_EXTENT and f["n"] >= VEH_MIN_PTS:
        return "vehicle"                             # large + many returns
    if f["v_spread"] > PED_VSPREAD or f["extent"] <= VEH_EXTENT:
        return "pedestrian"                          # small + limb micro-Doppler
    return "unknown"


def cluster_dict(grp, include_points=False):
    """Point group -> the standard cluster dict (features + label).
    Also used to rebuild clusters after reflectivity-based splitting
    (radar_material.split_mixed_clusters)."""
    f = features(grp)
    c = {
        "label": classify(f),
        "range_m": round(f["range"], 2),
        "doppler_mps": round(f["v_mean"], 2),
        "extent_m": round(f["extent"], 2),
        "n_points": f["n"],
        "centroid": (round(f["cx"], 2), round(f["cy"], 2), round(f["cz"], 2)),
    }
    if include_points:
        c["points"] = grp
    return c


def classify_frame(points, include_points=False):
    """Full frame: cluster -> features -> label. Returns list of dicts.

    Points may be (x,y,z,v) or (x,y,z,v,snr_db) — extra fields pass through.
    include_points=True attaches the member points to each cluster (needed by
    radar_material for reflectivity/material estimation).
    """
    return [cluster_dict(grp, include_points) for grp in cluster(points)]


# ---- OpenMV N6 usage (uncomment on device) ----------------------------------
# from machine import UART
# from iwr1843_uart import RadarReader
# uart = UART(1, 921600, bits=8, parity=None, stop=1, timeout=5)
# radar = RadarReader()
# while True:
#     if uart.any():
#         for fr in radar.feed(uart.read()):
#             for obj in classify_frame(fr['points']):
#                 print(obj)   # -> {'label': 'pedestrian', 'range_m': 6.1, ...}
