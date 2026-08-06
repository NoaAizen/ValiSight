"""
Semantic scene classes the COCO camera detector cannot see: TREE and GROUND.

COCO has no 'tree'/'ground' class, so these are inferred by fusing:
  geometry from the radar track (pure Python, MCU-friendly):
    ground -> points lying at floor level (z ~= -radar_height), flat
    tree   -> tall vertical static cluster, weak-to-mid reflectivity
              (wood/foliage at 77 GHz), optionally leaf micro-motion
  color from the camera (PC only, cv2):
    vegetation is confirmed by the green fraction of the patch around the
    track's projected position.

Usage in the live loop:
    sem = semantic_from_geometry(tcluster, radar_height)
    if sem == "tree?":
        sem = "tree" if green_fraction(img, px, py, half) >= GREEN_MIN else None
"""
import math

# --- geometry tunables --------------------------------------------------------
GROUND_Z_TOL = 0.35     # |z_world| below this counts as floor level (m)
GROUND_MAX_EXT_Z = 0.5  # ground is flat: little vertical spread
GROUND_MIN_RANGE = 0.8  # too-close low returns are usually the rig itself
TREE_MIN_EXT_Z = 0.9    # canopy+trunk vertical spread (m)
TREE_STRONG_EXT_Z = 1.5 # this tall + static -> tree even without color proof
TREE_MAX_REFL = 10.0    # metal poles glint above the range-normalized
                        # baseline; trunks/foliage don't (score dB, matches
                        # radar_material.METAL_DB)

# --- color tunables -----------------------------------------------------------
GREEN_MIN = 0.18        # fraction of vegetation-colored pixels to confirm
# HSV vegetation window (OpenCV ranges: H 0-179, S/V 0-255)
HUE_LO, HUE_HI = 30, 90
SAT_MIN, VAL_MIN = 50, 40


def semantic_from_geometry(c, radar_height):
    """Radar-only semantic guess for a track-cluster dict.

    c must carry 'centroid', 'range_m', 'label', 'refl_db' and (if available)
    'ext_xyz'. Returns 'ground', 'tree?' (needs color confirmation),
    'tree' (geometry alone is conclusive), or None.
    """
    x, y, z = c["centroid"]
    ext = c.get("ext_xyz") or (c.get("extent_m", 0.0),) * 3
    z_world = z + radar_height              # height above the floor
    horiz = math.sqrt(x * x + y * y)        # distance along the floor
    refl = c.get("refl_db")

    # ground: at floor level, flat, not right under the rig
    if (abs(z_world) <= GROUND_Z_TOL and ext[2] <= GROUND_MAX_EXT_Z
            and horiz >= GROUND_MIN_RANGE and c.get("label") == "static"):
        return "ground"

    # tree: tall static column, not metallic
    if (c.get("label") == "static" and ext[2] >= TREE_MIN_EXT_Z
            and (refl is None or refl <= TREE_MAX_REFL)):
        return "tree" if ext[2] >= TREE_STRONG_EXT_Z else "tree?"
    return None


def green_fraction(img, px, py, half):
    """Fraction of vegetation-colored pixels in a patch around (px, py).

    PC-only helper (cv2/numpy). Returns 0.0 for empty/out-of-frame patches.
    """
    import cv2
    H, W = img.shape[:2]
    x0, x1 = max(px - half, 0), min(px + half, W)
    y0, y1 = max(py - half, 0), min(py + half, H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    patch = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((h >= HUE_LO) & (h <= HUE_HI) & (s >= SAT_MIN) & (v >= VAL_MIN))
    return float(mask.mean())


def resolve_semantics(tclusters, img, hfov_deg, yaw_offset_deg, radar_height):
    """Attach c['semantic'] to each track-cluster (in place) and return them.

    Projects each candidate into the camera frame for the green check; when
    the projection is unusable (dark frame, outside FOV) geometry decides.
    """
    if img is None:
        for c in tclusters:
            sem = semantic_from_geometry(c, radar_height)
            c["semantic"] = None if sem == "tree?" else sem
        return tclusters
    H, W = img.shape[:2]
    fx = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    for c in tclusters:
        sem = semantic_from_geometry(c, radar_height)
        if sem == "tree?":
            x, y, z = c["centroid"]
            az = math.atan2(-y, max(x, 0.01)) - math.radians(yaw_offset_deg)
            el = math.atan2(z, max(math.sqrt(x * x + y * y), 0.01))
            px = int(W / 2.0 + fx * math.tan(az))
            py = int(H / 2.0 - fx * math.tan(el))
            half = int(min(max(fx * c.get("extent_m", 0.5) /
                               (2.0 * max(c["range_m"], 0.3)), 15), W // 4))
            sem = "tree" if green_fraction(img, px, py, half) >= GREEN_MIN \
                else None
        c["semantic"] = sem
    return tclusters
