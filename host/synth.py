#!/usr/bin/env python3
"""Generate a synthetic frame pair with known ground truth.

Real captures are the goal, but they cannot tell you whether the pipeline is
*correct* - only whether the picture looks nice. This scene is built so every
stage has something falsifiable to check against:

  * a hot blob at a known thermal coordinate    -> registration + palette
  * a deliberately partial thermal footprint    -> the uncovered path
  * fine high-contrast structure in the luma    -> detail injection
  * flat luma under the blob                    -> lets the blob be asserted
                                                   without detail contaminating it
"""
import json
import os

import numpy as np

OUT_W, OUT_H = 640, 400
LOW_W, LOW_H = 160, 100
TH_W, TH_H = 160, 120

# The thermal camera sees only part of what the visible camera sees. Expressed
# in low-res grid coordinates; outside this box the LUT is marked invalid.
COV_X0, COV_X1 = 10, 150
COV_Y0, COV_Y1 = 5, 95

HOT_TX, HOT_TY, HOT_R = 100, 40, 14     # hot blob, thermal coords
COLD_CODE, WARM_CODE, HOT_CODE = 60, 95, 235

FLAT_BOX = (330, 90, 470, 230)          # flat luma region, output coords
OUT_DIR = "synth"

# The rows this unit returns pinned at 255 in every frame - the sensor defect
# fusion_thermal_prep() repairs. Reproduced here at the measured row numbers,
# including the two adjacent pairs (35/36 and 57/58), which is the case a naive
# nearest-live-row fill gets wrong.
DEAD_ROWS = (6, 12, 28, 35, 36, 55, 57, 58)


def make_luma():
    """An electrical-panel-ish scene: busbars, terminals, screws, fine labels."""
    y = np.full((OUT_H, OUT_W), 40, np.uint8)

    for x in range(0, OUT_W, 64):       # vertical busbars
        y[:, x:x + 26] = 150
    for row in range(30, OUT_H, 110):   # horizontal rails
        y[row:row + 12, :] = 95

    rng = np.random.default_rng(7)
    for cx in range(40, OUT_W - 20, 64):        # terminal blocks + screw heads
        for cy in range(60, OUT_H - 20, 110):
            y[cy:cy + 34, cx:cx + 30] = 200
            y[cy + 12:cy + 22, cx + 10:cx + 20] = 70
    for _ in range(500):                        # fine label text
        tx, ty = rng.integers(0, OUT_W - 6), rng.integers(0, OUT_H - 2)
        y[ty:ty + 2, tx:tx + 5] = 235

    # keep one region flat so the hot-blob assertion is not polluted by detail
    x0, y0, x1, y1 = FLAT_BOX
    y[y0:y1, x0:x1] = 120
    return y


def make_thermal():
    t = np.full((TH_H, TH_W), COLD_CODE, np.uint8)
    yy, xx = np.mgrid[0:TH_H, 0:TH_W]

    t[(xx > TH_W * 0.55)] = WARM_CODE                       # a warm half
    d2 = (xx - HOT_TX) ** 2 + (yy - HOT_TY) ** 2            # the hot spot
    t[d2 <= HOT_R ** 2] = HOT_CODE
    return t


def make_warp():
    """Low-res grid -> thermal coords, Q8, FUSION_INVALID outside the footprint."""
    INVALID = 0xFFFF
    lut = np.full((LOW_H, LOW_W, 2), INVALID, np.uint16)

    gx, gy = np.mgrid[0:LOW_W, 0:LOW_H]
    gx, gy = gx.T, gy.T
    inside = (gx >= COV_X0) & (gx < COV_X1) & (gy >= COV_Y0) & (gy < COV_Y1)

    u = (gx - COV_X0) * (TH_W - 1) / (COV_X1 - COV_X0 - 1)
    v = (gy - COV_Y0) * (TH_H - 1) / (COV_Y1 - COV_Y0 - 1)

    lut[..., 0] = np.where(inside, np.clip(u * 256, 0, (TH_W - 1) * 256), INVALID)
    lut[..., 1] = np.where(inside, np.clip(v * 256, 0, (TH_H - 1) * 256), INVALID)
    return lut


def hot_blob_output_box():
    """Where the hot blob must land in output pixels, per the warp above."""
    sx = (COV_X1 - COV_X0 - 1) / (TH_W - 1)
    sy = (COV_Y1 - COV_Y0 - 1) / (TH_H - 1)
    lx, ly = COV_X0 + HOT_TX * sx, COV_Y0 + HOT_TY * sy
    rx, ry = HOT_R * sx, HOT_R * sy
    f = OUT_W // LOW_W
    return [int(round(v)) for v in
            ((lx - rx * 0.4) * f, (ly - ry * 0.4) * f, (lx + rx * 0.4) * f, (ly + ry * 0.4) * f)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    y, t, lut = make_luma(), make_thermal(), make_warp()

    y.tofile(os.path.join(OUT_DIR, "y.raw"))
    t.tofile(os.path.join(OUT_DIR, "thermal.raw"))
    lut.tofile(os.path.join(OUT_DIR, "warp.lut"))

    dead = t.copy()
    dead[list(DEAD_ROWS)] = 255
    dead.tofile(os.path.join(OUT_DIR, "thermal_dead.raw"))

    # A flat hot object filling the top 40% of the field: every one of those rows
    # is featureless and far above the frame median, so each passes the dead-row
    # test individually. It is a scene, not a defect, and interpolating it away
    # would erase the hottest thing in the frame - so the count valve has to
    # decline the whole frame. 40% is chosen to trip the valve (>25%) while
    # leaving the median down in the scene, where an easier all-255 frame would
    # have pushed the median to 255 and made the test pass for the wrong reason.
    saturated = t.copy()
    saturated[:int(TH_H * 0.4)] = 255
    saturated.tofile(os.path.join(OUT_DIR, "thermal_hot.raw"))

    meta = {
        "out_w": OUT_W, "out_h": OUT_H, "low_w": LOW_W, "low_h": LOW_H,
        "th_w": TH_W, "th_h": TH_H,
        "coverage_low": [COV_X0, COV_Y0, COV_X1, COV_Y1],
        "uncovered_output_box": [0, 0, COV_X0 * (OUT_W // LOW_W), OUT_H],
        "hot_output_box": hot_blob_output_box(),
        "flat_box": list(FLAT_BOX),
        "codes": {"cold": COLD_CODE, "warm": WARM_CODE, "hot": HOT_CODE},
        "dead_rows": list(DEAD_ROWS),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("synth: y %dx%d, thermal %dx%d, warp %dx%d" % (OUT_W, OUT_H, TH_W, TH_H, LOW_W, LOW_H))
    print("  hot blob should land at output box %s" % (meta["hot_output_box"],))
    print("  uncovered strip: x < %d" % meta["uncovered_output_box"][2])


if __name__ == "__main__":
    main()
