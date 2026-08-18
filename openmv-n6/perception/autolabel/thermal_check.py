"""Map a visible-frame box into the thermal frame and score its heat.

Same composition live.py uses via fusion_temp_region(), reimplemented in numpy
against the warp LUT so the autolabel stage needs neither libfusion nor a
running pipeline. The LUT is the one calib.py emits: (100, 160, 2) uint16 Q8
thermal coordinates per low-res grid cell, 0xFFFF = no valid mapping.

Recorded live sessions carry NO radiometric scale (frames.jsonl has no
tmin/c_per_lsb), so absolute "31-39 C" body-heat gating is impossible offline.
The score is therefore RELATIVE: how far the box's warped maximum sits above
the frame's background median, in 8-bit counts. The threshold is chosen from
the measured distribution, not assumed.
"""
import os

import numpy as np

ART = os.path.join(os.path.dirname(__file__), '..', '..', 'calib-artifacts')
LOW_W, LOW_H = 160, 100
DECIMATION = 4
FUSION_INVALID = 0xFFFF
TH_W, TH_H = 160, 120


class ThermalBoxCheck:
    def __init__(self, lut_path=None):
        path = lut_path or os.path.join(ART, 'warp_3.5.lut')
        lut = np.fromfile(path, dtype=np.uint16).reshape(LOW_H, LOW_W, 2)
        self.valid = lut[..., 0] != FUSION_INVALID
        # Q8 -> integer thermal pixel to sample (same rounding as fusion.c)
        self.tu = np.clip((lut[..., 0].astype(np.int32) + 128) >> 8, 0, TH_W - 1)
        self.tv = np.clip((lut[..., 1].astype(np.int32) + 128) >> 8, 0, TH_H - 1)

    def region(self, thermal, x, y, w, h):
        """Visible-frame box -> stats over the warped thermal samples.

        Returns None when the box has no valid thermal coverage (outside the
        overlap ROI) - which downstream must treat as "unknown", never "cold".
        """
        gx0 = max(0, int(x) // DECIMATION)
        gy0 = max(0, int(y) // DECIMATION)
        gx1 = min(LOW_W, int(np.ceil((x + w) / DECIMATION)))
        gy1 = min(LOW_H, int(np.ceil((y + h) / DECIMATION)))
        if gx1 <= gx0 or gy1 <= gy0:
            return None
        m = self.valid[gy0:gy1, gx0:gx1]
        if not m.any():
            return None
        vals = thermal[self.tv[gy0:gy1, gx0:gx1][m],
                       self.tu[gy0:gy1, gx0:gx1][m]]
        return {'n': int(m.sum()),
                'max': int(vals.max()), 'mean': float(vals.mean()),
                'p90': float(np.percentile(vals, 90))}

    @staticmethod
    def background(thermal):
        """Frame-level background level: the median is robust to one person
        occupying even a third of the (much wider) thermal FOV."""
        return float(np.median(thermal))
