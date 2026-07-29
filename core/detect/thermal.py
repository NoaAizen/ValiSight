"""Warm-blob person detection on a repaired Lepton frame.

Input is the DECISION-PATH image: ``lepton_fix.decode -> destripe -> repair``
(plus the per-pixel FPN subtraction when a map exists), i.e. the 120x160
uint8 AGC output where hotter = higher DN. It is NOT the display image —
Regain, palettes and JPEG are presentation and never reach this module.

Synthetic pixels never become evidence. ``repair()`` invents the dead rows by
interpolation, so the caller passes them as ``invalid_rows`` and the detector
treats them as glue only: they may BRIDGE a blob that straddles a dead-row
run (their values came from the blob's own neighbours anyway), but they
contribute nothing to area, contrast, or confidence. A blob living entirely
in invalid rows does not exist. This is the row-55-63 problem made explicit:
a person crossing that 9-row band stays one detection, with confidence
discounted by exactly the fraction of them the sensor never saw.

Confidence is thermal's own failure signature, quantified: it is high only
when a blob stands well clear of the background (contrast term) with enough
real pixels (area term, valid rows only). A flat field, low contrast, or an
occluded person all map to weak/absent boxes — which is precisely what
core.fusion.uncertainty.thermal_estimate() turns into high sigma so radar
can carry the track.

Pure module: numpy + stdlib. Rule-based analytic scaffold, same status as
the uncertainty constants: hand-set starting points, not calibrated values.
"""
import math
from dataclasses import dataclass

import numpy as np

# --- tunables ----------------------------------------------------------------
# ANALYTIC SCAFFOLD — hand-set, in DN of the AGC 8-bit output (pre-Regain,
# where the whole scene spans ~70 levels and row-to-row detail is ~1-5 DN).
NOISE_K = 3.0                 # threshold: background + K robust sigmas...
MIN_CONTRAST_DN = 8.0         # ...but never less than this above background
MIN_AREA_PX = 12              # valid pixels; smaller is noise / FPN residue
MAX_AREA_FRAC = 0.6           # bigger than this is a scene change, not a target
MIN_VALID_FRACTION = 0.25     # blob mostly made of synthetic rows is not evidence
FULL_CONF_CONTRAST_DN = 25.0  # contrast term saturates here
FULL_CONF_AREA_PX = 120.0     # area term saturates here (~person at mid-range)


@dataclass
class Box:
    """One warm blob. ``confidence`` is the field thermal_estimate() reads."""
    x: int                    # bounding box, image coordinates
    y: int
    w: int
    h: int
    confidence: float         # in [0,1]
    contrast_dn: float        # mean valid-pixel excess over background
    area_px: int              # VALID pixels only — synthetic rows excluded
    valid_fraction: float     # share of the blob the sensor actually saw
    mean_dn: float = 0.0      # mean valid-pixel level (for absolute labelling
    max_dn: float = 0.0       # by a radiometric caller; meaningless after AGC)
    cx: float = 0.0           # valid-pixel centroid, image coordinates
    cy: float = 0.0


def detect(frame, invalid_rows=()):
    """Repaired frame (H x W, uint8-ish) -> list of Box, best first.

    ``invalid_rows``: rows whose pixels are interpolation, not measurement
    (lepton_fix.DEAD_ROWS for this module). They bridge connectivity but
    carry zero evidence.
    """
    img = np.asarray(frame, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("expected a 2D grayscale frame, got shape %r"
                         % (img.shape,))
    h, w = img.shape
    valid = np.ones(h, dtype=bool)
    if len(invalid_rows):
        rows = np.asarray(invalid_rows, dtype=int)
        if rows.min() < 0 or rows.max() >= h:
            raise ValueError("invalid_rows outside the frame")
        valid[rows] = False
    vmask = np.zeros_like(img, dtype=bool)
    vmask[valid] = True

    # Robust background from real pixels only: median + MAD-derived sigma.
    pool = img[vmask]
    bg = float(np.median(pool))
    sigma = 1.4826 * float(np.median(np.abs(pool - bg)))
    thr = bg + max(MIN_CONTRAST_DN, NOISE_K * sigma)

    hot = img > thr                       # synthetic rows included: glue only
    boxes = []
    max_area = MAX_AREA_FRAC * float(vmask.sum())
    for comp in _components(hot):
        comp_valid = comp & vmask
        area = int(comp_valid.sum())
        total = int(comp.sum())
        if area < MIN_AREA_PX or area > max_area:
            continue
        valid_fraction = area / float(total)
        if valid_fraction < MIN_VALID_FRACTION:
            continue
        vals = img[comp_valid]
        contrast = float(vals.mean()) - bg
        ys, xs = np.nonzero(comp)
        ys_v, xs_v = np.nonzero(comp_valid)
        conf = (min(1.0, contrast / FULL_CONF_CONTRAST_DN)
                * min(1.0, area / FULL_CONF_AREA_PX)
                * valid_fraction)
        boxes.append(Box(x=int(xs.min()), y=int(ys.min()),
                         w=int(xs.max() - xs.min() + 1),
                         h=int(ys.max() - ys.min() + 1),
                         confidence=_clamp01(conf),
                         contrast_dn=contrast,
                         area_px=area,
                         valid_fraction=valid_fraction,
                         mean_dn=float(vals.mean()),
                         max_dn=float(vals.max()),
                         cx=float(xs_v.mean()),
                         cy=float(ys_v.mean())))
    boxes.sort(key=lambda b: b.confidence, reverse=True)
    return boxes


def best_box(frame, invalid_rows=()):
    """The single box thermal_estimate() should see, or None (thermal blind)."""
    boxes = detect(frame, invalid_rows)
    return boxes[0] if boxes else None


def _components(mask):
    """8-connected components of a boolean mask, yielded as boolean masks."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        comp = np.zeros_like(mask, dtype=bool)
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = comp[y0, x0] = True
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < h and 0 <= xx < w
                            and mask[yy, xx] and not seen[yy, xx]):
                        seen[yy, xx] = comp[yy, xx] = True
                        stack.append((yy, xx))
        yield comp


def _clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
