"""Host-side repair for the Lepton 3.5 on this board.

Two separate problems, fixed in two separate places.

THE FLICKER is fixed on the sensor, not here. It was the AGC, but not because
AGC is wrong -- because this module ships with

    AGC_HEQ_DAMPENING_FACTOR = 0

i.e. no temporal damping at all, so the histogram equalisation was recomputed
from scratch on every frame and every frame was normalised against a slightly
different mapping. Raising it leaves the AGC doing its job and only slows how
fast the mapping may move (measured over 22 frames each, quiet scene):

    damping    0(stock)   32     64     128    200
    max |dmean|   12.19  22.07   4.70   3.78   1.70   DN
    pixel tstd     6.00   6.06   2.90   3.96   2.63   DN
    scene contrast 11.3   11.7   11.7   11.4   11.2   DN   <- unchanged

Turning AGC off instead (measurement mode + fixed celsius range) also kills the
flicker, but it kills the contrast with it: an indoor scene then lands on ~40 of
255 levels and the picture is a flat wash. Winding it back with a host-side
stretch works numerically and looks terrible -- the scene occupies so little of
the range that the stretch amplifies fixed-pattern noise about ninefold. Do not
go back down that path; the register is the answer.

Narrowing AGC_ROI to dodge the dead rows was also tried, on the theory that 14
rows pinned at the top of the histogram must be distorting it. It measured
slightly worse (max |dmean| 12.48, contrast 10.8) and was dropped.

THE DEAD ROWS are fixed here. 14 rows on this module carry no scene data at all
-- 2 unique values across 160 pixels, against 18-26 for a healthy row, spatial
std 0.45 against 4.0-7.3. Same rows in every capture, both reset paths, before
and after a reseat, so the defect is inside the module and cannot be recovered.
It can only be concealed, which is what this file does.

Concealment quality, measured against 82 healthy rows hidden and rebuilt (DN of
a 15-45 C range, 0.118 C per DN):

    nearest  1.45    linear  1.13    cubic  1.18    edge-directed  1.18
    p95      2.44            1.61           1.74                   1.36

1.13 DN is exactly the typical difference between two adjacent rows: the error
equals the detail the row itself carried and no more, and sits 5x below the
scene's spatial contrast. Edge-directed loses slightly on RMS but wins on the
tail, which is what shows up as a visible streak across an edge, so isolated
rows use it. Runs of 2+ rows get plain linear -- there is no local gradient left
to steer with.

Rows 55-63 are a 9-row run through the middle of the frame. They are filled like
everything else and the result looks clean, but nothing is recovered: an object
living only in that band is gone, and no error metric here will say so.
"""
import numpy as np

W, H = 160, 120

# Lepton CCI attribute IDs, written at init by live_server.
CID_AGC_ROI = 0x0108
CID_AGC_HEQ_DAMPENING_FACTOR = 0x0124
AGC_DAMPING = 200

# This module has TWO row defects, and they need opposite treatments.
#
# DEAD rows carry no image at all: 2 unique values across 160 pixels, spatial
# std 0.46 against 5.33 for a healthy row, sitting ~65 DN above the whole scene
# in AGC output. Nothing to recover, so they are interpolated away.
#
# OFFSET rows carry the scene perfectly well at the wrong level. Their spatial
# std is 11-14 against 14.9 healthy, and after a single subtraction they
# correlate 0.967-0.984 with their neighbours. Measured RMS against the
# neighbour average, on data/captures/diag/step1_soft_*:
#
#     row            5     11     31     38     43     54    healthy control
#     raw         19.38   7.43  18.55   8.85   6.36   6.80      2.47
#     offset fix   3.21   2.80   2.76   2.58   2.64   4.14      2.42
#     gain+offset  2.92   2.69   2.57   2.45   2.37   3.49      2.39
#
# An offset lands on the healthy control; fitting a gain too buys ~0.3 DN and
# is not worth the extra failure mode. Interpolating them would be actively
# wrong -- it would discard six rows of real, well-formed scene data.
DEAD_ROWS = (6, 12, 28, 35, 36, 55, 56, 57, 58, 59, 60, 61, 62, 63)
OFFSET_ROWS = (5, 11, 31, 38, 43, 54)


def _local_baseline(rowmean, exclude, half=6):
    """Median row level around each row, ignoring rows already known bad."""
    H_ = len(rowmean)
    out = np.empty(H_)
    for y in range(H_):
        pool = [rowmean[k] for k in range(max(0, y - half), min(H_, y + half + 1))
                if k not in exclude]
        out[y] = np.median(pool) if pool else rowmean[y]
    return out


def detect_bad_rows(frames, max_flagged=30):
    """Find both defect classes from a few settled frames.

    Detection beats a hard-coded list because a replacement module will have a
    different defect map, and a silently stale list would correct healthy rows
    while leaving broken ones on screen -- the failure that looks like it is
    working. It is only trusted when the answer looks like a defect map rather
    than like a failed measurement.

    Returns (dead, offset), or None if the result is not plausible.
    """
    if not frames:
        return None
    stds = np.stack([f.std(axis=1) for f in frames])            # frames, H
    med_sd = np.median(stds)
    dead = set(int(y) for y in
               np.flatnonzero((stds < max(1.0, 0.30 * med_sd)).all(axis=0)))

    # Level outliers, measured against a local baseline so that a genuine warm
    # or cold band in the scene does not read as a defect. A row only counts if
    # it is an outlier in EVERY frame -- a person walking through moves rows for
    # a moment, a miscalibrated row is wrong always.
    rowmeans = np.stack([f.mean(axis=1) for f in frames])
    votes = np.zeros(rowmeans.shape[1], int)
    for rm in rowmeans:
        dev = rm - _local_baseline(rm, dead)
        healthy = np.array([d for y, d in enumerate(dev) if y not in dead])
        mad = np.median(np.abs(healthy - np.median(healthy))) or 1.0
        votes += (np.abs(dev) > max(3.0, 5.0 * mad))
    # Tolerate one dissenting frame: row 43 is only ~5 DN low and drops out of a
    # unanimous vote. A false positive here is cheap -- destripe shifts a healthy
    # row by the ~0 DN it is already off by -- so the vote leans towards catching
    # the mild ones.
    offset = set(int(y) for y in np.flatnonzero(votes >= len(frames) - 1)) - dead
    # A row with no structure left is dead, not merely mis-levelled.
    offset = {y for y in offset if stds[:, y].mean() > 0.5 * med_sd}

    if not dead or len(dead) + len(offset) > max_flagged:
        return None
    return tuple(sorted(dead)), tuple(sorted(offset))


def _runs(rows):
    """(6,12,35,36) -> [(6,6), (12,12), (35,36)]"""
    out = []
    for y in sorted(rows):
        if out and y == out[-1][1] + 1:
            out[-1][1] = y
        else:
            out.append([y, y])
    return [(a, b) for a, b in out]


def _neighbour_fn(bad):
    def neighbour(y, step):
        """Nearest row not in `bad` in one direction, or None at the edge."""
        y += step
        while 0 <= y < H:
            if y not in bad:
                return y
            y += step
        return None
    return neighbour


def destripe(frame, dead=DEAD_ROWS, offset=OFFSET_ROWS):
    """Level-correct the rows that carry the scene at the wrong offset.

    Runs BEFORE repair(), because several of these sit directly next to a dead
    row: interpolating row 6 from rows 5 and 7 while row 5 is still 19 DN low
    would push that error straight into the replacement.

    The shift is measured per frame from the row's own median against its
    neighbours', so it tracks whatever the AGC is doing rather than assuming a
    calibration constant.
    """
    if not len(offset):
        return frame
    out = frame.astype(np.float32, copy=True)
    neighbour = _neighbour_fn(set(dead) | set(offset))
    for y in sorted(offset):
        up, dn = neighbour(y, -1), neighbour(y, +1)
        if up is None and dn is None:
            continue
        ref = out[up] if dn is None else out[dn] if up is None else 0.5 * (out[up] + out[dn])
        out[y] += np.median(ref) - np.median(out[y])
    return np.clip(out, 0, 255).astype(np.uint8)


def repair(frame, dead=DEAD_ROWS):
    """Fill the dead rows from their healthy neighbours. Returns a new array."""
    if not len(dead):
        return frame
    out = frame.astype(np.float32, copy=True)
    bad = set(dead)
    neighbour = _neighbour_fn(bad)

    for y0, y1 in _runs(dead):
        up, dn = neighbour(y0, -1), neighbour(y1, +1)
        if up is None and dn is None:
            continue
        if up is None or dn is None:                       # run touches an edge
            out[y0:y1 + 1] = out[up if dn is None else dn]
            continue
        a, b = out[up], out[dn]
        n = y1 - y0 + 1
        if n == 1:
            # Edge-directed: weight towards whichever side is locally flatter,
            # so a hard horizontal edge is not smeared symmetrically across the
            # gap. Falls back to the plain average when a second neighbour on
            # either side is itself dead.
            up2, dn2 = neighbour(up, -1), neighbour(dn, +1)
            if up2 is not None and dn2 is not None:
                wu = np.abs(out[up2] - a)
                wd = np.abs(out[dn2] - b)
                t = wd / (wu + wd + 1e-6)
                out[y0] = t * a + (1.0 - t) * b
            else:
                out[y0] = 0.5 * (a + b)
        else:
            for k in range(n):
                out[y0 + k] = a + (b - a) * (k + 1) / (n + 1)
    return np.clip(out, 0, 255).astype(np.uint8)


class Regain:
    """Reclaim the output levels the dead rows were occupying.

    With the dead rows excluded, the AGC's 8-bit output only spans about 71 of
    255 levels -- it is spending the top of its range on 14 rows that are not
    scene. Rescaling to the healthy rows' own percentiles gives that back:
    measured spatial sd 15.6 -> 51.8, p1..p99 span 71 -> 237. Without it the
    picture is the washed-out grey that made this look broken.

    The endpoints are damped, for exactly the reason the sensor's AGC needed
    damping. Recomputing a mapping per frame is what flicker IS, and this stage
    applies a gain of ~3.4, so it would amplify its own endpoint wobble rather
    than the scene. Damping here is the same fix as AGC_HEQ_DAMPENING_FACTOR,
    one stage further down the pipe.

    This is not the host-side stretch that was tried and rejected earlier. That
    one ran on non-equalised measurement-mode data, needed ~9x, and amplified
    fixed-pattern noise into the picture. This runs on an already-equalised
    image and only undoes a known, quantified loss.
    """

    SPAN_TAU = 5.0          # the contrast must hold still
    MAX_GAIN = 5.0

    def __init__(self, lo_pct=0.5, hi_pct=99.5, min_span=32):
        self.lo_pct, self.hi_pct = lo_pct, hi_pct
        self.min_span = min_span
        self.span = None

    def apply(self, frame, dead=(), dt=1.0 / 8.7):
        """Centre on the frame's own median; damp only the span.

        Level and span need opposite treatment, and getting that wrong is what
        made this stage a flicker source in its own right. Any global wobble
        left in the AGC output is multiplied by this stage's gain, so tracking
        the level through a lagging endpoint turned a 15 DN input swing into a
        67 DN output swing -- four times the flicker the damping register had
        just removed. Damping the level faster only helped halfway (p99 26.8),
        because the low endpoint was a 0.5th percentile, which is itself noisy
        frame to frame.

        Subtracting the frame's own median removes a global shift exactly
        rather than chasing it, and the median is robust enough not to add
        noise of its own. The span is the only thing carried between frames,
        because the span IS the contrast and a span recomputed per frame is
        the definition of flicker.
        """
        if not len(dead):
            return frame
        keep = np.ones(frame.shape[0], bool)
        keep[list(dead)] = False
        pool = frame[keep]
        lo, hi = np.percentile(pool, [self.lo_pct, self.hi_pct])
        # Widen, never bail out. Returning the frame unscaled when the scene goes
        # flat means the output alternates between full gain and unity gain as it
        # drifts across the threshold: measured as 69 DN jumps.
        span = max(hi - lo, self.min_span)
        if self.span is None:
            self.span = span
        else:
            self.span += (1.0 - np.exp(-max(dt, 1e-3) / self.SPAN_TAU)) * (span - self.span)
        g = min(255.0 / max(self.span, 1e-6), self.MAX_GAIN)
        centred = frame.astype(np.float32) - np.median(pool)
        return np.clip(centred * g + 128.0, 0, 255).astype(np.uint8)


def _gray_lut(invert):
    x = np.arange(256, dtype=np.uint8)
    v = (255 - x) if invert else x
    return np.stack([v, v, v], axis=1)


def _ironbow_lut():
    """The colours the board's to_ironbow() was supposed to apply.

    It never did: every thermal JPEG this project produced before the raw path
    is single-component grayscale, because to_ironbow() needs a second frame
    buffer that will not fit and the caller swallowed the exception. Applying
    it here removes the allocation from the problem entirely.
    """
    stops = [(0.00, (0, 0, 0)), (0.15, (30, 0, 80)), (0.30, (110, 0, 120)),
             (0.45, (190, 20, 80)), (0.60, (240, 80, 20)), (0.75, (255, 150, 0)),
             (0.90, (255, 220, 80)), (1.00, (255, 255, 255))]
    x = np.linspace(0.0, 1.0, 256)
    pos = [p for p, _ in stops]
    return np.stack([np.interp(x, pos, [c[i] for _, c in stops])
                     for i in range(3)], axis=1).astype(np.uint8)


PALETTES = {
    "blackhot": _gray_lut(invert=True),     # hot = dark
    "whitehot": _gray_lut(invert=False),
    "ironbow": _ironbow_lut(),
}


def colourise(frame, palette="blackhot"):
    """160x120 grayscale -> 160x120x3 RGB."""
    return PALETTES.get(palette, PALETTES["blackhot"])[frame]


def decode(buf):
    """Raw VoSPI bytes off the board -> 120x160 array, or None if truncated."""
    if buf is None or len(buf) < W * H:
        return None
    return np.frombuffer(buf[:W * H], dtype=np.uint8).reshape(H, W)


# --- per-pixel fixed-pattern correction -------------------------------------
#
# destripe() corrects whole ROWS and nothing else. Measured over 200 recorded
# frames, the fixed pattern that survives repair() splits as
#
#     per-pixel 2.06 DN    rows 0.86 DN    columns 0.55 DN
#
# so the axis the pipeline corrects is the smallest of the three, and the
# largest is untouched. regain() then stretches whatever is left by ~4.4x, which
# is why it reaches the eye. This is the missing correction.
#
# WHY THE CAMERA MUST BE MOVING. The pattern is fixed to the SENSOR and the
# scene is not, so panning decorrelates them: average enough frames while moving
# and the scene averages toward a smooth field while the pattern stays exactly
# where it is. Hold the camera still instead and the scene IS the average -- the
# map then contains the wall you were pointing at, and subtracting it burns a
# permanent negative of that wall into every future frame. That failure looks
# like a beautifully clean image of the wrong thing, so it is refused rather
# than warned about.

FPN_MIN_FRAMES = 40
# How much the SCENE must have changed between the start and the end of the
# capture, as 1 - correlation of the two low-passed frames. 0 means the view is
# identical; 1 means unrecognisable.
#
# This deliberately does NOT measure per-pixel variation over time. That was the
# first attempt and it fails silently in the worst way: this sensor's temporal
# noise is ~6 DN against a spatial spread of the same order, so a camera sitting
# still on a desk scored 0.84 on a 0.35 threshold and the capture was accepted.
# The metric was reading noise and calling it motion. Correlating LOW-PASSED
# frames removes noise from the comparison and leaves only structure, which is
# the thing that has to move.
#
# KNOWN HOLE (2026-07-29, src/cfg/lepton_fpn_REJECTED_person_burnin_*.npy is
# the specimen): a PERSON moving in front of a STATIC camera also changes the
# low-passed structure, so the gate passes while the background never moves —
# and the person's time-averaged silhouette burns into the map (measured std
# 12.4 DN against ~2 DN for real FPN). The gate proves the view changed, not
# that the CAMERA moved. Until it also demands whole-field change, the
# procedure is on the operator: pan across a varied scene with NOBODY in frame.
FPN_MIN_SCENE_CHANGE = 0.12


def _lowpass(a, k=5, passes=2):
    out = a.astype(np.float32, copy=True)
    ker = np.ones(k, dtype=np.float32) / k
    for _ in range(passes):
        out = np.apply_along_axis(lambda v: np.convolve(v, ker, "same"), 1, out)
        out = np.apply_along_axis(lambda v: np.convolve(v, ker, "same"), 0, out)
    return out


def measure_pixel_offsets(frames, dead=DEAD_ROWS):
    """Frames captured while panning -> per-pixel offset map, or (None, why).

    Returns (offsets float32 HxW, note). Subtracting `offsets` removes the
    sensor's per-pixel non-uniformity. Dead rows are pinned to zero: they carry
    no scene, repair() invents them, and "correcting" an interpolation would
    only fight the interpolator.
    """
    if frames is None or len(frames) < FPN_MIN_FRAMES:
        return None, ("need >= %d frames, got %d"
                      % (FPN_MIN_FRAMES, 0 if frames is None else len(frames)))
    A = np.stack(frames).astype(np.float32)

    # Did the view actually change? Compare the STRUCTURE at the start against
    # the structure at the end, low-passed so that sensor noise -- which is the
    # same order as the scene contrast here -- cannot masquerade as movement.
    n = len(A)
    a0 = _lowpass(A[:max(3, n // 8)].mean(axis=0))
    a1 = _lowpass(A[-max(3, n // 8):].mean(axis=0))
    x, y = a0.ravel() - a0.mean(), a1.ravel() - a1.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    corr = float((x * y).sum() / denom) if denom > 1e-9 else 1.0
    change = 1.0 - corr
    if change < FPN_MIN_SCENE_CHANGE:
        return None, ("the view barely changed (%.3f, need %.2f) -- pan the "
                      "camera slowly across a varied scene for the whole "
                      "capture. Held still, the map records the scene instead "
                      "of the sensor and burns a negative of it into every "
                      "later frame." % (change, FPN_MIN_SCENE_CHANGE))

    mean = A.mean(axis=0)
    offsets = (mean - _lowpass(mean)).astype(np.float32)
    if len(dead):
        offsets[np.asarray(dead, dtype=int), :] = 0.0
    offsets -= float(offsets.mean())          # no global level shift
    return offsets, ("from %d frames, scene change %.3f, residual sd %.2f DN"
                     % (len(A), change, float(offsets.std())))


def apply_pixel_offsets(frame, offsets):
    """Subtract the per-pixel map. Returns a new uint8 frame."""
    if offsets is None:
        return frame
    out = frame.astype(np.float32) - offsets
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
