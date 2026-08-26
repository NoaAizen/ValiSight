#!/usr/bin/env python3
"""Refuse to call a frame empty when something warm is standing in it.

`--verified-negative SESS` used to stamp `label_state = 0` on every frame of a
session, on one human's word that the room was empty. Measured 2026-08-26,
straight off the recorded thermal:

    radar3-dark-negative1   306 of 630 frames (48.6%) hold a person, frames
                            115-418, filling the frame. It was the ONLY dark
                            negative in v3's and v4's val split.
    negative1-clean         373 of 2145 frames (17.4%), at both ends of the
                            session and in four runs in the middle. Train
                            negative in v2, v3 and v4.

What that cost: v4's thermal student reports `false_positive_rate 0.995`
(fp 597, tn 3) and v3's reports a total collapse. Neither number is about the
model - both are the mislabel measured back. And `train_students.py` selects
its best checkpoint on val, so both runs chose a checkpoint against a poisoned
signal. There is no way to notice this downstream: an empty label on an
occupied frame is indistinguishable from a model that got it wrong.

So the operator's word is now a claim about the SESSION, and this is the
per-frame check on it.

WHAT IT DOES, AND THE ONE DIRECTION IT MOVES

It only ever demotes NEGATIVE -> UNKNOWN. It never promotes anything to
POSITIVE and never invents a box. That asymmetry is the whole safety argument:
a false alarm here costs a negative frame, and the project has thousands; a
miss here poisons supervision in exactly the conditions the rig exists for.
Missing evidence is not negative evidence - the same rule `label_state()`
already applies to a teacher that emitted no box.

THE TEST, AND WHY EACH PART IS THERE

    1. Background: the 20th percentile per pixel over a rolling +/-300 frame
       (~34 s) window. Not the session median - a person makes a pixel hotter
       and never colder, so a low percentile survives a session that is half
       occupied, which is the case that matters (dark-negative1 is 48.6%).
       Rolling and not whole-session because the sun moves: measured on
       radar3-negative1, a whole-session background flags 27.5% of a genuinely
       empty patio as occupied purely from the ground warming through the
       afternoon. A rolling one flags 9.9%.

    2. Per-frame global offset removed. A sensor that warms, or an FFC, lifts
       every pixel together. That is not a person and this subtracts it.

    3. Warm mask at +3.0 C over that background. Below the 1.7 C a person
       reflects off a 29 C ceiling and well under the 4.7 C a body clears
       directly, because this gate is not trying to identify a person - it is
       trying to notice that the frame is not empty.

    4. A 3x3 cross erosion before counting. Measured on radar3-negative1 the
       remaining flags were a 1-3 px bright line along the roof edge warming
       in the sun. A person's torso is 6 px wide at 15 m on this sensor
       (0.3125 deg/px at HFOV 50) and survives losing one pixel a side; a
       structural edge does not. This takes the empty patio from 9.9% to 5.7%
       and leaves both occupied sessions untouched (47.9% / 19.0%).

    5. +/-9 frames (~1 s) of temporal dilation around every flag, for the
       frames where somebody is halfway out of shot at the edge of a run.

WHAT IT STILL CANNOT DO, STATED RATHER THAN HIDDEN

A person who is in shot for most of a +/-34 s window, motionless, is absorbed
into the rolling background and will not be flagged. The defence against that
is not a threshold - it is that a session recorded as empty with somebody
standing in it for a minute is a session the operator should not have called
empty, and step 1's low percentile is what buys the margin. `--negative-audit`
prints the flagged runs so a human can read them back against what they
remember recording.

Its false alarms are honest ones and are reported, not tuned away: on
radar3-negative1 the 5.7% is the roof edge, the two torn frames at session
start, and - reading the frames - the operator walking up at the end. Torn and
FFC frames flag too, which is correct: they are not usable evidence of an
empty room either.

Nothing here runs on the live path. It is an offline gate on what may enter
training as "there is nobody here".
"""
import numpy as np

try:
    import cv2
except ImportError:                                    # pragma: no cover
    cv2 = None

THERMAL_H, THERMAL_W = 120, 160

# Degrees above the rolling background before a pixel counts as warm. See §3.
WARM_C = 3.0
# Eroded warm pixels before the frame is called occupied. A person at 15 m is
# about 6x21 px on this sensor; one erosion leaves ~4x19.
MIN_PX = 20
# Rolling background: half-window, spacing between estimates, and the
# percentile taken within the window. See §1.
BG_HALF = 300
BG_STEP = 100
BG_PCT = 20.0
# Frames either side of a flag that are demoted with it. See §5.
DILATE = 9

# uint16 sessions carry absolute centi-kelvin, so their scale is known without
# meta.json: C = n/100 - 273.15.
CK_PER_LSB = 0.01


class NegativeGateError(Exception):
    """The session cannot be checked, so it cannot be trusted as empty."""


def _celsius_scale(meta, dtype):
    """Degrees per count for this session, or refuse.

    A uint8 plane is counts on a window recorded in meta.json and nowhere in
    the data. Without that window there is no threshold in degrees to apply,
    and guessing one is exactly the silent failure thermal_io exists to stop.
    """
    if np.dtype(dtype).itemsize == 2:
        return CK_PER_LSB
    c = meta.get('c_per_lsb')
    if c is None:
        tmin, tmax = meta.get('tmin'), meta.get('tmax')
        if tmin is not None and tmax is not None:
            c = (tmax - tmin) / 255.0
    if not c or c <= 0:
        raise NegativeGateError(
            'no c_per_lsb (and no tmin/tmax) in meta.json: the 8-bit plane is '
            'intensity, not temperature, so "3 C above background" has no '
            'meaning here. Re-record with --range, or drop the session from '
            '--verified-negative rather than trusting it')
    return float(c)


def _rolling_background(frames, half=BG_HALF, step=BG_STEP, pct=BG_PCT,
                        sub=3):
    """(estimates, index-per-frame). One estimate per `step` frames."""
    n = len(frames)
    centres = list(range(0, max(n, 1), step))
    bgs = np.empty((len(centres), THERMAL_H, THERMAL_W), np.float32)
    for k, c in enumerate(centres):
        lo, hi = max(0, c - half), min(n, c + half)
        window = frames[lo:hi:sub]
        if not len(window):
            window = frames[lo:hi]
        bgs[k] = np.percentile(window, pct, axis=0)
    idx = np.clip(np.round(np.arange(n) / step).astype(int),
                  0, len(centres) - 1)
    return bgs, idx


def _warm_counts(frames, c_per_lsb):
    """Eroded warm-pixel count per frame, over the rolling background."""
    bgs, idx = _rolling_background(frames)
    kernel = (cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
              if cv2 is not None else None)
    counts = np.empty(len(frames), np.int32)
    for i, f in enumerate(frames):
        d = (f - bgs[idx[i]]) * c_per_lsb
        d -= np.median(d)                       # whole-scene drift, not a body
        mask = (d > WARM_C).astype(np.uint8)
        if kernel is not None:
            mask = cv2.erode(mask, kernel)
        else:                                   # pragma: no cover - cv2 absent
            m = mask.astype(bool)
            mask = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1]
                    & m[1:-1, :-2] & m[1:-1, 2:]).astype(np.uint8)
        counts[i] = int(mask.sum())
    return counts


def _dilate(flags, width=DILATE):
    """Spread each flag `width` frames either side."""
    if width <= 0 or not flags.any():
        return flags
    out = flags.copy()
    for shift in range(1, width + 1):
        out[shift:] |= flags[:-shift]
        out[:-shift] |= flags[shift:]
    return out


def runs_of(flags, min_len=1):
    """Contiguous flagged ranges as (first, last) frame ordinals."""
    idx = np.flatnonzero(flags)
    if not len(idx):
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i > prev + 1:
            out.append((int(start), int(prev)))
            start = i
        prev = i
    out.append((int(start), int(prev)))
    return [r for r in out if r[1] - r[0] + 1 >= min_len]


def occupied_frames(session):
    """Which frames of a session are NOT empty. -> (set of frame i, stats).

    `session` is a `perception.dataset.LiveSession`. The returned set holds
    `frames.jsonl` `i` values - the same key `export_session` joins on - so a
    caller never has to reason about thermal.bin ordinals.
    """
    rows = [m for m in session.frames if m.get('thermal_off') is not None]
    if not rows:
        raise NegativeGateError(
            'no thermal frames: a session recorded in a fused view carries no '
            'measurement to check, and cannot be trusted as empty')
    dtype = session.thermal_dtype(rows[0])
    c_per_lsb = _celsius_scale(session.meta, dtype)
    frames = np.empty((len(rows), THERMAL_H, THERMAL_W), np.float32)
    for k, m in enumerate(rows):
        frames[k] = session.thermal_frame(m)

    counts = _warm_counts(frames, c_per_lsb)
    flags = _dilate(counts >= MIN_PX)
    occupied = {int(rows[k]['i']) for k in np.flatnonzero(flags)}
    stats = {
        'frames_checked': len(rows),
        'frames_occupied': int(flags.sum()),
        'fraction': float(flags.mean()),
        'c_per_lsb': c_per_lsb,
        'warm_c': WARM_C,
        'min_px': MIN_PX,
        'runs': [(int(rows[a]['i']), int(rows[b]['i']))
                 for a, b in runs_of(flags, min_len=3)],
    }
    return occupied, stats
