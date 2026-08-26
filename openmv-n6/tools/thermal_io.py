#!/usr/bin/env python3
"""Reading a session's thermal.bin without guessing what is in it.

A session's thermal plane comes in one of two encodings and they are not
distinguishable by looking at the numbers:

  uint8   the board's own frame, the Lepton's radiometric output clamped to the
          session's [tmin,tmax] window and scaled onto 0..255. A count is
          tmin + n * (tmax - tmin) / 255 degrees, so what a count means depends
          on a window recorded in meta.json and nowhere in the data.

  uint16  the Lepton's words themselves, kept by the raw-passthrough firmware
          before that conversion. Absolute temperature in centi-kelvin:
          C = n / 100 - 273.15, with no window to look up and nothing clipped.

Both are little-endian, both are 120x160, and a uint16 file read as uint8 does
not fail - it yields twice as many frames, each one a mangled interleave of a
real one. That is the failure this module exists to prevent: every reader here
takes the dtype from meta.json and refuses to proceed on a mismatch.

The conversion runs one way only. codes8() turns 16-bit words into the 8-bit
codes the same session's window would have produced, so tools written against
8-bit thresholds keep working on a 16-bit recording. There is no inverse: the
8-bit plane is lossy and the words are not recoverable from it.
"""
import json
import os

import numpy as np

THERMAL_H, THERMAL_W = 120, 160
THERMAL_PIXELS = THERMAL_H * THERMAL_W

DTYPES = {'uint8': np.dtype('u1'), 'uint16_le': np.dtype('<u2')}


def dtype_of(meta):
    """The recorded frame's dtype, from what the session declared.

    Fails closed. A session that declares nothing is uint8 only if its frame
    size says so; anything else is an error rather than a guess, because the
    guess that is wrong here is silent.
    """
    declared = meta.get('thermal_dtype')
    nbytes = meta.get('thermal_frame_bytes')

    if declared is not None:
        if declared not in DTYPES:
            raise ValueError('unsupported thermal_dtype %r' % declared)
        dt = DTYPES[declared]
        if nbytes is not None and nbytes != THERMAL_PIXELS * dt.itemsize:
            raise ValueError('thermal_dtype=%s wants %d bytes/frame, meta says %d'
                             % (declared, THERMAL_PIXELS * dt.itemsize, nbytes))
        return dt

    if nbytes == THERMAL_PIXELS:
        return DTYPES['uint8']
    if nbytes == 2 * THERMAL_PIXELS:
        return DTYPES['uint16_le']
    raise ValueError('cannot tell the thermal encoding: no thermal_dtype and '
                     'thermal_frame_bytes=%r' % (nbytes,))


def load_meta(session_dir):
    with open(os.path.join(session_dir, 'meta.json')) as f:
        return json.load(f)


def open_session(session_dir, meta=None):
    """-> (frames, meta). frames is (N, 120, 160) in the file's own dtype."""
    meta = load_meta(session_dir) if meta is None else meta
    dt = dtype_of(meta)
    frames = np.memmap(os.path.join(session_dir, 'thermal.bin'), dt, mode='r')
    return frames.reshape(-1, THERMAL_H, THERMAL_W), meta


def window_ck(meta):
    """The session's sensor window in centi-kelvin, the unit the words use.

    (C + 273.15) * 100 is the driver's own conversion - imlib.c does exactly
    this before scaling - so reproducing it here rather than working in degrees
    keeps the rounding identical instead of merely close.
    """
    tmin = meta.get('tmin', -10)
    tmax = meta.get('tmax', 140)
    return (int(round((tmin + 273.15) * 100.0)),
            int(round((tmax + 273.15) * 100.0)))


def codes8(raw16, window):
    """16-bit centi-kelvin -> the 8-bit codes that window would have produced.

    Identical to imlib_fill_image_from_lepton() rather than merely equivalent,
    because the thresholds downstream were tuned against the real thing - the
    warm margin in the walk validator, fusion.c's deband and dead-row constants,
    the students' input distribution. A conversion that was only close would move
    all of them by an amount nobody had measured.

    Identical includes the tie-break, which is where this first got it wrong.
    The firmware rounds with fast_roundf(), and on the board that is a single
    `vcvtr.S32.F32` (fmath.h) - VCVTR takes its rounding mode from FPSCR, whose
    reset value is round-to-nearest-EVEN. Not round-half-up, which is what the
    same header falls back to off-target and what any obvious implementation
    here would do. The difference only shows on words landing exactly on a bin
    boundary, so it is invisible on a wide window and unmissable on a narrow one:
    measured against the board over 8 frames at a 1100 cK auto-ranged window,
    half-up disagreed on 161 of 153,600 pixels and every single one of them sat
    at exactly .5.

    Done in integers for the same reason - (word-lo)*255 and the window span are
    both small, so the quotient, the remainder and the tie are all exact, and
    there is no float precision left to argue about.
    """
    lo, hi = window
    den = hi - lo
    if den <= 0:
        return np.zeros(np.shape(raw16), np.uint8)

    num = (np.clip(np.asarray(raw16).astype(np.int64), lo, hi) - lo) * 255
    q, r = np.divmod(num, den)
    # 2r > den rounds up, 2r < den rounds down, 2r == den goes to the even side.
    up = (2 * r > den) | ((2 * r == den) & (q % 2 == 1))
    return (q + up).astype(np.uint8)


def celsius(raw16):
    """16-bit centi-kelvin -> degrees C, at the sensor's own 0.01C step."""
    return np.asarray(raw16).astype(np.float32) * 0.01 - 273.15


def as_codes8(frame, meta, window=None):
    """One recorded frame as 8-bit codes, whatever the session stored.

    The pass-through for a uint8 session is deliberate: a tool that calls this
    on every session behaves exactly as it always did on the old ones.
    """
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    return codes8(frame, window_ck(meta) if window is None else window)
