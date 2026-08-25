#!/usr/bin/env python3
"""Where the whole picture went, so the tracker can tell it from where a person went.

The tracker's `moved` test asks whether a candidate has GONE anywhere, because
that is what separates a person from the lobby's warm glass door when no
temperature can. Turn the rig and everything in the frame goes somewhere, the
door included - so without this, sweeping the rig is a way to manufacture people
out of furniture, and the rule written to prevent that produces it instead.

This measures the one number that fixes it: the translation the whole frame
underwent since the last one. Phase correlation on the plain visible luma - the
same plane the detector reads, for the same reason (the fused frame's tone moves
with the thermal scene, and a correlator cannot tell that from the camera
moving).

WHY THIS AND NOT THE IMU. The IMU is the better instrument in principle and is
coming; it is cheap, it cannot be fooled by what the scene does, and it needs no
texture. Two things are not ready for it: the rotation between the IMU and the
camera has never been solved (the 2026-08-18 holds fixed the IMU's own axes, but
the calibration board was not in frame), and getting samples off the board means
adding fields to the per-frame header, whose size is load-bearing - the GC block
it lands in decides the per-frame leak, and the leak is what eventually fires a
collect that wedges the Lepton. Both are answerable, neither is free, and this
needs neither: it measures the displacement in the plane where the tracker lives,
in the units the tracker wants, with nothing to calibrate.

MEASURED, on captures/handwave3 (bolted rig, a hand waving through the frame)
and on the same frames translated by known amounts:

    known shift 1..90 px    recovered to within 0.13 px
    bolted rig              |du|,|dv| <= 0.27 px per frame - the noise floor
    cost                    1.1 ms per frame at 160x100, resize included

HOW IT FAILS, which matters more than how it works. Phase correlation reports
the DOMINANT translation, and a large enough moving object is the dominant one.
Measured with a synthetic mover crossing a real frame while the rig is still:

    object over 25% of the width, moving 25 px   ->  reads +0.69 px, response 0.65
    over 50%                                     ->  reads +1.57 px, response 0.46
    over 70%                                     ->  reads +3.17 px, response 0.44

So it under-reports rather than inventing - it never handed back the mover's own
25 px - and the response collapses while it does. A trustworthy frame (bolted
rig, or a clean pan) measured 0.83..0.99. MIN_RESPONSE sits in the gap between
those two populations, and a rejected frame returns None, which the tracker
reads as "nobody measured it" rather than as zero.
"""
import cv2
import numpy as np

# 160x100 from 640x400. Measured against 320x200: a fifth of the cost (1.1 ms
# against 5.3), no worse on known shifts (0.13 px against 0.24), and a noise
# floor twice as high in absolute terms (0.27 px against 0.12) which is still
# an order of magnitude under anything the tracker acts on.
SCALE = 4

# The gap between the two measured populations above: 0.83 at worst when the
# answer was trustworthy, 0.65 at best when a mover had taken over the frame.
MIN_RESPONSE = 0.70

# Per frame, full-resolution pixels. At the thermal rate this is roughly a
# thousand pixels a second - a whipped pan, during which nothing in the frame
# is worth tracking anyway. Past this, a wrapped correlation peak is likelier
# than a real sweep.
MAX_SHIFT_PX = 120.0


class GlobalShift:
    """Frame-to-frame translation of the whole picture, in display pixels."""

    def __init__(self, scale=SCALE, min_response=MIN_RESPONSE,
                 max_shift_px=MAX_SHIFT_PX):
        self.scale = int(scale)
        self.min_response = float(min_response)
        self.max_shift_px = float(max_shift_px)
        self._prev = None
        self._win = None
        # Diagnostics, for the panel: a measurement nobody can see is a
        # measurement nobody can doubt.
        self.response = 0.0
        self.rejected = 0
        self.measured = 0
        self.last = (0.0, 0.0)

    def reset(self):
        """Forget the previous frame. After a stream restart the next pair are
        not consecutive views of anything, and their correlation is noise."""
        self._prev = None

    def measure(self, y):
        """(du, dv) in full-resolution pixels, or None when it cannot be trusted.

        `y` is the plain visible luma, full resolution, uint8. None is returned
        for the first frame of a stream, for a frame whose correlation peak is
        too weak to believe, and for a shift too large to be a real sweep.
        """
        if y is None:
            return None
        h, w = y.shape[:2]
        sw, sh = w // self.scale, h // self.scale
        small = cv2.resize(y, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._win is None or self._win.shape != small.shape:
            # Hanning, or the frame's own edges correlate with themselves and
            # every answer is pulled towards zero.
            self._win = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
        prev, self._prev = self._prev, small
        if prev is None or prev.shape != small.shape:
            return None
        (dx, dy), response = cv2.phaseCorrelate(prev, small, self._win)
        du, dv = dx * self.scale, dy * self.scale
        self.response = float(response)
        self.last = (du, dv)
        if response < self.min_response or max(abs(du), abs(dv)) > self.max_shift_px:
            self.rejected += 1
            return None
        self.measured += 1
        return (du, dv)

    def as_dict(self):
        return {"du": round(self.last[0], 2), "dv": round(self.last[1], 2),
                "response": round(self.response, 3),
                "measured": self.measured, "rejected": self.rejected}
