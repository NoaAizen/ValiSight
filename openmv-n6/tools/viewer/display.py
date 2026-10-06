"""Display-only composition; sensor measurements stay in the native pipeline."""
import time

import cv2
import numpy as np

import live_channels
import thermal_io
from .constants import OUT_W, OUT_H

# ---------------------------------------------------------------- registration views
#
# Judging registration from the fused picture alone is close to impossible, and
# that is not a UI complaint - it is the pipeline working as designed. The guided
# filter deliberately borrows the visible camera's edges to sharpen the thermal
# layer, so a *misregistered* frame still comes out with crisp edges in all the
# right places. It just colours the wrong side of them. The image looks fine and
# the measurement is wrong, which is the worst failure mode available.
#
# So these views break the two layers apart again:
#
#   blink   alternate visible and fused. Misalignment shows up as things jumping;
#           the eye is far better at spotting motion than at spotting offset.
#   mix     the same comparison, continuous, for judging how much offset there is
#   edges   thermal edges from t_reg drawn over the plain visible image. The
#           strongest test of the three: those edges belong to the thermal camera,
#           so if they land on the visible object's outline the warp is right.
#   operator preserve visible luminance while carrying thermal information mostly
#           in colour. Unlike fused, this is meant for a person watching the scene,
#           not for reading a temperature back from the rendered pixel.
#
# All host-side numpy. None of this belongs in fusion.c - that file compiles into
# firmware, and these are display questions asked while calibrating.

VIEWS = ("fused", "visible", "blink", "mix", "edges", "operator")
CHANNELS = live_channels.CHANNELS


def thermal_edges(treg, w, h, pct=97.0, exclude=None):
    """Gradient magnitude of the registered thermal plane, thresholded.

    The threshold is a percentile rather than a constant because thermal contrast
    varies enormously between scenes: a fixed one either paints the whole frame on
    a high-contrast scene or nothing at all on a flat wall. The consequence is
    worth knowing when reading the view: a percentile always paints about
    (100-pct)% of the frame, so on a scene with no thermal structure what you see
    is noise, not edges. FLOOR below is the guard against that.

    `exclude` is a low-res mask of cells sampled from rebuilt rows. Those are
    dropped before the threshold is chosen as well as after, since otherwise the
    artefact's own gradients set the level for everything else.
    """
    # A gradient of this many codes per output pixel is roughly one thermal code
    # across one thermal pixel - below it there is no structure, only sensor
    # noise, and painting it would be inventing edges the camera did not see.
    FLOOR = 6.0

    t = cv2.resize(treg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    gx = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
    m = np.hypot(gx, gy)

    keep = np.ones((h, w), bool)
    if exclude is not None and exclude.any():
        # Dilate by two cells, not one. The artefact is at the *boundary* of a
        # rebuilt block and it spreads: the bilinear upsample to full resolution
        # reaches a cell either side, and the Sobel kernel another output pixel
        # beyond that. Measured with one cell, the two busiest painted rows in the
        # frame were still splice boundaries. Two cells costs coverage - about a
        # quarter of this sensor's frame is excluded - but a quarter of its
        # thermal rows really are reconstructed, so that is the honest number.
        grown = cv2.dilate(exclude.astype(np.uint8), np.ones((5, 5), np.uint8))
        keep = cv2.resize(grown, (w, h), interpolation=cv2.INTER_NEAREST) == 0

    vals = m[keep]
    if vals.size == 0:
        return np.zeros((h, w), bool)

    thr = max(float(np.percentile(vals, pct)), FLOOR)
    if float(vals.max()) < FLOOR:
        return np.zeros((h, w), bool)       # nothing in the thermal frame to draw
    return (m >= thr) & keep


def coverage_outline(cover, w, h):
    """Boundary of the thermal footprint: where the thermal camera stops seeing.

    Worth having on screen whatever the view. Without it, the plain-luma fallback
    outside the footprint reads as 'the thermal layer is grey here', when what it
    actually means is 'there is no thermal data here at all'.
    """
    c = cv2.resize(cover, (w, h), interpolation=cv2.INTER_NEAREST)
    k = np.ones((3, 3), np.uint8)
    return cv2.dilate(c, k) != cv2.erode(c, k)


def operator_fusion(fused, y, cover=None, mix=60):
    """Operator-friendly fusion: visible structure with thermal colour.

    The radiometric fused view intentionally lets temperature own the whole
    low-frequency image and adds only visible high-frequency detail. That is an
    honest thermogram, but it hides large visible structures behind an opaque
    false-colour sheet. This view keeps the visible luminance and borrows the
    thermal layer's colour, with two deliberately display-only adaptations:

      * flat/low-contrast visible areas trust thermal luminance more, while
        textured areas retain more of the camera image;
      * the thermal footprint is feathered into visible grey instead of ending
        at a hard rectangular seam.

    Temperatures still come from temp_at()/temp_region(), never from this image.
    mix is the nominal thermal weight and remains one live control shared with
    the simple mix view.
    """
    gray = np.repeat(y[:, :, None], 3, 2)
    base = max(0.0, min(1.0, mix / 100.0))

    # Split fused into luma and colour residual directly. This has the useful
    # property of LAB/YCbCr (visible brightness and thermal colour can be mixed
    # independently) without two expensive full-frame colour conversions.
    fused_luma = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY).astype(np.float32)
    yf = y.astype(np.float32)
    local_mean = cv2.blur(y, (17, 17), borderType=cv2.BORDER_REFLECT)
    local_detail = cv2.absdiff(y, local_mean).astype(np.float32)
    visible_confidence = np.clip((local_detail - 2.0) / 22.0, 0.0, 1.0)
    thermal_luma = np.clip(base + (0.5 - visible_confidence) * 0.30, 0.20, 0.90)

    target_luma = fused_luma * thermal_luma + yf * (1.0 - thermal_luma)
    chroma = 0.25 + 0.75 * base
    out = target_luma[..., None] + (
        fused.astype(np.float32) - fused_luma[..., None]) * chroma
    out = np.clip(out, 0, 255).astype(np.uint8)

    if cover is not None:
        # Roughly a 20 px transition at 640x400: wide enough not to look like a
        # calibration cut, narrow enough not to imply thermal coverage far past
        # the last measured cell. Blur on the 160x100 grid first: the result is
        # the same scale, with a quarter of each dimension to process.
        footprint = cv2.GaussianBlur(cover.astype(np.float32), (0, 0),
                                     sigmaX=2.0, sigmaY=2.0)
        footprint = cv2.resize(footprint, (out.shape[1], out.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        footprint = np.clip(footprint, 0.0, 1.0)[..., None]
        out = (out.astype(np.float32) * footprint
               + gray.astype(np.float32) * (1.0 - footprint)).astype(np.uint8)
    return out


def agc8(raw16, rng):
    """The board's 8-bit plane, rebuilt from the 16-bit words it was made of.

    This is what makes --raw16 additive: everything downstream - the warp LUT,
    fusion.c's deband and dead-row thresholds, the students, the detector - was
    tuned against the plane the board used to send, so this reproduces it rather
    than improving on it. thermal_io.codes8() is the firmware's own arithmetic,
    kept in one place so the live path and the offline readers cannot drift.
    """
    tmin, tmax = rng if rng else (-10, 140)
    window = thermal_io.window_ck({"tmin": tmin, "tmax": tmax})
    return thermal_io.codes8(raw16, window).tobytes()


def compose(view, fused, y, treg=None, cover=None, mix=50, phase=True,
            exclude=None, outline=True):
    """Build the frame the browser sees. `fused` is never modified in place."""
    if view == "operator":
        out = operator_fusion(fused, y, cover=cover, mix=mix)
    elif view == "mix":
        a = max(0.0, min(1.0, mix / 100.0))
        out = (fused.astype(np.float32) * a
               + np.repeat(y[:, :, None], 3, 2).astype(np.float32) * (1.0 - a)).astype(np.uint8)
    elif view == "visible" or (view == "blink" and not phase):
        out = np.repeat(y[:, :, None], 3, 2)
    elif view == "edges":
        out = np.repeat(y[:, :, None], 3, 2)
        if treg is not None:
            out[thermal_edges(treg, out.shape[1], out.shape[0], exclude=exclude)] = (80, 255, 255)
    else:
        out = fused.copy()

    if cover is not None and outline:
        out[coverage_outline(cover, out.shape[1], out.shape[0])] = (255, 210, 40)
    return out


def display_sharpen(gray, strength, valid=None):
    """Bounded detail boost with a dead band for quantisation/sensor noise."""
    amount = min(100, max(0, strength)) / 100.0
    if not amount:
        return gray.copy()
    source = gray.astype(np.float32)
    detail = source - cv2.GaussianBlur(source, (5, 5), 0.9)
    # Small fluctuations receive no gain; large edges have a capped boost to
    # keep JPEG blocking and high-contrast halos from dominating the picture.
    boost = np.sign(detail) * np.maximum(np.abs(detail) - 2.0, 0.0)
    boost = np.clip(boost * amount, -6.0, 6.0)
    if valid is not None:
        interior = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
        boost *= interior
    return np.clip(np.rint(source + boost), 0, 255).astype(np.uint8)


def display_visible(pipe, y):
    """Bring out local visible contrast without changing the inference input."""
    amount = min(100, max(0, pipe.visible_detail)) / 100.0
    if not amount:
        return y.copy()
    local = pipe.display_clahe.apply(y)
    # A bounded blend keeps dark flat regions from turning into noisy texture.
    delta = np.clip(local.astype(np.float32) - y.astype(np.float32), -24, 24)
    enhanced = np.clip(np.rint(y + delta * amount * 0.5), 0, 255).astype(np.uint8)
    return display_sharpen(enhanced, pipe.visible_detail)


def compose_channel(pipe, fused, y, now=None):
    """Build the clean image base for the selected product channel.

    Sensor/AI overlays are intentionally absent.  Keeping this as a pure-ish
    boundary makes the five channel contracts testable without a board, radar
    or detector, and keeps calibration recording able to copy the clean frame
    before any annotation is burned into it.
    """
    channel = live_channels.get(pipe.channel)
    if channel.base == "thermal":
        out = pipe.thermal_image()
        if pipe.outline:
            out[coverage_outline(pipe.cover_grid(), OUT_W, OUT_H)] = (255, 210, 40)
        return out
    if channel.base == "visible":
        out = np.repeat(display_visible(pipe, y)[:, :, None], 3, 2)
        if pipe.outline:
            out[coverage_outline(pipe.cover_grid(), OUT_W, OUT_H)] = (255, 210, 40)
        return out

    out = fused
    # Fusion already has a detail layer. Only refine its display luminance;
    # preserve palette chroma and leave calibration views unenhanced so edges
    # and blink remain faithful registration diagnostics.
    if pipe.visible_detail and pipe.view in ("fused", "operator", "mix"):
        lab = cv2.cvtColor(fused, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = display_sharpen(lab[:, :, 0], pipe.visible_detail)
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if pipe.view != "fused" or pipe.outline:
        edges = pipe.view == "edges"
        needs_cover = pipe.outline or pipe.view == "operator"
        clock = time.time() if now is None else now
        display_y = display_visible(pipe, y) if pipe.view in ("visible", "operator", "mix") else y
        out = compose(pipe.view, out, display_y,
                      treg=pipe.treg_grid() if edges else None,
                      exclude=pipe.repaired_grid() if edges else None,
                      cover=pipe.cover_grid() if needs_cover else None,
                      mix=pipe.mix,
                      phase=int(clock / pipe.blink_period) % 2 == 0,
                      outline=pipe.outline)
    return out


