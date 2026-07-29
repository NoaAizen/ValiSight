"""The unified thermal detector: relative detection, absolute labels.

Each test is one of the documented failure modes of the old absolute-band
segmentation, exercised through the public detect() / thermal_detections()
entry points on synthetic radiometric frames. The old detector's caveat
block named these exact cases; now they are pinned instead of documented.
"""
import numpy as np

from fusion.detector import detect, thermal_detections, temp_to_g

DEAD_ROWS = (6, 12, 28, 35, 36, 55, 56, 57, 58, 59, 60, 61, 62, 63)


def frame(ambient_c, noise=0.0, seed=7):
    img = np.full((120, 160), float(temp_to_g(ambient_c)), np.float32)
    if noise:
        img += np.random.default_rng(seed).normal(0.0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def stamp(img, temp_c, y0=50, y1=70, x0=70, x1=90):
    img = img.copy()
    img[y0:y1, x0:x1] = temp_to_g(temp_c)
    return img


def test_hot_ambient_no_longer_floods_the_human_band():
    """Ambient 33 C sits INSIDE the 30-38 C human band. The old absolute
    segmentation flagged the whole frame; relatively, only the person stands
    out from the background."""
    boxes = detect(stamp(frame(33.0), 37.0), "thermal")
    assert len(boxes) == 1
    b = boxes[0]
    assert (b.x, b.y, b.w, b.h) == (70, 50, 20, 20)
    assert b.label == "human"


def test_distant_cool_person_is_found_and_clamped_to_warm():
    """Sub-pixel fill drops apparent temperature: a person reading 22 C is
    below every band and used to vanish. Relative detection finds them; the
    label clamps to the bottom band."""
    boxes = detect(stamp(frame(15.0), 22.0), "thermal")
    assert len(boxes) == 1
    assert boxes[0].label == "warm"


def test_thermal_crossover_yields_nothing():
    """Target at ambient temperature has zero contrast. No threshold fixes
    physics — the honest output is no box (the uncertainty layer then makes
    thermal high-sigma and radar carries)."""
    assert detect(stamp(frame(33.0), 33.0), "thermal") == []


def test_score_is_detection_confidence_not_band_position():
    """Radiometric mapping is ~8.5 DN/C, so the contrast term saturates ~3 C
    above ambient; the comparison must sit below that. Under the old
    band-position score the 22 C blob (dead centre of nothing) and a 34 C
    blob scored by different rules entirely."""
    hot = detect(stamp(frame(20.0), 34.0), "thermal")[0]
    dim = detect(stamp(frame(20.0), 22.0), "thermal")[0]
    for b in (hot, dim):
        assert 0.0 < b.score <= 1.0
    assert hot.score > dim.score


def test_blob_living_only_in_dead_rows_is_rejected():
    img = stamp(frame(20.0), 37.0, y0=55, y1=64, x0=40, x1=100)
    assert detect(img, "thermal", invalid_rows=DEAD_ROWS) == []
    assert detect(img, "thermal") != []          # same frame, rows trusted


def test_record_shape_matches_on_device_emitter_plus_confidence():
    d = thermal_detections(stamp(frame(20.0), 34.0))[0]
    assert set(d) == {"label", "rect", "cx", "cy", "t_mean", "t_max",
                      "confidence"}
    assert abs(d["t_mean"] - 34.0) < 0.5
    assert 69.0 <= d["cx"] <= 90.0 and 49.0 <= d["cy"] <= 70.0
