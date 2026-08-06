"""fusion.app mode selector: darkness must NEVER be served by RGB
(acceptance test from the task brief), + hysteresis behaviour."""
import numpy as np

from fusion.app import select_mode, ModeSelector, DARK_ENTER, DARK_EXIT
from fusion.detector import detect, Box, temp_to_g


def test_lit_scene_uses_rgb():
    mode, dark = select_mode(120.0, thermal_available=False)
    assert mode == "rgb" and not dark


def test_dark_with_thermal_uses_thermal():
    mode, dark = select_mode(5.0, thermal_available=True)
    assert mode == "thermal" and dark


def test_dark_without_thermal_falls_back_to_radar_only():
    mode, dark = select_mode(5.0, thermal_available=False)
    assert mode == "radar-only" and dark


def test_rgb_never_selected_in_the_dark():
    """The HARD constraint: no luminance below the dark threshold may ever
    route to the RGB detector, with or without thermal."""
    for luma in range(0, int(DARK_ENTER)):
        for thermal in (False, True):
            mode, _ = select_mode(float(luma), thermal)
            assert mode != "rgb", "RGB selected at luma %d" % luma


def test_hysteresis_between_thresholds():
    # between ENTER and EXIT the previous state wins - no mode flapping
    mid = (DARK_ENTER + DARK_EXIT) / 2.0
    assert select_mode(mid, False, was_dark=True)[0] == "radar-only"
    assert select_mode(mid, False, was_dark=False)[0] == "rgb"


def test_mode_selector_transitions():
    sel = ModeSelector(thermal_available=False)
    assert sel.update(120.0) == "rgb"
    assert sel.update(10.0) == "radar-only"
    assert sel.update((DARK_ENTER + DARK_EXIT) / 2.0) == "radar-only"  # sticky
    assert sel.update(DARK_EXIT + 10.0) == "rgb"


# --- thermal detector (the darkness detection path) ------------------------------

def thermal_frame(blob_temp_c=34.0, ambient_c=20.0):
    """160x120 radiometric frame with one warm blob at the centre."""
    img = np.full((120, 160), temp_to_g(ambient_c), np.uint8)
    img[50:70, 70:90] = temp_to_g(blob_temp_c)
    return img


def test_thermal_detector_finds_human_temperature_blob():
    boxes = detect(thermal_frame(34.0), "thermal")
    assert len(boxes) == 1
    b = boxes[0]
    assert b.label == "human"
    assert (b.x, b.y, b.w, b.h) == (70, 50, 20, 20)


def test_thermal_detector_separates_bands():
    labels = {b.label for b in detect(thermal_frame(42.0), "thermal")}
    assert labels == {"hot"}                    # engine/fire, not a body


def test_thermal_detector_ignores_none_frame():
    assert detect(None, "thermal") == []        # Lepton not wired yet
