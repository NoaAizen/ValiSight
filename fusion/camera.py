"""Camera abstraction — the key to darkness-readiness.

RGB (works only in light) and thermal (works in darkness/smoke) sit behind
one interface, so fusion/app.py can swap sources at runtime without touching
detection or fusion code. The ThermalCamera is a stub until the FLIR Lepton
3.5 arrives; its on-device counterpart is thermal_heatmap_n6.py.
"""
import os

import cv2
import numpy as np

from calibrate_radar_camera import load_calib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RGB_CALIB = os.path.join(_ROOT, "calib_rgb.json")
THERMAL_CALIB = os.path.join(_ROOT, "calib_thermal.json")


class Camera:
    """Interface: get_frame() -> np.ndarray|None, intrinsics() -> (K, dist).

    calib() returns the full radar<->camera calibration dict (or None when
    the camera has not been calibrated yet)."""

    kind = "abstract"

    def get_frame(self):
        raise NotImplementedError

    def intrinsics(self):
        c = self.calib()
        return (c["K"], c["dist"]) if c else (None, None)

    def calib(self):
        return None

    def available(self):
        return True

    def release(self):
        pass


class RGBCamera(Camera):
    """OpenCV VideoCapture webcam. Detection is valid in adequate light ONLY."""

    kind = "rgb"

    def __init__(self, index=0, calib_path=RGB_CALIB, warmup=20):
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError("cannot open RGB camera index %d" % index)
        for _ in range(warmup):          # auto-exposure warm-up
            self._cap.read()
        self._calib = load_calib(calib_path) \
            if os.path.isfile(calib_path) else None

    def get_frame(self):
        ok, img = self._cap.read()
        return img if ok else None

    def calib(self):
        return self._calib

    def release(self):
        self._cap.release()


class ThermalCamera(Camera):
    """FLIR Lepton 3.5 (LWIR) — the real darkness/smoke sensor.

    STUB until the hardware arrives: available() is False and get_frame()
    returns None, which makes the mode selector fall back to radar-only.
    When wired: return the radiometric frame as uint8 gray mapped
    MIN_C..MAX_C -> 0..255 (same mapping as thermal_heatmap_n6.py), load
    calib_thermal.json, and flip available() — nothing else changes.
    """

    kind = "thermal"

    def __init__(self, calib_path=THERMAL_CALIB):
        self._calib = load_calib(calib_path) \
            if os.path.isfile(calib_path) else None

    def get_frame(self):
        return None                      # no Lepton yet

    def calib(self):
        return self._calib

    def available(self):
        return False


def make_camera(kind, **kwargs):
    """Factory: make_camera('rgb', index=1) / make_camera('thermal')."""
    if kind == "rgb":
        return RGBCamera(**kwargs)
    if kind == "thermal":
        return ThermalCamera(**kwargs)
    raise ValueError("unknown camera kind: %r" % (kind,))
