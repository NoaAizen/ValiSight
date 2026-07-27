"""Dual-camera streamer — runs ON the OpenMV N6, paired with view_thermal_rgb.py.

Grabs frames from BOTH sensors on the board — the FLIR Lepton 3.5 (radiometric
thermal) and the stock PAG7936 RGB — JPEG-compresses each, and prints them as
base64 lines over the USB VCP. The PC side (view_thermal_rgb.py) launches this
script via mpremote and decodes the lines into a live side-by-side view.

Line protocol (one record per line):
    T:<base64 jpeg>   thermal frame, grayscale, MIN_C..MAX_C -> 0..255
    R:<base64 jpeg>   RGB frame
    D:<json>          thermal detections [{label, rect, t_mean, t_max}, ...]
    S:<json>          thermal frame stats {min, mean, max} in deg C

N6 firmware notes (verified on-device):
  - a bare csi.CSI() binds the PAG7936, so the Lepton must be selected
    explicitly with cid=csi.LEPTON
  - IOCTL_LEPTON_SET_MODE takes TWO args: (measurement_mode, high_temp_mode)
  - snapshot() may return None right after init — skip and retry
"""
import csi
import time
import json
import ubinascii

MIN_C = 15.0
MAX_C = 45.0

# same bands as thermal_heatmap_n6.py
CLASSES = [
    ("warm", 25.0, 30.0),
    ("human", 30.0, 38.0),
    ("hot", 38.0, 45.0),
]


def temp_to_g(t):
    t = min(max(t, MIN_C), MAX_C)
    return int((t - MIN_C) * 255.0 / (MAX_C - MIN_C))


def g_to_temp(g):
    return (g * (MAX_C - MIN_C) / 255.0) + MIN_C


BANDS = [(lbl, [(temp_to_g(a),
                 temp_to_g(b) - (0 if i == len(CLASSES) - 1 else 1))])
         for i, (lbl, a, b) in enumerate(CLASSES)]

# --- thermal: FLIR Lepton 3.5 in radiometric mode ---------------------------
lep = csi.CSI(cid=csi.LEPTON)
lep.reset()
lep.pixformat(csi.GRAYSCALE)
lep.framesize(csi.QQVGA)
lep.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)
lep.ioctl(csi.IOCTL_LEPTON_SET_RANGE, MIN_C, MAX_C)

# --- rgb: stock PAG7936 -----------------------------------------------------
rgb = csi.CSI(cid=csi.PAG7936)
rgb.reset()
rgb.pixformat(csi.RGB565)
rgb.framesize(csi.QVGA)

time.sleep_ms(5000)          # Lepton settle + first FFC


def _v(x):
    """Firmware compat: blob/statistics fields are properties on some builds,
    methods on others."""
    return x() if callable(x) else x


def grab(cam):
    for _ in range(10):
        img = cam.snapshot()
        if img is not None:
            return img
        time.sleep_ms(10)
    return None


def emit(tag, img, quality):
    data = ubinascii.b2a_base64(img.compress(quality=quality).bytearray())
    print(tag + ":" + data[:-1].decode())


# The Lepton is hard-capped at ~8.7 Hz (VoSPI) and grab(lep) blocks until its
# next frame, so the loop is paced by the thermal sensor no matter what — the
# thermal rate cannot be raised, only protected. The RGB sensor is NOT capped
# at 9 Hz, so to make the RGB panel smoother we send it more often (every 2nd
# thermal cycle instead of every 3rd) and drop its JPEG quality so the extra
# frame does not eat enough USB bandwidth to throttle the thermal rate.
RGB_EVERY = 2
RGB_QUALITY = 40
THERMAL_QUALITY = 80
n = 0

while True:
    t = grab(lep)
    if t is not None:
        st = t.get_statistics()
        print("S:" + json.dumps({"min": g_to_temp(_v(st.min)),
                                 "mean": g_to_temp(_v(st.mean)),
                                 "max": g_to_temp(_v(st.max))}))
        detections = []
        for label, thr in BANDS:
            for b in t.find_blobs(thr, pixels_threshold=8,
                                  area_threshold=8, merge=True):
                rect = tuple(_v(b.rect))
                s = t.get_statistics(thresholds=thr, roi=rect)
                detections.append({"label": label,
                                   "rect": rect,
                                   "t_mean": g_to_temp(_v(s.mean)),
                                   "t_max": g_to_temp(_v(s.max))})
        print("D:" + json.dumps(detections))
        emit("T", t, THERMAL_QUALITY)   # compress LAST — mutates the image

    n += 1
    if n % RGB_EVERY == 0:
        r = grab(rgb)
        if r is not None:
            emit("R", r, RGB_QUALITY)
