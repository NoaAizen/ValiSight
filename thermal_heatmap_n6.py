"""
Thermal heat-mapping — OpenMV N6 + FLIR Lepton 3.5 (radiometric).

IMPORTANT: the N6 uses the NEW `csi` module (firmware 4.5+ / MicroPython 1.28),
NOT the old `fir` / `sensor` module. Verified against OpenMV's official
lepton_get_object_temp_color.py example.

What it does: puts the Lepton in radiometric measurement mode, maps a fixed
temperature window to 0..255, then segments the image into temperature bands
("items by heat"), measures each region's real temperature, and classifies it
(warm / human-range / hot). The `detections` list is what you hand to the
radar-fusion / association step.
"""
import csi
import image
import time

# --- radiometric range mapped to grayscale 0..255 ---------------------------
MIN_C = 15.0        # low end of the mapped range (deg C)
MAX_C = 45.0        # high end (covers bodies + warm engines/exhaust)

csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(csi.GRAYSCALE)
csi0.framesize(csi.QQVGA)                               # Lepton native 160x120
csi0.ioctl(csi.IOCTL_LEPTON_SET_MODE, True)            # measurement (radiometric) mode
csi0.ioctl(csi.IOCTL_LEPTON_SET_RANGE, MIN_C, MAX_C)  # MIN_C..MAX_C -> 0..255
csi0.snapshot(time=5000)                               # settle + first FFC

print("Radiometry:", "Yes" if csi0.ioctl(csi.IOCTL_LEPTON_GET_RADIOMETRY) else "No")
print("Resolution: %dx%d" % (csi0.ioctl(csi.IOCTL_LEPTON_GET_WIDTH),
                             csi0.ioctl(csi.IOCTL_LEPTON_GET_HEIGHT)))


def temp_to_g(t):
    if t < MIN_C:
        t = MIN_C
    elif t > MAX_C:
        t = MAX_C
    return int((t - MIN_C) * 255.0 / (MAX_C - MIN_C))


def map_g_to_temp(g):
    return (g * (MAX_C - MIN_C) / 255.0) + MIN_C


# --- temperature classes -> grayscale windows -------------------------------
# (label, tmin_c, tmax_c). Bodies ~30-37 C; engines/exhaust/fire hotter.
CLASSES = [
    ("warm", 25.0, 30.0),
    ("human", 30.0, 38.0),
    ("hot", 38.0, 45.0),
]
BANDS = [(lbl, [(temp_to_g(a), temp_to_g(b))]) for (lbl, a, b) in CLASSES]

clock = time.clock()

while True:
    clock.tick()
    img = csi0.snapshot()

    # segment each temperature band into blobs ("items")
    detections = []
    for label, thr in BANDS:
        for b in img.find_blobs(thr, pixels_threshold=8, area_threshold=8, merge=True):
            st = img.get_statistics(thresholds=thr, roi=b.rect)
            detections.append({
                "label": label,
                "rect": b.rect,            # (x, y, w, h) in Lepton pixels
                "cx": b.cx(),
                "cy": b.cy(),
                "t_mean": map_g_to_temp(st.mean),
                "t_max": map_g_to_temp(st.max),
            })

    # colorize + annotate (for a live view / debugging)
    img.to_rainbow(color_palette=image.PALETTE_IRONBOW)
    for d in detections:
        img.draw_rectangle(d["rect"])
        img.draw_string(d["rect"][0], d["rect"][1] - 9,
                        "%s %.1fC" % (d["label"], d["t_mean"]))

    # `detections` = per-item heat map -> pass to radar association step.
    # FPA temp is handy to watch when the shutter (FFC) will fire:
    # print("FPS %.1f  FPA %.1fC" % (clock.fps(), csi0.ioctl(csi.IOCTL_LEPTON_GET_FPA_TEMP)))
