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
# One argument on purpose. SET_MODE(a, b) is (measurement_mode, high_temp_mode);
# the second selects LOW gain, which buys a 600C ceiling this window does not
# use and costs the tighter accuracy spec. Radiometry is the FIRST argument.
csi0.ioctl(csi.IOCTL_LEPTON_SET_MODE, True)            # measurement (radiometric) mode
csi0.ioctl(csi.IOCTL_LEPTON_SET_RANGE, MIN_C, MAX_C)  # MIN_C..MAX_C -> 0..255
csi0.snapshot(time=5000)                               # settle + first FFC

# Fatal, not informational. Without radiometry the frame is the Lepton's own
# scene-relative AGC output, map_g_to_temp is arithmetic on nothing, and every
# number below is fabricated - while still printing to two significant figures.
# Printing the state and carrying on is the worst of the three options.
if not csi0.ioctl(csi.IOCTL_LEPTON_GET_RADIOMETRY):
    raise RuntimeError("radiometry did not enable - every temperature below "
                       "would be fabricated. Refusing to run.")
print("Radiometry: Yes")
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

# Measurement windows are open at the top. Segmenting with the band is right -
# that is what defines the item - but measuring through it caps t_max at the
# band's own upper edge, so a "human" blob could never report above 38.0C
# however hot it really was. Under-reported by up to the band width.
STAT_BANDS = [(lbl, [(temp_to_g(a), 255)]) for (lbl, a, _b) in CLASSES]

# --- dead-row defect --------------------------------------------------------
# This part intermittently kills a fixed set of rows and then holds them for the
# rest of the session. They are NOT stuck at 255: they sit at a fixed high
# offset and only clip once the scene level rises, so a brightness threshold
# misses them on roughly half the frames. Flat AND lifted is the test that works.
# Unrepaired they land in the "hot" band and emit a full-width false target on
# every frame - 14 of them on the affected unit.
DEADROW_FLAT = 24        # max spread within the row, in codes
DEADROW_LIFT = 64        # min excess of the row mean over the frame median
DEADROW_RECHECK = 30     # re-detect every N frames (~3.4s); the defect can
                         # begin mid-session, so detecting once is not enough
_bad_rows = []


def find_dead_rows(img):
    """Rows that are flat and lifted relative to the frame median."""
    w, h = img.width(), img.height()
    hist = img.get_histogram()
    median = int(hist.get_percentile(0.5).value() * 255)
    bad = []
    for y in range(h):
        st = img.get_statistics(roi=(0, y, w, 1))
        if st.max() - st.min() <= DEADROW_FLAT and st.mean() - median >= DEADROW_LIFT:
            bad.append(y)
    return bad


def repair_rows(img, rows):
    """Flatten each condemned row to its neighbours' level.

    Deliberately a fill and not a per-pixel interpolation. The per-pixel version
    is 160 get_pixel + 160 set_pixel per row through the MicroPython interpreter
    -- on 14 rows that is ~4500 calls inside a 113ms frame budget, and this loop
    has no headroom to spend. A row that has been condemned carries no scene
    information to preserve anyway; the job here is only to stop it being
    segmented as a hot target. One C-side draw per row does that.
    """
    w, h = img.width(), img.height()
    for y in rows:
        up, dn = y - 1, y + 1
        while up >= 0 and up in rows:
            up -= 1
        while dn < h and dn in rows:
            dn += 1
        levels = []
        if up >= 0:
            levels.append(img.get_statistics(roi=(0, up, w, 1)).mean())
        if dn < h:
            levels.append(img.get_statistics(roi=(0, dn, w, 1)).mean())
        if not levels:
            continue
        img.draw_rectangle(0, y, w, 1, color=int(sum(levels) / len(levels)),
                           fill=True)


clock = time.clock()
frame_i = 0

while True:
    clock.tick()
    img = csi0.snapshot()
    frame_i += 1

    if frame_i % DEADROW_RECHECK == 1:
        _bad_rows = find_dead_rows(img)
    if _bad_rows:
        repair_rows(img, _bad_rows)

    # segment each temperature band into blobs ("items")
    # NOTE rect(), mean(), max(): these are METHODS on OpenMV's blob and
    # statistics objects. They were read as attributes here, which handed a
    # bound method to map_g_to_temp instead of a number - this loop could never
    # have run to completion.
    detections = []
    for (label, thr), (_l, stat_thr) in zip(BANDS, STAT_BANDS):
        for b in img.find_blobs(thr, pixels_threshold=8, area_threshold=8, merge=True):
            rect = b.rect()
            st = img.get_statistics(thresholds=stat_thr, roi=rect)
            t_max = map_g_to_temp(st.max())
            detections.append({
                "label": label,
                "rect": rect,              # (x, y, w, h) in Lepton pixels
                "cx": b.cx(),
                "cy": b.cy(),
                "t_mean": map_g_to_temp(st.mean()),
                "t_max": t_max,
                # The window clips at MAX_C, and a clipped reading is
                # indistinguishable from a real one at exactly MAX_C - a 200C
                # manifold and a 45C one both report 45.0. Say which it is.
                "saturated": st.max() >= 255,
                "repaired_rows": len(_bad_rows),
            })

    # colorize + annotate. to_rainbow rewrites the frame in place, so every
    # measurement above has to be finished before this line - the separation is
    # by ordering only, which is why the note matters.
    img.to_rainbow(color_palette=image.PALETTE_IRONBOW)
    for d in detections:
        img.draw_rectangle(d["rect"])
        img.draw_string(d["rect"][0], d["rect"][1] - 9,
                        "%s %.1fC" % (d["label"], d["t_mean"]))

    # `detections` = per-item heat map -> pass to radar association step.
    # FPA temp is handy to watch when the shutter (FFC) will fire:
    # print("FPS %.1f  FPA %.1fC" % (clock.fps(), csi0.ioctl(csi.IOCTL_LEPTON_GET_FPA_TEMP)))
