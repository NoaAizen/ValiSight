"""Dual-camera streamer — runs ON the OpenMV N6, paired with view_thermal_rgb.py.

Grabs frames from BOTH sensors on the board — the FLIR Lepton 3.5 (radiometric
thermal) and the stock PAG7936 RGB — JPEG-compresses each, and prints them as
base64 lines over the USB VCP. The PC side (view_thermal_rgb.py) launches this
script via mpremote and decodes the lines into a live side-by-side view.

Line protocol (one record per line). <ticks> is time.ticks_us() latched right
after that frame's snapshot() returned (src/frame_clock.py doctrine) — the N6
monotonic axis both sensors share, which is what makes the two streams
pairable offline for thermal<->RGB registration:
    T:<ticks>:<base64 raw>    thermal frame, 19200 B raw grayscale,
                              MIN_C..MAX_C -> 0..255. Raw and not JPEG for the
                              same reason live_server ships thermal raw:
                              a recording of compressed pixels can never be
                              re-processed, and registration centroids must
                              come from the sensor's exact bytes.
    R:<ticks>:<base64 jpeg>   RGB frame
    D:<json>                  thermal detections of the preceding T frame
    S:<json>                  thermal frame stats {min, mean, max} in deg C

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

# A sensor that hit a CSI overflow does NOT come back by retrying snapshot()
# — measured live: the Lepton stayed dead for a full minute while the RGB kept
# flowing. Recovery is a full re-init of that sensor, so the init sequences
# live in functions the main loop can call again.
#
# Reset discipline per the OpenMV Multispectral-Thermal module docs: the
# COLOUR sensor comes up FIRST with reset(hard=True) — that call brings the
# module's power rail up — and the Lepton is configured with reset(hard=False)
# so its driver only reprograms the chip without re-toggling reset. Plain
# reset() on both, Lepton first (the previous version of this file), is the
# prime suspect for the wedged dual mode measured at 0.1 fps.

def rgb_init(hard=True):
    rgb = csi.CSI(cid=csi.PAG7936)
    rgb.reset(hard=hard)
    rgb.pixformat(csi.RGB565)
    rgb.framesize(csi.QVGA)
    return rgb


def lepton_init():
    lep = csi.CSI(cid=csi.LEPTON)
    lep.reset(hard=False)
    lep.pixformat(csi.GRAYSCALE)
    lep.framesize(csi.QQVGA)
    lep.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)
    lep.ioctl(csi.IOCTL_LEPTON_SET_RANGE, MIN_C, MAX_C)
    return lep


rgb = rgb_init(hard=True)    # colour first: hard reset raises the rail
lep = lepton_init()          # then Lepton, soft-configured
time.sleep_ms(5000)          # Lepton settle + first FFC


def _v(x):
    """Firmware compat: blob/statistics fields are properties on some builds,
    methods on others."""
    return x() if callable(x) else x


def grab(cam):
    """(image, ticks_us) — ticks latched right after snapshot() returns, which
    is the instant frame_clock's mono_us axis is defined by. (None, None) if
    the sensor stalls.

    snapshot() can raise 'Frame buffer overflow' when the CSI FIFO fills while
    the CPU is busy pushing a 26 KB thermal line over the VCP — measured live,
    it killed the whole stream ~3 s in. A dropped frame is recoverable; a dead
    stream is not, so swallow-and-retry."""
    for _ in range(10):
        try:
            img = cam.snapshot()
        except RuntimeError:
            time.sleep_ms(5)
            continue
        if img is not None:
            return img, time.ticks_us()
        time.sleep_ms(10)
    return None, None


def emit(tag, ticks, data):
    b64 = ubinascii.b2a_base64(data)
    print(tag + ":" + str(ticks) + ":" + b64[:-1].decode())


# The Lepton is hard-capped at ~8.7 Hz (VoSPI) and grab(lep) blocks until its
# next frame, so the loop is paced by the thermal sensor no matter what — the
# thermal rate cannot be raised, only protected. The RGB sensor is NOT capped
# at 9 Hz, so to make the RGB panel smoother we send it more often (every 2nd
# thermal cycle instead of every 3rd) and drop its JPEG quality so the extra
# frame does not eat enough USB bandwidth to throttle the thermal rate.
# Raw thermal is ~26 KB/frame after base64 (vs ~8 KB as JPEG); the honest fps
# counter in view_thermal_rgb.py is the throttle alarm — if THERMAL drops
# below ~8.5, raise RGB_EVERY before touching anything else.
# RGB_EVERY=1 since the direct-pyserial transport: an RGB frame is grabbed
# right after EVERY thermal frame, so the fusion view never composites a
# moving person against a quarter-second-old background (the "same person
# twice" ghost). Costs ~+25 KB/s against a link measured sustaining ~270.
RGB_EVERY = 1
RGB_QUALITY = 40
REINIT_AFTER = 3             # consecutive dead grabs before re-initialising
n = 0
lep_miss = rgb_miss = 0

while True:
    t, t_ticks = grab(lep)
    if t is None:
        lep_miss += 1
        if lep_miss >= REINIT_AFTER:
            print("warn: lepton stalled - reinitialising")
            lep = lepton_init()
            time.sleep_ms(500)
            lep_miss = 0
    else:
        lep_miss = 0
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
        emit("T", t_ticks, t.bytearray())   # raw sensor bytes, uncompressed

    n += 1
    if n % RGB_EVERY == 0:
        r, r_ticks = grab(rgb)
        if r is None:
            rgb_miss += 1
            if rgb_miss >= REINIT_AFTER:
                print("warn: rgb stalled - reinitialising")
                # soft mid-run: a hard reset re-toggles the module rail and
                # would take the Lepton down with it
                rgb = rgb_init(hard=False)
                time.sleep_ms(200)
                rgb_miss = 0
        else:
            rgb_miss = 0
            # compress LAST — it mutates the image
            emit("R", r_ticks, r.compress(quality=RGB_QUALITY).bytearray())
