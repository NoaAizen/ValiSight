# Milestone 0 - verify simultaneous RGB + thermal capture on OpenMV N6.
# Runs on the board. Reports geometry, formats, radiometry and capture timing.
import csi, time

PAG, LEP = 0x7936, 0x5435


def show(tag, fn):
    try:
        print("  %-22s %s" % (tag, fn()))
    except Exception as e:
        print("  %-22s ERR %s" % (tag, e))


print("=== RGB: PAG7936 ===")
rgb = csi.CSI(cid=PAG)
rgb.reset()
for pf in ("YUV422", "RGB565", "GRAYSCALE", "BAYER"):
    try:
        rgb.pixformat(getattr(csi, pf))
        print("  pixformat %-10s OK" % pf)
    except Exception as e:
        print("  pixformat %-10s ERR %s" % (pf, e))
rgb.pixformat(csi.RGB565)
for fs in ("VGA", "WVGA", "HD", "WXGA", "QVGA"):
    try:
        rgb.framesize(getattr(csi, fs))
        print("  framesize %-10s -> %dx%d" % (fs, rgb.width(), rgb.height()))
    except Exception as e:
        print("  framesize %-10s ERR %s" % (fs, e))
rgb.framesize(csi.VGA)
show("triggered mode get", lambda: rgb.ioctl(csi.IOCTL_GET_TRIGGERED_MODE))

print("=== THERMAL: Lepton 3.5 ===")
lep = csi.CSI(cid=LEP)
lep.reset()
for pf in ("GRAYSCALE", "RGB565"):
    try:
        lep.pixformat(getattr(csi, pf))
        print("  pixformat %-10s OK" % pf)
    except Exception as e:
        print("  pixformat %-10s ERR %s" % (pf, e))
lep.pixformat(csi.GRAYSCALE)
show("native width", lambda: lep.ioctl(csi.IOCTL_LEPTON_GET_WIDTH))
show("native height", lambda: lep.ioctl(csi.IOCTL_LEPTON_GET_HEIGHT))
show("radiometry", lambda: lep.ioctl(csi.IOCTL_LEPTON_GET_RADIOMETRY))
show("refresh (Hz)", lambda: lep.ioctl(csi.IOCTL_LEPTON_GET_REFRESH))
show("resolution (bits)", lambda: lep.ioctl(csi.IOCTL_LEPTON_GET_RESOLUTION))
show("range (min,max)", lambda: lep.ioctl(csi.IOCTL_LEPTON_GET_RANGE))
show("FPA temp (C)", lambda: lep.ioctl(csi.IOCTL_LEPTON_GET_FPA_TEMP))
show("AUX temp (C)", lambda: lep.ioctl(csi.IOCTL_LEPTON_GET_AUX_TEMP))
try:
    lep.framesize(csi.QQVGA)
except Exception:
    pass
print("  frame geometry         %dx%d" % (lep.width(), lep.height()))

print("=== SIMULTANEOUS CAPTURE ===")
try:
    for i in range(5):
        t0 = time.ticks_us()
        a = rgb.snapshot()
        t1 = time.ticks_us()
        b = lep.snapshot()
        t2 = time.ticks_us()
        print("  %d  rgb %dx%d %5.1fms | thermal %dx%d %5.1fms" % (
            i, a.width(), a.height(), time.ticks_diff(t1, t0) / 1000.0,
            b.width(), b.height(), time.ticks_diff(t2, t1) / 1000.0))
    print("  RESULT: dual capture OK")
except Exception as e:
    print("  RESULT: FAILED ->", e)
