# Isolate the dual-capture timeout: each sensor alone, then both together.
import csi, time

PAG, LEP = 0x7936, 0x5435


def attempt(tag, fn):
    t0 = time.ticks_us()
    try:
        img = fn()
        dt = time.ticks_diff(time.ticks_us(), t0) / 1000.0
        print("  %-26s OK  %dx%d  %s  %.1fms" % (tag, img.width(), img.height(), img.format(), dt))
        return img
    except Exception as e:
        dt = time.ticks_diff(time.ticks_us(), t0) / 1000.0
        print("  %-26s ERR %s  (%.1fms)" % (tag, e, dt))
        return None


print("--- 1. thermal alone ---")
lep = csi.CSI(cid=LEP)
lep.reset()
lep.pixformat(csi.GRAYSCALE)
print("  framebuffers:", end=" ")
try:
    print(lep.framebuffers())
except Exception as e:
    print("err", e)
for i in range(3):
    attempt("lepton snapshot %d" % i, lep.snapshot)

print("--- 2. rgb alone (GRAYSCALE VGA) ---")
rgb = csi.CSI(cid=PAG)
rgb.reset()
rgb.pixformat(csi.GRAYSCALE)
rgb.framesize(csi.VGA)
time.sleep_ms(500)
for i in range(3):
    attempt("pag snapshot %d" % i, rgb.snapshot)

print("--- 3. rgb alone (RGB565 VGA) ---")
rgb.pixformat(csi.RGB565)
rgb.framesize(csi.VGA)
time.sleep_ms(500)
for i in range(3):
    attempt("pag rgb565 %d" % i, rgb.snapshot)

print("--- 4. interleaved ---")
for i in range(4):
    attempt("rgb  %d" % i, rgb.snapshot)
    attempt("therm %d" % i, lep.snapshot)

print("--- 5. lepton radiometry controls ---")
for tag, args in [
    ("GET_MODE", (csi.IOCTL_LEPTON_GET_MODE,)),
    ("GET_RADIOMETRY", (csi.IOCTL_LEPTON_GET_RADIOMETRY,)),
    ("GET_RANGE", (csi.IOCTL_LEPTON_GET_RANGE,)),
]:
    try:
        print("  %-16s %s" % (tag, lep.ioctl(*args)))
    except Exception as e:
        print("  %-16s ERR %s" % (tag, e))
for tag, args in [
    ("SET_MODE(True)", (csi.IOCTL_LEPTON_SET_MODE, True)),
    ("SET_MODE(True,True)", (csi.IOCTL_LEPTON_SET_MODE, True, True)),
    ("SET_RANGE(-10,140)", (csi.IOCTL_LEPTON_SET_RANGE, -10, 140)),
]:
    try:
        print("  %-20s -> %s" % (tag, lep.ioctl(*args)))
    except Exception as e:
        print("  %-20s ERR %s" % (tag, e))
try:
    print("  after: radiometry=%s range=%s" % (
        lep.ioctl(csi.IOCTL_LEPTON_GET_RADIOMETRY), lep.ioctl(csi.IOCTL_LEPTON_GET_RANGE)))
except Exception as e:
    print("  after: err", e)
