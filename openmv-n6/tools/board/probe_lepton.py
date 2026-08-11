# Bring the Lepton 3.5 up cleanly: radiometry on, framesize set, VoSPI synced.
import csi, time

LEP = 0x5435

lep = csi.CSI(cid=LEP)
lep.reset()
lep.pixformat(csi.GRAYSCALE)

# radiometric (TLinear) mode + measurement range suited to electrical inspection.
# Second arg is high_temp_mode = LOW gain, not radiometry - keep it False.
lep.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)
lep.ioctl(csi.IOCTL_LEPTON_SET_RANGE, -10, 140)
print("radiometry :", lep.ioctl(csi.IOCTL_LEPTON_GET_RADIOMETRY))
print("mode       :", lep.ioctl(csi.IOCTL_LEPTON_GET_MODE))
print("range      :", lep.ioctl(csi.IOCTL_LEPTON_GET_RANGE))

for fs in ("QQVGA", "QQQVGA", "QVGA", "VGA"):
    try:
        lep.framesize(getattr(csi, fs))
        print("framesize %-7s -> %dx%d" % (fs, lep.width(), lep.height()))
    except Exception as e:
        print("framesize %-7s ERR %s" % (fs, e))

lep.framesize(csi.QQVGA)
print("settling for VoSPI sync + FFC ...")
time.sleep_ms(2000)

ok = 0
for i in range(12):
    t0 = time.ticks_us()
    try:
        img = lep.snapshot()
        dt = time.ticks_diff(time.ticks_us(), t0) / 1000.0
        stats = img.get_statistics()
        print("  %2d OK  %dx%d %6.1fms  min=%d max=%d mean=%d" % (
            i, img.width(), img.height(), dt, stats.min(), stats.max(), stats.mean()))
        ok += 1
    except Exception as e:
        dt = time.ticks_diff(time.ticks_us(), t0) / 1000.0
        print("  %2d ERR %s (%.1fms)" % (i, e, dt))
        time.sleep_ms(200)

print("captured %d/12" % ok)
print("FPA temp   :", lep.ioctl(csi.IOCTL_LEPTON_GET_FPA_TEMP))
