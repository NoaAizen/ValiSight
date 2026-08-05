# Milestone 0 verification: real pixel data, true refresh, dual capture, auto-range.
import csi, time

PAG, LEP = 0x7936, 0x5435


class Lepton:
    def __init__(self, tmin=-10, tmax=140):
        self.tmin, self.tmax = tmin, tmax
        c = csi.CSI(cid=LEP)
        c.reset()
        c.pixformat(csi.GRAYSCALE)
        c.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, True)
        c.ioctl(csi.IOCTL_LEPTON_SET_RANGE, tmin, tmax)
        c.framesize(csi.QQVGA)
        self.csi = c
        time.sleep_ms(2000)
        c.snapshot()                      # absorb slow first sync frame

    def set_range(self, tmin, tmax):
        self.tmin, self.tmax = tmin, tmax
        self.csi.ioctl(csi.IOCTL_LEPTON_SET_RANGE, int(tmin), int(tmax))

    def to_c(self, p):
        return self.tmin + p * (self.tmax - self.tmin) / 255.0

    def step_c(self):
        return (self.tmax - self.tmin) / 255.0

    def snapshot(self):
        return self.csi.snapshot()


print("=== lepton bring-up (wide range) ===")
lep = Lepton(-10, 140)
img = lep.snapshot()
s = img.get_statistics()
print("  %dx%d  codes %d..%d of 0..255  step=%.2f C/LSB" % (
    img.width(), img.height(), s.min, s.max, lep.step_c()))
print("  scene  %.1fC .. %.1fC  (mean %.1fC)" % (
    lep.to_c(s.min), lep.to_c(s.max), lep.to_c(s.mean)))

print("=== auto-range to the scene ===")
lo_c, hi_c = lep.to_c(s.min), lep.to_c(s.max)
pad = max(3.0, (hi_c - lo_c) * 0.15)
lep.set_range(int(lo_c - pad), int(hi_c + pad) + 1)
time.sleep_ms(300)
lep.snapshot()
img = lep.snapshot()
s2 = img.get_statistics()
print("  range now %.0f..%.0fC -> step=%.3f C/LSB (was %.3f)" % (
    lep.tmin, lep.tmax, lep.step_c(), 150.0 / 255.0))
print("  codes now %d..%d of 0..255   scene %.1fC .. %.1fC" % (
    s2.min, s2.max, lep.to_c(s2.min), lep.to_c(s2.max)))

print("=== true thermal refresh ===")
prev, distinct, t0 = None, 0, time.ticks_ms()
N = 40
for _ in range(N):
    b = lep.snapshot().bytearray()
    sig = (b[9600] << 16) | (b[4800] << 8) | b[14400]
    if sig != prev:
        distinct += 1
        prev = sig
el = time.ticks_diff(time.ticks_ms(), t0)
print("  %d snapshots in %dms -> %.1f snap/s, %d distinct (~%.1f Hz real)" % (
    N, el, N * 1000.0 / el, distinct, distinct * 1000.0 / el))

print("=== rgb bring-up ===")
rgb = csi.CSI(cid=PAG)
rgb.reset()
rgb.pixformat(csi.GRAYSCALE)
rgb.framesize(csi.VGA)
time.sleep_ms(300)
rgb.snapshot()
print("  %dx%d grayscale" % (rgb.width(), rgb.height()))

print("=== DUAL CAPTURE ===")
ok = 0
for i in range(8):
    try:
        t0 = time.ticks_us()
        a = rgb.snapshot()
        t1 = time.ticks_us()
        b = lep.snapshot()
        t2 = time.ticks_us()
        print("  %d  rgb %dx%d %5.1fms (%d B) | therm %dx%d %5.1fms (%d B)" % (
            i, a.width(), a.height(), time.ticks_diff(t1, t0) / 1000.0, a.size(),
            b.width(), b.height(), time.ticks_diff(t2, t1) / 1000.0, b.size()))
        ok += 1
    except Exception as e:
        print("  %d  ERR %s" % (i, e))
print("RESULT: %d/8 dual frames" % ok)
