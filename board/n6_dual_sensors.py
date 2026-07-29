"""Bring up both OpenMV N6 cameras: PAG7936 RGB + FLIR Lepton 3.5 thermal.

Copy to the board (/flash or /sdcard) and run it there -- this is MicroPython,
not CPython.

Two things cost an afternoon to work out, so they are spelled out here:

1. Construct the Lepton with the FAMILY constant, csi.LEPTON (0x54). The id that
   csi.devices() reports for it is 0x5435 (family 0x54 + revision 3.5), and
   passing that full id to csi.CSI() fails with "Sensor control failed".

2. reset() alone leaves the sensor at 0x0. Geometry only appears once BOTH
   pixformat() and framesize() have been set.

Radiometry (absolute temperatures) can be turned on with
    csi0.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, True)
which does flip IOCTL_LEPTON_GET_RADIOMETRY to 1 -- but on this unit the
temperature readbacks still return sentinels (FPA -167.78 C, AUX -273.15 C),
so treat the thermal frame as relative intensity until that is sorted out.
"""
import csi
import time

THERMAL_SIZE = csi.QQVGA   # Lepton 3.x native: 160x120
RGB_SIZE = csi.QVGA


def init_thermal():
    c = csi.CSI(cid=csi.LEPTON)
    c.reset()
    c.pixformat(csi.GRAYSCALE)
    c.framesize(THERMAL_SIZE)
    return c


def init_rgb():
    c = csi.CSI(cid=csi.PAG7936)
    c.reset()
    c.pixformat(csi.RGB565)
    c.framesize(RGB_SIZE)
    return c


thermal = init_thermal()
rgb = init_rgb()

print("thermal : %dx%d (cid %s)" % (thermal.width(), thermal.height(), hex(thermal.cid())))
print("rgb     : %dx%d (cid %s)" % (rgb.width(), rgb.height(), hex(rgb.cid())))

frames = 0
last = time.ticks_ms()
while True:
    t_img = thermal.snapshot()
    r_img = rgb.snapshot()
    frames += 1

    now = time.ticks_ms()
    if time.ticks_diff(now, last) >= 1000:
        st = t_img.get_statistics()
        print("%d fps | thermal min/mean/max %d/%d/%d" %
              (frames, st.min(), st.mean(), st.max()))
        frames = 0
        last = now
