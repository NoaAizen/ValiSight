"""Thermal-ONLY streamer — runs ON the OpenMV N6 over raw REPL.

Same T-line protocol as n6_dual_stream.py (T:<ticks>:<b64 raw 19200B>), no RGB
at all. Exists for jobs that must not share the CSI with the colour sensor:
NUC/FPN calibration capture (tools/calibrate_nuc.py) and thermal-only live
view. Init mirrors n6_dual_stream.lepton_init — if the Lepton setup changes
there, change it here too (the two files are deliberately small so the
duplication stays visible).
"""
import csi
import time
import ubinascii

MIN_C = 15.0
MAX_C = 45.0

lep = csi.CSI(cid=csi.LEPTON)
lep.reset(hard=False)
lep.pixformat(csi.GRAYSCALE)
lep.framesize(csi.QQVGA)
lep.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)
lep.ioctl(csi.IOCTL_LEPTON_SET_RANGE, MIN_C, MAX_C)
time.sleep_ms(5000)              # settle + first FFC

ok = err = 0
while True:
    try:
        img = lep.snapshot()
    except RuntimeError:         # CSI FIFO overflow — drop, don't die
        err += 1
        time.sleep_ms(5)
        continue
    if img is None:
        continue
    ok += 1
    t = time.ticks_us()
    d = ubinascii.b2a_base64(img.bytearray())
    print("T:" + str(t) + ":" + d[:-1].decode())
    if ok % 100 == 0:
        print("stat: ok=%d err=%d" % (ok, err))
