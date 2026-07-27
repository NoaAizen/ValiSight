"""Standalone thermal + RGB viewer to run INSIDE the OpenMV IDE (on the N6).

Open this file in the OpenMV IDE and press the green Run button. It shows a
live view in the IDE frame buffer: the RGB (PAG7936) frame with the Lepton 3.5
thermal drawn black-hot (hot = black, cold = white) as an inset in the corner.

Running it in the IDE also HALTS whatever else was running (e.g. a stuck
n6_radar_bridge main.py), so it doubles as a way to free the board. To stop the
bridge from auto-starting again on the next power-up, delete main.py from the
board (OpenMV IDE: open the board's files and remove main.py).

No PC script, no mpremote, no wiring — just this file, run in the IDE.
"""
import csi
import time

MIN_C = 15.0
MAX_C = 45.0

# --- thermal: FLIR Lepton 3.5 (radiometric) ---------------------------------
lep = csi.CSI(cid=csi.LEPTON)
lep.reset()
lep.pixformat(csi.GRAYSCALE)
lep.framesize(csi.QQVGA)                       # 160x120
lep.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)
lep.ioctl(csi.IOCTL_LEPTON_SET_RANGE, MIN_C, MAX_C)

# --- rgb: stock PAG7936 ------------------------------------------------------
rgb = csi.CSI(cid=csi.PAG7936)
rgb.reset()
rgb.pixformat(csi.RGB565)
rgb.framesize(csi.QVGA)                         # 320x240 (the displayed canvas)

time.sleep_ms(5000)                             # Lepton settle + first FFC
clock = time.clock()

while True:
    clock.tick()
    frame = rgb.snapshot()                      # this is what the IDE displays
    therm = lep.snapshot()                      # 160x120 grayscale, MIN..MAX C
    therm.invert()                              # black-hot: hot -> black
    stats = therm.get_statistics()
    frame.draw_image(therm, 0, 0)               # thermal inset, top-left
    frame.draw_rectangle(0, 0, 160, 120, color=(255, 255, 0))
    frame.draw_string(2, 2, "THERMAL (black-hot)", color=(255, 255, 0))
    frame.draw_string(2, 124, "RGB", color=(255, 255, 0))
    print("fps=%.1f  thermal max=%.0f" % (clock.fps(), stats.max()))
