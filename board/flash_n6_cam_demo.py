"""Live camera demo — OpenMV N6 with the stock PAG7936 RGB sensor.
Continuous capture; prints FPS once a second. Stop with Ctrl+C."""
import csi
import time

c = csi.CSI()
c.reset()
c.pixformat(csi.RGB565)
c.framesize(csi.VGA)
c.snapshot(time=2000)  # let AE/AWB settle

print("N6 camera running: %dx%d" % (c.width(), c.height()))

clock = time.clock()
frames = 0
last = time.ticks_ms()
while True:
    clock.tick()
    img = c.snapshot()
    frames += 1
    now = time.ticks_ms()
    if time.ticks_diff(now, last) >= 1000:
        print("frames: %d  FPS: %.1f" % (frames, clock.fps()))
        last = now
