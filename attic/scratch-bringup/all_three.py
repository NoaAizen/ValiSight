"""Prove radar, RGB and thermal are live at the same moment.

The radar is already streaming from the previous run, so this only listens.
Camera frames stay on the OpenMV board -- only their geometry and intensity
statistics cross the REPL, since pushing pixels over 115200 baud is pointless
for a liveness check.
"""
import glob, os, sys, time

sys.path.insert(0, "/workspace/host_test")

import serial
from iwr1843_uart import RadarReader
import radar_gate, radar_classify_n6

BY_ID = "/dev/serial/by-id"
PROMPT = b">>> "


def by_id(pattern):
    for p in sorted(glob.glob(os.path.join(BY_ID, pattern))):
        return os.path.realpath(p)
    return None


def read_to_prompt(ser, timeout=40):
    buf, t0 = b"", time.time()
    while time.time() - t0 < timeout:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
            if buf.endswith(PROMPT):
                break
        else:
            time.sleep(0.05)
    return buf


def repl(ser, line, timeout=40):
    ser.write(b"\x03")
    time.sleep(0.15)
    ser.reset_input_buffer()
    ser.write(line.encode() + b"\r\n")
    raw = read_to_prompt(ser, timeout).decode(errors="replace")
    out = raw.split("\r\n", 1)[-1]
    for tail in (">>> ", "... "):
        while out.endswith(tail):
            out = out[: -len(tail)]
    return out.strip().replace("\r\n", " | ")


def guarded(expr):
    return 'exec("try:\\n %s\\nexcept Exception as e:\\n print(\'ERR\',e)")' % expr


radar_data = by_id("*XDS110*if03")
openmv = by_id("*MicroPython*if00")
print("radar DATA : %s" % radar_data)
print("openmv     : %s" % openmv)
print()

print("== cameras (on-board) ==")
omv = serial.Serial(openmv, 115200, timeout=1)
time.sleep(0.5)
omv.write(b"\x03")
time.sleep(0.2)
omv.reset_input_buffer()
omv.write(b"\r\n")
read_to_prompt(omv, 3)

print("  " + repl(omv, guarded(
    "import csi; "
    "t=csi.CSI(cid=csi.LEPTON); t.reset(); t.pixformat(csi.GRAYSCALE); t.framesize(csi.QQVGA); "
    "r=csi.CSI(cid=csi.PAG7936); r.reset(); r.pixformat(csi.RGB565); r.framesize(csi.QVGA); "
    "ti=t.snapshot(); ri=r.snapshot(); s=ti.get_statistics(); "
    "print('thermal %dx%d min/mean/max %d/%d/%d | rgb %dx%d' % "
    "(ti.width(), ti.height(), s.min(), s.mean(), s.max(), ri.width(), ri.height()))"), 60))
omv.close()

print()
print("== radar (live, 6s) ==")
reader = RadarReader()
n_frames = n_raw = n_kept = 0
labels = {}
t0 = time.time()
with serial.Serial(radar_data, 921600, timeout=0.1) as ser:
    ser.reset_input_buffer()
    while time.time() - t0 < 6.0:
        for fr in reader.feed(ser.read(8192)):
            n_frames += 1
            n_raw += len(fr["points"])
            kept, _ = radar_gate.gate_points(fr["points"], fr["snr"], fr["noise"])
            n_kept += len(kept)
            for c in radar_classify_n6.classify_frame(kept):
                labels[c["label"]] = labels.get(c["label"], 0) + 1

print("  frames %d | points %d -> %d kept | %s" % (n_frames, n_raw, n_kept, labels or "no clusters"))
