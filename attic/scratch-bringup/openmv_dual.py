"""Check whether the N6 can hold the RGB and Lepton CSI instances at once.

A fusion pipeline needs both live in the same script, so this captures from each
in turn from a single session rather than opening them independently.
Also queries the Lepton mode, since radiometry reported 0 (disabled) and the FPA
temperature is meaningless until it is on.
"""
import serial, sys, time

PORT = sys.argv[1]
PROMPT = b">>> "


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


def run(ser, line, timeout=40):
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


ser = serial.Serial(PORT, 115200, timeout=1)
time.sleep(0.5)
ser.write(b"\x03")
time.sleep(0.2)
ser.reset_input_buffer()
ser.write(b"\r\n")
read_to_prompt(ser, 3)

print("== lepton mode / range ==")
print("  get_mode :", run(ser, guarded(
    "import csi; c=csi.CSI(cid=csi.LEPTON); c.reset(); print(c.ioctl(csi.IOCTL_LEPTON_GET_MODE))")))
print("  set_mode :", run(ser, guarded(
    "import csi; c=csi.CSI(cid=csi.LEPTON); c.reset(); c.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, True); "
    "print('mode now', c.ioctl(csi.IOCTL_LEPTON_GET_MODE), 'radio', c.ioctl(csi.IOCTL_LEPTON_GET_RADIOMETRY))")))
print("  fpa temp :", run(ser, guarded(
    "import csi; c=csi.CSI(cid=csi.LEPTON); c.reset(); c.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, True); "
    "print('fpa %.2f C  aux %.2f C' % (c.ioctl(csi.IOCTL_LEPTON_GET_FPA_TEMP), c.ioctl(csi.IOCTL_LEPTON_GET_AUX_TEMP)))")))

print()
print("== both sensors in one session ==")
print(run(ser, guarded(
    "import csi; "
    "t=csi.CSI(cid=csi.LEPTON); t.reset(); t.pixformat(csi.GRAYSCALE); t.framesize(csi.QQVGA); "
    "r=csi.CSI(cid=csi.PAG7936); r.reset(); r.pixformat(csi.RGB565); r.framesize(csi.QVGA); "
    "ti=t.snapshot(); ri=r.snapshot(); "
    "print('thermal %dx%d  rgb %dx%d' % (ti.width(), ti.height(), ri.width(), ri.height()))"),
    timeout=60))

ser.close()
