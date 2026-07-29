"""Bring up the suspected FLIR Lepton on the OpenMV N6.

csi.devices() reports 0x5435 (LEPTON=0x54 plus a 0x35 revision nibble pair,
i.e. a Lepton 3.5), but constructing with that full id yields "Sensor control
failed". The OpenMV Lepton examples pass the base constant csi.LEPTON, so try
that first, then fall back to the raw id.
"""
import serial, sys, time

PORT = sys.argv[1]
PROMPT = b">>> "


def read_to_prompt(ser, timeout=30):
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


def run(ser, line, timeout=30):
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

print("devices now   :", run(ser, guarded("import csi; print([hex(d) for d in csi.devices()])")))

print()
print("-- cid=csi.LEPTON --")
print("  construct   :", run(ser, guarded(
    "import csi; c=csi.CSI(cid=csi.LEPTON); print('cid', hex(c.cid()), 'size', c.width(), 'x', c.height())")))
print("  reset       :", run(ser, guarded(
    "import csi; c=csi.CSI(cid=csi.LEPTON); c.reset(); print('size', c.width(), 'x', c.height())")))
print("  grayscale   :", run(ser, guarded(
    "import csi; c=csi.CSI(cid=csi.LEPTON); c.reset(); c.pixformat(csi.GRAYSCALE); c.framesize(csi.QQVGA); "
    "print('size', c.width(), 'x', c.height())")))
print("  snapshot    :", run(ser, guarded(
    "import csi; c=csi.CSI(cid=csi.LEPTON); c.reset(); c.pixformat(csi.GRAYSCALE); c.framesize(csi.QQVGA); "
    "img=c.snapshot(); print('FRAME', img.width(), 'x', img.height(), 'bytes', img.size())"), timeout=45))

print()
print("-- radiometry --")
print("  ioctl       :", run(ser, guarded(
    "import csi; c=csi.CSI(cid=csi.LEPTON); c.reset(); "
    "print('w', c.ioctl(csi.IOCTL_LEPTON_GET_WIDTH), 'h', c.ioctl(csi.IOCTL_LEPTON_GET_HEIGHT), "
    "'radio', c.ioctl(csi.IOCTL_LEPTON_GET_RADIOMETRY), 'fpa', c.ioctl(csi.IOCTL_LEPTON_GET_FPA_TEMP))")))

ser.close()
