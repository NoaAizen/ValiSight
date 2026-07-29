"""Try to bring up CSI device 0x5435 on the OpenMV N6 (suspected thermal).

Follows the same reset -> pixformat -> framesize -> snapshot sequence that
n6_cam_demo.py uses for the RGB sensor. Each step is separate and guarded so a
failure names the step that failed instead of aborting the run.
"""
import serial, sys, time

PORT = sys.argv[1]
CID = "0x5435"
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

print("reset      :", run(ser, guarded(
    "import csi; c=csi.CSI(cid=%s); c.reset(); print('after reset', c.width(), 'x', c.height())" % CID)))

print("grayscale  :", run(ser, guarded(
    "import csi; c=csi.CSI(cid=%s); c.reset(); c.pixformat(csi.GRAYSCALE); "
    "print('size', c.width(), 'x', c.height())" % CID)))

print("snapshot   :", run(ser, guarded(
    "import csi; c=csi.CSI(cid=%s); c.reset(); c.pixformat(csi.GRAYSCALE); "
    "img=c.snapshot(); print('frame', img.width(), 'x', img.height(), 'bytes', img.size())" % CID),
    timeout=40))

print("lepton ioctl:", run(ser, guarded(
    "import csi; c=csi.CSI(cid=%s); "
    "print('w', c.ioctl(csi.IOCTL_LEPTON_GET_WIDTH), 'h', c.ioctl(csi.IOCTL_LEPTON_GET_HEIGHT), "
    "'radiometric', c.ioctl(csi.IOCTL_LEPTON_GET_RADIOMETRY))" % CID)))

ser.close()
