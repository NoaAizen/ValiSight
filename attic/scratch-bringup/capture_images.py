"""Capture real RGB and thermal frames on the OpenMV N6 and pull them off.

The board writes JPEGs to its own filesystem first, then they come back over the
REPL as base64 chunks -- the same transfer that already worked for source files.
Compressing on-device keeps the serial transfer small; raw RGB565 at 320x200
would be 128 KB, which is a minute of wall-clock at 115200 baud.
"""
import base64, os, serial, sys, time

PORT = sys.argv[1]
OUT = sys.argv[2]
PROMPT = b">>> "
CHUNK = 512


def read_to_prompt(ser, timeout=60):
    buf, t0 = b"", time.time()
    while time.time() - t0 < timeout:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
            if buf.endswith(PROMPT):
                break
        else:
            time.sleep(0.02)
    return buf


def run(ser, line, timeout=60):
    ser.write(b"\x03")
    time.sleep(0.15)
    ser.reset_input_buffer()
    ser.write(line.encode() + b"\r\n")
    raw = read_to_prompt(ser, timeout).decode(errors="replace")
    out = raw.split("\r\n", 1)[-1]
    for tail in (">>> ", "... "):
        while out.endswith(tail):
            out = out[: -len(tail)]
    return out.strip()


def guarded(expr):
    return 'exec("try:\\n %s\\nexcept Exception as e:\\n print(\'ERR\',e)")' % expr


def pull(ser, path, dest):
    size_s = run(ser, "import os; print(os.stat('%s')[6])" % path)
    try:
        size = int(size_s.strip().splitlines()[-1])
    except Exception:
        print("  pull %s: cannot stat (%s)" % (path, size_s[:60]))
        return False
    data = b""
    for off in range(0, size, CHUNK):
        r = run(ser,
                "import binascii; f=open('%s','rb'); f.seek(%d); "
                "print(binascii.b2a_base64(f.read(%d)).decode().strip()); f.close()"
                % (path, off, CHUNK))
        line = r.strip().splitlines()[-1] if r.strip() else ""
        try:
            data += base64.b64decode(line)
        except Exception as e:
            print("  pull %s: chunk %d failed (%s)" % (path, off, e))
            return False
    with open(dest, "wb") as fh:
        fh.write(data)
    print("  %s -> %s (%d bytes)" % (path, os.path.basename(dest), len(data)))
    return True


ser = serial.Serial(PORT, 115200, timeout=1)
time.sleep(0.5)
ser.write(b"\x03")
time.sleep(0.2)
ser.reset_input_buffer()
ser.write(b"\r\n")
read_to_prompt(ser, 3)

os.makedirs(OUT, exist_ok=True)

print("== capture RGB ==")
print(" ", run(ser, guarded(
    "import csi; c=csi.CSI(cid=csi.PAG7936); c.reset(); c.pixformat(csi.RGB565); "
    "c.framesize(csi.VGA); c.snapshot(time=1500); img=c.snapshot(); "
    "img.save('/flash/cap_rgb.jpg', quality=92); "
    "print('saved rgb %dx%d' % (img.width(), img.height()))"), 90))

print("== capture thermal ==")
print(" ", run(ser, guarded(
    "import csi; t=csi.CSI(cid=csi.LEPTON); t.reset(); t.pixformat(csi.GRAYSCALE); "
    "t.framesize(csi.QQVGA); t.snapshot(); img=t.snapshot(); "
    "img.save('/flash/cap_thermal.jpg', quality=92); "
    "print('saved thermal %dx%d' % (img.width(), img.height()))"), 90))

print("== capture thermal (rainbow palette) ==")
print(" ", run(ser, guarded(
    "import csi; t=csi.CSI(cid=csi.LEPTON); t.reset(); t.pixformat(csi.GRAYSCALE); "
    "t.framesize(csi.QQVGA); t.snapshot(); img=t.snapshot(); "
    "r=img.to_rainbow(color_palette=None); r.save('/flash/cap_thermal_rgb.jpg', quality=92); "
    "print('saved thermal-rainbow %dx%d' % (r.width(), r.height()))"), 90))

print()
print("== pulling files ==")
pull(ser, "/flash/cap_rgb.jpg", os.path.join(OUT, "cap_rgb.jpg"))
pull(ser, "/flash/cap_thermal.jpg", os.path.join(OUT, "cap_thermal.jpg"))
pull(ser, "/flash/cap_thermal_rgb.jpg", os.path.join(OUT, "cap_thermal_rgb.jpg"))

ser.close()
