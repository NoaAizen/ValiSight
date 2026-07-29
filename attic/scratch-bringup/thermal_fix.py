"""Try to get a valid frame out of the Lepton, and prove whether it is valid.

The first capture came back as vertical banding -- the signature of an
unsynchronised VoSPI stream. A Lepton needs time after power-up and a flat-field
correction before its output means anything, so this warms up, runs an FFC, then
checks the pixel statistics: a real scene varies across the frame, whereas a
desynchronised stream tends to be near-random with an extreme spread.
"""
import base64, os, serial, sys, time

PORT = sys.argv[1]
OUT = sys.argv[2]
PROMPT = b">>> "
CHUNK = 512


def read_to_prompt(ser, timeout=90):
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


def run(ser, line, timeout=90):
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
        print("  cannot stat %s" % path)
        return False
    data = b""
    for off in range(0, size, CHUNK):
        r = run(ser, "import binascii; f=open('%s','rb'); f.seek(%d); "
                     "print(binascii.b2a_base64(f.read(%d)).decode().strip()); f.close()"
                     % (path, off, CHUNK))
        line = r.strip().splitlines()[-1] if r.strip() else ""
        try:
            data += base64.b64decode(line)
        except Exception:
            print("  chunk %d failed" % off)
            return False
    with open(dest, "wb") as fh:
        fh.write(data)
    print("  %s (%d bytes)" % (os.path.basename(dest), len(data)))
    return True


ser = serial.Serial(PORT, 115200, timeout=1)
time.sleep(0.5)
ser.write(b"\x03")
time.sleep(0.2)
ser.reset_input_buffer()
ser.write(b"\r\n")
read_to_prompt(ser, 3)
os.makedirs(OUT, exist_ok=True)

print("== warm up + FFC ==")
print(" ", run(ser, guarded(
    "import csi, time; t=csi.CSI(cid=csi.LEPTON); t.reset(); "
    "t.pixformat(csi.GRAYSCALE); t.framesize(csi.QQVGA); "
    "[t.snapshot() for _ in range(40)]; time.sleep_ms(2000); "
    "print('warmed'); "
    "t.ioctl(csi.IOCTL_LEPTON_RUN_COMMAND, 0x0242); time.sleep_ms(2000); "
    "print('ffc done')"), 120))

print("== frame statistics over 5 captures ==")
print(" ", run(ser, guarded(
    "import csi, time; t=csi.CSI(cid=csi.LEPTON); t.reset(); "
    "t.pixformat(csi.GRAYSCALE); t.framesize(csi.QQVGA); "
    "[t.snapshot() for _ in range(20)]; "
    "rows=[]\\n"
    " for i in range(5):\\n"
    "  im=t.snapshot(); s=im.get_statistics(); "
    "rows.append((s.min(), s.mean(), s.max(), s.stdev()))\\n"
    "  time.sleep_ms(300)\\n"
    " print(rows)"), 120))

print("== save a warmed frame ==")
print(" ", run(ser, guarded(
    "import csi, time; t=csi.CSI(cid=csi.LEPTON); t.reset(); "
    "t.pixformat(csi.GRAYSCALE); t.framesize(csi.QQVGA); "
    "[t.snapshot() for _ in range(30)]; time.sleep_ms(1000); "
    "im=t.snapshot(); im.save('/flash/cap_thermal2.jpg', quality=95); "
    "print('saved %dx%d' % (im.width(), im.height()))"), 120))

print()
print("== pull ==")
pull(ser, "/flash/cap_thermal2.jpg", os.path.join(OUT, "cap_thermal2.jpg"))
ser.close()
