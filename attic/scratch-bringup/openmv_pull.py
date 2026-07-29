"""Pull source files off an OpenMV board over its USB REPL.

Read-only: opens files and nothing else, so it cannot disturb the sensors.
Content moves as base64 in 512-byte chunks -- raw text through a REPL gets
mangled by echo and control characters, and one huge line risks truncation.
"""
import base64, os, serial, sys, time

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/probe/board"
PROMPT = b">>> "
CHUNK = 512

FILES = [
    "/flash/main.py",
    "/flash/n6_cam_demo.py",
    "/flash/README.txt",
    "/sdcard/iwr1843_uart.py",
    "/sdcard/radar_classify_n6.py",
    "/sdcard/radar_gate.py",
]


def read_to_prompt(ser, timeout=15):
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


def run(ser, line, timeout=15):
    ser.write(b"\x03")
    time.sleep(0.1)
    ser.reset_input_buffer()
    ser.write(line.encode() + b"\r\n")
    raw = read_to_prompt(ser, timeout).decode(errors="replace")
    out = raw.split("\r\n", 1)[-1]
    for tail in (">>> ", "... "):
        while out.endswith(tail):
            out = out[: -len(tail)]
    return out.strip()


ser = serial.Serial(PORT, 115200, timeout=1)
time.sleep(0.5)
ser.write(b"\x03")
time.sleep(0.2)
ser.reset_input_buffer()
ser.write(b"\r\n")
read_to_prompt(ser, 3)

os.makedirs(OUT, exist_ok=True)

for path in FILES:
    size_s = run(ser, "import os; print(os.stat('%s')[6])" % path)
    try:
        size = int(size_s.strip().splitlines()[-1])
    except Exception:
        print("%-32s SKIP (%s)" % (path, size_s[:60]))
        continue

    data = b""
    ok = True
    for off in range(0, size, CHUNK):
        r = run(
            ser,
            "import binascii; f=open('%s','rb'); f.seek(%d); "
            "print(binascii.b2a_base64(f.read(%d)).decode().strip()); f.close()"
            % (path, off, CHUNK),
        )
        line = r.strip().splitlines()[-1] if r.strip() else ""
        try:
            data += base64.b64decode(line)
        except Exception as e:
            print("%-32s CHUNK FAIL at %d: %s" % (path, off, e))
            ok = False
            break

    if ok:
        dest = os.path.join(OUT, path.strip("/").replace("/", "_"))
        with open(dest, "wb") as fh:
            fh.write(data)
        print("%-32s %6d bytes -> %s" % (path, len(data), os.path.basename(dest)))

ser.close()
