"""Identify the OpenMV N6's RGB sensor and hunt for the thermal one.

fir.init() auto-detect already failed, so this tries each supported FIR type
explicitly and scans the I2C buses -- a thermal sensor that is wired but of an
unexpected type will still answer on the bus.
"""
import serial, sys, time

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
PROMPT = b">>> "


def read_to_prompt(ser, timeout=20):
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


def run(ser, line, timeout=20):
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

print("== RGB sensor ==")
print("  devices  :", run(ser, guarded("import csi; print(csi.devices())")))
print("  snapshot :", run(ser, guarded(
    "import csi; c=csi.CSI(); c.reset(); c.pixformat(csi.RGB565); c.framesize(csi.QVGA); "
    "img=c.snapshot(); print('frame', img.width(), 'x', img.height(), 'bytes', img.size())"), timeout=30))

print()
print("== thermal: explicit type attempts ==")
for t in ("FIR_MLX90640", "FIR_MLX90641", "FIR_MLX90621", "FIR_AMG8833", "FIR_SHIELD"):
    print("  %-14s %s" % (t, run(ser, guarded(
        "import fir; fir.deinit()\\n fir.init(fir.%s)\\n print('OK', fir.width(), 'x', fir.height())" % t), timeout=25)))

print()
print("== I2C scan (MLX90640=0x33, AMG8833=0x68/0x69) ==")
for bus in (0, 1, 2, 3, 4):
    r = run(ser, guarded(
        "from machine import I2C; b=I2C(%d); d=b.scan(); print([hex(x) for x in d] if d else 'empty')" % bus), timeout=15)
    print("  bus %d : %s" % (bus, r))

print()
print("== /flash/main.py ==")
print(run(ser, guarded("print(open('/flash/main.py').read()[:1200])"), timeout=25))

ser.close()
