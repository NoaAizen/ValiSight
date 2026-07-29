"""Scan the OpenMV N6's I2C buses for a thermal sensor.

MLX90640/MLX90641 answer at 0x33, MLX90621 at 0x60, AMG8833 at 0x68 or 0x69.
fir.init() auto-detect already failed, so the question is whether the part is on
the bus at all (wiring) or present but not auto-detected (needs an explicit
type). Scanning is read-only; nothing here calls fir.deinit(), which is what
hung the board earlier.
"""
import serial, sys, time

PORT = sys.argv[1]
PROMPT = b">>> "

KNOWN = {
    0x33: "MLX90640 / MLX90641",
    0x60: "MLX90621",
    0x68: "AMG8833 (addr low)",
    0x69: "AMG8833 (addr high)",
    0x2A: "FLIR Lepton (CCI)",
}


def read_to_prompt(ser, timeout=25):
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


def run(ser, line, timeout=25):
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

print("== machine.I2C buses ==")
hits = {}
for bus in range(0, 6):
    r = run(ser, guarded(
        "from machine import I2C; b=I2C(%d); d=b.scan(); print('SCAN', [hex(x) for x in d])" % bus))
    print("  bus %d : %s" % (bus, r))
    if "SCAN" in r:
        for a, name in KNOWN.items():
            if hex(a) in r:
                hits[a] = (bus, name)

print()
print("== pyb.I2C fallback ==")
for bus in range(0, 5):
    print("  bus %d : %s" % (bus, run(ser, guarded(
        "import pyb; b=pyb.I2C(%d, pyb.I2C.MASTER); print('SCAN', [hex(x) for x in b.scan()])" % bus))))

print()
if hits:
    print("THERMAL CANDIDATES FOUND:")
    for a, (bus, name) in hits.items():
        print("  %s on bus %d -> %s" % (hex(a), bus, name))
else:
    print("No known thermal address answered on any bus.")

ser.close()
