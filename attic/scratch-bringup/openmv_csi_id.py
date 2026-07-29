"""Identify the CSI devices the OpenMV N6 reports.

Deliberately narrow: touches only the csi module. The earlier fir.deinit() call
hung the board's USB stack and needed a physical reset, so nothing here
de-initialises anything or touches fir.
"""
import serial, sys, time

PORT = sys.argv[1]
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

print("devices    :", run(ser, guarded("import csi; print([hex(d) for d in csi.devices()])")))
print("CSI methods:", run(ser, guarded("import csi; c=csi.CSI(); print([m for m in dir(c) if not m.startswith('_')])")))

# Map each reported id back to the csi.<NAME> constant that equals it.
print("id -> name :", run(ser, guarded(
    "import csi; ds=csi.devices(); "
    "print([(hex(d), [n for n in dir(csi) if getattr(csi,n,None)==d]) for d in ds])")))

ser.close()
