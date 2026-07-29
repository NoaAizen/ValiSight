"""Work out what CSI device 0x5435 on the OpenMV N6 is.

Step 1 dumps every csi.<NAME> integer constant -- pure introspection, no
hardware touched -- so the mapping survives even if step 2 misbehaves.
Step 2 attempts to open 0x5435 and read its geometry.
"""
import serial, sys, time

PORT = sys.argv[1]
PROMPT = b">>> "


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

print("== sensor-name constants and their ids ==")
print(run(ser, guarded(
    "import csi; "
    "names=('LEPTON','BOSON320','BOSON640','PAG7936','PAG7920','GC2145','OV5640','OV2640',"
    "'MT9M114','FROGEYE2020','GENX320','HM01B0','HM0360','SOFTCSI'); "
    "print([(n, hex(getattr(csi,n))) for n in names if hasattr(csi,n)])")))

print()
print("== open 0x5435 ==")
print("  geometry :", run(ser, guarded(
    "import csi; c=csi.CSI(cid=0x5435); print('cid', hex(c.cid()), 'size', c.width(), 'x', c.height())"), timeout=30))

ser.close()
