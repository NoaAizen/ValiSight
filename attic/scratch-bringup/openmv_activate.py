"""Bring up the OpenMV N6's RGB and thermal sensors and report what each one is.

Same friendly-REPL, one-line-per-command discipline as openmv_probe.py.
"""
import serial, sys, time

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
PROMPT = b">>> "


def read_to_prompt(ser, timeout=15):
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


def run(ser, line, timeout=15):
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
    """Wrap an expression so a failure prints the reason instead of a traceback."""
    return 'exec("try:\\n %s\\nexcept Exception as e:\\n print(\'ERR\',e)")' % expr


ser = serial.Serial(PORT, 115200, timeout=1)
time.sleep(0.5)
ser.write(b"\x03")
time.sleep(0.2)
ser.reset_input_buffer()
ser.write(b"\r\n")
read_to_prompt(ser, 3)

print("== stored code ==")
for d in ("/flash", "/sdcard"):
    print("  %-8s %s" % (d, run(ser, guarded("import os; print(os.listdir('%s'))" % d))))

print()
print("== RGB / CSI ==")
print("  devices :", run(ser, guarded("import csi; print(csi.devices)")))
print("  bringup :", run(ser, guarded("import csi; c=csi.CSI(); print('id', c.id(), 'name', c.name())"), ))
print("  frame   :", run(ser, guarded(
    "import csi; c=csi.CSI(); c.reset(); c.pixformat(csi.RGB565); c.framesize(csi.QVGA); "
    "img=c.snapshot(); print('rgb frame', img.width(), 'x', img.height(), 'bpp', img.bpp())"), timeout=25))

print()
print("== thermal / FIR ==")
print("  init    :", run(ser, guarded("import fir; fir.init(); print('type', fir.type(), 'size', fir.width(), 'x', fir.height())"), timeout=25))
print("  read    :", run(ser, guarded(
    "import fir; ta, ir, tmin, tmax = fir.read_ir(); "
    "print('ambient %.2fC  min %.2fC  max %.2fC  pixels %d' % (ta, tmin, tmax, len(ir)))"), timeout=25))

ser.close()
