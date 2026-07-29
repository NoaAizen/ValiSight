"""Query an idle OpenMV board over its USB REPL.

Friendly REPL only, one physical line per command. Multi-line blocks put the
REPL into continuation mode ("..."), which never returns the ">>> " we wait on,
so anything conditional is packed into exec("...\\n...") on a single line.
Ctrl-C is used to clear state -- Ctrl-D at this prompt is a soft reset.
"""
import serial, sys, time

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
PROMPT = b">>> "


def read_to_prompt(ser, timeout=6):
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


def run(ser, line, timeout=8):
    ser.write(b"\x03")                 # clear any continuation state
    time.sleep(0.15)
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

print("platform :", run(ser, "import sys; print(sys.platform, sys.implementation.version)"))
print("files    :", run(ser, "import os; print(os.listdir('/'))"))

avail = []
for m in ("sensor", "csi", "image", "fir", "ml", "tv", "display", "audio", "imu"):
    r = run(ser, 'exec("try:\\n import %s\\n print(\'YES\')\\nexcept Exception as e:\\n print(\'NO\',e)")' % m)
    print("  %-8s -> %s" % (m, r.replace("\r\n", " | ")))
    if r.startswith("YES"):
        avail.append(m)

if "fir" in avail:
    print("fir attrs:", run(ser, "import fir; print([d for d in dir(fir) if not d.startswith('_')])"))
if "csi" in avail:
    print("csi attrs:", run(ser, "import csi; print([d for d in dir(csi) if not d.startswith('_')])"))

ser.close()
