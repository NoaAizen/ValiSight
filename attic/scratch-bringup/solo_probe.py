"""Confirm each camera streams fine on its own after a clean soft reset.

Ctrl-D at the friendly REPL is MicroPython's soft reset -- it clears the
interpreter state, which is the only way to undo a CSI init. Each sensor then
gets a fresh session to itself.
"""
import serial, sys, time

P = sys.argv[1]
PR = b">>> "


def rp(s, t=40):
    b, t0 = b"", time.time()
    while time.time() - t0 < t:
        n = s.in_waiting
        if n:
            b += s.read(n)
            if b.endswith(PR):
                break
        else:
            time.sleep(0.02)
    return b


def run(s, l, t=40):
    s.write(b"\x03")
    time.sleep(0.15)
    s.reset_input_buffer()
    s.write(l.encode() + b"\r\n")
    r = rp(s, t).decode(errors="replace")
    o = r.split("\r\n", 1)[-1]
    for x in (">>> ", "... "):
        while o.endswith(x):
            o = o[: -len(x)]
    return o.strip().replace("\r\n", " | ")


def g(e):
    return 'exec("try:\\n %s\\nexcept Exception as e:\\n print(\'ERR\',e)")' % e


def soft_reset(s):
    s.write(b"\x03")
    time.sleep(0.2)
    s.write(b"\x04")          # soft reset from the friendly prompt
    time.sleep(4.0)
    s.reset_input_buffer()
    s.write(b"\r\n")
    rp(s, 6)


s = serial.Serial(P, 115200, timeout=1)
time.sleep(0.5)
s.write(b"\x03")
time.sleep(0.2)
s.reset_input_buffer()
s.write(b"\r\n")
rp(s, 3)

print("== RGB alone, fresh session ==")
soft_reset(s)
print(" ", run(s, g(
    "import csi; r=csi.CSI(cid=csi.PAG7936); r.reset(); r.pixformat(csi.RGB565); "
    "r.framesize(csi.QQVGA)\\n"
    " for i in range(10):\\n"
    "  b=r.snapshot()\\n"
    " print('ok 10 frames', b.width(), 'x', b.height())"), 90))

print("== thermal alone, fresh session ==")
soft_reset(s)
print(" ", run(s, g(
    "import csi; t=csi.CSI(cid=csi.LEPTON); t.reset(); t.pixformat(csi.GRAYSCALE); "
    "t.framesize(csi.QQVGA)\\n"
    " for i in range(10):\\n"
    "  b=t.snapshot()\\n"
    " print('ok 10 frames', b.width(), 'x', b.height())"), 90))

print("== back to RGB after soft reset ==")
soft_reset(s)
print(" ", run(s, g(
    "import csi; r=csi.CSI(cid=csi.PAG7936); r.reset(); r.pixformat(csi.RGB565); "
    "r.framesize(csi.QQVGA)\\n"
    " for i in range(10):\\n"
    "  b=r.snapshot()\\n"
    " print('ok 10 frames', b.width(), 'x', b.height())"), 90))

s.close()
