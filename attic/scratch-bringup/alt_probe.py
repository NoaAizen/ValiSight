"""Work out why alternating thermal and RGB captures kills the RGB one.

A single thermal-then-RGB pair works. It only fails in the live loop, so the
question is which ingredient breaks it: the alternation itself, the ironbow
conversion, or missing a per-sensor reset. Each strategy runs several rounds,
because the failure did not show up on the first iteration.
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


s = serial.Serial(P, 115200, timeout=1)
time.sleep(0.5)
s.write(b"\x03")
time.sleep(0.2)
s.reset_input_buffer()
s.write(b"\r\n")
rp(s, 3)

for line in ("import csi, binascii",
             "r = csi.CSI(cid=csi.PAG7936); r.reset(); r.pixformat(csi.RGB565); r.framesize(csi.QQVGA)",
             "t = csi.CSI(cid=csi.LEPTON); t.reset(); t.pixformat(csi.GRAYSCALE); t.framesize(csi.QQVGA)"):
    run(s, line)

STRATS = [
    ("plain alternate",
     "for i in range(4):\\n"
     "  a=t.snapshot(); b=r.snapshot()\\n"
     " print('ok', a.width(), b.width())"),
    ("with ironbow",
     "for i in range(4):\\n"
     "  a=t.snapshot().to_ironbow(); b=r.snapshot()\\n"
     " print('ok', a.width(), b.width())"),
    ("reset before rgb",
     "for i in range(4):\\n"
     "  a=t.snapshot()\\n"
     "  r.pixformat(csi.RGB565); r.framesize(csi.QQVGA); b=r.snapshot()\\n"
     " print('ok', a.width(), b.width())"),
    ("ironbow + reset",
     "for i in range(4):\\n"
     "  a=t.snapshot().to_ironbow()\\n"
     "  r.reset(); r.pixformat(csi.RGB565); r.framesize(csi.QQVGA); b=r.snapshot()\\n"
     " print('ok', a.width(), b.width())"),
    ("rgb only x8",
     "for i in range(8):\\n"
     "  b=r.snapshot()\\n"
     " print('ok', b.width(), b.height())"),
]

for name, body in STRATS:
    print("%-18s %s" % (name, run(s, g(body), 90)))

s.close()
