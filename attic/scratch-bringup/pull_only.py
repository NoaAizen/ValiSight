"""Pull the three paced-capture JPEGs, printing what the REPL actually returns
when a chunk will not decode.
"""
import base64, os, serial, sys, time

PORT, OUT = sys.argv[1], sys.argv[2]
PR = b">>> "


def rp(s, t=30):
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


def run(s, l, t=30):
    s.write(b"\x03")
    time.sleep(0.15)
    s.reset_input_buffer()
    s.write(l.encode() + b"\r\n")
    r = rp(s, t).decode(errors="replace")
    o = r.split("\r\n", 1)[-1]
    for x in (">>> ", "... "):
        while o.endswith(x):
            o = o[: -len(x)]
    return o.strip()


s = serial.Serial(PORT, 115200, timeout=1)
time.sleep(0.5)
s.write(b"\x03")
time.sleep(0.2)
s.reset_input_buffer()
s.write(b"\r\n")
rp(s, 5)

for name, path in (("pace_a_rested", "/flash/pa.jpg"),
                   ("pace_b_paced", "/flash/pb.jpg"),
                   ("pace_c_fast", "/flash/pc.jpg")):
    ss = run(s, "import os; print(os.stat('%s')[6])" % path)
    try:
        size = int(ss.strip().splitlines()[-1])
    except Exception:
        print("%-14s stat -> %r" % (name, ss[:120]))
        continue
    data, bad = b"", False
    for off in range(0, size, 512):
        r = run(s, "import binascii; f=open('%s','rb'); f.seek(%d); "
                   "print(binascii.b2a_base64(f.read(512)).decode().strip()); f.close()"
                   % (path, off))
        lines = [l for l in r.splitlines() if l.strip()]
        ln = lines[-1] if lines else ""
        try:
            data += base64.b64decode(ln)
        except Exception as e:
            print("%-14s chunk %d failed: %s" % (name, off, e))
            print("   raw tail: %r" % r[-200:])
            bad = True
            break
    if not bad:
        dest = os.path.join(OUT, name + ".jpg")
        open(dest, "wb").write(data)
        print("%-14s %d bytes -> %s" % (name, len(data), os.path.basename(dest)))

s.close()
