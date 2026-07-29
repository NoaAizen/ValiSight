"""Does the Lepton band because it is being read faster than it produces?

A Lepton 3.5 delivers ~8.7 frames/s. Everything so far hammered snapshot() as
fast as the REPL allowed, which is a plausible cause of segment desync. Three
conditions, same sensor, same session order:

  A  long idle after init, then a single frame
  B  paced at ~8.7 Hz
  C  as fast as possible

Each frame is saved on the board and pulled back, so the images can be compared
directly rather than argued about.
"""
import base64, os, serial, sys, time

PORT, OUT = sys.argv[1], sys.argv[2]
PR = b">>> "


def rp(s, t=180):
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


def run(s, l, t=180):
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


def pull(s, p, d):
    ss = run(s, "import os; print(os.stat('%s')[6])" % p, 30)
    try:
        size = int(ss.strip().splitlines()[-1])
    except Exception:
        print("    stat failed for %s" % p)
        return
    data = b""
    for off in range(0, size, 512):
        r = run(s, "import binascii; f=open('%s','rb'); f.seek(%d); "
                   "print(binascii.b2a_base64(f.read(512)).decode().strip()); f.close()"
                   % (p, off), 30)
        ln = r.strip().splitlines()[-1] if r.strip() else ""
        try:
            data += base64.b64decode(ln)
        except Exception:
            print("    chunk failed")
            return
    open(d, "wb").write(data)
    print("    saved %s (%d bytes)" % (os.path.basename(d), len(data)))


s = serial.Serial(PORT, 115200, timeout=1)
time.sleep(0.5)
s.write(b"\x03")
time.sleep(0.2)
s.reset_input_buffer()
s.write(b"\r\n")
rp(s, 5)
os.makedirs(OUT, exist_ok=True)

# Clear any CSI state left by the live server.
s.write(b"\x03")
time.sleep(0.2)
s.write(b"\x04")
time.sleep(5.0)
s.reset_input_buffer()
s.write(b"\r\n")
rp(s, 8)

print("init + 25 s idle, then FFC")
print(" ", run(s, g(
    "import csi, time; c=csi.CSI(cid=csi.LEPTON); c.reset(); "
    "c.pixformat(csi.GRAYSCALE); c.framesize(csi.QQVGA); "
    "time.sleep(25); c.ioctl(csi.IOCTL_LEPTON_RUN_COMMAND, 0x0242); time.sleep(3); "
    "print('rested')"), 240))

print("A: single frame after rest")
print(" ", run(s, g(
    "im=c.snapshot(); im.save('/flash/pa.jpg', quality=95); print('A ok')"), 90))
pull(s, "/flash/pa.jpg", os.path.join(OUT, "pace_a_rested.jpg"))

print("B: paced at ~8.7 Hz for 6 s")
print(" ", run(s, g(
    "import time\\n"
    " for i in range(52):\\n"
    "  im=c.snapshot(); time.sleep_ms(115)\\n"
    " im.save('/flash/pb.jpg', quality=95); print('B ok')"), 240))
pull(s, "/flash/pb.jpg", os.path.join(OUT, "pace_b_paced.jpg"))

print("C: unpaced, as fast as possible")
print(" ", run(s, g(
    "for i in range(60):\\n"
    "  im=c.snapshot()\\n"
    " im.save('/flash/pc.jpg', quality=95); print('C ok')"), 240))
pull(s, "/flash/pc.jpg", os.path.join(OUT, "pace_c_fast.jpg"))

s.close()
