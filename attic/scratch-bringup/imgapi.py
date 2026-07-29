"""Find how this OpenMV firmware exposes JPEG compression.

img.compressed() does not exist on firmware v5.0.0, but img.save(...) works, so
some compression path is there -- this lists the candidate method names and
tries the two most likely calls.
"""
import serial, sys, time

P = sys.argv[1]
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
    return o.strip().replace("\r\n", " | ")


def g(e):
    return 'exec("try:\\n %s\\nexcept Exception as e:\\n print(\'ERR\',e)")' % e


SETUP = ("import csi; c=csi.CSI(cid=csi.PAG7936); c.reset(); "
         "c.pixformat(csi.RGB565); c.framesize(csi.QQVGA); im=c.snapshot(); ")

s = serial.Serial(P, 115200, timeout=1)
time.sleep(0.5)
s.write(b"\x03")
time.sleep(0.2)
s.reset_input_buffer()
s.write(b"\r\n")
rp(s, 3)

print("candidates :", run(s, g(
    SETUP + "print([m for m in dir(im) if any(k in m for k in ('comp','jpeg','byte','buf','size','to_'))])")))

print("compress() :", run(s, g(
    SETUP + "j=im.compress(quality=50); print('ok', type(j), im.size())")))

print("to_jpeg()  :", run(s, g(
    SETUP + "j=im.to_jpeg(quality=50); print('ok', type(j), j.size())")))

print("bytearray():", run(s, g(
    SETUP + "im.compress(quality=50); b=im.bytearray(); print('ok len', len(b))")))

s.close()
