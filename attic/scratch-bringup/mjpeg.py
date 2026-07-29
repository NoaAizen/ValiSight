"""Parse the MJPEG stream by its Content-Length headers.

Scanning for FFD8/FFD9 is wrong: those byte pairs occur inside entropy-coded
data, and once the frames carried real contrast they occurred often enough to
shred the stream into ~5.7 KB fragments that still counted as 'frames'. That is
what produced the impossible 34-45 fps readings.
"""
import time
import urllib.request


def frames(url, seconds):
    r = urllib.request.urlopen(url, timeout=seconds + 25)
    buf, out, t0 = b"", [], time.time()
    try:
        while time.time() - t0 < seconds:
            hdr_end = buf.find(b"\r\n\r\n")
            if hdr_end < 0:
                c = r.read(8192)
                if not c:
                    break
                buf += c
                continue
            head = buf[:hdr_end].decode("latin1")
            n = None
            for line in head.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    n = int(line.split(":", 1)[1])
            if n is None:
                buf = buf[hdr_end + 4:]
                continue
            need = hdr_end + 4 + n
            if len(buf) < need:
                c = r.read(max(8192, need - len(buf)))
                if not c:
                    break
                buf += c
                continue
            out.append((time.time(), buf[hdr_end + 4:need]))
            buf = buf[need:]
    finally:
        r.close()
    return out
