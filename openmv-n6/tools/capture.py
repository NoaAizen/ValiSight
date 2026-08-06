#!/usr/bin/env python3
"""Capture synchronized RGB + thermal frame pairs from an OpenMV N6.

Default mode buffers frames in board RAM, then transfers once capture is done.

The rule that actually matters here is in the bring-up, not the phasing: never
hard-reset one sensor while the other is streaming. The reset line is shared
across all CSI devices, and on firmware older than 2026-07-18 the PAG7936 locks
up if it is reset without being idled first (OpenMV PR #3208). A locked sensor
stalls the interpreter; USB on this board is serviced by TinyUSB from the
MicroPython scheduler, so the board then vanishes from the bus with no error.
Hence: bring the visible sensor up first, and give the Lepton reset(hard=False).

    ./capture.py -n 20 -o captures/calib     # 20 pairs
    ./capture.py -n 1 --preview              # write PNGs to eyeball
    ./capture.py -n 4 --stream               # stream while capturing
    ./capture.py --sd / --fetch-only         # stage on the SD card instead

Each pair yields <NNNN>_rgb.raw, <NNNN>_thermal.raw and <NNNN>.json holding the
Lepton measurement range needed to turn 8-bit codes back into degrees C.
"""
import argparse, json, os, sys, time

import serial

PORT = "/dev/ttyACM0"
PAG, LEP = 0x7936, 0x5435
BOARD_DIR = "/sdcard/cap"

# ---------------------------------------------------------------- board code

_BRINGUP = r'''
import csi, time, sys, gc, os

PAG, LEP = 0x7936, 0x5435
TMIN, TMAX = __TMIN__, __TMAX__

# Visible sensor FIRST. OMV_CSI_RESET_PIN (PE3) is shared board-wide, so
# rgb.reset() hard-resets the Lepton too - doing it after the Lepton has synced
# costs a full ~1.2s VoSPI resync on the next thermal frame.
rgb = csi.CSI(cid=PAG)
rgb.reset()
rgb.pixformat(csi.GRAYSCALE)
rgb.framesize(csi.VGA)
time.sleep_ms(300)
rgb.snapshot()

lep = csi.CSI(cid=LEP)
# hard=False is essential, not a nicety. reset() defaults to hard=True, and
# OMV_CSI_RESET_PIN is shared board-wide (omv_csi.c: "hard-reset is shared
# between all CSIs"), so a hard reset here lands on the PAG7936 while it is
# streaming MIPI. OpenMV PR #3208 documents that the PAG7936 locks up when reset
# without being idled first; that fix is dated 2026-07-18 and the firmware on
# this board is 2026-07-02, so the guard does not exist here. A locked sensor
# stalls the interpreter, which starves TinyUSB's tud_task(), and the board
# silently drops off USB. This single flag is what makes capture survive.
# OpenMV's own dual-sensor example does the same: "no hardware reset - just
# configure lepton".
lep.reset(hard=False)
lep.pixformat(csi.GRAYSCALE)
lep.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, True)
lep.ioctl(csi.IOCTL_LEPTON_SET_RANGE, TMIN, TMAX)
lep.framesize(csi.QQVGA)
time.sleep_ms(2000)
lep.snapshot()                      # absorb the ~1.7s VoSPI sync frame

# ---- torn-frame rejection -------------------------------------------------
# The Lepton 3.5 delivers 120 rows as four 30-row VoSPI segments. When a segment
# is dropped or mismatched the frame stitches together moments that do not belong
# together, and the seam lands exactly on a segment boundary. Detect it by
# comparing the row-to-row step at rows 29/59/89 against the typical step
# everywhere else - a real scene edge can sit anywhere, a tear only ever sits on
# a boundary. Costs ~4800 byte reads per frame, which is nothing at 8.7Hz.
SEG = 30


def _row_means(b, w, h, step=4):
    out = []
    for y in range(h):
        base = y * w
        s = 0
        for x in range(0, w, step):
            s += b[base + x]
        out.append(s / (w // step))
    return out


def is_torn(img):
    b = img.bytearray()
    w, h = img.width(), img.height()
    rm = _row_means(b, w, h)
    jumps = [abs(rm[i + 1] - rm[i]) for i in range(h - 1)]
    boundaries = [jumps[i - 1] for i in range(SEG, h, SEG) if i - 1 < len(jumps)]
    if not boundaries:
        return False
    ordered = sorted(jumps)
    typical = ordered[len(ordered) // 2]
    return max(boundaries) > max(6.0, 5.0 * typical)


def good_snapshot(csi_dev, tries=6):
    """A thermal frame with no segment seam, or the last one tried."""
    img = csi_dev.snapshot()
    for _ in range(tries):
        if not is_torn(img):
            return img, False
        img = csi_dev.snapshot()
    return img, True


if __AUTORANGE__:
    # Percentile clip off a real histogram. Two summary-statistic attempts failed
    # here: min/max let a few hot pixels stretch the range fivefold (40 of 255
    # codes used on a flat wall), and quartile whiskers clipped both ends of a
    # scene with a person in it. Percentiles do neither - the 1st/99.5th cover
    # the scene while letting a genuinely hot target saturate, which is the
    # behaviour you want on an inspection camera anyway.
    _b = lep.snapshot().bytearray()
    _h = [0] * 256
    for _i in range(0, len(_b), 3):
        _h[_b[_i]] += 1
    _n = sum(_h)

    def _pct(f):
        target = _n * f
        c = 0
        for _k in range(256):
            c += _h[_k]
            if c >= target:
                return _k
        return 255

    step = (TMAX - TMIN) / 255.0
    lo = TMIN + _pct(0.010) * step
    hi = TMIN + _pct(0.995) * step
    pad = max(1.5, (hi - lo) * 0.10)
    lo, hi = lo - pad, hi + pad

    # Floor the span at the sensor's own noise. Lepton 3.5 NETD is ~50mK, so a
    # range finer than 255 * 0.05 = 12.75C is not resolving temperature any more,
    # it is just displaying NETD noise at full contrast - which is what made a
    # flat wall look torn. Tightening past the noise floor buys nothing.
    NETD_C = 0.05
    span_min = 255.0 * NETD_C
    if hi - lo < span_min:
        mid = (hi + lo) / 2.0
        lo, hi = mid - span_min / 2.0, mid + span_min / 2.0
    TMIN, TMAX = int(lo), int(hi) + 1
    lep.ioctl(csi.IOCTL_LEPTON_SET_RANGE, TMIN, TMAX)
    time.sleep_ms(300)
    lep.snapshot()
'''

RECORD_CODE = _BRINGUP + r'''
try:
    os.mkdir("__DIR__")
except OSError:
    pass
for f in os.listdir("__DIR__"):
    os.remove("__DIR__/" + f)

meta = {"tmin": TMIN, "tmax": TMAX, "rgb_w": rgb.width(), "rgb_h": rgb.height(),
        "th_w": lep.width(), "th_h": lep.height(), "burst": __BURST__}
with open("__DIR__/meta.json", "w") as f:
    f.write(str(meta).replace("'", '"').replace("True", "true").replace("False", "false"))

for i in range(__NPAIRS__):
    t = lep.snapshot()               # blocks ~113ms -> paces the loop at the thermal rate
    with open("__DIR__/%04d_thermal.raw" % i, "wb") as f:
        f.write(t.bytearray())
    for b in range(__BURST__):
        with open("__DIR__/%04d_rgb%d.raw" % (i, b), "wb") as f:
            f.write(rgb.snapshot().bytearray())
    sys.stdout.write("#REC %d\n" % i)
    gc.collect()

# No shutdown(True): OMV_CSI_POWER_PIN is shared, so powering one sensor down
# cuts the other, and the second shutdown then talks I2C to an unpowered part.
sys.stdout.write("#RECDONE\n")
'''

RAM_CODE = _BRINGUP + r'''
RGB_W, RGB_H = rgb.width(), rgb.height()
TH_W, TH_H = lep.width(), lep.height()

# No IOCTL_LEPTON_GET_*_TEMP here. Those are blocking CCI/I2C transactions on the
# shared I2C3 bus, and issuing one while VoSPI is streaming can stall the
# interpreter. On this board USB is serviced by TinyUSB from the MicroPython
# scheduler (mp_usbd.c -> mp_usbd_task_callback -> tud_task_ext), so a long
# blocking call starves tud_task(), the host stops getting answers, and the
# device drops off the bus with no Python-level error at all.

frames = []
torn_total = 0
for i in range(__NPAIRS__):
    t, was_torn = good_snapshot(lep)
    torn_total += 1 if was_torn else 0
    # bytearray() is a view onto the framebuffer, which the next snapshot
    # overwrites - copy it out or every frame ends up identical
    frames.append(("%04d_thermal" % i, bytes(t.bytearray())))
    for b in range(__BURST__):
        frames.append(("%04d_rgb%d" % (i, b), bytes(rgb.snapshot().bytearray())))
    sys.stdout.write("#REC %d\n" % i)
    if __INTERVAL_MS__:
        time.sleep_ms(__INTERVAL_MS__)

sys.stdout.write("#TORN %d\n" % torn_total)

# No shutdown(True): OMV_CSI_POWER_PIN (PE1) is shared, so powering one sensor
# down cuts the other too, and the second shutdown then talks I2C to an
# unpowered part. Just stop pulling frames.
gc.collect()

sys.stdout.write("#META %s\n" % str({
    "tmin": TMIN, "tmax": TMAX, "rgb_w": RGB_W, "rgb_h": RGB_H,
    "th_w": TH_W, "th_h": TH_H, "burst": __BURST__,
}).replace("'", '"'))

out = sys.stdout.buffer
CHUNK = 4096
for name, data in frames:
    sys.stdout.write("#FILE %s.raw %d\n" % (name, len(data)))
    m = memoryview(data)
    off = 0
    while off < len(m):
        out.write(m[off:off + CHUNK])
        off += CHUNK
sys.stdout.write("#FETCHDONE\n")
'''

FETCH_CODE = r'''
import os, sys, gc

out = sys.stdout.buffer
CHUNK = 4096
names = sorted(os.listdir("__DIR__"))
sys.stdout.write("#LIST %d\n" % len(names))
buf = bytearray(CHUNK)
mv = memoryview(buf)
for n in names:
    p = "__DIR__/" + n
    sz = os.stat(p)[6]
    sys.stdout.write("#FILE %s %d\n" % (n, sz))
    with open(p, "rb") as f:
        while True:
            k = f.readinto(buf)
            if not k:
                break
            out.write(mv[:k])         # chunked: never park in one huge blocking write
    gc.collect()
sys.stdout.write("#FETCHDONE\n")
'''

STREAM_CODE = _BRINGUP + r'''
out = sys.stdout.buffer
CHUNK = 4096


def emit(name, img):
    sys.stdout.write("#FRAME %s %d %d %d\n" % (name, img.width(), img.height(), img.size()))
    m = memoryview(img.bytearray())
    off, n = 0, len(m)
    while off < n:
        out.write(m[off:off + CHUNK])
        off += CHUNK


sys.stdout.write("#META %s\n" % str({
    "tmin": TMIN, "tmax": TMAX, "rgb_w": rgb.width(), "rgb_h": rgb.height(),
    "burst": __BURST__,
}).replace("'", '"'))

for i in range(__NPAIRS__):
    t = lep.snapshot()
    sys.stdout.write("#PAIR %d %d\n" % (i, time.ticks_ms()))
    emit("thermal", t)
    for b in range(__BURST__):
        emit("rgb%d" % b, rgb.snapshot())
    gc.collect()
sys.stdout.write("#DONE\n")
'''

# ---------------------------------------------------------------- transport


class Board:
    """Raw-REPL transport that never consumes more of the stream than it returns."""

    def __init__(self, port=PORT):
        if not os.path.exists(port):
            raise SystemExit(
                "%s is not present - the board is not enumerated.\n"
                "Replug the USB cable (or tap RESET), wait for the port, then re-run." % port)
        self.s = serial.Serial(port, 115200, timeout=0.2, write_timeout=10)
        self.buf = bytearray()

    # -- low level

    def _fill(self, timeout):
        """Drain in small reads with a short yield, matching the transport that
        survives. pyserial's read(n) blocks for the full port timeout collecting
        up to n bytes, so asking for 8192 at a time drains in 0.2s steps. That is
        slow enough to back-pressure the board's CDC TX; a blocked out.write()
        starves TinyUSB's tud_task(), which is serviced from the MicroPython
        scheduler, and the board falls off the bus with no error."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            chunk = self.s.read(4096)
            if chunk:
                self.buf += chunk
                return True
            time.sleep(0.05)
        return False

    def _take(self, n, skip):
        out = bytes(self.buf[:n])
        del self.buf[:n + skip]
        return out

    def read_until(self, token, timeout=30):
        while True:
            i = self.buf.find(token)
            if i >= 0:
                return self._take(i, len(token))
            if not self._fill(timeout):
                raise TimeoutError("waiting for %r (have %r)" % (token, bytes(self.buf[:120])))

    def read_line(self, timeout=60):
        """A line, or the board's error marker with \\x04 prefixed so callers can spot it."""
        while True:
            nl, eot = self.buf.find(b"\n"), self.buf.find(b"\x04")
            if eot >= 0 and (nl < 0 or eot < nl):
                return b"\x04" + self._take(eot, 1)
            if nl >= 0:
                return self._take(nl, 1)
            if not self._fill(timeout):
                raise TimeoutError("waiting for a line (have %r)" % bytes(self.buf[:120]))

    def read_exact(self, n, timeout=60):
        while len(self.buf) < n:
            if not self._fill(timeout):
                raise TimeoutError("short read: %d of %d bytes" % (len(self.buf), n))
        return self._take(n, 0)

    def drain(self):
        out = bytes(self.buf) + (self.s.read_all() or b"")
        self.buf = bytearray()
        return out

    # -- high level

    def run(self, code, soft_reset=False):
        """Run `code` in the raw REPL, from a clean interpreter by default.

        soft_reset defaults OFF, and should stay that way. A ctrl-D reboot looks
        like it should give a cleaner start, but ports/stm32/main.c guards
        omv_csi_init() with `if (first_soft_reset)`, so afterwards the CSI
        subsystem is never re-probed or reconfigured and XCLK is never
        reprogrammed. It makes bring-up less deterministic, not more.
        """
        if soft_reset:
            self.s.write(b"\r\x03\x03")             # interrupt whatever is running
            time.sleep(0.2)
            self.s.write(b"\x02")                   # make sure we are in the friendly REPL
            time.sleep(0.2)
            self.s.reset_input_buffer()
            self.s.write(b"\x04")                   # soft reboot
            time.sleep(1.5)
            self.s.write(b"\r\x03\x03")             # and stop main.py if it auto-started
            time.sleep(0.5)
        else:
            self.s.write(b"\r\x03\x03")
            time.sleep(0.3)

        self.s.reset_input_buffer()
        self.buf = bytearray()
        self.s.write(b"\x01")                       # raw REPL
        time.sleep(0.3)
        self.s.read_all()
        self.s.write(code.encode() + b"\x04")
        self.read_until(b"OK", timeout=10)

    def close(self):
        """Interrupt and drain before releasing: closing on a board that is mid-write
        leaves it stuck in that write until it is physically replugged."""
        try:
            deadline = time.time() + 3
            while time.time() < deadline:
                self.s.write(b"\x03")
                drained = False
                for _ in range(64):
                    if not self.s.read(8192):
                        break
                    drained = True
                if not drained:
                    break
            self.s.write(b"\x02")
        except Exception as e:
            print("warning: unclean shutdown: %s" % e, file=sys.stderr)
        finally:
            self.s.close()


def subst(code, args, npairs=None):
    return (code
            .replace("__NPAIRS__", str(npairs if npairs is not None else args.pairs))
            .replace("__BURST__", str(args.burst))
            .replace("__TMIN__", str(args.tmin))
            .replace("__TMAX__", str(args.tmax))
            .replace("__AUTORANGE__", str(not args.no_autorange))
            .replace("__INTERVAL_MS__", str(int(args.interval * 1000)))
            .replace("__DIR__", BOARD_DIR))


def check_error(line, b):
    if line.startswith("\x04"):
        raise SystemExit("\nBOARD ERROR:\n" + line.lstrip("\x04") +
                         b.drain().decode("utf-8", "replace"))

# ---------------------------------------------------------------- phases


def record(b, args):
    print("recording %d pair(s) to %s (sensors up, no USB traffic)..." % (args.pairs, BOARD_DIR),
          file=sys.stderr)
    b.run(subst(RECORD_CODE, args))
    while True:
        line = b.read_line(timeout=90).decode("utf-8", "replace").strip()
        check_error(line, b)
        if line.startswith("#REC "):
            n = int(line.split()[1]) + 1
            print("\r  recorded %d/%d" % (n, args.pairs), end="", file=sys.stderr)
        elif line.startswith("#RECDONE"):
            print("\n  sensors released", file=sys.stderr)
            return


def fetch(b, args):
    print("transferring (sensors down)...", file=sys.stderr)
    b.run(FETCH_CODE.replace("__DIR__", BOARD_DIR))
    files, total = {}, 0
    while True:
        line = b.read_line(timeout=60).decode("utf-8", "replace").strip()
        check_error(line, b)
        if line.startswith("#LIST"):
            total = int(line.split()[1])
            print("  %d file(s) on card" % total, file=sys.stderr)
        elif line.startswith("#FILE"):
            _, name, size = line.split()
            files[name] = b.read_exact(int(size))
            print("  %-24s %8d B" % (name, int(size)), file=sys.stderr)
        elif line.startswith("#FETCHDONE"):
            return files


def ram(b, args):
    """Capture into board RAM, release the sensors, then transfer.

    Writing frames to the SD card while both sensors are live reliably drops this
    board off the USB bus; the same bring-up and dual capture without SD writes
    runs clean. 25MB of heap holds ~90 pairs, so the card is not needed at all.
    """
    print("capturing %d pair(s) to board RAM (sensors up, no USB, no SD)..." % args.pairs,
          file=sys.stderr)
    b.run(subst(RAM_CODE, args))

    meta, files = {}, {}
    while True:
        line = b.read_line(timeout=120).decode("utf-8", "replace").strip()
        check_error(line, b)
        if line.startswith("#REC "):
            print("\r  captured %d/%d" % (int(line.split()[1]) + 1, args.pairs),
                  end="", file=sys.stderr)
        elif line.startswith("#TORN"):
            n = int(line.split()[1])
            if n:
                print("\n  warning: %d frame(s) still torn after retries" % n, file=sys.stderr)
        elif line.startswith("#META"):
            meta = json.loads(line[5:].strip())
            print("\n  sensors released, transferring...", file=sys.stderr)
        elif line.startswith("#FILE"):
            _, name, size = line.split()
            files[name] = b.read_exact(int(size))
            print("  %-24s %8d B" % (name, int(size)), file=sys.stderr)
        elif line.startswith("#FETCHDONE"):
            return meta, unpack_fetched(files, meta)


def stream(b, args):
    print("streaming %d pair(s) (single phase, higher peak load)..." % args.pairs, file=sys.stderr)
    b.run(subst(STREAM_CODE, args))
    meta, idx, planes, pairs = {}, None, {}, []
    while True:
        line = b.read_line().decode("utf-8", "replace").strip()
        check_error(line, b)
        if line.startswith("#META"):
            meta = json.loads(line[5:].strip())
        elif line.startswith("#PAIR"):
            if planes:
                pairs.append((idx, planes))
            idx, planes = int(line.split()[1]), {}
        elif line.startswith("#FRAME"):
            _, name, w, h, n = line.split()
            planes[name] = (int(w), int(h), b.read_exact(int(n)))
            print("  pair %s  %-8s %sx%s" % (idx, name, w, h), file=sys.stderr)
        elif line.startswith("#DONE"):
            if planes:
                pairs.append((idx, planes))
            return meta, pairs

# ---------------------------------------------------------------- output


def write_pairs(outdir, meta, pairs, do_preview):
    for idx, planes in pairs:
        for name, (w, h, data) in planes.items():
            with open(os.path.join(outdir, "%04d_%s.raw" % (idx, name)), "wb") as f:
                f.write(data)
        side = dict(meta)
        side["planes"] = {n: {"w": w, "h": h} for n, (w, h, _) in planes.items()}
        side["c_per_lsb"] = (meta["tmax"] - meta["tmin"]) / 255.0
        with open(os.path.join(outdir, "%04d.json" % idx), "w") as f:
            json.dump(side, f, indent=2)
        if do_preview:
            preview(outdir, idx, planes)


def unpack_fetched(files, meta):
    """Group the flat file listing off the card back into per-index pairs."""
    pairs = {}
    for name, data in files.items():
        if not name.endswith(".raw"):
            continue
        stem = name[:-4]
        idx, plane = int(stem[:4]), stem[5:]
        w, h = ((meta["th_w"], meta["th_h"]) if plane == "thermal"
                else (meta["rgb_w"], meta["rgb_h"]))
        pairs.setdefault(idx, {})[plane] = (w, h, data)
    return sorted(pairs.items())


def preview(outdir, idx, planes):
    from PIL import Image
    import numpy as np

    for name, (w, h, data) in planes.items():
        a = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
        if name == "thermal":
            x = np.linspace(0, 1, 256)
            ramp = np.stack([np.clip(x * 3.0 - k, 0, 1) * 255 for k in (0, 1, 2)], 1).astype(np.uint8)
            Image.fromarray(ramp[a]).resize((w * 4, h * 4), Image.NEAREST).save(
                os.path.join(outdir, "%04d_thermal.png" % idx))
        else:
            Image.fromarray(a).save(os.path.join(outdir, "%04d_%s.png" % (idx, name)))

# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--pairs", type=int, default=1)
    ap.add_argument("-b", "--burst", type=int, default=1,
                    help="RGB frames per thermal frame (for averaging studies)")
    ap.add_argument("-o", "--outdir", default="captures/session")
    ap.add_argument("-p", "--port", default=PORT)
    ap.add_argument("--tmin", type=int, default=-10)
    ap.add_argument("--tmax", type=int, default=140)
    ap.add_argument("--no-autorange", action="store_true")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="seconds to pause between pairs, so a person can reposition "
                         "something between shots (the loop otherwise runs at 8.7Hz)")
    ap.add_argument("--preview", action="store_true", help="also write PNGs")
    ap.add_argument("--sd", action="store_true",
                    help="stage frames on the SD card (known to destabilise this board)")
    ap.add_argument("--fetch-only", action="store_true", help="SD mode: skip recording")
    ap.add_argument("--stream", action="store_true", help="single-phase capture (higher peak load)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    b, t0 = Board(args.port), time.time()
    try:
        if args.stream:
            meta, pairs = stream(b, args)
        elif args.sd or args.fetch_only:
            if not args.fetch_only:
                record(b, args)
            files = fetch(b, args)
            if "meta.json" not in files:
                raise SystemExit("no meta.json on the card - nothing recorded")
            meta = json.loads(files["meta.json"].decode())
            pairs = unpack_fetched(files, meta)
        else:
            meta, pairs = ram(b, args)
        write_pairs(args.outdir, meta, pairs, args.preview)
        print("\n%d pair(s) in %.1fs -> %s\n  range %s..%sC  %.3f C/LSB" % (
            len(pairs), time.time() - t0, args.outdir, meta["tmin"], meta["tmax"],
            (meta["tmax"] - meta["tmin"]) / 255.0), file=sys.stderr)
    finally:
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
