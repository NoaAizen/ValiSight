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
# SET_MODE(a, b) is (measurement_mode, high_temp_mode) - the SECOND argument is
# not radiometry. Radiometry comes from the first (LEP_SetRadEnableState); the
# second selects LEP_SYS_GAIN_MODE_LOW (lepton.c:307). Passing True there ran the
# part in LOW gain, which is what every frame recorded before 2026-08-09 was shot
# in. Confirmed 2026-08-06 by reading GAIN_MODE (CID 0x0248) back: 1 = LOW.
#
# HIGH gain from here on. Low gain bought a 600C ceiling this project never uses
# (0.5-2m electrical inspection) and gave up the tighter accuracy spec for it. It
# costs no noise either way - NETD measured identical in both modes - so the
# ceiling was the only thing being traded, and it was not worth having.
#
# The pairing with SET_RANGE below is the part to keep in view, and it is safe at
# the default: SET_RANGE clamps to the ceiling of whichever gain mode is active,
# and the clamp was measured on the board 2026-08-09 at 140C high / 200C+ low.
# The default range is -10..140, exactly the high-gain ceiling, so it survives the
# switch untouched. Ask for more than 140 here and you will silently get 140.
lep.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)
# Must come AFTER SET_MODE, and does not touch I2C at all - it only writes two
# host-side floats. It also clamps silently, to a limit that depends on the gain
# mode just set (measured: 140C in high gain, 600C in low), and silently reorders
# an inverted pair. Requesting 143 in high gain returns 140 with no error.
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
STEP = 4

# This runs on every thermal frame forever, so it must allocate NOTHING. Measured
# 2026-08-09: the original built three Python lists per call and cost 3728 B/frame
# of the 3888 B/frame the whole stream loop leaked - 96% of it. At 8.8fps that
# exhausts the ~25MB heap in about eleven minutes, and what ends the run is not
# the exhaustion itself but the emergency gc.collect() it triggers: a collect on
# this board costs 1.06s whatever the garbage (it scales with heap size, and the
# heap is external SDRAM), the Lepton delivers every 114ms, and its queue backs up
# and raises "Frame buffer overflow". A 5892-frame run died exactly there.
#
# So: fixed buffers filled in place, and integer sums rather than float means.
# Storing a small int into a list is allocation-free in MicroPython; storing a
# float boxes it, and 120 boxed floats per frame was most of the leak. Working in
# sums just scales both sides of the test by the sample count, so the thresholds
# below carry that factor and the verdict is unchanged.
_ROWSUM = None
_JUMPS = None
_SORTBUF = None


def _alloc_tear_buffers(h):
    global _ROWSUM, _JUMPS, _SORTBUF
    _ROWSUM = [0] * h
    _JUMPS = [0] * (h - 1)
    _SORTBUF = [0] * (h - 1)


def is_torn(img):
    b = img.bytearray()
    w = img.width()
    h = img.height()
    ns = w // STEP

    # while, not `for x in range(...)`: a range object is allocated per loop, and
    # at one per row that was 120 of them a frame - measured at 1920 B/frame, the
    # entire remainder of the leak once the lists were gone. The outer loops below
    # build one range each and are left alone.
    rs = _ROWSUM
    y = 0
    while y < h:
        base = y * w
        end = base + w
        s = 0
        x = base
        while x < end:
            s += b[x]
            x += STEP
        rs[y] = s
        y += 1

    jm = _JUMPS
    for i in range(h - 1):
        d = rs[i + 1] - rs[i]
        jm[i] = d if d >= 0 else -d

    # The seam only ever lands on a segment boundary; a real scene edge can sit
    # anywhere. Worst boundary step, without building a list to hold them.
    worst = -1
    i = SEG
    while i < h:
        k = i - 1
        if k < h - 1 and jm[k] > worst:
            worst = jm[k]
        i += SEG
    if worst < 0:
        return False

    sb = _SORTBUF
    for i in range(h - 1):
        sb[i] = jm[i]
    sb.sort()                       # in place: no copy, unlike sorted()
    typical = sb[(h - 1) // 2]

    # 5x the typical step, floored - the floor is the old 6.0 threshold on means,
    # carried into sums by the ns factor that is deliberately never divided out.
    thr = 5 * typical
    floor = 6 * ns
    if thr < floor:
        thr = floor
    return worst > thr


# THE INVARIANT THIS LOOP MUST HOLD: never call gc.collect() while the Lepton is
# up. Not "keep up with the frame rate" - the collect itself is the hazard.
#
# Measured directly 2026-08-09, because the obvious theory was wrong and cost an
# afternoon. Plain delays between snapshots are harmless: 150, 300, 600, 1000,
# 1500 and 2500ms gaps via time.sleep_ms() all returned a frame afterwards. The
# sensor does not care that nobody collected its output. But a gc.collect() of
# 1.066s wedges it immediately, every time, and the collect costs that 1.06s
# whatever the garbage - it scales with heap size, and the heap is 25MB of
# external SDRAM. So it is the collect, not the pause.
#
# The wedge is PERMANENT within the process. Against a wedged sensor these were
# all tried and all raised:
#   - 60 retries over 7.2s of waiting
#   - re-arming with framesize(csi.QQVGA)
#   - a full soft re-init: reset(hard=False) + pixformat + SET_MODE + SET_RANGE
#     + framesize + 2s VoSPI settle
# The PAG7936 keeps delivering 640x400 throughout, so this is the Lepton's buffer
# specifically and not the board running out of memory. Only tearing the whole
# bring-up down and running it again brings it back - the host's job, not this
# loop's, so there is deliberately no retry here. A retry costs a second and
# cannot work.
#
# That makes prevention the entire defence, and prevention means the automatic
# collector must never fire. Hence is_torn() above allocating nothing: it was
# 3728 B/frame of the loop's 3888, which filled the heap in about eleven minutes
# and ended a 5892-frame run with exactly this fault. The loop now leaks ~208
# B/frame, which is roughly 3.7 hours - better, still bounded, so a session
# meant to outlive that needs the host to restart the bring-up on purpose rather
# than be surprised by it.
#
# FFC is NOT this failure and must not be confused with it: the sensor stops
# delivering for 1824ms every ~183s, but snapshot() blocks, so the consumer is
# still parked in the driver. A 5892-frame run spans 3.7 FFC intervals and did
# not die at the first one.


def good_snapshot(csi_dev, tries=6):
    """A thermal frame with no segment seam, or the last one tried."""
    img = csi_dev.snapshot()
    for _ in range(tries):
        if not is_torn(img):
            return img, False
        img = csi_dev.snapshot()
    return img, True


_alloc_tear_buffers(lep.height())


if __AUTORANGE__:
    # Percentile clip off a real histogram. Two summary-statistic attempts failed
    # here: min/max let a few hot pixels stretch the range fivefold (40 of 255
    # codes used on a flat wall), and quartile whiskers clipped both ends of a
    # scene with a person in it. Percentiles do neither - the 1st/99.5th cover
    # the scene while letting a genuinely hot target saturate, which is the
    # behaviour you want on an inspection camera anyway.
    _b = lep.snapshot().bytearray()
    _W, _H = lep.width(), lep.height()

    # Condemn the lifted rows BEFORE building the histogram, or they decide it.
    # This part intermittently kills 14 rows and holds them for the session, at
    # a fixed high offset - so they are 14/120 = 11.67% of the pixels sitting at
    # the TOP of the distribution, and any percentile above the 88.33rd must
    # land inside them. That is arithmetic: no choice of clip point avoids it,
    # and the comment above about outliers does not apply to a population this
    # size. Measured on the recorded frames, _pct(0.995) read 242 with them in
    # and 106 with them out - the range came out more than twice as wide as the
    # scene needed, costing ~4x of the resolution this whole narrow-range
    # exercise exists to buy. Flat AND lifted, the same test repair_rows uses in
    # fusion.c: a threshold on brightness alone misses them before they clip.
    _h = [0] * 256
    _rlo, _rhi, _rsum = [255] * _H, [0] * _H, [0] * _H
    for _y in range(_H):
        _o = _y * _W
        _lo, _hi, _s = 255, 0, 0
        for _x in range(_W):
            _v = _b[_o + _x]
            _h[_v] += 1
            _s += _v
            if _v < _lo:
                _lo = _v
            if _v > _hi:
                _hi = _v
        _rlo[_y], _rhi[_y], _rsum[_y] = _lo, _hi, _s

    _n = _W * _H
    _c, _median = 0, 0
    for _k in range(256):
        _c += _h[_k]
        if _c * 2 >= _n:
            _median = _k
            break

    _dead = 0
    for _y in range(_H):
        if _rhi[_y] - _rlo[_y] <= 24 and (_rsum[_y] // _W) - _median >= 64:
            _o = _y * _W
            for _x in range(_W):
                _h[_b[_o + _x]] -= 1       # out of the histogram, not repaired
            _dead += 1
    _n -= _dead * _W
    sys.stdout.write("#DEADROWS %d\n" % _dead)

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

    # Floor the span at the sensor's own noise. Measured on this part 2026-08-06
    # over 250-frame runs: NETD is 33mK, not the ~50mK the datasheet implies, and
    # it is the same in both gain modes (33.0 high / 34.5 low; three runs landed
    # in 32-35mK). So the floor is 255 * 0.033 = 8.4C - below that the range is
    # no longer resolving temperature, it is displaying NETD noise at full
    # contrast, which is what made a flat wall look torn.
    #
    # This is a noise floor, not an accuracy floor. Over the same runs the
    # common-mode-removed temporal spread was 126-148mK, so two readings seconds
    # apart agree only to ~0.15C. A finer range is not a finer measurement.
    NETD_C = 0.033
    span_min = 255.0 * NETD_C
    if hi - lo < span_min:
        mid = (hi + lo) / 2.0
        lo, hi = mid - span_min / 2.0, mid + span_min / 2.0
    # floor, not truncate: int() rounds toward zero, so a negative lo would move
    # UP and narrow the range on the cold side - the one direction that clips
    # real scene content rather than padding it.
    TMIN, TMAX = int(lo // 1), int(hi // 1) + 1
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
    # No gc.collect() here, deliberately. It wedges the Lepton - see the invariant
    # above good_snapshot(). This loop is bounded by __NPAIRS__ rather than running
    # forever, so it does not need one; RAM_CODE below collects after its loop, and
    # that is the pattern to copy.

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
    # No gc.collect() here either - it wedges the Lepton. See good_snapshot().
sys.stdout.write("#DONE\n")
gc.collect()
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
