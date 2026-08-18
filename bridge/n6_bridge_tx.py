"""N6 -> Jetson bridge, SENDER side. Runs ON the OpenMV N6 (MicroPython).

Replaces the base64 line protocol of n6_dual_stream.py with binary records
(bridge_protocol.py) over the USB VCP: thermal frames, RGB JPEG frames, IMU
samples and a 1 Hz HELLO heartbeat, every one with a running seq number and a
timestamp on the N6 clock, so the Jetson can detect loss and pair streams.

Layout: `Bridge` is hardware-free (takes a write() and a clock) so it can be
unit-tested on the PC (test_n6_bridge_tx.py). `main()` binds it to the real
sensors and is what the launcher (n6_bridge_start.py) executes on the board.
Camera init/recovery discipline is copied from n6_dual_stream.py, which was
measured live — see the comments there before changing reset order.
"""
import bridge_protocol as bp

RGB_QUALITY = 40
REINIT_AFTER = 3          # consecutive dead grabs before re-initialising a sensor
HELLO_EVERY_TICKS = 1000000   # ~1 s in ticks_us units
MEM_PROBE = [0, 0]        # [mem_alloc after warm-up, iterations] — read from the REPL for diagnostics
IMU_HZ = 200              # timer rate, as measured stable in src/live_server.py
IMU_RING = 1024           # ~5 s at 200 Hz: rides out a multi-second USB host stall (3.9 s measured 2026-08-17)


class Bridge:
    """Builds and writes records; counts everything it could not deliver."""

    def __init__(self, write, ticks, ts_src=bp.TS_TICKS_US, hello_every=HELLO_EVERY_TICKS):
        self.write = write            # bytes -> number of bytes actually written
        self.ticks = ticks            # () -> current N6 clock value
        self.ts_src = ts_src
        self.hello_every = hello_every
        self.seq = 0
        self.sent = 0
        self.drops = 0                # records the USB link refused (host not draining)
        self.imu_overflow = 0         # IMU samples the ring could not hold (informational)
        self.last_hello = None

    def _diff(self, a, b):
        """a - b on the wrapping clock (same semantics as time.ticks_diff)."""
        p = bp.ts_period(self.ts_src)
        d = (a - b) & (p - 1)
        return d - p if d >= p // 2 else d

    def _send(self, rtype, ts, payload, prefix=b""):
        """Write one record as header(+prefix) then payload — the payload is
        NOT copied. On the N6 every copy of a 19 KB frame is garbage, and a
        MicroPython gc.collect() on this board takes ~1 s (25 MB heap), during
        which the Lepton FIFO overflows and the sensor must be re-initialised
        (measured 2026-08-17: 45 s cadence, ~6 s outage each). So: no copies."""
        hdr = bp.pack_header(rtype, self.ts_src, self.seq, ts, prefix, payload)
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        n = self.write(hdr + prefix) if prefix else self.write(hdr)
        n += self.write(payload)
        if n != len(hdr) + len(prefix) + len(payload):
            self.drops += 1
            return False
        self.sent += 1
        return True

    def hello(self, now=None):
        now = self.ticks() if now is None else now
        self.last_hello = now
        return self._send(bp.T_HELLO, now, bp.hello_payload(self.drops, self.imu_overflow))

    def maybe_hello(self, now=None):
        now = self.ticks() if now is None else now
        if self.last_hello is None or self._diff(now, self.last_hello) >= self.hello_every:
            return self.hello(now)
        return None

    def thermal(self, pixels, ts, w=160, h=120, flags=0):
        return self._send(bp.T_THERMAL, ts, pixels, bp.image_prefix(w, h, bp.PIX_GRAY8, flags))

    def rgb_jpeg(self, jpeg, ts, w=320, h=240):
        return self._send(bp.T_RGB, ts, jpeg, bp.image_prefix(w, h, bp.PIX_JPEG))

    def imu(self, accel_mg, gyro_mdps, ts):
        """One IMU sample, already in milli-g / milli-deg-per-second (the units
        imu.acceleration_mg() / imu.angular_rate_mdps() return on the N6)."""
        ax, ay, az = accel_mg
        gx, gy, gz = gyro_mdps
        return self._send(bp.T_IMU, ts, bp.imu_payload(ax, ay, az, gx, gy, gz))

    def imu_ring(self, ring):
        """Drain an ImuRing: one IMU record per buffered sample, oldest first,
        all packed into the ring's preallocated out-buffer and written in ONE
        call (no per-sample allocation). Returns how many were sent."""
        n = ring.pack_records(self.ts_src, self.seq)
        if n == 0:
            return 0
        self.seq = (self.seq + n) & 0xFFFFFFFF
        total = n * ring.REC
        if self.write(ring.outmv[:total]) != total:
            self.drops += n
            return 0
        self.sent += n
        return n


class ImuRing:
    """Fixed-size buffer filled from a timer callback, drained from the main loop.

    Recipe measured live on this board in src/live_server.py (200 Hz): the
    callback must be tiny and allocation-free, so it writes into preallocated
    arrays; the main loop copies them out between camera frames. Samples that
    arrive while the ring is full are COUNTED (overflow), never silently lost.
    """

    REC = bp.HDR_LEN + 24       # one packed IMU record

    def __init__(self, capacity, sample, ticks):
        import array
        self.n = capacity
        self.ts = array.array('i', [0] * capacity)
        self.v = array.array('i', [0] * (capacity * 6))
        self.out = bytearray(capacity * self.REC)     # packed records, reused forever
        self.outmv = memoryview(self.out)
        self.ix = 0
        self.overflow = 0
        self.busy = False
        self.sample = sample        # () -> (accel_mg(3), gyro_mdps(3))
        self.ticks = ticks

    def tick(self, _t=None):
        """Timer callback: no allocation beyond the sample tuples themselves."""
        if self.busy:
            return
        i = self.ix
        if i >= self.n:
            self.overflow += 1
            return
        a, g = self.sample()
        self.ts[i] = self.ticks()
        j = 6 * i
        self.v[j] = int(a[0]); self.v[j + 1] = int(a[1]); self.v[j + 2] = int(a[2])
        self.v[j + 3] = int(g[0]); self.v[j + 4] = int(g[1]); self.v[j + 5] = int(g[2])
        self.ix = i + 1

    def _compact(self, n):
        """Drop samples [0, n) and slide the ones that arrived meanwhile to the
        front. This is the ONLY critical section (a few samples at most), so the
        timer keeps sampling during the long pack/drain loops. Measured live
        2026-08-18: holding `busy` across the whole pack loop cost ~30 ms per
        camera frame = 6 samples lost per frame (148 Hz instead of 200)."""
        self.busy = True
        k = self.ix - n
        if k > 0:
            self.ts[0:k] = self.ts[n:n + k]
            self.v[0:6 * k] = self.v[6 * n:6 * (n + k)]
        self.ix = k
        self.busy = False

    def pack_records(self, ts_src, seq0):
        """Pack every buffered sample as a full record into self.out (in place).
        Returns the number of records. Ticks that arrive while packing are kept
        (they land at index >= n and are slid down by _compact)."""
        n = self.ix
        for i in range(n):
            j = 6 * i
            bp.pack_imu_into(self.outmv, i * self.REC, ts_src, (seq0 + i) & 0xFFFFFFFF, self.ts[i] & 0xFFFFFFFF,
                             self.v[j], self.v[j + 1], self.v[j + 2], self.v[j + 3], self.v[j + 4], self.v[j + 5])
        self._compact(n)
        return n

    def drain(self):
        n = self.ix
        out = []
        for i in range(n):
            j = 6 * i
            out.append((self.ts[i] & 0xFFFFFFFF,
                        (self.v[j], self.v[j + 1], self.v[j + 2]),
                        (self.v[j + 3], self.v[j + 4], self.v[j + 5])))
        self._compact(n)
        return out


# ------------------------------------------------------------ hardware side
def main():
    import csi
    import time

    MIN_C, MAX_C = 15.0, 45.0

    def rgb_init(hard=True):
        cam = csi.CSI(cid=csi.PAG7936)
        cam.reset(hard=hard)
        cam.pixformat(csi.RGB565)
        cam.framesize(csi.QVGA)
        return cam

    def lepton_init():
        cam = csi.CSI(cid=csi.LEPTON)
        cam.reset(hard=False)
        cam.pixformat(csi.GRAYSCALE)
        cam.framesize(csi.QQVGA)
        cam.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)
        cam.ioctl(csi.IOCTL_LEPTON_SET_RANGE, MIN_C, MAX_C)
        return cam

    LEP_SYS_FFC_STATUS = 0x0244        # Lepton SDK: LEP_CID_SYS_FFC_STATUS -> LEP_SYS_STATUS_E (1 = BUSY = FFC running)

    def ffc_flags(cam):
        """bit0 = FFC in progress, bit1 = status known. One I2C read, ~3.9 ms
        (measured 2026-08-18; after a forced FFC the flag stays up ~1.35 s)."""
        try:
            st = cam.ioctl(csi.IOCTL_LEPTON_GET_ATTRIBUTE, LEP_SYS_FFC_STATUS, 2)
            return bp.IMG_FLAG_FFC_KNOWN | (bp.IMG_FLAG_FFC if st[0] == 1 else 0)
        except Exception:
            return 0

    def grab(cam):
        for _ in range(10):
            try:
                img = cam.snapshot()
            except RuntimeError:
                time.sleep_ms(5)
                continue
            if img is not None:
                return img, time.ticks_us()
            time.sleep_ms(10)
        return None, None

    # OpenMV v5 / MicroPython 1.28 on the N6 has no pyb.USB_VCP; the binary-safe
    # path is sys.stdout.buffer (verified live 2026-08-17: bytes pass unchanged).
    import sys
    out = sys.stdout.buffer

    def write(b):
        return out.write(b) or 0

    ring = None
    try:
        import imu as imu_mod
        import machine
        imu_mod.acceleration_mg()                     # the real N6 API (mg / mdps)
        ring = ImuRing(IMU_RING, lambda: (imu_mod.acceleration_mg(), imu_mod.angular_rate_mdps()),
                       time.ticks_us)
        machine.Timer(-1).init(freq=IMU_HZ, callback=ring.tick)
    except Exception:
        ring = None                                   # no IMU on this build: stream cameras only

    rgb = rgb_init(hard=True)     # colour first: hard reset raises the module rail
    lep = lepton_init()
    time.sleep_ms(5000)           # Lepton settle + first FFC

    import gc
    gc.collect()
    MEM_PROBE[0] = gc.mem_alloc(); MEM_PROBE[1] = 0   # allocated after warm-up, iterations
    br = Bridge(write, time.ticks_us)
    br.hello()
    lep_miss = rgb_miss = 0
    while True:
        t, t_ts = grab(lep)
        if t is None:
            lep_miss += 1
            if lep_miss >= REINIT_AFTER:
                lep = lepton_init()
                time.sleep_ms(500)
                lep_miss = 0
        else:
            lep_miss = 0
            br.thermal(t.bytearray(), t_ts, t.width(), t.height(), ffc_flags(lep))

        if ring is not None:
            br.imu_ring(ring)                         # ~22 samples per thermal frame at 200 Hz

        r, r_ts = grab(rgb)
        if r is None:
            rgb_miss += 1
            if rgb_miss >= REINIT_AFTER:
                rgb = rgb_init(hard=False)   # soft: a hard reset would take the Lepton down too
                time.sleep_ms(200)
                rgb_miss = 0
        else:
            rgb_miss = 0
            w, h = r.width(), r.height()
            br.rgb_jpeg(r.compress(quality=RGB_QUALITY).bytearray(), r_ts, w, h)   # compress LAST: it mutates the image

        if ring is not None:
            br.imu_ring(ring)                         # drain again after the RGB grab
        br.imu_overflow = ring.overflow if ring is not None else 0
        br.maybe_hello()
        MEM_PROBE[1] += 1


if __name__ == "__main__":
    main()
