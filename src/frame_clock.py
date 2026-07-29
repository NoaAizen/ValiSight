"""Frame timestamp arithmetic. No hardware, no serial, no imports from drivers.

Acquisition is an adapter concern (live_server talks to the board). Everything
here is pure: give it raw tick values and it gives back a continuous time axis,
so it can be tested without a Lepton attached.

TWO CLOCKS, deliberately kept apart:

  mono_us   the N6's own time.ticks_us() latched right after snapshot() returns.
            Monotonic, unaffected by anyone setting a clock, and the only value
            timing maths may use. This is the sync master's axis.

  host_wall the Jetson's wall-clock at the moment the frame was received. For
            humans and log correlation ONLY. It includes USB transfer and REPL
            latency and can jump backwards when NTP steps the Jetson.

Never subtract one from the other and call the result a duration.

WRAPAROUND: MicroPython's ticks counter is modulo 2**30. At microsecond
resolution that wraps every 1073.7 s -- about 17.9 minutes -- which is well
inside a normal session, so raw tick values cannot be compared directly or
stored as-is. Unwrap() turns them into a continuous 64-bit axis.
"""

TICKS_PERIOD = 1 << 30          # verified on-board: time.ticks_add(0, -1) == 2**30 - 1
TICKS_HALF = TICKS_PERIOD // 2


def ticks_diff(a, b, period=TICKS_PERIOD):
    """Signed a - b on a wrapping counter, matching MicroPython's semantics."""
    half = period // 2
    return ((a - b + half) % period) - half


class Unwrapper:
    """Raw wrapping ticks -> a continuous, always-increasing axis.

    Rides on ticks_diff rather than on a bare `<` comparison, so it stays
    correct whether or not a wrap happened between two samples, and tolerates
    the occasional dropped frame. It only breaks if more than half a period
    passes with no sample at all, which would mean the source has stopped and
    the gap is the least of the problems.

    The period is a parameter because there are two wrapping counters on this
    rig with different widths: the N6's time.ticks_us() at 2**30 (17.9 min) and
    the radar's timeCpuCycles at 2**32 (21.5 s at its measured 200 MHz). The
    radar one wraps 50x faster, so it cannot be an afterthought.
    """

    def __init__(self, period=TICKS_PERIOD):
        self.period = period
        self.total = 0
        self.last = None

    def unwrap(self, ticks):
        ticks %= self.period
        if self.last is None:
            self.last, self.total = ticks, ticks
            return self.total
        self.total += ticks_diff(ticks, self.last, self.period)
        self.last = ticks
        return self.total

    def reset(self):
        self.total, self.last = 0, None


class LinearFit:
    """y = slope*x + offset by lower-envelope regression.

    Ordinary least squares is the wrong estimator for a timing relationship.
    Transport delay is one-sided -- a sample can only ever arrive LATE, never
    early -- so the noise is not symmetric about the line and OLS is pulled up
    by the tail. The true relationship lives along the lower envelope of the
    scatter, which is what an unloaded, uncontended transfer looks like.

    Fit by OLS first, then keep only the points in the lowest `keep` quantile of
    residual and refit on those. Two passes is enough; the first pass only has
    to be good enough to rank residuals.
    """

    def __init__(self, keep=0.25):
        self.keep = keep
        self.slope = 1.0
        self.offset = 0.0
        self.n = 0
        self.resid_sd = 0.0

    @staticmethod
    def _ols(xs, ys):
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0:
            return 0.0, my
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx
        return slope, my - slope * mx

    def fit(self, xs, ys):
        xs, ys = list(xs), list(ys)
        self.n = len(xs)
        if self.n < 2:
            return self
        slope, offset = self._ols(xs, ys)
        if self.n >= 8:
            resid = [y - (slope * x + offset) for x, y in zip(xs, ys)]
            cut = sorted(resid)[max(1, int(len(resid) * self.keep)) - 1]
            sel = [(x, y) for x, y, r in zip(xs, ys, resid) if r <= cut]
            if len(sel) >= 2:
                slope, offset = self._ols([p[0] for p in sel], [p[1] for p in sel])
        self.slope, self.offset = slope, offset
        r = [y - self.predict(x) for x, y in zip(xs, ys)]
        mr = sum(r) / len(r)
        self.resid_sd = (sum((v - mr) ** 2 for v in r) / len(r)) ** 0.5
        return self

    def predict(self, x):
        return self.slope * x + self.offset


class RateMeter:
    """Bytes and events per second over a sliding report window.

    Reports the link rate the board is actually sustaining, which is what the
    clock configuration has to be judged against -- a nominal baud number says
    nothing once REPL turnaround and base64 expansion are in the path.
    """

    def __init__(self, window=1.0):
        self.window = window
        self.bytes = self.events = 0
        self.t0 = None
        self.bps = self.eps = 0.0
        self.peak_bps = 0.0

    def add(self, nbytes, now, events=1):
        if self.t0 is None:
            # Anchor the window without counting this sample: its bytes arrived
            # over an interval that started before we were watching, and folding
            # them in makes the first report read high (n+1 samples' bytes over
            # n samples' worth of time). Every later window opens at the exact
            # instant the previous one closed, so nothing is lost after this.
            self.t0 = now
            return False
        self.bytes += nbytes
        self.events += events
        dt = now - self.t0
        if dt >= self.window:
            self.bps = self.bytes / dt
            self.eps = self.events / dt
            self.peak_bps = max(self.peak_bps, self.bps)
            self.bytes = self.events = 0
            self.t0 = now
            return True
        return False


def stamp(seq, mono_us, host_wall, nbytes, sensor, epoch=0):
    """One frame's timing record, in the shape the log and replay both use.

    EPOCH is not decoration. Switching cameras soft-resets MicroPython, which
    restarts the board's ticks counter at zero, so mono_us alone is NOT ordered
    across a session -- a log spanning one switch reads as going backwards in
    time. The ordering key is (epoch, mono_us); epoch increments on every camera
    init. Only compare mono_us values that share an epoch.
    """
    return {"seq": seq, "sensor": sensor, "epoch": epoch, "mono_us": mono_us,
            "host_wall": host_wall, "bytes": nbytes}
