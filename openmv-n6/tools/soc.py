#!/usr/bin/env python3
"""Host SoC telemetry, read straight out of sysfs.

    import soc; s = soc.Soc(); s.read()

Why a thermal camera's viewer watches the machine it runs on: host load is the
one thing on this side that can reach the Lepton. The board writes a frame in
4KB chunks and out.write() gives up after 500ms of no progress, discarding the
tail; while it sits in that call it is not calling snapshot(), and the part is
only measured safe up to a 2500ms gap (live.py's LEPTON_SAFE_GAP_MS). So a host
that stops draining the port for long enough does not merely drop frames - it
leaves the sensor unserviced past anything anyone has tested.

live.py already reports that damage after the fact, from the board's own clock:
the "thermal load" check counts write stalls and starve events. What it cannot
say is *why* the host stopped draining, and by the time the starve counter moves
the answer is already in the past. These are the numbers that move first - a
core saturated, a thermal throttle, memory about to run out.

Everything here is a sysfs read of a few dozen bytes. Measured on this Orin, a
full read() is well under a millisecond, which is why it can be taken on the
same 1 Hz poll as everything else rather than needing a thread. Deliberately not
tegrastats: that is a subprocess per sample, it prints a line that has to be
parsed by position, and its format has changed between L4T releases.

Nothing here is Jetson-only by construction. Every field is None when its source
is absent, and CPU, memory and load work on any Linux host - so a laptop running
the viewer against a recording gets a smaller panel rather than a broken one.
"""
import glob
import os
import threading
import time

TICKS = os.sysconf("SC_CLK_TCK")

# A zone reading below this is the driver saying "no reading", not a cold SoC.
# The Tegra zones return -256000 (-256 C) while a sensor is unavailable.
MIN_SANE_MC = -40000


def _read(path, cast=int):
    """A sysfs read that cannot take the viewer down with it.

    The broad except is not laziness. Measured on this Orin 2026-08-16, the
    cv0/cv1/cv2 thermal zones belong to power domains that are gated off, and
    reading them does not raise OSError - the failing read reaches the io codec
    as a None and comes back as `TypeError: can't concat NoneType to bytes`.
    Three of this board's nine zones do that, every time. Enumerating which
    exception each unavailable sysfs node happens to produce is not a contract
    anyone maintains, and none of these readings is worth a traceback.
    """
    try:
        with open(path) as fp:
            return cast(fp.read().strip())
    except Exception:
        return None


class Soc:
    """One sampler. Holds the previous counters, since CPU use is a rate.

    Not thread-safe by accident: /ui and /health can both land on this from
    different handler threads, and two readers a millisecond apart would each
    take a delta over a near-zero interval and report nonsense percentages. The
    lock plus MIN_INTERVAL makes the second caller share the first one's sample.
    """

    # Below this the deltas are dominated by the granularity of the jiffy
    # counter: at 100 Hz, a 0.1s window is a single tick and the percentage
    # lands on a multiple of 16.7% with six cores.
    MIN_INTERVAL = 0.4

    def __init__(self):
        self.lock = threading.Lock()
        self.ncpu = os.cpu_count() or 1
        self._prev = None
        self._last = None

        # --- discovered once. These paths are fixed for the life of a boot, and
        # globbing them every second would be the most expensive thing here.
        self.zones = []
        for t in sorted(glob.glob("/sys/class/thermal/thermal_zone*/type")):
            name = _read(t, str)
            if name:
                self.zones.append((name.replace("-thermal", ""), t[:-4] + "temp"))

        # The temperature at which the SoC starts protecting itself. Worth having
        # because an absolute number is meaningless without it: 84 C is alarming
        # on a part that trips at 90 and unremarkable on one that trips at 105.
        crit = []
        for _, temp_path in self.zones:
            d = os.path.dirname(temp_path)
            for tp in glob.glob(os.path.join(d, "trip_point_*_type")):
                if _read(tp, str) in ("critical", "hot"):
                    v = _read(tp.replace("_type", "_temp"))
                    if v and v > 0:
                        crit.append(v / 1000.0)
        self.t_crit = min(crit) if crit else None

        # INA3221 on the carrier. Channel labels rather than positions: the rail
        # order differs across Jetson carriers, and reading channel 1 blind gives
        # you a module rail on one board and the total on another.
        self.rails = []
        for lbl in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"):
            name = _read(lbl, str)
            n = os.path.basename(lbl)[2:-6]        # in3_label -> 3
            volt = lbl.replace("_label", "_input")
            curr = os.path.join(os.path.dirname(lbl), "curr%s_input" % n)
            if name and os.path.exists(curr):
                self.rails.append((name, volt, curr))
        self.rails.sort()

        self.gpu_load = next((p for p in ("/sys/devices/platform/gpu.0/load",
                                          "/sys/devices/gpu.0/load")
                              if os.path.exists(p)), None)
        self.freq_cur = sorted(glob.glob(
            "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"))
        self.freq_max = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")

        # Prime the counters here, so the first read() already has something to
        # take a delta against. Without this the first sample reports no CPU use
        # at all, which is not "idle" - it is "unknown", and it lands on screen
        # in the seconds when somebody is watching to see whether the host can
        # keep up with the board at all.
        self._prev = (time.monotonic(),) + self._cpu_counters()

    # ------------------------------------------------------------------ sampling

    def _cpu_counters(self):
        """(busy, total) jiffies across all cores, and this process's own.

        iowait is counted as idle, not as busy. The distinction matters here more
        than usual: a recording session writes an mp4 while it streams, and a
        host blocked on the SD card is not a host that is too slow to drain the
        port - it is a host that is not being asked to do anything.
        """
        with open("/proc/stat") as fp:
            f = [float(x) for x in fp.readline().split()[1:]]
        total = sum(f)
        idle = f[3] + (f[4] if len(f) > 4 else 0.0)
        with open("/proc/self/stat") as fp:
            s = fp.read().rsplit(") ", 1)[1].split()      # comm can contain spaces
        return total - idle, total, float(s[11]) + float(s[12])

    def read(self):
        now = time.monotonic()
        with self.lock:
            if self._last is not None and now - self._last[0] < self.MIN_INTERVAL:
                return self._last[1]
            out = self._sample(now)
            self._last = (now, out)
            return out

    def _sample(self, now):
        busy, total, mine = self._cpu_counters()
        cpu_pct = self_pct = None
        if self._prev is not None:
            db, dt, dm = (busy - self._prev[1], total - self._prev[2],
                          mine - self._prev[3])
            if dt > 0:
                cpu_pct = round(100.0 * db / dt, 1)
                # Reported against the whole machine, the same denominator as
                # cpu_pct, so "the host is at 70% and 55 of it is me" is a
                # comparison you can make by eye. top's per-core number would
                # not be.
                self_pct = round(100.0 * (dm / TICKS) / ((now - self._prev[0]) * self.ncpu), 1)
        self._prev = (now, busy, total, mine)

        mem = {}
        with open("/proc/meminfo") as fp:
            for line in fp:
                k, v = line.split(":", 1)
                if k in ("MemTotal", "MemAvailable"):
                    mem[k] = int(v.split()[0]) / 1024.0        # MB
                    if len(mem) == 2:
                        break

        temps = {}
        for name, path in self.zones:
            v = _read(path)
            if v is not None and v > MIN_SANE_MC:
                temps[name] = round(v / 1000.0, 1)

        power = {}
        for name, volt, curr in self.rails:
            mv, ma = _read(volt), _read(curr)
            if mv is not None and ma is not None:
                power[name] = round(mv * ma / 1e6, 2)          # W

        gpu = _read(self.gpu_load) if self.gpu_load else None
        freq = [f for f in (_read(p) for p in self.freq_cur) if f]

        return {
            "cpu_pct": cpu_pct, "self_pct": self_pct, "ncpu": self.ncpu,
            "load1": os.getloadavg()[0],
            "mem_avail_mb": round(mem.get("MemAvailable", 0)),
            "mem_total_mb": round(mem.get("MemTotal", 0)),
            "gpu_pct": None if gpu is None else round(gpu / 10.0, 1),
            "temps": temps,
            # The hottest zone, which is the one that decides whether the part
            # throttles. Quoting a per-zone number invites picking the coolest.
            "t_max": max(temps.values()) if temps else None,
            "t_crit": self.t_crit,
            "power": power,
            # VDD_IN is the module total on every Jetson carrier that populates
            # it; the other rails are informative but do not add up to it.
            "power_w": next((v for k, v in power.items() if k.upper() == "VDD_IN"), None),
            "freq_mhz": round(max(freq) / 1000.0) if freq else None,
            "freq_max_mhz": round(self.freq_max / 1000.0) if self.freq_max else None,
        }


def checks(s):
    """Health rows for a reading, in live.py's {name, level, text} shape.

    None of these is allowed to be a 'fail' on CPU or heat alone, and that is a
    deliberate line rather than an oversight. A saturated host is a *risk* to the
    stream; whether it actually reached the sensor is a question the board's own
    clock answers, and live.py's "thermal load" check owns that verdict. Two
    checks failing on one event teaches the operator to discount both.

    Memory is the exception. An OOM kill during a recording is not a risk, it is
    a session that ends without the mp4 being finalised.
    """
    out = []
    if not s:
        return out

    if s["cpu_pct"] is not None:
        mine = "" if s["self_pct"] is None else " (%.0f%% of the machine is this process)" % s["self_pct"]
        if s["cpu_pct"] > 85:
            out.append(("host cpu", "warn",
                        "%.0f%% of %d cores busy%s - the reader thread has to be "
                        "scheduled within 500ms or the board discards a frame tail"
                        % (s["cpu_pct"], s["ncpu"], mine)))
        else:
            out.append(("host cpu", "ok", "%.0f%% of %d cores%s, load %.2f"
                        % (s["cpu_pct"], s["ncpu"], mine, s["load1"])))

    avail = s["mem_avail_mb"]
    if avail:
        if avail < 250:
            out.append(("host memory", "fail", "%d MB available - an OOM kill here "
                        "ends the session with an unfinalised mp4" % avail))
        elif avail < 600:
            out.append(("host memory", "warn", "%d MB available" % avail))
        else:
            out.append(("host memory", "ok", "%d MB of %d MB free"
                        % (avail, s["mem_total_mb"])))

    if s["t_max"] is not None:
        # Headroom, not the raw number: the temperature only matters relative to
        # where the part starts slowing itself down, and a throttled host is a
        # host that stops draining the port.
        if s["t_crit"]:
            head = s["t_crit"] - s["t_max"]
            if head < 8:
                # Still only a warning at 1 C from the trip, which looks too mild
                # until you hold it against what the levels mean here: a hot SoC
                # does not make the numbers on screen wrong, it makes them likely
                # to stop. The wrongness, if it arrives, arrives as write stalls
                # and a starved sensor, and those checks say so in their own
                # words rather than borrowing this one's.
                out.append(("host thermal", "warn",
                            "%.0f C, %.0f C below the %.0f C trip%s - a throttled "
                            "host stops draining the port on time"
                            % (s["t_max"], head, s["t_crit"],
                               " and throttling now" if head < 2 else "")))
            else:
                out.append(("host thermal", "ok", "%.0f C, %.0f C below the %.0f C trip"
                            % (s["t_max"], head, s["t_crit"])))
        else:
            out.append(("host thermal", "ok", "%.0f C" % s["t_max"]))

    if s["power_w"]:
        # No level of its own. Power is not a failure, it is the context that
        # explains one: a rig that browns out under load fails everywhere else
        # first, and this is the number that says why.
        out.append(("host power", "ok", "%.1f W total (%s)" % (
            s["power_w"], ", ".join("%s %.1f W" % (k.replace("VDD_", "").lower(), v)
                                    for k, v in sorted(s["power"].items())
                                    if k.upper() != "VDD_IN"))))

    return [{"name": n, "level": l, "text": t} for n, l, t in out]


if __name__ == "__main__":
    s = Soc()
    s.read()
    time.sleep(1)
    r = s.read()
    for k, v in r.items():
        print("%-14s %s" % (k, v))
    print()
    for c in checks(r):
        print("  %-14s %-5s %s" % (c["name"], c["level"], c["text"]))
