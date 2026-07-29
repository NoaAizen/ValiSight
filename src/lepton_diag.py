"""Headless Lepton 3.5 diagnostics, driven from the Jetson over the OpenMV REPL.

There is no OpenMV IDE here, so every frame is pulled back as raw pixel bytes and
analysed numerically on the host. That is better than eyeballing anyway: the
question "which rows are bad, and are they static?" is a measurement, not a look.

    python3 lepton_diag.py probe                  # firmware + every LEPTON ioctl
    python3 lepton_diag.py capture [--hard] [--frames N] [--ffc] [--tag NAME]
    python3 lepton_diag.py measure [--frames N] [--tag NAME]   # radiometric path

Frames are written to data/captures/diag/<tag>_fNN.bin plus a .json of row stats.

The OpenMV board is a MicroPython VCP; resolve it through /dev/serial/by-id
because /dev/ttyACM* numbering follows enumeration order.
"""
import argparse, base64, glob, json, os, sys, time

import serial

BY_ID = "/dev/serial/by-id"
PROMPT = b">>> "
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "captures", "diag")


def by_id(pattern="*MicroPython*if00"):
    for p in sorted(glob.glob(os.path.join(BY_ID, pattern))):
        return os.path.realpath(p)
    return None


def read_to_prompt(ser, timeout=30):
    buf, t0 = b"", time.time()
    while time.time() - t0 < timeout:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
            if buf.endswith(PROMPT):
                break
        else:
            time.sleep(0.005)
    return buf


def exchange(ser, line, timeout=30):
    ser.write(line.encode() + b"\r\n")
    raw = read_to_prompt(ser, timeout).decode(errors="replace")
    out = raw.split("\r\n", 1)[-1]
    for tail in (">>> ", "... "):
        while out.endswith(tail):
            out = out[: -len(tail)]
    return out.strip()


def repl(ser, line, timeout=30):
    """Interrupt first, then run. Use for setup, not for the hot path."""
    ser.write(b"\x03")
    time.sleep(0.05)
    ser.reset_input_buffer()
    return exchange(ser, line, timeout)


def block(lines):
    """Multi-line board-side code as one REPL line: exec('a\\nb\\nc')."""
    return "exec(%r)" % "\n".join(lines)


def soft_reset(ser):
    """Ctrl-D at the friendly prompt -- the only way to clear a CSI init."""
    ser.write(b"\x03")
    time.sleep(0.2)
    ser.write(b"\x04")
    time.sleep(4.0)
    ser.reset_input_buffer()
    ser.write(b"\r\n")
    read_to_prompt(ser, 8)


def connect():
    port = by_id()
    if not port:
        print("OpenMV not found under %s" % BY_ID)
        sys.exit(1)
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(0.4)
    print("openmv : %s" % port)
    soft_reset(ser)
    return ser


def last_line(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------- board setup

def init_lepton(ser, hard=False, ffc=False, measurement=False, settle=3.0):
    """Bring the Lepton up alone at its native 160x120, nothing else touched."""
    print("\n== init (hard=%s, measurement=%s) ==" % (hard, measurement))
    for line in ("import csi, binascii, time",
                 "c = csi.CSI(cid=csi.LEPTON)",
                 "c.reset(hard=%s)" % hard):
        out = repl(ser, line, timeout=45)
        if out:
            print("  %s -> %s" % (line, out))
    if measurement:
        # SET_MODE(measurement, high_temp). In measurement mode the driver turns
        # the sensor's own AGC OFF (LEP_SetAgcEnableState(!measurement)) and maps
        # the 14/16-bit TLinear values through a FIXED celsius range instead of a
        # per-frame stretch -- that is the closest thing to raw this API exposes.
        out = repl(ser, "c.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)", 40)
        print("  SET_MODE(measurement=True) -> %s" % (out or "ok"))
    for line in ("c.pixformat(csi.GRAYSCALE)",
                 "c.framesize(csi.QQVGA)"):
        out = repl(ser, line, timeout=30)
        if out:
            print("  %s -> %s" % (line, out))
    print("  geometry -> %s" % repl(ser, "print(c.width(), c.height())", 15))
    time.sleep(settle)
    if ffc:
        print("  FFC -> %s" % (repl(ser, "c.ioctl(csi.IOCTL_LEPTON_RUN_COMMAND, 0x0242)", 30) or "ok"))
        time.sleep(1.2)


def grab_bytes(ser, timeout=60):
    """One frame back as raw pixel bytes: base64 in a single round trip."""
    out = last_line(exchange(
        ser, "b = c.snapshot().bytearray(); "
             "print(binascii.b2a_base64(b).decode().strip())", timeout))
    if not out or out.startswith("Traceback") or len(out) < 64:
        return None, out
    try:
        return base64.b64decode(out), None
    except Exception as e:
        return None, "b64: %s (%s)" % (e, out[:80])


# ------------------------------------------------------------- host analysis

def row_stats(buf, w, h, bpp=1):
    """Per-row mean/min/max. bpp=2 reads little-endian 16-bit."""
    rows = []
    for y in range(h):
        if bpp == 1:
            vals = list(buf[y * w:(y + 1) * w])
        else:
            base = y * w * 2
            vals = [buf[base + 2 * x] | (buf[base + 2 * x + 1] << 8) for x in range(w)]
        rows.append({"y": y, "mean": round(sum(vals) / len(vals), 2),
                     "min": min(vals), "max": max(vals),
                     "const": len(set(vals)) == 1})
    return rows


def segment_of(y):
    return y // 30


def report(tag, frames, w, h, bpp=1):
    """Print the two things that decide this: which rows are bad, and do they move."""
    stats = [row_stats(f, w, h, bpp) for f in frames]
    ref = stats[0]
    means = [r["mean"] for r in ref]
    med = sorted(means)[len(means) // 2]

    print("\n== %s : row profile (frame 0) ==" % tag)
    print("  median row mean = %.1f" % med)
    print("  %-4s %-8s %-6s %-6s %-6s %s" % ("row", "mean", "min", "max", "seg", "flag"))
    hot = []
    for r in ref:
        dev = r["mean"] - med
        flag = ""
        if r["const"]:
            flag = "CONSTANT"
        elif abs(dev) > 12:
            flag = "BRIGHT" if dev > 0 else "DARK"
        if flag:
            hot.append(r["y"])
        if flag or r["y"] % 10 == 0:
            print("  %-4d %-8.1f %-6d %-6d %-6d %s"
                  % (r["y"], r["mean"], r["min"], r["max"], segment_of(r["y"]), flag))

    if hot:
        print("\n  anomalous rows : %s" % compact(hot))
        segs = sorted({segment_of(y) for y in hot})
        print("  segments hit   : %s  (segment N = rows %s)"
              % (segs, ", ".join("%d:%d-%d" % (s, s * 30, s * 30 + 29) for s in segs)))
        for s in segs:
            n = len([y for y in hot if segment_of(y) == s])
            print("    segment %d: %d/30 rows anomalous%s"
                  % (s, n, "  <-- WHOLE SEGMENT" if n >= 28 else ""))
    else:
        print("\n  no anomalous rows -- profile is flat")

    if len(frames) > 1:
        print("\n== %s : static or moving? (%d frames) ==" % (tag, len(frames)))
        base = frames[0]
        for i, f in enumerate(frames[1:], 1):
            if len(f) != len(base):
                print("  frame %d: SIZE MISMATCH %d vs %d" % (i, len(f), len(base)))
                continue
            diff = sum(1 for a, b in zip(base, f) if a != b)
            print("  frame %d vs 0 : %d/%d bytes differ (%.1f%%)"
                  % (i, diff, len(base), 100.0 * diff / len(base)))
        if hot:
            print("  anomalous-row means across frames:")
            for y in hot[:8]:
                print("    row %-4d %s" % (y, " ".join("%7.1f" % s[y]["mean"] for s in stats)))
    return hot, stats


def compact(ys):
    """[0,1,2,5] -> '0-2, 5'"""
    out, start, prev = [], None, None
    for y in ys + [None]:
        if start is None:
            start = prev = y
            continue
        if y is not None and y == prev + 1:
            prev = y
            continue
        out.append("%d" % start if start == prev else "%d-%d" % (start, prev))
        start = prev = y
    return ", ".join(out)


def save(tag, frames, stats):
    os.makedirs(OUTDIR, exist_ok=True)
    for i, f in enumerate(frames):
        with open(os.path.join(OUTDIR, "%s_f%02d.bin" % (tag, i)), "wb") as fh:
            fh.write(f)
    with open(os.path.join(OUTDIR, "%s_rows.json" % tag), "w") as fh:
        json.dump(stats, fh)
    print("\n  saved %d frame(s) to %s" % (len(frames), os.path.normpath(OUTDIR)))


def to_png(tag, buf, w, h, index=0):
    """Grayscale PNG so the bands can also be looked at, zlib only, no PIL."""
    import struct, zlib
    raw = b"".join(b"\x00" + buf[y * w:(y + 1) * w] for y in range(h))

    def chunk(kind, data):
        c = kind + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "%s_f%02d.png" % (tag, index))
    with open(path, "wb") as fh:
        fh.write(png)
    return path


# ------------------------------------------------------------------ commands

def cmd_probe(ser, args):
    print("\n== firmware ==")
    print(repl(ser, "import sys, csi; print(sys.implementation); print(sys.version)", 20))
    print("\n== every LEPTON symbol in csi (this firmware, not the docs) ==")
    print(repl(ser, "print(sorted(n for n in dir(csi) if 'LEPTON' in n))", 20))
    print("\n== CSI methods ==")
    print(repl(ser, "print(sorted(n for n in dir(csi.CSI) if not n.startswith('__')))", 20))
    print("\n== detected sensors ==")
    print(repl(ser, "print(csi.devices())", 20))


def cmd_capture(ser, args):
    init_lepton(ser, hard=args.hard, ffc=args.ffc, settle=args.settle)
    w, h = [int(v) for v in last_line(repl(ser, "print(c.width(), c.height())", 15)).split()]
    frames, errs = [], []
    for i in range(args.frames):
        buf, err = grab_bytes(ser)
        if buf is None:
            errs.append("frame %d: %s" % (i, err))
        else:
            frames.append(buf)
        time.sleep(args.interval)
    print("\n  got %d/%d frames, %d bytes each (expect %d for %dx%d 8-bit)"
          % (len(frames), args.frames, len(frames[0]) if frames else 0, w * h, w, h))
    for e in errs:
        print("  %s" % e)
    if not frames:
        return 1
    hot, stats = report(args.tag, frames, w, h, bpp=1)
    save(args.tag, frames, stats)
    print("  png: %s" % to_png(args.tag, frames[0], w, h))
    return 0


def cmd_measure(ser, args):
    """Radiometric path: is the badness in the sensor data or only in the AGC?"""
    init_lepton(ser, hard=args.hard, measurement=True, settle=args.settle)
    print("  SET_RANGE(%.1f, %.1f)        -> %s"
          % (args.min, args.max,
             repl(ser, "c.ioctl(csi.IOCTL_LEPTON_SET_RANGE, %f, %f)" % (args.min, args.max), 20) or "ok"))
    for label, expr in (("GET_MODE (measurement, high_temp)", "csi.IOCTL_LEPTON_GET_MODE"),
                        ("GET_RANGE (celsius)              ", "csi.IOCTL_LEPTON_GET_RANGE"),
                        ("GET_RADIOMETRY                   ", "csi.IOCTL_LEPTON_GET_RADIOMETRY"),
                        ("GET_RESOLUTION (bits)            ", "csi.IOCTL_LEPTON_GET_RESOLUTION"),
                        ("GET_REFRESH (hz)                 ", "csi.IOCTL_LEPTON_GET_REFRESH"),
                        ("GET_FPA_TEMP (celsius)           ", "csi.IOCTL_LEPTON_GET_FPA_TEMP"),
                        ("GET_AUX_TEMP (celsius)           ", "csi.IOCTL_LEPTON_GET_AUX_TEMP")):
        print("  %s -> %s" % (label, last_line(repl(ser, "print(c.ioctl(%s))" % expr, 20))))
    w, h = [int(v) for v in last_line(repl(ser, "print(c.width(), c.height())", 15)).split()]
    frames = []
    for _ in range(args.frames):
        buf, err = grab_bytes(ser)
        if buf is None:
            print("  grab failed: %s" % err)
        else:
            frames.append(buf)
        time.sleep(args.interval)
    if not frames:
        return 1
    bpp = 2 if len(frames[0]) >= w * h * 2 else 1
    print("\n  %d bytes/frame -> %d bpp" % (len(frames[0]), bpp))
    f0 = frames[0]
    print("  clipped low (0)   : %.1f%%   clipped high (255): %.1f%%"
          % (100.0 * f0.count(0) / len(f0), 100.0 * f0.count(255) / len(f0)))
    hot, stats = report(args.tag, frames, w, h, bpp=bpp)
    save(args.tag, frames, stats)
    print("  png: %s" % to_png(args.tag, frames[0], w, h))
    return 0


TELEMETRY_CIDS = {
    "SYS_TELEMETRY_ENABLE_STATE": (0x0218, 2),
    "SYS_TELEMETRY_LOCATION": (0x021C, 2),
    "SYS_FFC_SHUTTER_MODE_OBJ": (0x023C, 16),
    "SYS_FFC_STATUS": (0x0244, 2),
    "SYS_GAIN_MODE": (0x0248, 2),
    "SYS_SCENE_STATISTICS": (0x022C, 4),
    "OEM_VIDEO_OUTPUT_SOURCE": (0x482C, 2),
    "RAD_TLINEAR_ENABLE_STATE": (0x4EC0, 2),
    "RAD_SPOTMETER_ROI": (0x4ECC, 4),
    "VID_FOCUS_CALC_ENABLE": (0x030C, 2),
}


def cmd_telemetry(ser, args):
    """Read the sensor's own CCI registers -- telemetry, shutter mode, gain."""
    init_lepton(ser, hard=args.hard, settle=1.0)
    print("\n== CCI attributes (LEP_GetAttribute via IOCTL_LEPTON_GET_ATTRIBUTE) ==")
    for name, (cid, words) in TELEMETRY_CIDS.items():
        out = last_line(repl(ser, "print(bytes(c.ioctl(csi.IOCTL_LEPTON_GET_ATTRIBUTE, 0x%04X, %d)))"
                             % (cid, words), 25))
        val = ""
        if out.startswith("b'") or out.startswith('b"'):
            try:
                raw = eval(out)
                val = "  = %s" % [raw[i] | (raw[i + 1] << 8) for i in range(0, len(raw), 2)]
            except Exception:
                pass
        print("  %-28s 0x%04X -> %s%s" % (name, cid, out, val))
    print("\n  TELEMETRY_ENABLE_STATE: [0,...] = disabled, [1,...] = ENABLED")
    print("  frame geometry -> %s" % last_line(repl(
        ser, "print(c.ioctl(csi.IOCTL_LEPTON_GET_WIDTH), c.ioctl(csi.IOCTL_LEPTON_GET_HEIGHT))", 20)))
    return 0


CID_RAD_SPOTMETER_ROI = 0x4ECC
CID_RAD_SPOTMETER_OBJ = 0x4ED0     # value, max, min, population (Kelvin x100)


def spot_read(ser, r0, c0, r1, c1):
    """Point the sensor's OWN spotmeter at a band and read it back over CCI.

    This is the test that separates sensor from transfer: the spotmeter is
    computed inside the Lepton, before a single bit goes out over VoSPI. If a
    band reads anomalous here too, the SPI link cannot be what created it.
    """
    words = [r0, c0, r1, c1]
    payload = ",".join(str(v) for v in words)
    repl(ser, "c.ioctl(csi.IOCTL_LEPTON_SET_ATTRIBUTE, 0x%04X, "
              "bytes([%s]))" % (CID_RAD_SPOTMETER_ROI,
                                ",".join("%d,%d" % (w & 0xFF, w >> 8) for w in words)), 25)
    time.sleep(0.4)
    back = last_line(repl(ser, "print(bytes(c.ioctl(csi.IOCTL_LEPTON_GET_ATTRIBUTE, 0x%04X, 4)))"
                          % CID_RAD_SPOTMETER_ROI, 25))
    out = last_line(repl(ser, "print(bytes(c.ioctl(csi.IOCTL_LEPTON_GET_ATTRIBUTE, 0x%04X, 4)))"
                         % CID_RAD_SPOTMETER_OBJ, 25))
    try:
        raw = eval(out)
        vals = [raw[i] | (raw[i + 1] << 8) for i in range(0, len(raw), 2)]
    except Exception:
        vals = None
    return payload, back, vals


def cmd_spot(ser, args):
    init_lepton(ser, measurement=True, settle=2.0)
    repl(ser, "c.snapshot()", 30)
    print("\n== on-chip spotmeter: bad-row bands vs good-row bands ==")
    print("  (value/max/min in Kelvin x100; population = pixels in ROI)\n")
    bands = [("BAD  rows 55-59", 55, 10, 59, 150),
             ("BAD  rows 60-63", 60, 10, 63, 150),
             ("BAD  row  12    ", 12, 10, 12, 150),
             ("good rows 70-74", 70, 10, 74, 150),
             ("good rows 100-104", 100, 10, 104, 150),
             ("good rows 20-24", 20, 10, 24, 150)]
    for name, r0, c0, r1, c1 in bands:
        repl(ser, "c.snapshot()", 30)
        payload, back, vals = spot_read(ser, r0, c0, r1, c1)
        if vals:
            print("  %-18s roi=[%s] readback=%s  value %.2f K  max %.2f  min %.2f  pop %d"
                  % (name, payload, back, vals[0] / 100.0, vals[1] / 100.0,
                     vals[2] / 100.0, vals[3]))
        else:
            print("  %-18s roi=[%s] -> no reading (%s)" % (name, payload, back))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, tag):
        p.add_argument("--frames", type=int, default=4)
        p.add_argument("--interval", type=float, default=0.4)
        p.add_argument("--settle", type=float, default=3.0)
        p.add_argument("--hard", action="store_true")
        p.add_argument("--tag", default=tag)

    sub.add_parser("probe")
    cap = sub.add_parser("capture")
    common(cap, "capture")
    cap.add_argument("--ffc", action="store_true")
    mea = sub.add_parser("measure")
    common(mea, "measure")
    mea.add_argument("--min", type=float, default=15.0)
    mea.add_argument("--max", type=float, default=45.0)
    tel = sub.add_parser("telemetry")
    tel.add_argument("--hard", action="store_true")
    spot = sub.add_parser("spot")
    spot.add_argument("--hard", action="store_true")
    spot.add_argument("--settle", type=float, default=2.0)

    args = ap.parse_args()
    ser = connect()
    try:
        return {"probe": cmd_probe, "capture": cmd_capture, "measure": cmd_measure,
                "telemetry": cmd_telemetry, "spot": cmd_spot}[args.cmd](ser, args) or 0
    finally:
        ser.close()


if __name__ == "__main__":
    sys.exit(main())
