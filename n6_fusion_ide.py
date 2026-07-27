"""Radar + thermal FUSION view, to run INSIDE the OpenMV IDE (on the N6).

Shows the Lepton 3.5 thermal (black-hot) with radar clusters overlaid by their
azimuth. Open in the OpenMV IDE and press Run.

RADAR SOURCE (honest):
  * USE_SIM = True  (default): a clearly-labelled SIMULATED target sweeps across
    the view so you can SEE the fused layout with NO wiring. The marker is drawn
    with "SIM" on screen — it is NOT real radar data.
  * USE_SIM = False: reads a real IWR1843 over UART1. Requires the radar DATA
    UART wired to the N6:  radar DATA_UART TX -> N6 UART1 RX, shared GND (3.3V).
    Then it parses the TLV point cloud on-board and overlays the real returns.

Why UART and not USB: in the IDE the USB carries the frame buffer, so live radar
must come in on a separate UART — that is the wire above.
"""
import csi
import time
import math

USE_SIM = True                 # <-- set False after wiring the radar to UART1
RADAR_UART = 1
RADAR_BAUD = 921600
THERMAL_HFOV_DEG = 57.0
W = 160                        # Lepton width (QQVGA)
MIN_C, MAX_C = 15.0, 45.0

# --- thermal -----------------------------------------------------------------
lep = csi.CSI(cid=csi.LEPTON)
lep.reset(hard=False)
lep.pixformat(csi.GRAYSCALE)
lep.framesize(csi.QQVGA)
lep.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)
lep.ioctl(csi.IOCTL_LEPTON_SET_RANGE, MIN_C, MAX_C)

# --- radar (only when wired) -------------------------------------------------
MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
uart = None
_buf = b""
if not USE_SIM:
    from machine import UART
    uart = UART(RADAR_UART, RADAR_BAUD, bits=8, parity=None, stop=1, timeout=5)


def az_to_x(az_deg):
    f = (W / 2.0) / math.tan(math.radians(THERMAL_HFOV_DEG / 2.0))
    return int(W / 2.0 + f * math.tan(math.radians(az_deg)))


def parse_radar():
    """Return live radar clusters [{az, range_m}] from UART1 (best-effort)."""
    global _buf
    import ustruct as struct
    d = uart.read()
    if d:
        _buf += d
    if len(_buf) > 16384:
        _buf = _buf[-16384:]
    out = []
    while True:
        i = _buf.find(MAGIC)
        if i < 0 or len(_buf) - i < 40:
            break
        total = struct.unpack_from("<I", _buf, i + 12)[0]
        if total < 40 or total > 16384:
            _buf = _buf[i + 8:]
            continue
        if len(_buf) - i < total:
            break
        pkt = _buf[i:i + total]
        _buf = _buf[i + total:]
        n_obj = struct.unpack_from("<I", pkt, 8 + 20)[0]
        n_tlv = struct.unpack_from("<I", pkt, 8 + 24)[0]
        off = 40
        for _ in range(n_tlv):
            if off + 8 > len(pkt):
                break
            ttype, tlen = struct.unpack_from("<2I", pkt, off)
            off += 8
            if ttype == 1:
                for k in range(min(n_obj, tlen // 16)):
                    xr, yf, zu, v = struct.unpack_from("<4f", pkt, off + 16 * k)
                    x, y = yf, -xr                 # TI -> project (x fwd, y left)
                    rng = math.sqrt(x * x + y * y + zu * zu)
                    if 0.3 < rng < 9.0:
                        az = math.degrees(math.atan2(-y, x))
                        out.append({"az": az, "range_m": rng})
            off += tlen
    return out


def sim_radar(tsec):
    az = 30.0 * math.sin(tsec * 0.8)               # a target sweeping L<->R
    return [{"az": az, "range_m": 3.2}]


clock = time.clock()
t0 = time.ticks_ms()
while True:
    clock.tick()
    img = lep.snapshot()
    img.invert()                                   # black-hot: hot -> black
    tsec = time.ticks_diff(time.ticks_ms(), t0) / 1000.0

    if USE_SIM:
        clusters = sim_radar(tsec)
        tag = "RADAR: SIM (no wire)"
    else:
        clusters = parse_radar()
        tag = "RADAR: LIVE (UART%d)" % RADAR_UART

    for c in clusters:
        x = az_to_x(c["az"])
        img.draw_line(x, 22, x, 120, color=200)
        img.draw_circle(x, 40, 4, color=255)
        img.draw_string(min(max(x - 22, 0), W - 60), 26,
                        "%.1fm%s" % (c["range_m"], " SIM" if USE_SIM else ""),
                        color=255)
    img.draw_string(2, 2, tag, color=255)
    img.draw_string(2, 110, "THERMAL black-hot + radar az", color=255)
    print("fps=%.1f  clusters=%d" % (clock.fps(), len(clusters)))
