"""
IWR1843BOOST UART frame parser (TI mmWave SDK 3.x demo output).

Works on CPython (PC, pyserial) and MicroPython (OpenMV N6) — pure struct,
no numpy.

The demo firmware streams binary frames on the DATA/AUX UART (921600 baud):
  magic word (8B) -> header (32B) -> TLVs.
TLV type 1 = detected points: numObj * (x, y, z, doppler) float32.

TI coordinate frame: x = right, y = forward, z = up.
This parser converts to the project convention used by radar_classify_n6:
  x = forward, y = left, z = up   (so azimuth right = atan2(-y, x) > 0).

Usage (PC):
    import serial
    from iwr1843_uart import RadarReader
    port = serial.Serial("COM5", 921600, timeout=0.05)
    rr = RadarReader()
    while True:
        for fr in rr.feed(port.read(4096)):
            print(fr["frame"], fr["points"])
"""
import struct

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HDR_LEN = 40                      # magic(8) + header(32)
TLV_DETECTED_POINTS = 1
TLV_SIDE_INFO = 7                 # snr/noise per point (int16 each, 0.1 dB)


class RadarReader:
    """Incremental parser: feed() raw bytes, get back complete frames."""

    def __init__(self, max_buffer=65536):
        self._buf = b""
        self._max = max_buffer

    def feed(self, data):
        """data: bytes (or None). Returns list of frame dicts:
        {'frame': int, 'n_obj': int, 'points': [(x_fwd, y_left, z_up, v), ...],
         'snr': [snr_db, ...] or None, 'noise': [noise_db, ...] or None}

        snr is relative to the local noise floor (CFAR output); noise is the
        floor estimate itself. snr + noise = absolute received signal level,
        the quantity that actually follows the radar range equation — use it
        (not snr alone) for reflectivity/material estimation.
        """
        if data:
            self._buf += data
        if len(self._buf) > self._max:            # runaway guard
            self._buf = self._buf[-self._max:]

        frames = []
        while True:
            idx = self._buf.find(MAGIC)
            if idx < 0:
                # keep a magic-sized tail in case the magic word is split
                self._buf = self._buf[-(len(MAGIC) - 1):]
                break
            if idx:
                self._buf = self._buf[idx:]
            if len(self._buf) < HDR_LEN:
                break
            (version, total_len, platform, frame_no, cpu_cycles,
             n_obj, n_tlv, subframe) = struct.unpack_from("<8I", self._buf, 8)
            if total_len < HDR_LEN or total_len > self._max:
                self._buf = self._buf[len(MAGIC):]   # corrupt header, resync
                continue
            if len(self._buf) < total_len:
                break                                # wait for more bytes
            packet = self._buf[:total_len]
            self._buf = self._buf[total_len:]
            fr = self._parse_tlvs(packet, n_obj, n_tlv, frame_no)
            if fr is not None:
                frames.append(fr)
        return frames

    def _parse_tlvs(self, pkt, n_obj, n_tlv, frame_no):
        points, snr, noise = [], None, None
        off = HDR_LEN
        for _ in range(n_tlv):
            if off + 8 > len(pkt):
                return None
            tlv_type, tlv_len = struct.unpack_from("<2I", pkt, off)
            off += 8
            if off + tlv_len > len(pkt):
                return None
            if tlv_type == TLV_DETECTED_POINTS:
                n = min(n_obj, tlv_len // 16)
                for i in range(n):
                    xr, yf, zu, v = struct.unpack_from("<4f", pkt, off + 16 * i)
                    # TI (x right, y fwd) -> project (x fwd, y left)
                    points.append((yf, -xr, zu, v))
            elif tlv_type == TLV_SIDE_INFO:
                n = min(n_obj, tlv_len // 4)
                snr, noise = [], []
                for i in range(n):
                    s, nz = struct.unpack_from("<2h", pkt, off + 4 * i)
                    snr.append(s * 0.1)
                    noise.append(nz * 0.1)
            off += tlv_len
        # a truncated TLV1 with a full TLV7 would misalign snr[i] <-> point[i]
        # downstream (feeding material scoring) — trim side-info to the points
        if snr is not None and len(snr) > len(points):
            snr = snr[:len(points)]
            noise = noise[:len(points)]
        return {"frame": frame_no, "n_obj": n_obj, "points": points,
                "snr": snr, "noise": noise}


def send_config(cfg_serial, cfg_path, verbose=True, retries=2,
                line_map=None):
    """Send a .cfg file line-by-line over the CONFIG UART (115200 baud).

    Waits for the radar's actual Done/Error/Ignored response per command (a
    fixed sleep desynchronizes: slow commands like sensorStop answer late and
    every later response gets attributed to the wrong command). Failed
    commands are retried, which clears transient CLI errors on reconfigure.
    line_map, if given, may rewrite each command line before sending (used to
    flip lvdsStreamCfg on for DCA1000 raw capture without editing the file).
    Returns list of (command, response) tuples. CPython helper.
    """
    import time

    def send_line(line, timeout=2.0):
        cfg_serial.reset_input_buffer()          # drop stale bytes
        cfg_serial.write((line + "\n").encode())
        t0 = time.time()
        resp = ""
        while time.time() - t0 < timeout:
            n = cfg_serial.in_waiting
            if n:
                resp += cfg_serial.read(n).decode("ascii", "ignore")
                if "Done" in resp or "Error" in resp or "Ignored" in resp:
                    break
            else:
                time.sleep(0.01)
        return resp.strip()

    log = []
    with open(cfg_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line_map:
                line = line_map(line)
            resp = send_line(line)
            attempt = 0
            # an empty response (timeout) is NOT success: a lost cfarCfg means
            # the radar silently runs with the previous config's thresholds
            while ("Error" in resp or not resp) and attempt < retries:
                attempt += 1
                time.sleep(0.3)
                resp = send_line(line)
            log.append((line, resp))
            if verbose:
                short = resp.replace("\r", "").replace("\n", " | ")
                note = " (retry %d)" % attempt if attempt else ""
                print("  cfg> %-58s %s%s" % (line, short[:90], note))
            if "Error" in resp:
                raise RuntimeError("Radar rejected: %s -> %s" % (line, resp))
            if not resp:
                raise RuntimeError("Radar did not respond to: %s (check the "
                                   "CONFIG port and baud rate)" % line)
    return log


def find_com_ports():
    """Auto-detect the radar CONFIG and DATA COM ports (Windows/PC only).

    With TI drivers the ports are named "XDS110 Class Application/User UART"
    (config) and "XDS110 Class Auxiliary Data Port" (data). Without them they
    show as generic "USB Serial Device" — in that case the CLI port is found
    by probing with the harmless 'version' command; its sibling is the data
    port.

    Returns (cfg_port, data_port) names, or (None, None) if not found.
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        return None, None
    cfg = data = None
    generic = []
    for p in list_ports.comports():
        desc = (p.description or "") + " " + (p.interface or "")
        if "XDS110" in desc:
            if "Application" in desc or "User" in desc:
                cfg = p.device
            elif "Auxiliary" in desc or "Data" in desc:
                data = p.device
        elif "USB Serial Device" in desc:
            generic.append(p.device)
    if cfg and data:
        return cfg, data
    if len(generic) >= 2:                    # probe generic pairs for the CLI
        import serial, time
        for port in generic:
            try:
                s = serial.Serial(port, 115200, timeout=0.4)
                s.write(b"version\n")
                time.sleep(0.4)
                resp = s.read(256).decode("ascii", "ignore")
                s.close()
                if "mmWave" in resp or "xWR" in resp:
                    cfg = port
                    data = next(g for g in generic if g != port)
                    break
            except Exception:
                continue
    return cfg, data
