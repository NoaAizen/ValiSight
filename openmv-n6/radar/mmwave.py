"""IWR1843 mmWave demo (SDK 3.x) TLV parser — the portable core of the radar path.

Pure `struct`, no board imports and no numpy, so the SAME file runs three places:

    host   (Jetson)  radar on USB: CLI /dev/ttyACM1, DATA /dev/ttyACM2
    board  (N6)      radar on UART7, P13 = PE7 = RX  (see radar/README.md)
    tests  (pytest)  tools/tests/test_radar.py, no hardware

Wire format: magic (8B) -> header (32B, 8x uint32 LE) -> TLVs. Everything the
frame needs to be interpreted is in the frame; the .cfg is only needed for the
physical limits in CFG_10HZ.

TWO THINGS HERE ARE NOT COSMETIC, because both turn into silent calibration
errors rather than crashes:

1. Coordinate frame. TI reports x = right, y = forward, z = up. This module
   returns the *project* radar frame, x = forward, y = left, z = up, which is
   what `calibrate_radar_camera.solvePnP` and the archived classifier were
   built against. Pass `convention='ti'` to get the sensor's own numbers back.
   A frame mix-up looks exactly like a 90 deg mounting error in the extrinsics
   and will be absorbed by R without ever raising anything.

2. TLV length. The SDK is ambiguous about whether a TLV's length field counts
   its own 8-byte header. `walk_tlvs` resolves it per frame, by taking the
   reading that makes the walk land on totalPacketLen. Guessing wrong shifts
   every payload after the first TLV, which yields plausible-looking garbage
   points rather than an exception.
"""
import struct

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'
HEADER_FMT = '<8I'
HEADER_LEN = 32
FRAME_HEADER_LEN = len(MAGIC) + HEADER_LEN     # 40
_TOTAL_LEN_OFF = len(MAGIC) + 4                # header word 1 = totalPacketLen

TLV_POINTS = 1
TLV_RANGE_PROFILE = 2
TLV_NOISE_PROFILE = 3
TLV_AZIMUTH_HEATMAP = 4
TLV_RANGE_DOPPLER_HEATMAP = 5
TLV_STATS = 6
TLV_SIDE_INFO = 7
TLV_TEMPERATURE = 9

_MAX_PAD = 32          # the demo pads packets up to a 32-byte multiple
_MAX_TLVS = 32         # sanity bound; the demo emits <= 8
_MAX_OBJ = 4096

# Physical limits of radar/configs/radar_10hz.cfg. Measured on this board, not
# computed from the datasheet: 225k logged points fit a 0.0436 m range grid
# (0.058 mean residual, vs 0.240 at 0.0439), and the Doppler axis has exactly
# 16 distinct values from -0.649 to +0.568 in steps of 0.0812.
#
# hpf_blind_m is the one that decides whether this radar is usable for the
# 0.5-2 m inspection job at all: profileCfg hpfCornerFreq1/2 = 0/0 = 175 kHz,
# and 175 kHz maps to 0.375 m. Below that the IF high-pass attenuates signal
# and noise alike, so range bins 0-8 carry nothing to detect.
CFG_10HZ = {
    'frame_period_s': 0.100,
    'range_res_m': 0.0436,
    'range_max_unambiguous_m': 11.16,
    'v_max_m_s': 0.649,
    'v_res_m_s': 0.0812,
    # The HPF corner attenuates signal AND noise alike, so SNR is roughly
    # preserved and CFAR still fires below this: 1177 archived detections sit
    # at 0.261-0.350 m with SNR up to 32.1 dB. So this is where sensitivity
    # starts falling off, NOT a wall -- do not use it to reject points as
    # impossible. Note also that it is a function of freqSlopeConst, not a
    # property of the board: change the profile and this number moves with it.
    'hpf_blind_m': 0.375,
    # The MEASURED floor of the reported side-info SNR, not the cfg's
    # thresholdScale. Live on this radar at thresholdScale 15: min 11.2 dB over
    # 224 points, with 37.5% of them below 15.0 dB; archived data at
    # thresholdScale 12 floors at 9.0 dB. The relation is ~0.75x, so the cfg
    # number is not in dB of reported SNR. Filtering at 15.0 would throw away
    # more than a third of a real point cloud.
    'reported_snr_floor_db': 11.2,
    'cfar_threshold_scale': 15,
}


# --- framing ------------------------------------------------------------

class FrameSync:
    """Raw byte stream -> complete, validated frames, each stamped with the
    time its LAST BYTE ARRIVED.

    feed() returns [(t_complete, frame_bytes), ...].

    A magic word alone proves nothing: the pattern occurs inside payloads, and
    on a UART with no DMA (which is what the N6 has) a dropped byte turns one
    frame into two half-frames. A candidate is accepted only when both hold:

      - its TLV walk closes on totalPacketLen (catches a magic that was really
        payload data, and a corrupt length field), and
      - the next 8 bytes are the following frame's magic (catches the case the
        walk cannot see: a byte lost from INSIDE a payload, which leaves every
        declared length still summing to totalPacketLen while the frame quietly
        eats its successor's first byte).

    The second test needs the successor's bytes, so a frame followed by a
    partial magic is held until the ambiguity resolves. The one case it does
    not wait for is a frame that closes the buffer exactly, which is the
    ordinary live case and is safe for the reason given at that branch -- so on
    a live link this normally costs no latency at all, and on a replayed buffer
    it costs at most one frame.

    Latency, where it happens, does NOT reach the timestamp. t_complete is
    recorded when the
    frame's bytes finished arriving, not when validation released it, so the
    figure that pairing and time-offset estimation consume is unaffected by
    how long framing waited. Pass the read time into feed():

        t = time.monotonic()
        for t_complete, frame in fs.feed(port.read(4096), t):
            ...
    """

    def __init__(self, max_frame=16 * 1024, max_buffer=64 * 1024):
        self._buf = b''
        self._max_frame = max_frame
        self._max_buffer = max_buffer
        self._pending_t = None       # arrival time of the frame now at buf[0]
        self.resync_count = 0        # candidates rejected as not-a-frame
        self.dropped_bytes = 0       # bytes thrown away between good frames
        self.frame_count = 0

    def feed(self, data, t=None):
        """Feed raw bytes; return [(t_complete, frame_bytes), ...]."""
        if data:
            self._buf += bytes(data)
        if len(self._buf) > self._max_buffer:          # hard safety bound
            drop = len(self._buf) - self._max_buffer
            self.dropped_bytes += drop
            self._buf = self._buf[drop:]
            self._pending_t = None

        frames = []
        while True:
            idx = self._buf.find(MAGIC)
            if idx < 0:
                keep = len(MAGIC) - 1                  # maybe a split magic
                if len(self._buf) > keep:
                    self.dropped_bytes += len(self._buf) - keep
                    self._buf = self._buf[-keep:]
                break
            if idx > 0:
                self._drop(idx)
            if len(self._buf) < FRAME_HEADER_LEN:
                break                                   # header still arriving

            total = struct.unpack_from('<I', self._buf, _TOTAL_LEN_OFF)[0]
            if total < FRAME_HEADER_LEN or total > self._max_frame:
                if not self._resync():
                    break
                continue
            if len(self._buf) < total:
                break                                   # payload still arriving

            # All of the frame is here: this read is when it finished arriving,
            # whatever validation decides next.
            if self._pending_t is None:
                self._pending_t = t

            frame = self._buf[:total]
            if _resolve_length_mode(frame, total) < 0:
                if not self._resync():                  # magic inside a payload
                    break
                continue

            after = bytes(self._buf[total:total + len(MAGIC)])
            if after and after != MAGIC:
                if len(after) < len(MAGIC) and MAGIC.startswith(after):
                    break                               # undecided; wait
                if not self._resync():                  # truncated upstream
                    break
                continue
            # after is either the successor's magic, or empty because the frame
            # closes the buffer exactly. Empty is the ordinary live case -- at
            # 10 Hz the next frame is 100 ms away, so the read that completes
            # this one usually ends on its last byte -- and it is safe: a frame
            # short one byte never reaches `total` on its own, it reaches it
            # only once the successor's bytes arrive, and those arrive by the
            # thousand rather than one at a time.

            frames.append((self._pending_t, frame))
            self.frame_count += 1
            self._buf = self._buf[total:]
            self._pending_t = None

        return frames

    def flush(self):
        """End of stream: release a buffered frame that has no successor.

        Only for a finite source -- a recorded .bin, or a capture being closed.
        On a live link the frame that looks complete may simply be short one
        byte, which is the whole reason feed() waits.

        The trailing bytes are the subtle part. feed() holds a frame in exactly
        one situation: 1..7 bytes follow it that could be the start of the next
        magic. An earlier version of this required total == len(buf), i.e. NO
        trailing bytes -- which is the one case feed() has already released. It
        was therefore dead code that returned [] for every input while looking
        like an end-of-stream guarantee, and the test locked that in. At end of
        stream those trailing bytes are a truncated successor, not a successor,
        so the frame ahead of them is as good as it will ever get.
        """
        if len(self._buf) < FRAME_HEADER_LEN or not self._buf.startswith(MAGIC):
            return []
        total = struct.unpack_from('<I', self._buf, _TOTAL_LEN_OFF)[0]
        if total > len(self._buf):
            return []          # the frame never finished arriving
        if len(self._buf) - total >= len(MAGIC):
            return []          # a whole successor is present: feed() owns this
        if _resolve_length_mode(self._buf[:total], total) < 0:
            return []
        out = [(self._pending_t, self._buf[:total])]
        self.frame_count += 1
        self.dropped_bytes += len(self._buf) - total
        self._buf = b''
        self._pending_t = None
        return out

    def _drop(self, n):
        self.dropped_bytes += n
        self._buf = self._buf[n:]
        self._pending_t = None

    def _resync(self):
        """Current magic is not a frame start: jump to the next magic.

        Jumping rather than sliding one byte at a time matters on the board.
        A 4 KB burst of line noise costs one find() here; sliding costs 4096
        buffer copies, which is both O(n^2) and 4096 allocations, and the
        allocation is the part that eventually reclaims a CSI buffer the
        Lepton still needs (see openmv-n6/README.md on gc).
        """
        self.resync_count += 1
        nxt = self._buf.find(MAGIC, 1)
        if nxt < 0:
            keep = len(MAGIC) - 1
            if len(self._buf) > keep:
                self._drop(len(self._buf) - keep)
            self._pending_t = None
            return False
        self._drop(nxt)
        return True


class MagicScanner:
    """Stage-1 diagnostic: locate magics in a stream and check for lost bytes.

    Not a parser — it never looks at a payload. Its one job is to answer the
    only question stage 1 asks: did every byte the radar sent actually arrive?

    It answers it exactly rather than statistically. Each magic carries its own
    totalPacketLen, so the byte distance to the NEXT magic must equal it. A
    mismatch is bytes lost or bytes injected, full stop — which is what a UART
    with no DMA does when the reader was busy (a Lepton snapshot() blocks for
    113 ms; at 921600 baud that is ~10.4 KB arriving with nobody reading, and
    MicroPython's stm32 uart.c drops the overflow byte silently, no error).

    Byte distance is also immune to read batching, unlike wall-clock jitter.
    """

    def __init__(self):
        self._carry = b''
        self._pos = 0            # absolute offset of the start of _carry
        self.magics = 0
        self.good_gaps = 0       # gap == the previous frame's totalPacketLen
        self.bad_gaps = 0        # gap != it: bytes lost or injected
        self.lost_bytes = 0      # signed sum of (gap - totalPacketLen)
        self.gaps = []           # recent (gap, expected) pairs, capped
        self.max_gaps = 64
        self._last_pos = None
        self._last_total = None
        self.totals = []         # recent totalPacketLen values, capped

    def feed(self, data):
        if not data:
            return
        buf = self._carry + bytes(data)
        base = self._pos
        i = buf.find(MAGIC)
        while i >= 0:
            if i + _TOTAL_LEN_OFF + 4 > len(buf):
                break            # totalPacketLen not here yet; keep and retry
            total = struct.unpack_from('<I', buf, i + _TOTAL_LEN_OFF)[0]
            pos = base + i
            self.magics += 1
            if self._last_pos is not None:
                gap = pos - self._last_pos
                if len(self.gaps) < self.max_gaps:
                    self.gaps.append((gap, self._last_total))
                if gap == self._last_total:
                    self.good_gaps += 1
                else:
                    self.bad_gaps += 1
                    self.lost_bytes += gap - self._last_total
            self._last_pos, self._last_total = pos, total
            if len(self.totals) < self.max_gaps:
                self.totals.append(total)
            i = buf.find(MAGIC, i + len(MAGIC))

        if i >= 0:                       # incomplete header: replay it next time
            self._carry = buf[i:]
        else:
            self._carry = buf[-(len(MAGIC) - 1):]
        self._pos = base + len(buf) - len(self._carry)

    def window_reset(self):
        """Clear per-window samples, keep the cross-window position state."""
        self.gaps = []
        self.totals = []


# --- header and TLVs ----------------------------------------------------

def parse_header(frame):
    """Decode the 8x uint32 LE header. Raises ValueError."""
    if len(frame) < FRAME_HEADER_LEN or not bytes(frame[:8]) == MAGIC:
        raise ValueError('not an mmWave demo frame')
    w = struct.unpack_from(HEADER_FMT, frame, len(MAGIC))
    return {
        'version': w[0],
        'total_len': w[1],
        'platform': w[2],
        'frame_number': w[3],
        'time_cpu_cycles': w[4],
        'num_detected_obj': w[5],
        'num_tlvs': w[6],
        'subframe': w[7],
    }


def _tlv_end(frame, total, num_tlvs, includes_header):
    """Offset just past the last TLV under one length interpretation, or -1."""
    off = FRAME_HEADER_LEN
    for _ in range(num_tlvs):
        if off + 8 > total:
            return -1
        length = struct.unpack_from('<I', frame, off + 4)[0]
        plen = length - 8 if includes_header else length
        if plen < 0 or off + 8 + plen > total:
            return -1
        off += 8 + plen
    return off


def _resolve_length_mode(frame, total):
    """0 = length is the payload, 1 = length includes the 8-byte TL. -1 = neither.

    An exact landing on totalPacketLen wins over one that needs padding, so a
    frame that happens to close both ways is read the way that needs no excuse.

    THE PADDING IS NOT ZEROS. Measured on this radar 2026-08-09: totalPacketLen
    is rounded up to a 32-byte multiple (224 / 256 / 288 observed) and the
    leftover carries stale transmit-buffer content, e.g.
    b'\\x00\\x03\\x13\\x00\\x00\\x00\\x00\\x00y\\xde\\x00\\x00'. Requiring zeros
    there -- the obvious reading, and what the first version of this file did --
    rejected 70% of perfectly good frames while stage 1 reported a flawless
    link. So the only thing the tail is allowed to assert is its LENGTH.
    """
    if total > len(frame) or total < FRAME_HEADER_LEN:
        return -1
    num_tlvs = struct.unpack_from('<I', frame, len(MAGIC) + 24)[0]
    if num_tlvs > _MAX_TLVS:
        return -1
    padded = -1
    for mode in (0, 1):
        off = _tlv_end(frame, total, num_tlvs, mode)
        if off < 0:
            continue
        if off == total:
            return mode
        if padded < 0 and total - off < _MAX_PAD:
            padded = mode
    return padded


def walk_tlvs(frame):
    """Return ([(type, payload_bytes), ...], length_mode).

    length_mode is 'payload' or 'includes_header'. Raises ValueError if the
    walk closes under neither reading — which means the framing was wrong, not
    that the radar sent something new.
    """
    hdr = parse_header(frame)
    total = hdr['total_len']
    if total > len(frame):
        raise ValueError('frame shorter than totalPacketLen')
    mode = _resolve_length_mode(frame, total)
    if mode < 0:
        raise ValueError('TLV walk does not close the frame')

    tlvs = []
    off = FRAME_HEADER_LEN
    for _ in range(hdr['num_tlvs']):
        tlv_type, length = struct.unpack_from('<2I', frame, off)
        plen = length - 8 if mode else length
        tlvs.append((tlv_type, bytes(frame[off + 8:off + 8 + plen])))
        off += 8 + plen
    return tlvs, ('includes_header' if mode else 'payload')


# --- payloads -----------------------------------------------------------

def ti_to_project(x_right, y_fwd, z_up):
    """TI sensor frame -> project radar frame (x fwd, y left, z up).

    VERIFIED CORRECT 2026-08-10 with a corner reflector at 7 stations on
    both azimuth signs: projecting the recorded (x, y, z) through the
    canonical radar->camera rotation lands on the picked pixels within
    ~6 px for the clean holds (captures/holds1). The "flipped azimuth"
    anomaly of 2026-08-09 was NOT in this mapping - it is a display/veto
    convention mix in the calib tools (radar az printed left-positive,
    pixel az computed right-positive). Do not "fix" the sign here.
    """
    return y_fwd, -x_right, z_up


def parse_points(payload, convention='project'):
    """TLV 1 -> [{x, y, z, v, snr, noise}], metres and m/s.

    v is the FOLDED radial velocity as the radar reports it, not the true one;
    see fold_velocity. Positive v is away from the sensor in both conventions,
    because the sign lives on the range axis and the axis swap does not touch it.
    """
    n = len(payload) // 16
    pts = []
    for i in range(n):
        xr, yf, zu, v = struct.unpack_from('<4f', payload, 16 * i)
        if convention == 'ti':
            x, y, z = xr, yf, zu
        else:
            x, y, z = ti_to_project(xr, yf, zu)
        pts.append({'x': x, 'y': y, 'z': z, 'v': v,
                    'snr': None, 'noise': None})
    return pts


def parse_side_info(payload):
    """TLV 7 -> [(snr_db, noise_db)], converted from the wire's 0.1 dB int16.

    snr is above the local CFAR noise estimate; noise is that estimate. Their
    SUM is the absolute received level, and that is the quantity that follows
    the range equation — use the sum, not snr, for anything reflectivity-like.
    """
    n = len(payload) // 4
    out = []
    for i in range(n):
        s, nz = struct.unpack_from('<2h', payload, 4 * i)
        # Rounded to the wire's own quantum: the value IS an integer count of
        # 0.1 dB, so 27.200000000000003 is binary-float litter, not precision.
        out.append((round(s * 0.1, 1), round(nz * 0.1, 1)))
    return out


def parse_stats(payload):
    """TLV 6 -> the demo's own timing report, microseconds and percent.

    This is how the RADAR says it is struggling, as opposed to how the link
    says so. interframe_margin_us going to zero means the DSP did not finish
    inside the 100 ms frame period, and the frames that go missing then are
    missing before they ever reach a UART.
    """
    if len(payload) < 24:
        return None
    w = struct.unpack_from('<6I', payload, 0)
    return {
        'interframe_proc_us': w[0],
        'transmit_out_us': w[1],
        'interframe_margin_us': w[2],
        'interchirp_margin_us': w[3],
        'active_frame_cpu_load': w[4],
        'interframe_cpu_load': w[5],
    }


def parse_temperature(payload):
    """TLV 9 -> die temperatures in degrees C, or None if the report is invalid.

    Worth carrying rather than discarding: the range bias that
    compRangeBiasAndRxChanPhase corrects drifts with die temperature, so a
    calibration taken on a cold board and used on a warm one is a slowly
    moving depth offset. Recording this alongside the points is what makes
    that checkable later instead of guessable.
    """
    if len(payload) < 28:
        return None
    rc, t_ms = struct.unpack_from('<iI', payload, 0)
    s = struct.unpack_from('<10h', payload, 8)
    # tempReportValid is the RETURN CODE of rlRfGetTemperatureReport(), and
    # RL_RET_CODE_OK is 0 -- so zero means the report IS valid and non-zero is
    # an error code. `bool(rc)` is therefore exactly backwards, which is what
    # this code did until it was checked against the board: 251 consecutive
    # frames from a healthy radar all carried rc = 0, with sane die
    # temperatures of 42-45 C. Reading it the obvious way would have kept only
    # the failed reports -- the precise opposite of what this field is carried
    # for, which is checking whether a range calibration taken cold drifted.
    return {
        'valid': rc == 0,
        'return_code': rc,
        'time_ms': t_ms,
        'rx_c': list(s[0:4]),
        'tx_c': list(s[4:7]),
        'pm_c': s[7],
        'dig_c': list(s[8:10]),
    }


def parse_frame(frame, convention='project'):
    """One complete frame -> header + joined points.

    Side info is joined to points BY INDEX: the demo emits both lists in the
    same detection order. Missing or short side info leaves snr=None rather
    than raising, because a frame with points and no SNR is still a usable
    frame — but it is never silently zero-filled, since 0 dB SNR is a
    meaningful value and None is not.
    """
    hdr = parse_header(frame)
    tlvs, mode = walk_tlvs(frame)
    points = []
    side = stats = temperature = None
    for tlv_type, payload in tlvs:
        if tlv_type == TLV_POINTS:
            points = parse_points(payload, convention)
        elif tlv_type == TLV_SIDE_INFO:
            side = parse_side_info(payload)
        elif tlv_type == TLV_STATS:
            stats = parse_stats(payload)
        elif tlv_type == TLV_TEMPERATURE:
            temperature = parse_temperature(payload)
    if side:
        for i in range(min(len(points), len(side))):
            points[i]['snr'] = side[i][0]
            points[i]['noise'] = side[i][1]
    return {
        'header': hdr,
        'frame_number': hdr['frame_number'],
        'points': points,
        'stats': stats,
        'temperature': temperature,
        'tlv_types': [t for t, _ in tlvs],
        'length_mode': mode,
        'convention': convention,
    }


# --- Doppler ambiguity --------------------------------------------------
#
# FMCW measures velocity from chirp-to-chirp phase, unambiguous only over
# [-v_max, +v_max). With radar_10hz.cfg v_max is 0.649 m/s, so the alias period
# is 1.298 m/s: a person walking at 1.2-1.4 m/s folds to |v| <= 0.10 m/s and
# reads as STATIC. 85.2% of logged points sit at exactly zero Doppler. This is
# a property of the config, not a bug to fix in software — shortening idleTime
# (429 us) is what buys v_max back, at the cost of range.

def fold_velocity(v, vmax=None):
    """What the radar reports for a true velocity v."""
    if vmax is None:
        vmax = CFG_10HZ['v_max_m_s']
    span = 2.0 * vmax
    return ((v + vmax) % span) - vmax


def unfold_candidates(v_measured, vmax=None, kmax=2):
    """Every true velocity within +-kmax folds consistent with a reading."""
    if vmax is None:
        vmax = CFG_10HZ['v_max_m_s']
    return [v_measured + 2.0 * k * vmax for k in range(-kmax, kmax + 1)]


def _finite(x):
    return x == x and float('-inf') < x < float('inf')


def range_of(p):
    return (p['x'] * p['x'] + p['y'] * p['y'] + p['z'] * p['z']) ** 0.5


def validate_points(points, vmax=None, max_range=None, min_range=None):
    """Physics sanity. A violation means a parsing or units bug, not a target.

    Returns a list of issue strings; empty means every point is possible. This
    is the cheapest guard against the two failure modes that do not crash: a
    wrong TLV length shifting the float stream, and a frame read with the
    wrong endianness or stride.
    """
    if vmax is None:
        vmax = CFG_10HZ['v_max_m_s']
    if max_range is None:
        max_range = CFG_10HZ['range_max_unambiguous_m']
    issues = []
    for i, p in enumerate(points):
        if not (_finite(p['x']) and _finite(p['y']) and
                _finite(p['z']) and _finite(p['v'])):
            issues.append('point %d: non-finite field' % i)
            continue
        if vmax and abs(p['v']) > vmax * (1.0 + 1e-6):
            issues.append('point %d: |v|=%.3f exceeds unambiguous vmax=%.3f'
                          % (i, abs(p['v']), vmax))
        r = range_of(p)
        if max_range and r > max_range:
            issues.append('point %d: range %.2f m exceeds cfg max %.2f m'
                          % (i, r, max_range))
        if min_range and r < min_range:
            issues.append('point %d: range %.2f m is inside the %.2f m IF '
                          'high-pass blind zone' % (i, r, min_range))
    return issues
