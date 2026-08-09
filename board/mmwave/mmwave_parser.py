# mmwave_parser.py — portable core of the N6 <-> AWR1843 TLV library.
#
# Clean-room implementation from the TI mmWave demo (SDK 3.x) wire spec.
# Pure struct-based parsing, no board imports — the SAME file runs on the
# N6 under MicroPython and on the host under pytest (tests/mmwave/).
#
# Covers plan stages 2-5:
#   FrameSync        stage 2: byte stream -> whole frames, resync on garbage
#   parse_header     stage 3: 8x uint32 LE header
#   walk_tlvs        stage 3: TLV walk that must close the frame exactly;
#                    resolves the "does length include the TLV header?"
#                    SDK ambiguity empirically, tolerates 32-byte padding
#   parse_frame      stages 4-5: TLV type 1 points + type 7 SNR join
#
# Plus Doppler-ambiguity helpers (fold/unfold, physics validators) that the
# aliasing work (plan item 7 / tests 10-12) builds on.

import struct

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'
HEADER_FMT = '<8I'
HEADER_LEN = 32
TLV_POINTS = 1
TLV_SIDE_INFO = 7

_MIN_FRAME = len(MAGIC) + HEADER_LEN  # 40
_TOTAL_LEN_OFF = len(MAGIC) + 4       # header word 1 = totalPacketLen
_MAX_PAD = 32                         # demo pads packets to 32-byte multiple


class FrameSync:
    """Stage 2: accumulate raw UART bytes, emit complete validated frames.

    Framing is trusted only when totalPacketLen closes onto the next magic
    word (or the exact end of buffered data) — a magic word alone proves
    nothing, since the pattern can occur inside a payload.
    """

    def __init__(self, max_buffer=64 * 1024, max_frame=16 * 1024):
        self._buf = b''
        self._max_buffer = max_buffer
        self._max_frame = max_frame
        self.resync_count = 0
        self.dropped_bytes = 0

    def feed(self, data):
        """Feed raw bytes; return a list of complete frames (bytes)."""
        frames = []
        if data:
            self._buf += bytes(data)
        if len(self._buf) > self._max_buffer:  # hard safety bound
            drop = len(self._buf) - self._max_buffer
            self.dropped_bytes += drop
            self._buf = self._buf[drop:]

        while True:
            idx = self._buf.find(MAGIC)
            if idx < 0:
                # no magic: keep only a tail that could be a partial magic
                keep = len(MAGIC) - 1
                if len(self._buf) > keep:
                    self.dropped_bytes += len(self._buf) - keep
                    self._buf = self._buf[-keep:]
                break

            if idx > 0:
                self.dropped_bytes += idx
                self.resync_count += 1
                self._buf = self._buf[idx:]

            if len(self._buf) < _MIN_FRAME:
                break

            total = struct.unpack_from('<I', self._buf, _TOTAL_LEN_OFF)[0]
            if total < _MIN_FRAME or total > self._max_frame:
                # implausible length: noise that happened to look like magic
                self._skip_one()
                continue

            if len(self._buf) < total:
                break  # frame still arriving

            after = self._buf[total:total + len(MAGIC)]
            if len(after) == len(MAGIC):
                if after != MAGIC:
                    # frame would steal its successor's bytes (truncation
                    # upstream) or the length itself is corrupt — reject
                    self._skip_one()
                    continue
                frames.append(self._buf[:total])
                self._buf = self._buf[total:]
            elif not after:
                frames.append(self._buf[:total])  # closes buffer exactly
                self._buf = b''
            else:
                # 1..7 trailing bytes: maybe the next magic starting
                if MAGIC.startswith(after):
                    break  # wait for enough bytes to decide
                self._skip_one()

        return frames

    def _skip_one(self):
        self.dropped_bytes += 1
        self.resync_count += 1
        self._buf = self._buf[1:]


def parse_header(frame):
    """Stage 3: decode the 8x uint32 LE header. Raises ValueError."""
    if len(frame) < _MIN_FRAME or not bytes(frame).startswith(MAGIC):
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


def walk_tlvs(frame):
    """Stage 3: return ([(type, payload_bytes), ...], length_mode).

    length_mode is 'payload' or 'includes_header' — decided empirically:
    the interpretation that makes the walk land exactly on totalPacketLen
    (allowing <32 bytes of zero padding) wins. Raises ValueError if
    neither closes the frame.
    """
    hdr = parse_header(frame)
    total = hdr['total_len']
    if total > len(frame):
        raise ValueError('frame shorter than totalPacketLen')

    for mode in ('payload', 'includes_header'):
        tlvs = []
        off = _MIN_FRAME
        ok = True
        for _ in range(hdr['num_tlvs']):
            if off + 8 > total:
                ok = False
                break
            tlv_type, length = struct.unpack_from('<2I', frame, off)
            plen = length - 8 if mode == 'includes_header' else length
            if plen < 0 or off + 8 + plen > total:
                ok = False
                break
            tlvs.append((tlv_type, bytes(frame[off + 8:off + 8 + plen])))
            off += 8 + plen
        if not ok:
            continue
        tail = bytes(frame[off:total])
        if off == total or (len(tail) < _MAX_PAD and
                            tail == b'\x00' * len(tail)):
            return tlvs, mode

    raise ValueError('TLV walk does not close the frame')


def parse_points(payload):
    """Stage 4: TLV type 1 — (x, y, z, v) float32 per point, meters, m/s."""
    n = len(payload) // 16
    pts = []
    for i in range(n):
        x, y, z, v = struct.unpack_from('<4f', payload, 16 * i)
        pts.append({'x': x, 'y': y, 'z': z, 'v': v,
                    'snr': None, 'noise': None})
    return pts


def parse_side_info(payload):
    """Stage 5: TLV type 7 — (snr, noise) int16 per point, 0.1 dB units."""
    n = len(payload) // 4
    return [struct.unpack_from('<2h', payload, 4 * i) for i in range(n)]


def parse_frame(frame):
    """Stages 3-5 combined: one complete frame -> header + joined points.

    Side info is joined to points BY INDEX (the demo emits both lists in
    the same detection order). Missing/short side info leaves snr=None —
    never a crash.
    """
    hdr = parse_header(frame)
    tlvs, mode = walk_tlvs(frame)
    points = []
    side = None
    for tlv_type, payload in tlvs:
        if tlv_type == TLV_POINTS:
            points = parse_points(payload)
        elif tlv_type == TLV_SIDE_INFO:
            side = parse_side_info(payload)
    if side:
        n = min(len(points), len(side))
        for i in range(n):
            points[i]['snr'] = side[i][0]
            points[i]['noise'] = side[i][1]
    return {
        'header': hdr,
        'points': points,
        'tlv_types': [t for t, _ in tlvs],
        'length_mode': mode,
    }


# --- Doppler ambiguity helpers (plan item 7 groundwork) -----------------
#
# FMCW measures velocity from chirp-to-chirp phase, unambiguous only in
# [-v_max, +v_max): true velocities outside fold modulo 2*v_max. With the
# rig's cfg (v_max ~= 0.67 m/s) a 1.4 m/s pedestrian reads as ~0.06 m/s.

def fold_velocity(v, vmax):
    """What the radar reports for a true velocity v."""
    span = 2.0 * vmax
    return ((v + vmax) % span) - vmax


def unfold_candidates(v_measured, vmax, kmax=2):
    """All true velocities (up to +-kmax folds) consistent with a reading."""
    return [v_measured + 2.0 * k * vmax for k in range(-kmax, kmax + 1)]


def _finite(x):
    return x == x and float('-inf') < x < float('inf')


def validate_points(points, vmax=None, max_range=None):
    """Physics sanity: impossible values mean a parsing/units bug.

    Returns a list of issue strings (empty = all sane).
    """
    issues = []
    for i, p in enumerate(points):
        if not (_finite(p['x']) and _finite(p['y']) and
                _finite(p['z']) and _finite(p['v'])):
            issues.append('point %d: non-finite field' % i)
            continue
        if vmax is not None and abs(p['v']) > vmax * (1.0 + 1e-6):
            issues.append('point %d: |v|=%.3f exceeds unambiguous '
                          'vmax=%.3f' % (i, abs(p['v']), vmax))
        if max_range is not None:
            r = (p['x'] * p['x'] + p['y'] * p['y'] +
                 p['z'] * p['z']) ** 0.5
            if r > max_range:
                issues.append('point %d: range %.2f m exceeds cfg max '
                              '%.2f m' % (i, r, max_range))
    return issues
