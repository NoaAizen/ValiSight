"""ValiSight bridge protocol — Python mirror of bridge_protocol.h.

Runs on BOTH MicroPython (the N6 sender) and CPython (tests / decoders), so it
uses only struct + binascii. Keep the constants and layout identical to the .h;
test_bridge_protocol.py checks the two agree byte-for-byte.

Record = 24-byte header + payload.  See bridge_protocol.h for the field table.
"""
try:
    import ustruct as struct          # MicroPython
    import ubinascii as binascii
except ImportError:                   # CPython
    import struct
    import binascii

PROTO_VER = 1
MAGIC = b"VSB1"
HDR_LEN = 24
MAX_PAYLOAD = 256 * 1024

T_HELLO, T_THERMAL, T_RGB, T_IMU = 0, 1, 2, 3
TS_TICKS_US, TS_TIM2_500NS = 0, 1
PIX_GRAY8, PIX_JPEG = 0, 1

# header after the magic: type u8, ts_src u8, reserved u16, seq u32, ts u32, len u32
_HDR_BODY = "<BBHIII"          # 16 bytes: offsets 4..20
_CRC = "<I"                    # offset 20..24
_IMG = "<HHBB"                 # w, h, fmt, pad
_IMU = "<iiiiii"               # ax ay az [mg], gx gy gz [mdeg/s]
_HELLO = "<III"                # proto_ver, sender_drops, imu_overflow


def _crc(body, payload):
    return binascii.crc32(payload, binascii.crc32(body)) & 0xFFFFFFFF


def pack(rtype, ts_src, seq, ts, payload):
    """Build one complete record (bytes) ready to write to the USB port."""
    body = struct.pack(_HDR_BODY, rtype, ts_src, 0, seq & 0xFFFFFFFF, ts & 0xFFFFFFFF, len(payload))
    return MAGIC + body + struct.pack(_CRC, _crc(body, payload)) + payload


def image_payload(w, h, fmt, data):
    return struct.pack(_IMG, w, h, fmt, 0) + data


def image_prefix(w, h, fmt):
    """The 6-byte image sub-header alone (for zero-copy sends: header+prefix, then pixels)."""
    return struct.pack(_IMG, w, h, fmt, 0)


def pack_header(rtype, ts_src, seq, ts, prefix, payload):
    """24-byte header for a record whose payload is prefix+payload, WITHOUT
    concatenating them (crc is chained over body, prefix, payload)."""
    ln = len(prefix) + len(payload)
    body = struct.pack(_HDR_BODY, rtype, ts_src, 0, seq & 0xFFFFFFFF, ts & 0xFFFFFFFF, ln)
    crc = binascii.crc32(payload, binascii.crc32(prefix, binascii.crc32(body))) & 0xFFFFFFFF
    return MAGIC + body + struct.pack(_CRC, crc)


_IMU_REC = "<4sBBHIIII" + "iiiiii"     # magic, body, crc, then the 24-byte IMU payload


def pack_imu_into(buf, off, ts_src, seq, ts, ax, ay, az, gx, gy, gz):
    """Pack a complete IMU record in place at buf[off:] (no allocation beyond
    the crc computation over a memoryview)."""
    struct.pack_into(_HDR_BODY, buf, off + 4, T_IMU, ts_src, 0, seq & 0xFFFFFFFF, ts & 0xFFFFFFFF, 24)
    struct.pack_into(_IMU, buf, off + HDR_LEN, ax, ay, az, gx, gy, gz)
    buf[off:off + 4] = MAGIC
    crc = binascii.crc32(buf[off + HDR_LEN:off + HDR_LEN + 24], binascii.crc32(buf[off + 4:off + 20])) & 0xFFFFFFFF
    struct.pack_into(_CRC, buf, off + 20, crc)


def imu_payload(ax_mg, ay_mg, az_mg, gx_mdps, gy_mdps, gz_mdps):
    return struct.pack(_IMU, int(ax_mg), int(ay_mg), int(az_mg), int(gx_mdps), int(gy_mdps), int(gz_mdps))


def hello_payload(sender_drops, imu_overflow=0):
    return struct.pack(_HELLO, PROTO_VER, sender_drops & 0xFFFFFFFF, imu_overflow & 0xFFFFFFFF)


def ts_period(ts_src):
    """Wrap period of the sender clock, in ticks."""
    return 1 << 31 if ts_src == TS_TIM2_500NS else 1 << 30


def ts_seconds_per_tick(ts_src):
    return 500e-9 if ts_src == TS_TIM2_500NS else 1e-6


# ---------------------------------------------------------------- decoding
class Decoder:
    """Feed raw bytes in any chunking; get back complete records.

    Robust to garbage before/after: scans for MAGIC, drops records whose crc
    fails and rescans one byte later. Counts what it dropped so nothing is
    silently lost.
    """

    def __init__(self):
        self.buf = b""
        self.bad_crc = 0
        self.resyncs = 0

    def feed(self, chunk):
        self.buf += chunk
        out = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                # keep a tail in case the magic is split across chunks
                self.buf = self.buf[-(len(MAGIC) - 1):] if len(self.buf) >= len(MAGIC) else self.buf
                return out
            if i > 0:
                self.resyncs += 1
                self.buf = self.buf[i:]
            if len(self.buf) < HDR_LEN:
                return out
            rtype, ts_src, _res, seq, ts, ln = struct.unpack(_HDR_BODY, self.buf[4:20])
            (crc,) = struct.unpack(_CRC, self.buf[20:24])
            if ln > MAX_PAYLOAD:
                self.bad_crc += 1
                self.buf = self.buf[1:]
                continue
            if len(self.buf) < HDR_LEN + ln:
                return out
            payload = self.buf[HDR_LEN:HDR_LEN + ln]
            if _crc(self.buf[4:20], payload) != crc:
                self.bad_crc += 1
                self.buf = self.buf[1:]
                continue
            self.buf = self.buf[HDR_LEN + ln:]
            out.append((rtype, ts_src, seq, ts, payload))


def unpack_image(payload):
    w, h, fmt, _ = struct.unpack(_IMG, payload[:6])
    return w, h, fmt, payload[6:]


def unpack_imu(payload):
    return struct.unpack(_IMU, payload[:24])


def unpack_hello(payload):
    """(proto_ver, sender_drops, imu_overflow) — tolerates the older 8-byte HELLO."""
    if len(payload) >= 12:
        return struct.unpack(_HELLO, payload[:12])
    v, d = struct.unpack("<II", payload[:8])
    return v, d, 0
