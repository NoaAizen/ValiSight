#!/usr/bin/env python3
"""Self-test: pack/unpack round-trip, chunked feeding, garbage, bad crc, and
byte-for-byte agreement with the constants in bridge_protocol.h."""
import os, re, sys, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bridge_protocol as bp

# 1. constants agree with the header
h = open(os.path.join(os.path.dirname(__file__), "bridge_protocol.h")).read()
def cdef(name):
    m = re.search(r"#define\s+%s\s+([^\n/]+)" % name, h); assert m, name
    return m.group(1).strip()
assert cdef("BRIDGE_HDR_LEN") == str(bp.HDR_LEN)
assert cdef("BRIDGE_MAGIC") == '"%s"' % bp.MAGIC.decode()
assert cdef("BRIDGE_T_IMU").rstrip("u") == str(bp.T_IMU)
assert cdef("BRIDGE_TS_TIM2_500NS").rstrip("u") == str(bp.TS_TIM2_500NS)

# 2. round trip of every record type
recs = [
    bp.pack(bp.T_HELLO,   bp.TS_TICKS_US, 0, 100, bp.hello_payload(0)),
    bp.pack(bp.T_THERMAL, bp.TS_TICKS_US, 1, 200, bp.image_payload(160, 120, bp.PIX_GRAY8, bytes(range(256)) * 75)),
    bp.pack(bp.T_RGB,     bp.TS_TICKS_US, 2, 300, bp.image_payload(320, 200, bp.PIX_JPEG, b"\xff\xd8" + b"x" * 5000 + b"\xff\xd9")),
    bp.pack(bp.T_IMU,     bp.TS_TICKS_US, 3, 400, bp.imu_payload(12, -7, 1004, 150, -20, 3)),
]
stream = b"junk\x04>" + b"".join(recs) + b"VSB"        # garbage before, split magic after
d = bp.Decoder()
got = []
for i in range(0, len(stream), 700):                    # arbitrary chunking
    got += d.feed(stream[i:i + 700])
assert [g[0] for g in got] == [0, 1, 2, 3], got
assert [g[2] for g in got] == [0, 1, 2, 3]
w, hh, fmt, px = bp.unpack_image(got[1][4]); assert (w, hh, fmt, len(px)) == (160, 120, 0, 19200)
assert bp.unpack_imu(got[3][4]) == (12, -7, 1004, 150, -20, 3)
assert bp.unpack_hello(got[0][4]) == (1, 0)
assert d.resyncs == 1 and d.bad_crc == 0

# 3. a flipped payload byte is rejected, and the NEXT record still decodes
bad = bytearray(recs[3]); bad[-1] ^= 0xFF
d2 = bp.Decoder(); got2 = d2.feed(bytes(bad) + recs[0])
assert [g[0] for g in got2] == [0] and d2.bad_crc == 1

# 4. header layout: seq at offset 8, ts at 12, len at 16 (matches the .h table)
r = recs[3]
assert struct.unpack_from("<I", r, 8)[0] == 3 and struct.unpack_from("<I", r, 12)[0] == 400
assert struct.unpack_from("<I", r, 16)[0] == 24
print("bridge_protocol: all checks PASS  (records:", len(got), " total bytes:", len(stream), ")")
