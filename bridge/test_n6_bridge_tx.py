#!/usr/bin/env python3
"""Tests for the sender core (Bridge) — run on the PC, no hardware.

Covers: seq is contiguous, every record decodes back with the right type and
timestamp, a short USB write is counted as a drop (and seq still advances so
the receiver SEES the gap), HELLO fires once per interval and carries the drop
count, and the wrapping-clock difference is right across the wrap.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bridge_protocol as bp
from n6_bridge_tx import Bridge, ImuRing


class FakeLink:
    def __init__(self):
        self.buf = b""
        self.short_next = False          # simulate USB refusing the next write
    def write(self, b):
        if self.short_next:
            self.short_next = False
            return len(b) // 2            # partial write, like a stuffed VCP
        self.buf += b
        return len(b)


class Clock:
    def __init__(self, t=0): self.t = t
    def __call__(self): return self.t & ((1 << 30) - 1)


def decode(link):
    d = bp.Decoder(); recs = d.feed(link.buf); return recs, d


def test_records_and_seq():
    link, clk = FakeLink(), Clock(1000)
    br = Bridge(link.write, clk)
    br.hello()
    br.thermal(bytes(19200), 1001)
    br.imu((12, -7, 1004), (150000, -20000, 3000), 1002)
    br.rgb_jpeg(b"\xff\xd8" + b"x" * 3000 + b"\xff\xd9", 1003)
    recs, d = decode(link)
    assert [r[0] for r in recs] == [bp.T_HELLO, bp.T_THERMAL, bp.T_IMU, bp.T_RGB]
    assert [r[2] for r in recs] == [0, 1, 2, 3], "seq must be contiguous"
    assert [r[3] for r in recs] == [1000, 1001, 1002, 1003]
    assert bp.unpack_imu(recs[2][4]) == (12, -7, 1004, 150000, -20000, 3000)
    w, h, fmt, px = bp.unpack_image(recs[1][4]); assert (w, h, fmt, len(px)) == (160, 120, bp.PIX_GRAY8, 19200)
    assert d.bad_crc == 0 and d.resyncs == 0
    assert br.sent == 4 and br.drops == 0


def test_short_write_counts_as_drop_and_leaves_a_gap():
    link, clk = FakeLink(), Clock()
    br = Bridge(link.write, clk)
    br.thermal(bytes(100), 1)          # seq 0 ok
    link.short_next = True
    ok = br.thermal(bytes(100), 2)     # seq 1 lost
    br.thermal(bytes(100), 3)          # seq 2 ok
    assert ok is False and br.drops == 1 and br.sent == 2
    recs, _ = decode(link)
    assert [r[2] for r in recs] == [0, 2], "receiver must see the seq gap"
    br.hello()
    recs, _ = decode(link)
    assert bp.unpack_hello(recs[-1][4]) == (bp.PROTO_VER, 1, 0), "HELLO reports the sender's drop count"


def test_hello_cadence():
    link, clk = FakeLink(), Clock(0)
    br = Bridge(link.write, clk, hello_every=1000)
    assert br.maybe_hello() is True         # first call always sends
    clk.t = 500;  assert br.maybe_hello() is None
    clk.t = 999;  assert br.maybe_hello() is None
    clk.t = 1000; assert br.maybe_hello() is True
    recs, _ = decode(link)
    assert [r[0] for r in recs] == [bp.T_HELLO, bp.T_HELLO]


def test_imu_ring_drains_in_order_and_counts_overflow():
    """The timer callback fills the ring; the main loop drains it. Samples
    beyond capacity are counted, not silently dropped; drain returns oldest
    first with each sample's own timestamp."""
    clk = Clock(100)
    samples = iter([((i, -i, 1000 + i), (10 * i, 0, -5)) for i in range(10)])
    ring = ImuRing(4, lambda: next(samples), clk)
    for k in range(6):                # 6 ticks into a 4-slot ring
        clk.t = 100 + k; ring.tick()
    assert ring.ix == 4 and ring.overflow == 2
    link = FakeLink(); br = Bridge(link.write, clk)
    assert br.imu_ring(ring) == 4 and ring.ix == 0
    recs, _ = decode(link)
    assert [r[0] for r in recs] == [bp.T_IMU] * 4
    assert [r[3] for r in recs] == [100, 101, 102, 103], "each sample keeps its own timestamp"
    assert [bp.unpack_imu(r[4])[0] for r in recs] == [0, 1, 2, 3], "oldest first"
    br.imu_overflow = ring.overflow; br.hello()
    recs, _ = decode(link)
    assert bp.unpack_hello(recs[-1][4]) == (bp.PROTO_VER, 0, 2), "HELLO carries the ring overflow"
    # busy flag: a tick during drain is skipped, not corrupting
    ring.busy = True; ring.tick(); assert ring.ix == 0; ring.busy = False


def test_clock_wrap():
    br = Bridge(lambda b: len(b), lambda: 0)
    p = 1 << 30
    assert br._diff(5, p - 5) == 10, "across the ticks_us wrap"
    assert br._diff(p - 5, 5) == -10
    br2 = Bridge(lambda b: len(b), lambda: 0, ts_src=bp.TS_TIM2_500NS)
    assert br2._diff(3, (1 << 31) - 2) == 5


def test_imu_ring_keeps_ticks_that_arrive_while_packing():
    """Live 2026-08-18: holding `busy` across the whole pack loop lost ~6 samples
    per camera frame (148 Hz instead of 200). A tick that fires mid-pack must
    survive into the next drain, in order, with its own timestamp."""
    import bridge_protocol as bp_
    clk = Clock(0)
    ring = ImuRing(8, lambda: ((1, 2, 3), (4, 5, 6)), clk)
    for k in range(3):
        clk.t = k; ring.tick()
    orig = bp_.pack_imu_into
    fired = []
    def pack_and_tick(*a, **kw):          # simulate the timer firing during the pack loop
        r = orig(*a, **kw)
        if not fired:
            fired.append(1); clk.t = 50; ring.tick()
        return r
    bp_.pack_imu_into = pack_and_tick
    try:
        n = ring.pack_records(0, 0)
    finally:
        bp_.pack_imu_into = orig
    assert n == 3 and ring.ix == 1, "the mid-pack sample is kept, not dropped"
    assert ring.ts[0] == 50, "and slid to the front with its own timestamp"
    assert ring.pack_records(0, 3) == 1 and ring.ix == 0
    # same for drain()
    clk.t = 60; ring.tick(); clk.t = 61; ring.tick()
    out = ring.drain(); assert [o[0] for o in out] == [60, 61] and ring.ix == 0



def test_thermal_carries_ffc_flags():
    """5.2 for Yael: the FFC state read from the Lepton rides in the image flags byte."""
    link = FakeLink(); br = Bridge(link.write, lambda: 0)
    br.thermal(b"\x00" * 6, 5, w=3, h=2, flags=bp.IMG_FLAG_FFC | bp.IMG_FLAG_FFC_KNOWN)
    br.thermal(b"\x00" * 6, 6, w=3, h=2, flags=bp.IMG_FLAG_FFC_KNOWN)
    br.thermal(b"\x00" * 6, 7, w=3, h=2)                      # legacy: unknown
    recs, _ = decode(link)
    assert [bp.image_flags(r[4]) for r in recs] == [3, 2, 0]
    assert bp.unpack_image(recs[0][4])[:3] == (3, 2, bp.PIX_GRAY8)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, "-", e)
    sys.exit(1 if fails else 0)
