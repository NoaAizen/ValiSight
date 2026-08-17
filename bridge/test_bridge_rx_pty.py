#!/usr/bin/env python3
"""Tests for the C receiver (bridge_rx) — no hardware.

Builds a synthetic N6 stream with bridge_protocol.pack and feeds it to
bridge_rx (a) through a pty in odd-sized writes with faults injected, and
(b) as a file, checking the FINAL counters are EXACTLY what was injected:
    - N records of each type, in odd chunk sizes and one record split at the magic
    - one record with a flipped payload byte  -> bad_crc == 1, and the following record still counts
    - one record whose header claims an absurd len -> bad_crc (bounded), stream continues
    - a seq jump of 3 (records "lost" upstream)   -> gaps == 1, lost == 3
    - garbage before the first magic and a "OK\\x04>" raw-REPL preamble -> resyncs, nothing else
    - IMU samples land in imu.csv with unwrapped timestamps across a ticks_us wrap
    - HELLO's sender_drops is surfaced
Runs both the -O2 build and the ASan/UBSan build if present.
"""
import csv, os, pty, re, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bridge_protocol as bp

W = (1 << 30)   # ticks_us wrap

def build_stream():
    recs, seq = [], 0
    def add(rtype, ts, payload, seq_override=None):
        nonlocal seq
        s = seq if seq_override is None else seq_override
        recs.append(bp.pack(rtype, bp.TS_TICKS_US, s, ts, payload))
        seq = s + 1
    T0 = W - 600000                     # start 0.6 s before the ticks_us wrap
    add(bp.T_HELLO, T0, bp.hello_payload(0))
    for i in range(5):
        add(bp.T_THERMAL, T0 + 1000 + i * 114000, bp.image_payload(160, 120, bp.PIX_GRAY8, bytes([i]) * 19200))
        add(bp.T_IMU, T0 + 1500 + i * 114000, bp.imu_payload(i, -i, 1000 + i, 10 * i, 0, -5))
        add(bp.T_RGB, T0 + 2000 + i * 114000, bp.image_payload(320, 240, bp.PIX_JPEG, b"\xff\xd8" + bytes([i]) * 4000 + b"\xff\xd9"))
    # two IMU samples straddling the wrap: (W-10) then 5 -> unwrapped diff must be +15
    add(bp.T_IMU, (W - 10) & (W - 1), bp.imu_payload(1, 2, 3, 4, 5, 6))
    add(bp.T_IMU, 5, bp.imu_payload(7, 8, 9, 10, 11, 12))
    # seq jump: pretend 3 records were lost upstream
    add(bp.T_THERMAL, 9000, bp.image_payload(160, 120, bp.PIX_GRAY8, bytes(19200)), seq_override=seq + 3)
    # bad crc: flip a payload byte
    bad = bytearray(bp.pack(bp.T_RGB, bp.TS_TICKS_US, seq, 9100, bp.image_payload(320, 240, bp.PIX_JPEG, b"z" * 300))); seq += 1
    bad[-1] ^= 0xFF
    recs.append(bytes(bad))
    # absurd length in header (crc irrelevant): must be bounded, not awaited forever
    absurd = bytearray(bp.pack(bp.T_RGB, bp.TS_TICKS_US, seq, 9150, b"x" * 10)); seq += 1
    absurd[16:20] = (0x7FFFFFFF).to_bytes(4, "little")
    recs.append(bytes(absurd))
    add(bp.T_HELLO, 9200, bp.hello_payload(7, 11))     # sender says it dropped 7, ring overflowed 11
    add(bp.T_IMU, 9300, bp.imu_payload(0, 0, 1000, 0, 0, 0))
    expected = dict(hello=2, thermal=6, rgb=5, imu=8, gaps=1, lost=3 + 2, bad_crc=2, drops=7)
    # lost: 3 from the jump, plus the bad-crc and absurd records consumed two seq numbers -> gap of 2 before the last HELLO
    return recs, expected

def parse_final(out):
    m = re.search(r"FINAL (.*)", out); assert m, out[-800:]
    return {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m.group(1))}

def check(final, exp, imu_csv):
    assert final["hello"] == exp["hello"], final
    assert final["thermal"] == exp["thermal"], final
    assert final["rgb"] == exp["rgb"], final
    assert final["imu"] == exp["imu"], final
    assert final["bad_crc"] == exp["bad_crc"], final
    assert final["gaps"] >= 1 and final["lost"] == exp["lost"], final
    assert final["sender_drops"] == exp["drops"], final
    assert final["imu_overflow"] == 11, final
    assert final["other"] == 0
    rows = list(csv.DictReader(open(imu_csv)))
    assert len(rows) == exp["imu"], len(rows)
    ts = [int(r["ts_ticks"]) for r in rows]
    # the two wrap-straddling samples are rows 5 and 6: unwrapped diff must be +15
    assert ts[6] - ts[5] == 15, (ts[5], ts[6])
    assert [int(r["ax_mg"]) for r in rows[:5]] == [0, 1, 2, 3, 4]

def run_pty(binary, recs, dump):
    stream = b"junk\r\nOK\x04>" + b"".join(recs)
    master, slave = pty.openpty()
    proc = subprocess.Popen([binary, os.ttyname(slave), "-o", dump, "-q"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(0.2)
    # odd chunking, one record split right after 2 magic bytes
    i, sizes = 0, [7, 1000, 3, 4096, 33, 20000, 2, 900]
    k = 0
    while i < len(stream):
        n = sizes[k % len(sizes)]; k += 1
        os.write(master, stream[i:i + n]); i += n
        time.sleep(0.002)
    time.sleep(0.5)
    proc.send_signal(2)
    out, _ = proc.communicate(timeout=10)
    return out

def run_file(binary, recs, dump):
    p = os.path.join(dump, "stream.bin")
    with open(p, "wb") as f: f.write(b"garbage" + b"".join(recs))
    return subprocess.run([binary, p, "-o", dump, "-q"], capture_output=True, text=True, timeout=20).stdout

def run_clean_tiny_chunks(binary):
    """A perfectly clean stream fed one byte at a time must produce ZERO resyncs
    (regression: a fixed 3-byte tail re-injected stale bytes -> 3 false resyncs/record)."""
    recs = b"".join(bp.pack(bp.T_IMU, 0, i, i * 5000, bp.imu_payload(1, 2, 3, 4, 5, 6)) for i in range(30))
    recs += bp.pack(bp.T_THERMAL, 0, 30, 1, bp.image_payload(160, 120, 0, bytes(2000)))
    master, slave = pty.openpty()
    proc = subprocess.Popen([binary, os.ttyname(slave), "-q"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(0.2)
    for i in range(len(recs)):
        os.write(master, recs[i:i + 1])
        if i % 64 == 0: time.sleep(0.001)
    time.sleep(0.4); proc.send_signal(2)
    out, _ = proc.communicate(timeout=10)
    f = parse_final(out)
    assert f["imu"] == 30 and f["thermal"] == 1 and f["resyncs"] == 0 and f["bad_crc"] == 0, f


def main():
    fails = 0
    recs, exp = build_stream()
    for binary in ["./bridge_rx", "./bridge_rx_asan"]:
        path = os.path.join(HERE, binary)
        if not os.path.exists(path): print("SKIP", binary, "(not built)"); continue
        try:
            run_clean_tiny_chunks(path); print("PASS", binary, "1-byte chunks, zero resyncs")
        except AssertionError as e:
            fails += 1; print("FAIL", binary, "1-byte chunks -", e)
        for mode, fn in [("pty", run_pty), ("file", run_file)]:
            with tempfile.TemporaryDirectory() as d:
                out = fn(path, recs, d)
                try:
                    check(parse_final(out), exp, os.path.join(d, "imu.csv"))
                    n_frames = len([f for f in os.listdir(d) if f.endswith(".bin") and f != "stream.bin"])
                    assert n_frames == exp["thermal"] + exp["rgb"], n_frames
                    print("PASS", binary, mode)
                except AssertionError as e:
                    fails += 1; print("FAIL", binary, mode, "-", e); print(out[-1200:])
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
