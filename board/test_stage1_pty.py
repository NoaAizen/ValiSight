#!/usr/bin/env python3
# Feed the C stage1 tool a synthetic radar stream through a pty and check the
# magic count, including a magic deliberately split across two writes.
import os, pty, subprocess, sys, time

MAGIC = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])
FRAME = MAGIC + bytes(range(256)) * 4   # ~1KB fake frame

master, slave = pty.openpty()
proc = subprocess.Popen(
    ["./mmwave_stage1_raw_uart",
     os.ttyname(slave)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(0.3)

sent = 0
# 10 whole frames at ~10 Hz
for _ in range(10):
    os.write(master, FRAME)
    sent += 1
    time.sleep(0.1)

# one magic split across two writes: 3 bytes, pause, remaining 5 + payload
os.write(master, FRAME[:3])
time.sleep(0.05)
os.write(master, FRAME[3:])
sent += 1
time.sleep(0.1)

# garbage that must NOT count (magic bytes shuffled)
os.write(master, bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x07, 0x08]) * 3)
time.sleep(2.2)         # let a stats window fire

proc.terminate()
out, _ = proc.communicate(timeout=5)
print(out[-600:])
ok = f"magics total {sent}" in out
print("EXPECTED", sent, "->", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
