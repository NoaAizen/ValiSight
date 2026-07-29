"""Is the IWR1843 already configured and streaming?

Only read-only traffic: 'version' on the CLI is a query, and the DATA port is
just listened to. Nothing is configured or started here -- that needs the
project's .cfg file.
"""
import glob, os, serial, sys, time

BY_ID = "/dev/serial/by-id"
MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"


def find(suffix):
    for p in glob.glob(os.path.join(BY_ID, "*XDS110*%s" % suffix)):
        return os.path.realpath(p)
    return None


cli, data = find("if00"), find("if03")
print("CLI  port :", cli)
print("DATA port :", data)
if not cli or not data:
    raise SystemExit("radar ports not found under %s" % BY_ID)

print()
print("== CLI: version ==")
s = serial.Serial(cli, 115200, timeout=0.5)
s.reset_input_buffer()
s.write(b"version\n")
time.sleep(1.0)
resp = s.read(2048).decode("ascii", "ignore")
s.close()
print("\n".join("  " + l for l in resp.strip().splitlines()) or "  <no response>")

print()
print("== DATA: listening 5s ==")
d = serial.Serial(data, 921600, timeout=0.5)
d.reset_input_buffer()
buf, t0 = b"", time.time()
while time.time() - t0 < 5.0:
    buf += d.read(8192)
d.close()
print("  bytes    :", len(buf))
print("  magics   :", buf.count(MAGIC))
if buf.count(MAGIC):
    print("  -> radar IS configured and streaming")
else:
    print("  -> silent: needs a .cfg sent over the CLI port (sensorStart)")
