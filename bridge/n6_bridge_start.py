#!/usr/bin/env python3
"""Push the bridge sender to the OpenMV N6 over raw REPL and start it.

    n6_bridge_start.py                       # start the sender, then exit (C receiver opens the port)
    n6_bridge_start.py --record out.bin --seconds 10   # also read the stream: stats + raw dump

Raw-REPL handling (drain-while-interrupting, paced paste) is the routine that
survived live fire in hagai-live3/view_thermal_rgb.py — same reasons, see there.
bridge_protocol.py is written to the board's flash first so the sender can
`import` it; the sender itself is exec'd (not stored), so a reboot returns the
board to a clean state.
"""
import argparse, glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bridge_protocol as bp

HERE = os.path.dirname(os.path.abspath(__file__))
N6_GLOB = "/dev/serial/by-id/usb-MicroPython_Pyboard_Virtual_Comm_Port*"


def find_port():
    hits = glob.glob(N6_GLOB)
    if len(hits) != 1:
        sys.exit("expected exactly one N6 VCP, found %r — pass --port" % hits)
    return hits[0]


class RawRepl:
    def __init__(self, port):
        import serial
        self.ser = serial.Serial(port, 115200, timeout=0.2, write_timeout=2)
        deadline, sent = time.time() + 5.0, False
        while time.time() < deadline:
            n = self.ser.in_waiting
            if n:
                self.ser.read(n)
            if not sent:
                try:
                    self.ser.write(b"\x03\x03"); sent = True
                except serial.SerialTimeoutException:
                    continue
            elif not n:
                break
            time.sleep(0.05)
        if not sent:
            raise RuntimeError("the N6 is not reading USB (wedged) — replug its cable and rerun")
        time.sleep(0.4)
        self.ser.reset_input_buffer()
        self.ser.write(b"\x01"); time.sleep(0.3)      # raw REPL
        self.ser.reset_input_buffer()

    def exec_and_wait(self, code, timeout=5.0):
        """Run `code`, wait for it to finish, return (stdout, stderr)."""
        self._paste(code)
        self.ser.write(b"\x04")
        buf, deadline = b"", time.time() + timeout
        while time.time() < deadline and buf.count(b"\x04") < 2:
            buf += self.ser.read(self.ser.in_waiting or 1)
        assert buf.startswith(b"OK"), "board did not ACK: %r" % buf[:80]
        out, _, rest = buf[2:].partition(b"\x04")
        err = rest.split(b"\x04")[0]
        return out, err

    def exec_background(self, code):
        """Run `code` and return immediately (it keeps running on the board)."""
        self._paste(code)
        self.ser.write(b"\x04")
        ack = self.ser.read(2)
        assert ack == b"OK", "board did not ACK: %r" % ack

    def _paste(self, code):
        for i in range(0, len(code), 256):            # no flow control: 256 B / 10 ms is safe here
            self.ser.write(code[i:i + 256]); time.sleep(0.01)


def put_file(repl, name, data):
    code = "f=open(%r,'wb')\nf.write(%r)\nf.close()\nprint('put',%r,%d)\n" % (name, data, name, len(data))
    out, err = repl.exec_and_wait(code.encode())
    if err:
        raise RuntimeError("writing %s failed: %r" % (name, err))
    return out.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None)
    ap.add_argument("--record", default=None, help="read the stream and dump raw bytes here")
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()
    port = a.port or find_port()

    proto = open(os.path.join(HERE, "bridge_protocol.py"), "rb").read()
    tx = open(os.path.join(HERE, "n6_bridge_tx.py"), "rb").read()
    repl = RawRepl(port)
    print(put_file(repl, "bridge_protocol.py", proto).decode())
    # MicroPython keeps imported modules cached across raw-REPL execs, so a
    # sender started earlier would still see the OLD bridge_protocol (bit us
    # 2026-08-17: TypeError from a stale hello_payload). Drop the cache first.
    repl.exec_and_wait(b"import sys\nsys.modules.pop('bridge_protocol', None)\nprint('cache cleared')\n")
    print("starting sender on", port)
    repl.exec_background(tx + b"\nmain()\n")

    if not a.record:
        repl.ser.close()
        print("sender running; open the port with the receiver now")
        return

    ser, dec = repl.ser, bp.Decoder()
    ser.timeout = 0.05
    counts = {bp.T_HELLO: 0, bp.T_THERMAL: 0, bp.T_RGB: 0, bp.T_IMU: 0}
    names = {bp.T_HELLO: "hello", bp.T_THERMAL: "thermal", bp.T_RGB: "rgb", bp.T_IMU: "imu"}
    last_seq, gaps, nbytes, first_ts = None, 0, 0, None
    t0 = time.time()
    with open(a.record, "wb") as f:
        while time.time() - t0 < a.seconds:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            f.write(chunk); nbytes += len(chunk)
            for rtype, ts_src, seq, ts, payload in dec.feed(chunk):
                counts[rtype] = counts.get(rtype, 0) + 1
                if last_seq is not None and seq != ((last_seq + 1) & 0xFFFFFFFF):
                    gaps += 1
                last_seq = seq
                if rtype == bp.T_HELLO:
                    ver, drops = bp.unpack_hello(payload)
                    print("  hello: proto v%d, sender_drops=%d" % (ver, drops))
    el = time.time() - t0
    print("--- %.1f s, %.1f KB/s" % (el, nbytes / el / 1024))
    for k in sorted(counts):
        print("  %-8s %5d  (%.1f Hz)" % (names.get(k, k), counts[k], counts[k] / el))
    print("  seq gaps: %d   bad crc: %d   resyncs: %d" % (gaps, dec.bad_crc, dec.resyncs))
    ser.write(b"\x03\x03"); ser.close()
    print("stopped sender; raw stream saved to", a.record)


if __name__ == "__main__":
    main()
