#!/usr/bin/env python3
"""Minimal raw-REPL runner for MicroPython/OpenMV over a USB VCP.

Usage:  mpx.py <script.py> [port]
        mpx.py -c "print(1)" [port]
"""
import os, sys, time, serial

PORT = "/dev/ttyACM0"


def run(code, port=PORT, timeout=float(os.environ.get("MPX_TIMEOUT", 120))):
    s = serial.Serial(port, 115200, timeout=0.2, write_timeout=5)
    try:
        s.write(b"\r\x03\x03")          # interrupt any running script
        time.sleep(0.3)
        s.reset_input_buffer()
        s.write(b"\x01")                # enter raw REPL
        time.sleep(0.3)
        s.read_all()

        s.write(code.encode() + b"\x04")  # submit + EOF

        buf, deadline = b"", time.time() + timeout
        while time.time() < deadline:
            buf += s.read(4096)
            if buf.count(b"\x04") >= 2:
                break
            time.sleep(0.05)

        s.write(b"\x02")                # back to friendly REPL
        return buf
    finally:
        s.close()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "-c":
        code, rest = args[1], args[2:]
    else:
        code, rest = open(args[0]).read(), args[1:]
    port = rest[0] if rest else PORT

    raw = run(code, port)
    text = raw.decode("utf-8", "replace")
    if text.startswith("OK"):
        text = text[2:]
    out, _, tail = text.partition("\x04")
    err = tail.partition("\x04")[0]

    sys.stdout.write(out)
    if err.strip():
        sys.stdout.write("\n--- ERROR ---\n" + err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
