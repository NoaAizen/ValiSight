"""PC bridge: forward radar bytes to the N6, which does the processing.

No radar->N6 wiring: the PC reads the radar DATA UART over its own USB (or
generates synthetic frames) and pushes the raw bytes to the N6 over USB, where
n6_radar_bridge.py (installed as main.py) parses, gates and classifies them and
sends back clusters. The PC just displays.

    python bridge_radar_to_n6.py --synthetic         # no radar: test the N6 path
    python bridge_radar_to_n6.py --radar-port COM7   # live radar over the bridge

The N6 must already be running n6_radar_bridge.py as main.py (see that file's
install steps). This script uses pyserial directly — do NOT run mpremote at the
same time (it would grab the port).
"""
import argparse
import json
import struct
import sys
import time

import serial
from serial.tools import list_ports

OPENMV_VID = 0x37C5
MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"


def find_n6(wait_s=0.0):
    """Find the N6 by USB VID, optionally polling up to wait_s for it to
    re-enumerate (it changes/re-appears after a reset)."""
    deadline = time.time() + wait_s
    while True:
        for p in list_ports.comports():
            if p.vid == OPENMV_VID:
                return p.device
        if time.time() >= deadline:
            return None
        time.sleep(0.5)


def synth_frame(frame_no):
    """A valid IWR1843 TLV frame: a 5-point cluster + one isolated ghost."""
    pts = [(0.3 - 0.05 * i, 3.0 + 0.05 * i, 0.1, 1.0) for i in range(5)]
    pts.append((3.0, 6.0, 0.0, 0.0))                    # isolated ghost
    n = len(pts)
    body = b"".join(struct.pack("<4f", *p) for p in pts)
    tlv1 = struct.pack("<2I", 1, n * 16) + body
    side = b"".join(struct.pack("<2h", 200, 50) for _ in pts)  # snr20/noise5 dB
    tlv7 = struct.pack("<2I", 7, n * 4) + side
    tlvs = tlv1 + tlv7
    hdr = struct.pack("<8I", 1, 40 + len(tlvs), 0, frame_no, 0, n, 2, 0)
    return MAGIC + hdr + tlvs


def send_chunk(n6, payload):
    n6.write(struct.pack("<I", len(payload)) + payload)


def drain(n6, show=True):
    """Print any pending C:/L: lines from the N6."""
    while n6.in_waiting:
        line = n6.readline().strip()
        if not line:
            continue
        if line[:2] == b"C:":
            rec = json.loads(line[2:])
            r = rec["reduction"]
            labels = ",".join(c["label"] for c in rec["clusters"]) or "-"
            print("frame %s: %d clusters [%s]  gate %d->%d (cut %.0f%%)"
                  % (rec["frame"], len(rec["clusters"]), labels,
                     r["in"], r["out"], 100 * r["ratio"]))
        elif show:
            print("[N6]", line.decode(errors="replace"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n6-port", default=None, help="N6 VCP (default: auto)")
    ap.add_argument("--radar-port", default=None, help="radar DATA COM (live)")
    ap.add_argument("--synthetic", action="store_true",
                    help="send synthetic frames instead of live radar")
    ap.add_argument("--frames", type=int, default=8,
                    help="synthetic frame count")
    ap.add_argument("--restore", action="store_true",
                    help="unlock the N6 (restore Ctrl-C, drop to REPL) so you "
                         "can reprogram it / bring the thermal camera back")
    args = ap.parse_args()

    n6_port = args.n6_port or find_n6(wait_s=15.0)   # tolerate reset re-enumeration
    if not n6_port:
        sys.exit("N6 not found — is it plugged in and running n6_radar_bridge?")
    n6 = serial.Serial(n6_port, 115200, timeout=1)

    if args.restore:
        n6.write(struct.pack("<I", 0xFFFFFFFF))      # exit sentinel
        time.sleep(0.5)
        print("sent restore; the N6 dropped to REPL — reprogram with mpremote "
              "(e.g. `mpremote connect %s rm :main.py`)." % n6_port)
        n6.close()
        return 0

    time.sleep(4.0)                    # wait out the device boot grace window
    drain(n6)                          # boot banner / "ready"

    if args.synthetic:
        for i in range(args.frames):
            send_chunk(n6, synth_frame(i))
            time.sleep(0.15)
            drain(n6)
        time.sleep(0.3)
        drain(n6)
        n6.close()
        return 0

    if not args.radar_port:
        sys.exit("give --radar-port COMx (the radar DATA UART) or --synthetic")
    radar = serial.Serial(args.radar_port, 921600, timeout=0.05)
    print("bridging %s (radar) -> %s (N6). Ctrl-C to stop." % (args.radar_port,
                                                               n6_port))
    try:
        while True:
            data = radar.read(4096)
            if data:
                send_chunk(n6, data)
            drain(n6, show=False)
    except KeyboardInterrupt:
        pass
    finally:
        radar.close()
        n6.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
