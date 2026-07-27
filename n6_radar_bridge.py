"""Radar processing on the N6, fed by the PC over USB — no radar->N6 wiring.

Installed on the N6 as main.py. It reads length-framed radar byte chunks from
its USB VCP (sent by bridge_radar_to_n6.py on the PC), runs the FULL radar
pipeline on the board — parse (RadarReader) -> gate (radar_gate, drop ghosts) ->
cluster+classify (radar_classify_n6) — and prints the resulting clusters back
over USB. The PC is only a data pipe; all the processing is on the N6.

Wire protocol on the VCP:
    PC -> N6 :  <u32 little-endian length><length bytes of raw radar stream>
    N6 -> PC :  L:<text>    a log line
                C:<json>    {"frame", "clusters":[...], "reduction":{...}}

Install (one time), then reset so it autostarts:
    mpremote connect COM12 cp iwr1843_uart.py :iwr1843_uart.py
    mpremote connect COM12 cp radar_classify_n6.py :radar_classify_n6.py
    mpremote connect COM12 cp core/radar_gate/gate.py :radar_gate.py
    mpremote connect COM12 cp n6_radar_bridge.py :main.py
    mpremote connect COM12 reset
To remove:  mpremote connect COM12 rm :main.py
"""
import sys
import time
import struct
import json
import micropython

from iwr1843_uart import RadarReader
from radar_classify_n6 import classify_frame
from radar_gate import gate_points          # flat copy of core/radar_gate/gate.py

_read = sys.stdin.buffer.read
MAX_CHUNK = 65536
BOOT_GRACE_S = 3        # window with Ctrl-C still live, so mpremote can reprogram
EXIT_SENTINEL = 0xFFFFFFFF   # a length of 0xFFFFFFFF = "restore Ctrl-C, drop to REPL"


def read_exact(n):
    """Block until exactly n bytes are read. On the N6 the VCP read is
    non-blocking and returns empty when no data is queued yet, so we WAIT on
    empty rather than treating it as EOF — otherwise main() would fall through
    to the REPL at boot before the PC has sent anything."""
    out = b""
    while len(out) < n:
        c = _read(n - len(out))
        if c:
            out += c
        else:
            time.sleep_ms(2)                  # no data yet — keep waiting
    return out


def main():
    # Grace window: keep Ctrl-C alive for a few seconds so a host (mpremote /
    # OpenMV IDE) can interrupt and reprogram before we go binary-only. After
    # this, radar bytes containing 0x03 would otherwise raise KeyboardInterrupt.
    print("L:bridge starting in %ds (Ctrl-C now for REPL)" % BOOT_GRACE_S)
    time.sleep(BOOT_GRACE_S)
    micropython.kbd_intr(-1)
    radar = RadarReader()
    print("L:N6 radar bridge ready")
    while True:
        (length,) = struct.unpack("<I", read_exact(4))
        if length == EXIT_SENTINEL:
            micropython.kbd_intr(3)           # restore Ctrl-C so mpremote works
            print("L:bridge exiting to REPL (reprogrammable now)")
            return
        if length == 0 or length > MAX_CHUNK:
            continue                          # framing desync — skip
        payload = read_exact(length)
        for fr in radar.feed(payload):
            kept, rep = gate_points(fr["points"], fr["snr"], fr["noise"])
            clusters = classify_frame(kept)
            print("C:" + json.dumps({
                "frame": fr["frame"], "clusters": clusters,
                "reduction": {"in": rep.n_in, "out": rep.n_out,
                              "ratio": rep.reduction_ratio,
                              "weak": rep.dropped_weak, "fov": rep.dropped_fov,
                              "isolated": rep.dropped_isolated}}))


main()
