"""Run the ValiSight chain on the Jetson: radar -> parse -> gate -> classify.

Uses the board's own modules unchanged (they are CPython-compatible by design)
and resolves ports through /dev/serial/by-id, because /dev/ttyACM* numbering
follows enumeration order and the radar and OpenMV have already swapped places.

    python3 run_system.py [--cfg FILE] [--seconds N] [--no-config]

--no-config skips sending the .cfg, for when the radar is already streaming.
"""
import argparse, glob, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serial
from iwr1843_uart import RadarReader, send_config
import radar_gate
import radar_classify_n6

BY_ID = "/dev/serial/by-id"


def find_radar_ports():
    """(cli, data) for the XDS110: -if00 is the CLI, -if03 the data stream."""
    def one(suffix):
        for p in sorted(glob.glob(os.path.join(BY_ID, "*XDS110*%s" % suffix))):
            return os.path.realpath(p)
        return None
    return one("if00"), one("if03")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "stock_iwr1843.cfg"))
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--no-config", action="store_true")
    args = ap.parse_args()

    cli_port, data_port = find_radar_ports()
    print("CLI  : %s" % cli_port)
    print("DATA : %s" % data_port)
    if not cli_port or not data_port:
        print("radar ports not found under %s" % BY_ID)
        return 1

    if not args.no_config:
        print("\n== sending %s ==" % os.path.basename(args.cfg))
        with serial.Serial(cli_port, 115200, timeout=2) as cfg_ser:
            try:
                send_config(cfg_ser, args.cfg)
            except RuntimeError as e:
                print("\nCONFIG FAILED: %s" % e)
                return 2
        print("config accepted")

    print("\n== reading detections for %.0fs ==" % args.seconds)
    reader = RadarReader()
    n_frames = n_points = n_kept = 0
    labels = {}
    t0 = time.time()

    with serial.Serial(data_port, 921600, timeout=0.1) as ser:
        ser.reset_input_buffer()
        while time.time() - t0 < args.seconds:
            for fr in reader.feed(ser.read(8192)):
                n_frames += 1
                pts = fr["points"]
                n_points += len(pts)
                kept, report = radar_gate.gate_points(pts, fr["snr"], fr["noise"])
                n_kept += len(kept)
                for c in radar_classify_n6.classify_frame(kept):
                    labels[c["label"]] = labels.get(c["label"], 0) + 1
                    if n_frames <= 60 and c["label"] != "static":
                        print("  frame %-6d %-11s range %5.2f m  doppler %+5.2f m/s  "
                              "extent %4.2f m  pts %d"
                              % (fr["frame"], c["label"], c["range_m"],
                                 c["doppler_mps"], c["extent_m"], c["n_points"]))

    print("\n== summary ==")
    print("  frames parsed  : %d" % n_frames)
    print("  points raw     : %d" % n_points)
    print("  points kept    : %d%s" % (n_kept, "" if not n_points else
                                       "  (gate dropped %.0f%%)" % (100.0 * (1 - n_kept / n_points))))
    print("  clusters       : %s" % (labels or "none"))
    if n_frames == 0:
        print("\n  No frames. The radar accepted the config but is not streaming --")
        print("  check that sensorStart succeeded and that DATA is the -if03 port.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
