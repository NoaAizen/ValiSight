# Runs ON the N6 via: mpremote connect COM12 mount . run n6_selftest_radar.py
# Feeds a synthetic IWR1843 TLV frame through the real device pipeline.
import struct, json
from iwr1843_uart import RadarReader
from radar_classify_n6 import classify_frame
try:
    from core.radar_gate import gate_points
except ImportError:
    gate_points = None

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
pts = [(0.3 - 0.05 * i, 3.0 + 0.05 * i, 0.1, 1.0) for i in range(5)]  # TI (x_r,y_f,z,v)
pts.append((3.0, 6.0, 0.0, 0.0))                                     # isolated ghost
n_obj = len(pts)
body = b"".join(struct.pack("<4f", *p) for p in pts)
tlv1 = struct.pack("<2I", 1, n_obj * 16) + body
sn = b"".join(struct.pack("<2h", 200, 50) for _ in pts)             # snr20,noise5 dB
tlv7 = struct.pack("<2I", 7, n_obj * 4) + sn
tlvs = tlv1 + tlv7
total_len = 40 + len(tlvs)
hdr = struct.pack("<8I", 1, total_len, 0, 42, 0, n_obj, 2, 0)
frame = MAGIC + hdr + tlvs

radar = RadarReader()
for fr in radar.feed(frame):
    p = fr["points"]
    if gate_points:
        kept, rep = gate_points(p, fr["snr"], fr["noise"])
        print("GATE in=%d out=%d ratio=%.2f" % (rep.n_in, rep.n_out, rep.reduction_ratio))
    else:
        kept = p
    print("CLUSTERS:" + json.dumps(classify_frame(kept)))
print("N6_RADAR_PIPELINE_OK")
