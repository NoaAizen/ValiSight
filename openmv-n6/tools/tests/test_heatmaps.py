#!/usr/bin/env python3
"""Heatmap TLV parsing (mmwave TLV 2/4/5) against synthetic frames.

Script-style like its siblings: build a wire-exact frame, push it through
FrameSync + walk_tlvs + the parsers, and assert round-trips. The frame is
sized like radar_people's worst case (256 range bins, 64 Doppler bins), which
is precisely the size the old 16 kB FrameSync bound rejected - so the "big
frame survives FrameSync" check is the regression test for that bound.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'radar'))
import mmwave  # noqa: E402

FAILS = []


def check(name, ok):
    print('  %s %s' % ('ok  ' if ok else 'FAIL', name))
    if not ok:
        FAILS.append(name)


def make_frame(frame_no, tlvs):
    """Wire-exact demo frame: magic + 8-word header + TLVs, 32-byte padded."""
    body = b''
    for tlv_type, payload in tlvs:
        body += struct.pack('<2I', tlv_type, len(payload)) + payload
    total = mmwave.FRAME_HEADER_LEN + len(body)
    total_padded = (total + 31) // 32 * 32
    hdr = struct.pack('<8I', 0x03060002, total_padded, 0x1843, frame_no,
                      12345, 0, len(tlvs), 0)
    return mmwave.MAGIC + hdr + body + b'\x00' * (total_padded - total)


N_RANGE, N_VANT, N_DOPPLER = 256, 8, 64

# range profile: bin index as the Q9 value, so the round-trip is exact
rp_payload = struct.pack('<%dH' % N_RANGE, *range(N_RANGE))

# azimuth heatmap: imag = -(range bin), real = antenna index, distinguishable
ra_vals = []
for r in range(N_RANGE):
    for a in range(N_VANT):
        ra_vals += [-r, a]                       # imag FIRST on the wire
ra_payload = struct.pack('<%dh' % len(ra_vals), *ra_vals)

# range-Doppler: value = range*100 + doppler, unique per cell
rd_vals = [min(r * 100 + d, 0xFFFF)
           for r in range(N_RANGE) for d in range(N_DOPPLER)]
rd_payload = struct.pack('<%dH' % len(rd_vals), *rd_vals)

frame = make_frame(7, [(mmwave.TLV_RANGE_PROFILE, rp_payload),
                       (mmwave.TLV_AZIMUTH_HEATMAP, ra_payload),
                       (mmwave.TLV_RANGE_DOPPLER_HEATMAP, rd_payload)])
follower = make_frame(8, [(mmwave.TLV_RANGE_PROFILE, rp_payload)])

print('framing')
check('worst-case heatmap frame is %d B (over the old 16 kB bound)'
      % len(frame), len(frame) > 16 * 1024)
fs = mmwave.FrameSync()
got = fs.feed(frame + follower, t=1.0)
check('FrameSync accepts the 33 kB frame', len(got) >= 1)
check('no resyncs on clean input', fs.resync_count == 0)

tlvs, mode = mmwave.walk_tlvs(got[0][1])
check('three TLVs walked (%s lengths)' % mode, len(tlvs) == 3)

print('parsers')
rp = mmwave.parse_range_profile(rp_payload)
check('range profile round-trips', rp == list(range(N_RANGE)))
check('Q9 constant: value 512 is one log2 = ~6.02 dB',
      abs(512 * mmwave.Q9_LOG2_TO_DB - 6.0206) < 1e-3)

n_vant, rows = mmwave.parse_azimuth_heatmap(ra_payload, N_RANGE)
check('RA antenna count derived as %d' % n_vant, n_vant == N_VANT)
check('RA wire order is (imag, real)',
      rows[3][5] == (-3, 5) and rows[200][0] == (-200, 0))
try:
    mmwave.parse_azimuth_heatmap(ra_payload[:-4], N_RANGE)
    check('RA rejects a non-dividing payload', False)
except ValueError:
    check('RA rejects a non-dividing payload', True)

n_dop, rd = mmwave.parse_range_doppler_heatmap(rd_payload, N_RANGE)
check('RD Doppler count derived as %d' % n_dop, n_dop == N_DOPPLER)
check('RD is range-major', rd[2][3] == 203 and rd[100][0] == 10000)

print()
if FAILS:
    print('%d FAILED: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('all heatmap parser checks passed')
