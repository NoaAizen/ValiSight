#!/usr/bin/env python3
"""Exercise radar/mmwave.py against synthetic frames, with no radar attached.

    ./test_radar.py

Everything the parser can get wrong here is silent: a coordinate swap, a TLV
length read the wrong way, a byte lost on the wire. None of them raise, and all
of them survive into the calibration as a plausible-looking number. So the
tests are built around the wire format rather than around the happy path --
frames are constructed byte by byte, damaged deliberately, and the parser is
asked to notice.

What this cannot cover is the link itself. That is radar_listen.py --stage1 on
the host and radar_stage1_n6.py on the board, and both need hardware.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'radar'))
import mmwave  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print('  ok   %s' % name)
    else:
        print('  FAIL %s %s' % (name, detail))
        FAILS.append(name)


# What the radar's padding actually looks like, lifted from a captured frame:
# stale transmit-buffer content, not zeros. Tests that pad with zeros pass
# against a parser that is wrong about this, which is how it survived the
# first round.
STALE_PAD = b'\x00\x03\x13\x00\x00\x00\x00\x00y\xde\x00\x00\x01\x00\x00\x00' \
            b'\xe8\x3d\x00\x08\x10\x00\x00\x00'


def build_frame(points_ti=(), side=None, frame_no=7, mode='payload', pad=0,
                stats=False, temperature=False, temperature_rc=0):
    """Assemble a demo frame. points_ti are (x_right, y_fwd, z_up, v).

    mode picks which of the two SDK readings of the TLV length field to encode,
    so the parser's empirical resolution is tested against both, not just the
    one this author happens to believe.
    """
    body = b''
    n_tlv = 0
    if points_ti:
        payload = b''.join(struct.pack('<4f', *p) for p in points_ti)
        length = len(payload) + (8 if mode == 'includes_header' else 0)
        body += struct.pack('<2I', mmwave.TLV_POINTS, length) + payload
        n_tlv += 1
    if side is not None:
        payload = b''.join(struct.pack('<2h', int(round(s * 10)),
                                       int(round(nz * 10))) for s, nz in side)
        length = len(payload) + (8 if mode == 'includes_header' else 0)
        body += struct.pack('<2I', mmwave.TLV_SIDE_INFO, length) + payload
        n_tlv += 1
    if stats:
        payload = struct.pack('<6I', 4300, 300, 95000, 200, 12, 8)
        body += struct.pack('<2I', mmwave.TLV_STATS,
                            len(payload) + (8 if mode == 'includes_header' else 0))
        body += payload
        n_tlv += 1
    if temperature:
        # rc = 0 is what a HEALTHY radar sends -- RL_RET_CODE_OK. Encoding 1
        # here, as this builder first did, tests the inverted reading against
        # itself and passes.
        payload = struct.pack('<iI', temperature_rc, 55123) + struct.pack(
            '<10h', 41, 42, 43, 44, 45, 46, 47, 48, 49, 50)
        body += struct.pack('<2I', mmwave.TLV_TEMPERATURE,
                            len(payload) + (8 if mode == 'includes_header' else 0))
        body += payload
        n_tlv += 1
    body += STALE_PAD[:pad]
    total = mmwave.FRAME_HEADER_LEN + len(body)
    hdr = mmwave.MAGIC + struct.pack('<8I', 0x03040000, total, 0x000A1843,
                                     frame_no, 12345, len(points_ti), n_tlv, 0)
    return hdr + body


# --- framing ------------------------------------------------------------

print('framing')

def bodies(out):
    """feed() returns (t_complete, frame); most checks only care about frames."""
    return [f for _, f in out]


pkt = build_frame([(1.0, 2.0, 0.5, -0.3)], side=[(15.0, 90.0)])
fs = mmwave.FrameSync()
check('one frame in one feed', bodies(fs.feed(pkt)) == [pkt])

fs = mmwave.FrameSync()
check('split across feeds, part 1', fs.feed(pkt[:20]) == [])
check('split across feeds, part 2', bodies(fs.feed(pkt[20:])) == [pkt])

fs = mmwave.FrameSync()
check('split mid-magic',
      fs.feed(pkt[:3]) == [] and bodies(fs.feed(pkt[3:])) == [pkt])

fs = mmwave.FrameSync()
frames = fs.feed(b'\xde\xad\xbe\xef garbage \x13' + pkt)
check('resync after leading garbage', bodies(frames) == [pkt])
check('leading garbage counted', fs.dropped_bytes == 14,
      '(got %d)' % fs.dropped_bytes)

a = build_frame([(0.0, 1.0, 0.0, 0.0)], frame_no=1)
b = build_frame([(0.0, 2.0, 0.0, 0.0)], frame_no=2)
fs = mmwave.FrameSync()
check('two frames in one feed', bodies(fs.feed(a + b)) == [a, b])

# The magic word is data, not a delimiter: eight bytes of a float payload can
# spell it. Framing must never look inside a frame it has already accepted.
evil = struct.unpack('<2f', mmwave.MAGIC)
pkt_evil = build_frame([evil + (0.0, 0.0)])
fs = mmwave.FrameSync()
frames = fs.feed(pkt_evil + b)
check('magic inside a payload is not a frame start',
      bodies(frames) == [pkt_evil, b])

# One byte lost on the wire, which is exactly what a UART with no DMA does.
# Every declared TLV length still sums to totalPacketLen, so the walk alone
# cannot see it -- only the missing successor magic can. Accepting the frame
# would hand downstream floats shifted by one byte AND eat the next frame.
damaged = a[:60] + a[61:]
fs = mmwave.FrameSync()
frames = fs.feed(damaged + b)
check('a dropped byte kills only its own frame', bodies(frames) == [b],
      '(got %d frames)' % len(frames))
check('the kill is counted as a resync', fs.resync_count >= 1)

corrupt = bytearray(a)
struct.pack_into('<I', corrupt, 12, 0xFFFFFF)      # implausible totalPacketLen
fs = mmwave.FrameSync()
check('implausible length resyncs to the next frame',
      bodies(fs.feed(bytes(corrupt) + b)) == [b])

# Ambiguity: a frame followed by three bytes that could be the next magic. It
# is held until the fourth byte decides, and then released.
fs = mmwave.FrameSync()
check('an ambiguous tail holds the frame', fs.feed(a + b[:3]) == [])
check('and releases it once decided', bodies(fs.feed(b[3:])) == [a, b])

# The timestamp is when the bytes arrived, not when validation let go of them.
# Anything else silently adds a frame period to the radar-camera time offset.
fs = mmwave.FrameSync()
fs.feed(a + b[:3], t=100.0)
out = fs.feed(b[3:], t=100.5)
check('t_complete is arrival, not release', out[0][0] == 100.0,
      '(got %r)' % (out[0][0],))
check('the successor is stamped at its own arrival', out[1][0] == 100.5)

# flush() exists for a finite source. The case it must handle is the ONLY case
# feed() holds: a complete frame followed by 1..7 bytes that could be the next
# magic. Requiring no trailing bytes at all -- which an earlier version did --
# makes flush() dead code, because feed() has already released that case.
fs = mmwave.FrameSync()
check('feed holds a frame with an ambiguous tail', fs.feed(a + b'\x02\x01', t=7.0) == [])
out = fs.flush()
check('flush releases it at end of stream', bodies(out) == [a],
      '(got %d frames -- flush is dead code)' % len(out))
check('flush keeps the arrival stamp', out and out[0][0] == 7.0)
check('flush counts the truncated tail as dropped', fs.dropped_bytes >= 2)
check('flush is idempotent', fs.flush() == [])

fs = mmwave.FrameSync()
fs.feed(a[:30], t=1.0)
check('flush releases nothing half-arrived', fs.flush() == [])

fs = mmwave.FrameSync(max_buffer=1024)
fs.feed(b'\x00' * 4096)
check('runaway buffer is bounded', fs.dropped_bytes >= 3072)


# --- header and TLVs ----------------------------------------------------

print('header and TLVs')

hdr = mmwave.parse_header(pkt)
check('frame number', hdr['frame_number'] == 7)
check('num_detected_obj', hdr['num_detected_obj'] == 1)
check('num_tlvs', hdr['num_tlvs'] == 2)
check('total_len is the frame length', hdr['total_len'] == len(pkt))

try:
    mmwave.parse_header(b'not a frame at all, but long enough' * 2)
    check('a non-frame raises', False)
except ValueError:
    check('a non-frame raises', True)

tlvs, mode = mmwave.walk_tlvs(pkt)
check('TLV types', [t for t, _ in tlvs] == [1, 7])
check('length mode: payload', mode == 'payload')

tlvs, mode = mmwave.walk_tlvs(build_frame([(1.0, 2.0, 0.5, -0.3)],
                                          side=[(15.0, 90.0)],
                                          mode='includes_header'))
check('length mode: includes_header', mode == 'includes_header')
check('same payloads either way', len(tlvs[0][1]) == 16 and len(tlvs[1][1]) == 4)

# The padding the radar really emits is stale buffer content. Insisting it be
# zeros is what rejected 70% of live frames while the link measured clean.
padded = build_frame([(1.0, 2.0, 0.5, 0.0)], pad=12)
check('non-zero padding tolerated', mmwave.walk_tlvs(padded)[0][0][0] == 1)
check('padded frames still frame',
      bodies(mmwave.FrameSync().feed(padded + b)) == [padded, b])
# The tail's content is unconstrained, so its LENGTH is the only check left.
# A frame claiming 40 bytes more than its TLVs account for is a corrupt length,
# not padding.
overrun = bytearray(a)
struct.pack_into('<I', overrun, 12, len(a) + 40)
check('a tail longer than the 32-byte pad is rejected',
      bodies(mmwave.FrameSync().feed(bytes(overrun) + b'\xaa' * 40 + b)) == [b])

# The live radar emits four TLVs -- 1, 7, 6, 9 -- not the two the wire spec
# examples show. An extra TLV must not disturb the points or the framing.
real = build_frame([(1.0, 2.0, 0.5, 0.0)], side=[(27.5, 49.9)],
                   stats=True, temperature=True, pad=20)
fr = mmwave.parse_frame(real)
check('four-TLV frame walks', fr['tlv_types'] == [1, 7, 6, 9],
      '(got %r)' % fr['tlv_types'])
check('points survive the extra TLVs', abs(fr['points'][0]['x'] - 2.0) < 1e-6)
check('snr survives the extra TLVs', abs(fr['points'][0]['snr'] - 27.5) < 1e-6)
check('stats decoded', fr['stats']['interframe_margin_us'] == 95000)
check('temperature decoded', fr['temperature']['rx_c'] == [41, 42, 43, 44])
# tempReportValid is a return code: 0 = RL_RET_CODE_OK = the report is good.
# Confirmed on the live radar -- 251 consecutive frames all carried 0, with die
# temperatures of 42-45 C. Reading it as a boolean inverts the whole log.
check('rc = 0 means valid', fr['temperature']['valid'] is True,
      '(0 is RL_RET_CODE_OK -- bool(rc) is backwards)')
bad = mmwave.parse_frame(build_frame([(1.0, 2.0, 0.5, 0.0)], temperature=True,
                                     temperature_rc=3))
check('a non-zero rc means invalid', bad['temperature']['valid'] is False)
check('the raw return code is kept', bad['temperature']['return_code'] == 3)


# --- points -------------------------------------------------------------

print('points')

fr = mmwave.parse_frame(pkt)
p = fr['points'][0]
# TI x=right y=forward -> project x=forward y=left. A target 1 m to the RIGHT
# and 2 m AHEAD is (fwd 2, left -1).
check('coordinate convention', (p['x'], p['y'], p['z']) == (2.0, -1.0, 0.5),
      '(got %r)' % ((p['x'], p['y'], p['z']),))
check('velocity is passed through', abs(p['v'] + 0.3) < 1e-6)
check('snr in dB, not raw counts', abs(p['snr'] - 15.0) < 1e-6,
      '(got %r -- wire units are 0.1 dB)' % p['snr'])
check('noise in dB', abs(p['noise'] - 90.0) < 1e-6)

fr_ti = mmwave.parse_frame(pkt, convention='ti')
q = fr_ti['points'][0]
check('ti convention is untouched', (q['x'], q['y'], q['z']) == (1.0, 2.0, 0.5))

# Side info missing entirely: points still parse, and snr stays None rather
# than 0.0, because 0 dB SNR is a real reading and None is not.
fr = mmwave.parse_frame(build_frame([(1.0, 2.0, 0.5, 0.0)]))
check('no side info leaves snr None', fr['points'][0]['snr'] is None)

# Short side info: the demo has emitted mismatched counts. Join by index as far
# as it goes, leave the rest None, never raise.
fr = mmwave.parse_frame(build_frame([(1.0, 2.0, 0.0, 0.0), (0.0, 3.0, 0.0, 0.0)],
                                    side=[(20.0, 80.0)]))
check('short side info joins what it can',
      fr['points'][0]['snr'] == 20.0 and fr['points'][1]['snr'] is None)

check('zero points is a valid frame',
      mmwave.parse_frame(build_frame([]))['points'] == [])


# --- byte accounting (the stage-1 gate) ---------------------------------

print('byte accounting')

stream = b''.join(build_frame([(0.0, float(i), 0.0, 0.0)], frame_no=i)
                  for i in range(1, 6))
scan = mmwave.MagicScanner()
scan.feed(stream)
check('every magic found', scan.magics == 5, '(got %d)' % scan.magics)
check('every gap matches totalPacketLen', scan.bad_gaps == 0 and scan.good_gaps == 4)
check('nothing reported lost', scan.lost_bytes == 0)

scan = mmwave.MagicScanner()
for i in range(0, len(stream), 7):          # arbitrary chunking, incl. mid-header
    scan.feed(stream[i:i + 7])
check('chunking does not change the count', scan.magics == 5 and scan.bad_gaps == 0,
      '(magics %d, bad %d)' % (scan.magics, scan.bad_gaps))

lossy = stream[:80] + stream[81:]           # one byte gone, as the ring buffer does
scan = mmwave.MagicScanner()
scan.feed(lossy)
check('a single lost byte is detected', scan.bad_gaps == 1,
      '(bad_gaps %d)' % scan.bad_gaps)
check('the loss is signed and exact', scan.lost_bytes == -1,
      '(got %d)' % scan.lost_bytes)


# --- Doppler and physics ------------------------------------------------

print('Doppler and physics')

vmax = mmwave.CFG_10HZ['v_max_m_s']
check('a walking pedestrian folds to nearly static',
      abs(mmwave.fold_velocity(1.4, vmax)) < 0.15,
      '(got %.3f)' % mmwave.fold_velocity(1.4, vmax))
check('inside the window nothing moves',
      abs(mmwave.fold_velocity(0.3, vmax) - 0.3) < 1e-9)
check('fold is periodic in 2*vmax',
      abs(mmwave.fold_velocity(0.3 + 4 * vmax, vmax) - 0.3) < 1e-9)
check('the true velocity is among the candidates',
      any(abs(c - 1.4) < 1e-6
          for c in mmwave.unfold_candidates(mmwave.fold_velocity(1.4, vmax),
                                            vmax, kmax=2)))

good = mmwave.parse_frame(build_frame([(0.3, 1.2, 0.1, 0.2)]))['points']
check('a plausible point raises no issue', mmwave.validate_points(good) == [])

# A TLV length read the wrong way shifts the float stream, and the tell is a
# velocity outside the unambiguous window or a range past the alias limit --
# neither of which the radar can physically report.
bad = mmwave.parse_frame(build_frame([(0.0, 1.0, 0.0, 9.9)]))['points']
check('impossible velocity is caught', len(mmwave.validate_points(bad)) == 1)
far = mmwave.parse_frame(build_frame([(0.0, 40.0, 0.0, 0.0)]))['points']
check('impossible range is caught', len(mmwave.validate_points(far)) == 1)
near = mmwave.parse_frame(build_frame([(0.0, 0.2, 0.0, 0.0)]))['points']
check('the 0.375 m high-pass blind zone is caught',
      len(mmwave.validate_points(
          near, min_range=mmwave.CFG_10HZ['hpf_blind_m'])) == 1)

nan = struct.unpack('<f', b'\x00\x00\xc0\x7f')[0]
check('non-finite is caught',
      len(mmwave.validate_points(
          mmwave.parse_frame(build_frame([(nan, 1.0, 0.0, 0.0)]))['points'])) == 1)


print()
if FAILS:
    print('%d FAILED: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('all radar parser checks passed')
