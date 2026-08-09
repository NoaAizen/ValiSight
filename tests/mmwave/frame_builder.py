"""Synthetic mmWave-demo (SDK 3.x) frame builder for host-side tests.

Encodes the TI wire format the library must parse:
  magic word: 02 01 04 03 06 05 08 07
  header:     8 x uint32 little-endian — version, totalPacketLen, platform,
              frameNumber, timeCpuCycles, numDetectedObj, numTLVs,
              subFrameNumber
  TLVs:       type (u32), length (u32), payload
    type 1 (detected points): numDetectedObj x (x, y, z, v) float32
    type 7 (side info):       numDetectedObj x (snr, noise) int16, 0.1 dB units

Two real-device quirks are reproducible on demand:
  * `length_includes_tlv_header` — the SDK-version ambiguity Stage 3 must
    resolve empirically (does TLV length cover the 8-byte TLV header?)
  * `pad_to_32` — the demo pads each packet with zeros to a 32-byte multiple
"""
import struct

MAGIC = bytes((0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07))
HEADER_FMT = '<8I'
HEADER_LEN = struct.calcsize(HEADER_FMT)  # 32
TLV_POINTS = 1
TLV_SIDE_INFO = 7


def build_frame(frame_number=1, points=(), side_info=None,
                length_includes_tlv_header=False, pad_to_32=False,
                extra_tlvs=(), num_detected_obj=None,
                version=0x03060000, platform=0x000A1843, subframe=0):
    """points: iterable of (x, y, z, v); side_info: iterable of (snr, noise).

    extra_tlvs: iterable of (type, payload_bytes) appended after the
    standard TLVs — useful for adversarial payloads (e.g. one containing
    the magic word).
    """
    tlvs = []
    if points:
        tlvs.append((TLV_POINTS,
                     b''.join(struct.pack('<4f', *p) for p in points)))
    if side_info is not None:
        tlvs.append((TLV_SIDE_INFO,
                     b''.join(struct.pack('<2h', *s) for s in side_info)))
    tlvs.extend(extra_tlvs)

    body = b''
    for tlv_type, payload in tlvs:
        length = len(payload) + (8 if length_includes_tlv_header else 0)
        body += struct.pack('<2I', tlv_type, length) + payload

    total = len(MAGIC) + HEADER_LEN + len(body)
    pad = b''
    if pad_to_32 and total % 32:
        pad = b'\x00' * (32 - total % 32)
        total += len(pad)

    n_obj = len(points) if num_detected_obj is None else num_detected_obj
    header = struct.pack(HEADER_FMT, version, total, platform, frame_number,
                         123456, n_obj, len(tlvs), subframe)
    return MAGIC + header + body + pad
