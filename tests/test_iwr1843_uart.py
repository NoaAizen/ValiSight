"""iwr1843_uart.RadarReader: frame parsing, chunking, resync, coordinates."""
import struct

from iwr1843_uart import RadarReader, MAGIC, HDR_LEN


def build_packet(points_ti, snr_noise=None, frame_no=7):
    """points_ti: [(x_right, y_fwd, z_up, v)] in TI's coordinate frame."""
    tlvs = b""
    tlvs += struct.pack("<2I", 1, len(points_ti) * 16)
    for p in points_ti:
        tlvs += struct.pack("<4f", *p)
    if snr_noise is not None:
        tlvs += struct.pack("<2I", 7, len(snr_noise) * 4)
        for s, nz in snr_noise:
            tlvs += struct.pack("<2h", int(s * 10), int(nz * 10))
    total = HDR_LEN + len(tlvs)
    n_tlv = 2 if snr_noise is not None else 1
    hdr = MAGIC + struct.pack("<8I", 0x0304, total, 0x1843, frame_no,
                              0, len(points_ti), n_tlv, 0)
    return hdr + tlvs


def test_parse_single_frame_and_coordinate_convention():
    # TI: x = right, y = forward -> project: x = forward, y = left
    pkt = build_packet([(1.0, 2.0, 0.5, -0.3)], snr_noise=[(15.0, 90.0)])
    frames = RadarReader().feed(pkt)
    assert len(frames) == 1
    fr = frames[0]
    assert fr["frame"] == 7 and fr["n_obj"] == 1
    x, y, z, v = fr["points"][0]
    assert (x, y, z) == (2.0, -1.0, 0.5)
    assert abs(v - (-0.3)) < 1e-6
    assert abs(fr["snr"][0] - 15.0) < 1e-6      # int16 * 0.1 dB
    assert abs(fr["noise"][0] - 90.0) < 1e-6


def test_frame_split_across_feeds():
    pkt = build_packet([(0.0, 3.0, 0.0, 1.0)])
    rr = RadarReader()
    assert rr.feed(pkt[:20]) == []              # header not complete yet
    frames = rr.feed(pkt[20:])
    assert len(frames) == 1
    assert frames[0]["points"][0][0] == 3.0


def test_resync_after_garbage():
    pkt = build_packet([(0.0, 4.0, 0.0, 0.0)])
    frames = RadarReader().feed(b"\x00\xff garbage \x13" + pkt)
    assert len(frames) == 1


def test_two_frames_in_one_feed():
    pkts = build_packet([(0.0, 1.0, 0.0, 0.0)], frame_no=1) + \
           build_packet([(0.0, 2.0, 0.0, 0.0)], frame_no=2)
    frames = RadarReader().feed(pkts)
    assert [f["frame"] for f in frames] == [1, 2]


def test_corrupt_length_resyncs_to_next_frame():
    good = build_packet([(0.0, 5.0, 0.0, 0.0)])
    corrupt = MAGIC + struct.pack("<8I", 0, 10, 0, 0, 0, 0, 0, 0)  # len < HDR
    frames = RadarReader().feed(corrupt + good)
    assert len(frames) == 1
    assert frames[0]["points"][0][0] == 5.0


def test_missing_side_info_yields_none_snr():
    fr = RadarReader().feed(build_packet([(0.0, 1.0, 0.0, 0.0)]))[0]
    assert fr["snr"] is None and fr["noise"] is None
