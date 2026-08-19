#!/usr/bin/env python3
"""Extract radar heatmaps (TLV 2/4/5) offline from a recorded session.

radar.bin already IS the raw capture: RadarReader writes every byte off the
DATA port before parsing, and each radar.jsonl row carries byte_offset - the
position of that frame's magic word in radar.bin. So enabling heatmaps in
guiMonitor changes NOTHING in the recorder; the extra TLVs land in radar.bin
on their own, and this module is where they become arrays. That is the same
division of labour as thermal.bin: the recording stores the measurement
byte-for-byte, interpretation happens offline where it can be re-run.

Sessions recorded with heatmaps off (every session before radar_people_ra.cfg)
yield frames whose 'ra'/'rd'/'range_profile' are None - present-but-empty is
the honest reading of "the config did not emit them", and lets one loader
serve both eras.

Usage:
    python3 perception/export/radar_heatmaps.py captures/<session> [--summary]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'radar'))
import mmwave                                                    # noqa: E402

N_RANGE_BINS_DEFAULT = 256      # profileCfg numAdcSamples of the people configs


def frames(session_dir, n_range_bins=N_RANGE_BINS_DEFAULT):
    """Yield one dict per radar.jsonl row, heatmaps decoded when present.

    {'frame', 't_mono', 'range_profile' (n_range,) float32 dB or None,
     'ra' (n_range, n_vant) complex64 or None,
     'rd' (n_range, n_doppler) float32 dB, fftshifted on Doppler, or None}

    Range-profile and RD values are converted to relative dB (Q9 log2 * the
    fixed factor); RA stays complex - the angle FFT is a consumer choice (see
    ra_image). Decoding trusts byte_offset and re-verifies the magic there:
    an offset that does not land on a magic means the row and the bin file
    disagree (a truncated tail write), and that frame is skipped with a count
    rather than parsed into garbage.
    """
    bin_path = os.path.join(session_dir, 'radar.bin')
    jsonl_path = os.path.join(session_dir, 'radar.jsonl')
    raw = np.memmap(bin_path, dtype=np.uint8, mode='r')
    bad_offsets = 0
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            off = row.get('byte_offset')
            if off is None:
                continue
            frame = _slice_frame(raw, off)
            if frame is None:
                bad_offsets += 1
                continue
            out = {'frame': row['frame'], 't_mono': row['t_mono'],
                   'range_profile': None, 'ra': None, 'rd': None}
            try:
                tlvs, _ = mmwave.walk_tlvs(frame)
            except ValueError:
                bad_offsets += 1
                continue
            for tlv_type, payload in tlvs:
                if tlv_type == mmwave.TLV_RANGE_PROFILE:
                    out['range_profile'] = (
                        np.frombuffer(payload, np.uint16).astype(np.float32)
                        * mmwave.Q9_LOG2_TO_DB)
                elif tlv_type == mmwave.TLV_AZIMUTH_HEATMAP:
                    out['ra'] = _decode_ra(payload, n_range_bins)
                elif tlv_type == mmwave.TLV_RANGE_DOPPLER_HEATMAP:
                    out['rd'] = _decode_rd(payload, n_range_bins)
            yield out
    if bad_offsets:
        print(f'[heatmaps] {session_dir}: {bad_offsets} rows whose byte_offset '
              f'did not yield a parseable frame (truncated tail write)',
              file=sys.stderr)


def _slice_frame(raw, off):
    if off + mmwave.FRAME_HEADER_LEN > len(raw):
        return None
    if bytes(raw[off:off + len(mmwave.MAGIC)]) != mmwave.MAGIC:
        return None
    total = int(np.frombuffer(raw[off + 12:off + 16].tobytes(), '<u4')[0])
    if total < mmwave.FRAME_HEADER_LEN or off + total > len(raw):
        return None
    return raw[off:off + total].tobytes()


def _decode_ra(payload, n_range_bins):
    """cmplx16ImRe rows -> (n_range, n_vant) complex64.

    The wire order is IMAG then real per sample (mmwave.parse_azimuth_heatmap
    documents why getting this backwards mirrors azimuth); frombuffer reads
    pairs in wire order, so column 0 is imag.
    """
    iq = np.frombuffer(payload, np.int16)
    if n_range_bins <= 0 or iq.size % (2 * n_range_bins):
        return None
    iq = iq.reshape(n_range_bins, -1, 2).astype(np.float32)
    return (iq[:, :, 1] + 1j * iq[:, :, 0]).astype(np.complex64)


def _decode_rd(payload, n_range_bins):
    v = np.frombuffer(payload, np.uint16)
    if n_range_bins <= 0 or v.size % n_range_bins:
        return None
    rd = v.reshape(n_range_bins, -1).astype(np.float32) * mmwave.Q9_LOG2_TO_DB
    # wire Doppler order is FFT bins (DC, +v..., -v...); put zero in the middle
    return np.fft.fftshift(rd, axes=1)


def ra_image(ra, n_angle=64):
    """(n_range, n_vant) complex -> (n_range, n_angle) magnitude dB.

    Zero-padded FFT across the antenna axis, fftshifted so boresight is the
    centre column. Angle bins map to sin(azimuth) uniformly, NOT to azimuth
    degrees - resampling to a uniform angle axis is a display choice left to
    the caller, and training on the sin-uniform image is fine as long as it
    is consistent.
    """
    spec = np.fft.fftshift(np.fft.fft(ra, n=n_angle, axis=1), axes=1)
    return (20.0 * np.log10(np.abs(spec) + 1e-6)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('session', help='session directory (holds radar.bin/.jsonl)')
    ap.add_argument('--n-range-bins', type=int, default=N_RANGE_BINS_DEFAULT)
    ap.add_argument('--summary', action='store_true',
                    help='count TLV availability over the whole session')
    a = ap.parse_args()
    n = have_rp = have_ra = have_rd = 0
    first_shapes = {}
    for fr in frames(a.session, a.n_range_bins):
        n += 1
        for k in ('range_profile', 'ra', 'rd'):
            if fr[k] is not None:
                if k == 'range_profile':
                    have_rp += 1
                elif k == 'ra':
                    have_ra += 1
                else:
                    have_rd += 1
                first_shapes.setdefault(k, fr[k].shape)
        if not a.summary and n >= 5:
            break
    print(f'{a.session}: {n} frames read; range_profile in {have_rp}, '
          f'RA in {have_ra}, RD in {have_rd}')
    for k, s in first_shapes.items():
        print(f'  {k}: shape {s}')
    if not (have_rp or have_ra or have_rd):
        print('  no heatmap TLVs - session predates radar_people_ra.cfg '
              '(expected for everything recorded before 2026-08-19)')


if __name__ == '__main__':
    main()
