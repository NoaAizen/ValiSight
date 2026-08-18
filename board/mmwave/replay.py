# replay.py — deterministic replay of a raw radar recording (plan item 8).
#
# HOST-ONLY tool (CPython; not meant to run on the N6). Feeds a recorded
# byte stream through the exact same FrameSync/parse_frame code the live
# path uses and emits one canonical JSON line per frame. The contract:
#
#   same recording  =>  bit-identical output, every run, any machine,
#                       and for ANY way the stream is sliced into chunks.
#
# Canonical output deliberately contains no wall-clock time and nothing
# host-dependent. Chunk-dependent diagnostics (resync_count) are reported
# on stderr but kept OUT of the canonical stream — resync accounting can
# legitimately differ with chunk boundaries while the frames do not.
#
# CLI:
#   python replay.py radar_raw.bin --chunk 4096
#   python replay.py radar_raw.bin --random-chunks 1217 --out replay.jsonl
#
# NOTE: chunks must stay well under FrameSync's max_buffer (64 KB) — that
# is the live UART regime. Feeding a whole recording in one call trips the
# front-drop safety bound and silently discards all but the last 64 KB,
# which is exactly the kind of footgun this tool exists to catch.

import argparse
import hashlib
import json
import random
import sys

from mmwave_parser import FrameSync, parse_frame


def iter_chunks(data, size):
    """Slice data into fixed-size chunks (the last one may be short)."""
    for off in range(0, len(data), size):
        yield data[off:off + size]


def iter_random_chunks(data, seed, lo=1, hi=4096):
    """Slice data into pseudo-random chunk sizes from a FIXED seed, so the
    'random' slicing is itself reproducible."""
    rng = random.Random(seed)
    off = 0
    while off < len(data):
        size = rng.randint(lo, hi)
        yield data[off:off + size]
        off += size


def canonical_record(parsed):
    """One frame -> canonical dict. Only stream-derived, deterministic
    fields; float32 values round-trip bit-exactly through repr/json."""
    hdr = parsed['header']
    return {
        'frame': hdr['frame_number'],
        'num_detected_obj': hdr['num_detected_obj'],
        'length_mode': parsed['length_mode'],
        'points': [[p['x'], p['y'], p['z'], p['v'], p['snr'], p['noise']]
                   for p in parsed['points']],
    }


def replay(chunks):
    """Feed chunks through FrameSync; return (jsonl_bytes, stats).

    jsonl_bytes is the canonical output — one sorted-key JSON line per
    successfully parsed frame. Frames that fail TLV validation are counted,
    not emitted (a corrupt frame must never poison the canonical stream).
    """
    sync = FrameSync()
    lines = []
    bad_frames = 0
    for chunk in chunks:
        for frame in sync.feed(chunk):
            try:
                parsed = parse_frame(frame)
            except ValueError:
                bad_frames += 1
                continue
            lines.append(json.dumps(canonical_record(parsed),
                                    sort_keys=True,
                                    separators=(',', ':')))
    out = ('\n'.join(lines) + '\n').encode() if lines else b''
    stats = {
        'frames': len(lines),
        'bad_frames': bad_frames,
        'resync_count': sync.resync_count,
        'dropped_bytes': sync.dropped_bytes,
    }
    return out, stats


def replay_file(path, chunk=4096, random_seed=None):
    with open(path, 'rb') as f:
        data = f.read()
    if random_seed is not None:
        chunks = iter_random_chunks(data, random_seed)
    else:
        chunks = iter_chunks(data, chunk)
    return replay(chunks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recording', help='raw byte-stream recording (.bin)')
    ap.add_argument('--chunk', type=int, default=4096,
                    help='fixed feed-chunk size in bytes (default 4096)')
    ap.add_argument('--random-chunks', type=int, metavar='SEED',
                    help='use pseudo-random chunk sizes from this seed '
                         '(overrides --chunk)')
    ap.add_argument('--out', help='write canonical jsonl here')
    args = ap.parse_args(argv)

    out, stats = replay_file(args.recording, chunk=args.chunk,
                             random_seed=args.random_chunks)
    digest = hashlib.sha256(out).hexdigest()

    if args.out:
        with open(args.out, 'wb') as f:
            f.write(out)

    print('sha256  %s' % digest)
    print('frames  %(frames)d  bad %(bad_frames)d  '
          'resyncs %(resync_count)d  dropped %(dropped_bytes)d B'
          % stats, file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
