#!/usr/bin/env python3
"""Task 8 -- replay determinism: the same radar.bin must parse to bit-identical
results on every run, no matter how the bytes are delivered.

    ./test_replay_determinism.py                 # every radar.bin shipped in the repo
    ./test_replay_determinism.py DIR [DIR ...]   # specific session dirs

Why this exists: when something misbehaves in the field we record the raw UART
stream (radar.bin) and debug on the recording back at the desk. That only works
if the replay is a faithful, repeatable copy of what the live reader did. Three
things could silently break that, and each gets its own check:

  1. run-to-run   the parse depends on nothing outside the bytes -- no clock,
                  no hash-seed, no leftover state. Checked in-process twice and
                  once more in a fresh interpreter with a different
                  PYTHONHASHSEED.
  2. chunking     a live serial port hands FrameSync arbitrary slices (1 byte,
                  4 kB, whatever in_waiting says). The frames that come out
                  must not depend on where the slices fall. 4096-byte chunks
                  are the reference; 1, 7, 64, 1024 and 65536 must agree.
  3. live match   the rows the live reader wrote to radar.jsonl while the
                  radar was attached (frame number, byte_offset, points) must
                  be exactly what the replay produces from radar.bin. This is
                  the check that ties the desk to the field.

t_mono is deliberately excluded from the comparison: it is wall-clock arrival
time, the one field that is allowed -- required, even -- to differ.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'radar'))
import mmwave  # noqa: E402

# 4096 is what a live serial read typically hands over, so it is the reference.
# There is deliberately no "whole file at once": FrameSync caps its buffer at
# 256 kB and drops the head beyond that (a host-stall guard), so a single feed()
# of a big file is a misuse, not a replay.
REF_CHUNK = 4096
CHUNKS = (1, 7, 64, 1024, 65536)
FAILS = []


def check(name, cond, detail=''):
    if cond:
        print('  ok   %s' % name)
    else:
        print('  FAIL %s %s' % (name, detail))
        FAILS.append(name)


# --- the replay ----------------------------------------------------------

def replay(path, chunk=REF_CHUNK):
    """radar.bin -> (canonical rows, counters). Mirrors RadarReader.run() in
    tools/radar_overlay.py: same FrameSync, same parse_frame, same byte_offset
    bookkeeping (a frame that fails to parse still advances the offset), plus
    flush() because a file has an end and a serial port does not."""
    with open(path, 'rb') as fh:
        raw = fh.read()
    sync = mmwave.FrameSync()
    rows, byte_pos, parse_errors = [], 0, 0

    def take(frames):
        nonlocal byte_pos, parse_errors
        for _t, frame in frames:
            off = byte_pos
            byte_pos += len(frame)
            try:
                fr = mmwave.parse_frame(frame)
            except ValueError:
                parse_errors += 1
                continue
            rows.append({
                'frame': fr['frame_number'],
                'byte_offset': off,
                'points': [[p['x'], p['y'], p['z'], p['v'], p['snr'], p['noise']]
                           for p in fr['points']],
                'tlv_types': fr['tlv_types'],
                'length_mode': fr['length_mode'],
                'stats': fr['stats'],
                'temperature': fr['temperature'],
            })

    for i in range(0, len(raw), chunk):
        take(sync.feed(raw[i:i + chunk]))
    take(sync.flush())
    counters = {'frames': len(rows), 'parse_errors': parse_errors,
                'dropped_bytes': sync.dropped_bytes,
                'resyncs': sync.resync_count}
    return rows, counters


def digest(rows, counters):
    """One hash over everything the replay produced. sort_keys so dict order
    can never leak in; json float repr is exact and stable in Python 3."""
    blob = json.dumps([rows, counters], sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()


# --- checks --------------------------------------------------------------

def check_run_to_run(path, ref):
    _rows, _c = replay(path)
    check('run-to-run (same process)', digest(_rows, _c) == ref)
    # Fresh interpreter, different hash seed: catches dict/set ordering and any
    # module-level state that an in-process repeat cannot see.
    env = dict(os.environ, PYTHONHASHSEED='4242')
    out = subprocess.run([sys.executable, __file__, '--digest', path],
                         capture_output=True, text=True, env=env)
    got = out.stdout.strip()
    check('run-to-run (fresh interpreter, PYTHONHASHSEED=4242)', got == ref,
          '\n       got %s\n       want %s\n       %s' % (got, ref, out.stderr.strip()))


# Below this size every chunk width runs; above it the 1- and 7-byte feeds are
# skipped (a 32 MB session fed one byte at a time is minutes of CPU and says
# nothing the 64-byte feed does not).
SLOW_CHUNK_LIMIT = 2 * 1024 * 1024


def check_chunking(path, ref):
    big = os.path.getsize(path) > SLOW_CHUNK_LIMIT
    for chunk in CHUNKS:
        if big and chunk < 64:
            print('  skip chunking: %d-byte chunks (file > %d MB)'
                  % (chunk, SLOW_CHUNK_LIMIT >> 20))
            continue
        t0 = time.monotonic()
        rows, c = replay(path, chunk)
        check('chunking: %5d-byte chunks == %d-byte -> %d frames (%.2fs)'
              % (chunk, REF_CHUNK, c['frames'], time.monotonic() - t0),
              digest(rows, c) == ref,
              'counters %s' % c)


WARNS = []


def check_live_match(session, rows, counters):
    """Replay vs the rows the live reader logged. Strict over the common prefix;
    a difference confined to the last few frames is reported as a WARN with its
    cause, not a FAIL: it is the recording's tail that is unreliable, not the
    parser. (Both directions occur in real sessions -- radar.bin 2 frames short
    of radar.jsonl, and radar.jsonl 3 frames short of radar.bin -- because
    live.py closed the two files while the reader thread was still writing.)"""
    jl = os.path.join(session, 'radar.jsonl')
    if not os.path.exists(jl):
        print('  skip live match: no radar.jsonl')
        return
    with open(jl) as fh:
        live = [json.loads(l) for l in fh if l.strip()]
    n = min(len(rows), len(live))
    bad = None
    for i in range(n):
        r, l = rows[i], live[i]
        if r['frame'] != l['frame'] or r['byte_offset'] != l['byte_offset']:
            bad = (i, 'frame/offset', (r['frame'], r['byte_offset']),
                   (l['frame'], l['byte_offset']))
            break
        # The live rows went through json.dumps once; round-trip ours the same
        # way so the comparison is on the identical representation.
        if json.loads(json.dumps(r['points'])) != l['points']:
            bad = (i, 'points', r['points'][:2], l['points'][:2])
            break
    check('live match: frame numbers, byte offsets, points identical over %d rows' % n,
          bad is None, 'first mismatch at row %s: %s replay=%s live=%s' % (bad or ('', '', '', '')))
    if bad is None and len(rows) != len(live):
        # Same prefix, different length: only the tail differs.
        if len(live) > len(rows):
            size = os.path.getsize(os.path.join(session, 'radar.bin'))
            msg = ('radar.bin ends at byte %d but radar.jsonl logged %d more '
                   'frame(s) from there (first: frame %d)'
                   % (size, len(live) - len(rows),
                      live[len(rows)]['frame']))
        else:
            msg = ('radar.bin holds %d more frame(s) at its end than radar.jsonl '
                   'logged (first: frame %d at offset %d)'
                   % (len(rows) - len(live), rows[len(live)]['frame'],
                      rows[len(live)]['byte_offset']))
        print('  WARN live match: tail differs -- %s. Recording shutdown race, '
              'not a parser difference (see live.py: radar.close() before the '
              'reader thread stops).' % msg)
        WARNS.append('%s: %s' % (session, msg))
    if live and len(live) <= len(rows):
        last = live[-1]
        check('live match: dropped_bytes %s == %s, resyncs %s == %s at the last logged row'
              % (counters['dropped_bytes'], last['dropped_bytes'],
                 counters['resyncs'], last['resyncs']),
              counters['dropped_bytes'] == last['dropped_bytes']
              and counters['resyncs'] == last['resyncs'])


def run_session(session):
    path = os.path.join(session, 'radar.bin')
    print('\n%s  (%d bytes)' % (session, os.path.getsize(path)))
    rows, counters = replay(path)
    ref = digest(rows, counters)
    print('  digest %s  %s' % (ref[:16], counters))
    check_run_to_run(path, ref)
    check_chunking(path, ref)
    check_live_match(session, rows, counters)


def default_sessions():
    roots = [os.path.join(HERE, '..', 'captures'),
             os.path.join(HERE, '..', '..', '..', 'captures')]
    out = []
    for root in roots:
        for d, _dirs, files in os.walk(root):
            if 'radar.bin' in files:
                out.append(os.path.normpath(d))
    return sorted(out)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--digest':
        rows, c = replay(sys.argv[2])
        print(digest(rows, c))
        sys.exit(0)
    sessions = sys.argv[1:] or default_sessions()
    if not sessions:
        print('no radar.bin found')
        sys.exit(2)
    for s in sessions:
        run_session(s)
    print('\n%d sessions, %d failures, %d warnings' % (len(sessions), len(FAILS), len(WARNS)))
    for f in FAILS:
        print('  FAIL ' + f)
    for w in WARNS:
        print('  WARN ' + w)
    sys.exit(1 if FAILS else 0)
