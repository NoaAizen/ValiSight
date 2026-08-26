#!/usr/bin/env python3
"""REJECTED 2026-08-25 by its own gate. Kept as the measurement, not as a stage.

    Do not wire this into export_shards.py. The rule below was built to
    convert the 32,813 UNKNOWN thermal rows of v3 (55% of the export) into
    NEGATIVE supervision, and it fails: simulated on the five hand-labeled
    sessions with the machine teacher deciding and the human scoring,
    24.5% of the frames it calls empty contain a person a human labeled
    (240 of 980, min_blob=24). Without the thermal veto it is 65-76%.

    Why it cannot be tuned out: the teacher's misses are SYSTEMATIC, not
    random. Widening the temporal window from 0 to 45 frames cuts the yield
    50-fold and leaves the error rate at 65% - the same hard person (occluded,
    distant, low-contrast) is missed across the whole window. And the thermal
    cannot rescue it: 60% of the missed people present fewer than 24 thermal
    pixels above 4.7 C, so the veto never fires (diagnosed on moving1).

    The tri-state comment at export_shards.py:75 - "missing evidence is never
    negative evidence" - was right, and this file is the measurement that
    says so. Reproduce with:
        python3 perception/autolabel/witness_negative.py --simulate-machine --sweep

    What the run did produce is in [[thermal-bar-person-appliance-overlap]]:
    the 4.7 C grade bar keeps 81.5% of people while still admitting 42.2% of
    laptops and monitors, and no bar does better. That is a shape problem,
    not a threshold problem.

Original intent, for the record:

Turn "the teacher was working and saw no person" into NEGATIVE supervision.

Today a frame with no person box is UNKNOWN, and that is 55% of the v3 export
(32,813 of 59,853 thermal rows) contributing no gradient at all. But most of
those frames are not evidence-free: the teacher emitted chairs, laptops and
monitors in them, which proves the detector was fed a scene it could read.
Absence of a person box in a frame where the detector demonstrably saw other
things is weak positive evidence of an empty scene - and this stage turns it
into a label only when the THERMAL agrees, so the negative rests on two
sensors the way every other label in this pipeline does.

The rule, stated once:

    NEGATIVE-for-person(frame) iff
      (1) the teacher emitted >= 1 detection of any class at conf >= CONF_MIN
          - the detector proved it can see. A dark session where YOLO emits
            nothing stays UNKNOWN, which is the whole point of the tri-state.
      (2) zero of those detections are 'person'
      (3) the thermal carries no unexplained warm blob

Condition (3) is what makes this a two-sensor statement. "Unexplained" is
measured against two references, because a room is full of legitimately hot
things that no detector will ever box:

  - a per-session STATIC HEAT REFERENCE (per-pixel temporal median). Ceiling
    downlights, a radiator, the standing lamp - anything that does not move
    cancels here. Measured on negative1-clean: a scalar-median threshold alone
    finds ~12 blobs per frame in a verifiably empty room, which would veto
    everything; against the static reference what survives is what changed.
  - the thermal footprint of the teacher's own non-person boxes, mapped
    through the same warp LUT build_dataset.py uses. A laptop that warms up
    mid-session moves in the static reference but is explained by its box.

A blob that survives both is something warm that appeared and that nothing
accounts for. That is exactly the shape of a person the teacher missed, so it
vetoes the negative and the frame stays UNKNOWN.

Note the asymmetry that makes this safe: a false veto costs one UNKNOWN frame
(what we have today), a missed veto teaches the model that a person is
background. The thresholds are therefore set from the measured person-blob
distribution, not from the empty-room one.

Usage:
    python3 perception/autolabel/witness_negative.py --sweep      # gate report
    python3 perception/autolabel/witness_negative.py --write      # emit json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import cv2

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)
from perception.dataset import LiveSession                       # noqa: E402
from perception.autolabel.thermal_check import (                 # noqa: E402
    ThermalBoxCheck, to_uint8_equivalent, DECIMATION, LOW_W, LOW_H)

AL = os.path.join(ROOT, 'perception', 'out', 'autolabel')
CAPTURES = os.path.join(ROOT, 'captures')

CONF_MIN = 0.40          # the teacher's own working point for "saw something"
HOT_DEG_C = 4.7          # same bar build_dataset.py grades a person by
HOT_COUNTS = 20.0        # legacy AGC sessions with no declared scale
MIN_BLOB_PX = 24         # thermal pixels; swept and reported by --sweep
STATIC_SAMPLE = 240      # frames drawn for the static heat reference


def session_hot_threshold(sess):
    """Warm-blob bar in uint8-equivalent counts for this session's scale."""
    c = sess.meta.get('c_per_lsb')
    if c is None:
        tmin, tmax = sess.meta.get('tmin'), sess.meta.get('tmax')
        if tmin is not None and tmax is not None:
            c = (tmax - tmin) / sess.meta.get('thermal_counts_max', 255)
    return (HOT_DEG_C / c) if c else HOT_COUNTS


def load_dets(name):
    """Merged per-frame detections: manual persons win, machine classes ride.

    A hand-labeled session keeps its machine run as <sess>.teacher-machine.jsonl
    and the manual labels as <sess>_teacher.jsonl. The manual file is gold for
    persons and empty of everything else, so persons come from it and the other
    classes from the machine file. Machine PERSON boxes are dropped outright -
    keeping them would reintroduce the false-positive tail the hand labeling
    was done to remove.
    """
    manual_p = os.path.join(AL, f'{name}_teacher.jsonl')
    machine_p = os.path.join(AL, f'{name}.teacher-machine.jsonl')
    if not os.path.exists(manual_p):
        return None, False
    manual, is_hand = {}, False
    with open(manual_p) as f:
        for line in f:
            r = json.loads(line)
            is_hand = is_hand or r.get('source') == 'manual'
            manual[r['i']] = r['dets']
    if not os.path.exists(machine_p):
        return manual, is_hand
    out = {}
    with open(machine_p) as f:
        for line in f:
            r = json.loads(line)
            out[r['i']] = [d for d in r['dets'] if d['cls'] != 'person']
    for i, dets in manual.items():
        out.setdefault(i, [])
        out[i] = [d for d in dets if d['cls'] == 'person'] + out[i]
    return out, is_hand


def machine_persons(name):
    """Person boxes as the MACHINE teacher saw them, for the honesty gate."""
    path = os.path.join(AL, f'{name}.teacher-machine.jsonl')
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r['i']] = [d for d in r['dets']
                           if d['cls'] == 'person' and d['conf'] >= CONF_MIN]
    return out


def static_reference(sess, rows):
    """Per-pixel temporal median: everything in the room that does not move.

    Sampled rather than exhaustive - multi5 is 25k frames and the median of
    240 evenly spaced ones is the same picture for two orders less work.
    """
    idx = np.unique(np.linspace(0, len(rows) - 1,
                                min(STATIC_SAMPLE, len(rows))).astype(int))
    stack = np.stack([to_uint8_equivalent(sess.thermal_frame(rows[k]), 255)
                      for k in idx])
    ref = np.median(stack, axis=0).astype(np.float32)
    return ref, float(np.median(ref))


def det_footprint(check, dets, shape):
    """Mask of the thermal pixels the teacher's non-person boxes explain.

    Dilated by one cell: the warp LUT is decimated by 4, so a box edge lands
    on a grid boundary and a blob may sit one pixel outside the sampled set.
    """
    mask = np.zeros(shape, np.uint8)
    for d in dets:
        if d['cls'] == 'person' or d.get('conf', 0.0) < CONF_MIN:
            continue
        gx0 = max(0, int(d['x']) // DECIMATION)
        gy0 = max(0, int(d['y']) // DECIMATION)
        gx1 = min(LOW_W, int(np.ceil((d['x'] + d['w']) / DECIMATION)))
        gy1 = min(LOW_H, int(np.ceil((d['y'] + d['h']) / DECIMATION)))
        if gx1 <= gx0 or gy1 <= gy0:
            continue
        m = check.valid[gy0:gy1, gx0:gx1]
        if not m.any():
            continue
        mask[check.tv[gy0:gy1, gx0:gx1][m],
             check.tu[gy0:gy1, gx0:gx1][m]] = 1
    if mask.any():
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return mask.astype(bool)


def unexplained_blobs(th, ref, ref_bg, hot, explained):
    """Areas of warm regions that are neither static scene nor a teacher box."""
    bg = float(np.median(th))
    now = (th - bg) >= hot
    was = (ref - ref_bg) >= hot
    mask = (now & ~was & ~explained).astype(np.uint8)
    if not mask.any():
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return [int(stats[k, cv2.CC_STAT_AREA]) for k in range(1, n)]


def scan_session(name, min_blob, simulate_machine=False):
    """Per-frame witness decision plus the numbers the gate is read from.

    simulate_machine answers the only question that matters for the sessions
    the rule will actually run on. In a hand-labeled session the rule sees the
    HUMAN's person boxes, so condition (2) excludes every person frame by
    construction and the rule can never be caught being wrong - a gate that
    passes for free. Under simulation the decision is taken from the machine
    teacher alone, exactly as it would be on multi5 or radar3-*, and scored
    against the human labels. A frame the human marked with a person and the
    simulated rule called empty is the poisoning case, and it is counted.
    """
    sess_dir = os.path.join(CAPTURES, name)
    if not os.path.exists(sess_dir):
        return None
    dets, is_hand = load_dets(name)
    if dets is None:
        return None
    if simulate_machine:
        if not is_hand:
            return None
        dets = {i: [d for d in v if d['cls'] != 'person']
                for i, v in dets.items()}
        mp = machine_persons(name) or {}
        for i in dets:
            dets[i] = mp.get(i, []) + dets[i]
    sess = LiveSession(sess_dir)
    rows = [m for m in sess.frames if m.get('thermal_off') is not None]
    if len(rows) < 8:
        return None
    check = ThermalBoxCheck(counts_max=sess.meta.get('thermal_counts_max', 255)
                            or 255)
    hot = session_hot_threshold(sess)
    ref, ref_bg = static_reference(sess, rows)
    mach_p = machine_persons(name)

    out = {'session': name, 'hand_labeled': is_hand, 'hot_counts': hot,
           'frames': 0, 'negative': 0, 'no_teacher_evidence': 0,
           'has_person': 0, 'vetoed': 0, 'hand_person_frames': 0,
           'hand_person_machine_blind': 0, 'poisoned': 0,
           'per_frame': {}}
    for m in rows:
        i = m['i']
        d = dets.get(i)
        if d is None:
            continue
        out['frames'] += 1
        seen = [x for x in d if x.get('conf', 0.0) >= CONF_MIN]
        persons = [x for x in seen if x['cls'] == 'person']
        if persons:
            out['has_person'] += 1
            continue
        if not seen:
            out['no_teacher_evidence'] += 1
            continue
        th = to_uint8_equivalent(sess.thermal_frame(m), 255)
        blobs = unexplained_blobs(th, ref, ref_bg, hot,
                                  det_footprint(check, seen, th.shape))
        if any(b >= min_blob for b in blobs):
            out['vetoed'] += 1
            continue
        out['negative'] += 1
        out['per_frame'][i] = True

    # Honesty gate, scored against the human. POISON counts frames a person
    # was actually in and the rule called empty - the only error that teaches
    # the student something false. It can only be non-zero under simulation;
    # without it the rule reads the human's own boxes and trivially agrees.
    if is_hand:
        hand = {}
        with open(os.path.join(AL, f'{name}_teacher.jsonl')) as f:
            for line in f:
                r = json.loads(line)
                hand[r['i']] = [x for x in r['dets'] if x['cls'] == 'person']
        for i, hp in hand.items():
            if not hp:
                continue
            out['hand_person_frames'] += 1
            if not (mach_p or {}).get(i):
                out['hand_person_machine_blind'] += 1
            if out['per_frame'].get(i):
                out['poisoned'] += 1
    return out


def sessions_on_disk():
    names = set()
    for p in glob.glob(os.path.join(AL, '*_teacher.jsonl')):
        names.add(os.path.basename(p).replace('_teacher.jsonl', ''))
    return sorted(n for n in names
                  if os.path.exists(os.path.join(CAPTURES, n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='*', default=None)
    ap.add_argument('--min-blob', type=int, default=MIN_BLOB_PX)
    ap.add_argument('--sweep', action='store_true',
                    help='report the gate over a range of blob sizes')
    ap.add_argument('--simulate-machine', action='store_true',
                    help='decide from the machine teacher only, score against '
                         'the human labels (the gate that can actually fail)')
    ap.add_argument('--write', action='store_true',
                    help='write <sess>_witness.json with the negative frames')
    a = ap.parse_args()

    names = a.sessions or sessions_on_disk()
    sizes = [12, 18, 24, 32, 48, 72] if a.sweep else [a.min_blob]
    for min_blob in sizes:
        print(f'\n=== min unexplained blob = {min_blob} thermal px ===')
        print(f"{'session':24s} {'frames':>7} {'NEG':>7} {'veto':>7} "
              f"{'noEvid':>7} {'person':>7} {'blind':>6} {'POISON':>7}")
        tot = np.zeros(7, np.int64)
        results = []
        for n in names:
            r = scan_session(n, min_blob, a.simulate_machine)
            if r is None:
                continue
            results.append(r)
            print(f"{r['session']:24s} {r['frames']:7d} {r['negative']:7d} "
                  f"{r['vetoed']:7d} {r['no_teacher_evidence']:7d} "
                  f"{r['has_person']:7d} {r['hand_person_machine_blind']:6d} "
                  f"{r['poisoned']:7d}")
            tot += np.array([r['frames'], r['negative'], r['vetoed'],
                             r['no_teacher_evidence'], r['has_person'],
                             r['poisoned'], r['hand_person_frames']])
        print(f"{'TOTAL':24s} {tot[0]:7d} {tot[1]:7d} {tot[2]:7d} "
              f"{tot[3]:7d} {tot[4]:7d} {'':6s} {tot[5]:7d}")
        if a.simulate_machine and tot[6]:
            rate = 100.0 * tot[5] / tot[6]
            print(f'POISON RATE (<=1% to pass): {tot[5]}/{tot[6]} '
                  f'= {rate:.2f}% -> {"PASS" if rate <= 1.0 else "FAIL"}')
        if a.write and not a.sweep:
            for r in results:
                path = os.path.join(AL, f"{r['session']}_witness.json")
                with open(path, 'w') as f:
                    json.dump({'min_blob_px': min_blob,
                               'hot_counts': r['hot_counts'],
                               'conf_min': CONF_MIN,
                               'negative_frames': sorted(r['per_frame'])}, f)
            print(f'wrote {len(results)} *_witness.json -> {AL}')


if __name__ == '__main__':
    main()
