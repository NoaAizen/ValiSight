#!/usr/bin/env python3
"""Gate 5: train + evaluate the radar track person/clutter classifier.

Input: perception/out/radar_ai/<sess>_labelled.json (label_tracks.py).
Labels are re-derived here from the raw counts with explicit, printed
thresholds (the teacher on this host is CPU yolov4-tiny with poor recall, so
'person' = matched >= --min-matched frames AND matched_frac >= --pos;
'clutter' = matched <= 1 with >= --min-inband in-band frames).
Evaluation is LEAVE-ONE-SESSION-OUT - the honest number for a sensor
classifier; a random split would leak the same walk into train and test.
Also scores the rule-based tools/radar_classify_n6.py logic on the same
track features (mapped onto its thresholds) as the baseline to beat.

Usage:
    python3 perception/radar_ai/train_gate5.py [--pos 0.2] [--min-matched 8]
        -> perception/out/radar_ai/gate5_model.pkl + gate5_report.json
"""
import argparse, glob, json, os, pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
OUT = os.path.join(ROOT, 'perception', 'out', 'radar_ai')
FEATS = ['v_abs_mean', 'v_spread_med', 'v_p90_10', 'rcs_mean', 'rcs_max', 'extent_med',
         'npts_mean', 'npts_cv', 'duration_s', 'ground_speed_med', 'ground_speed_p90',
         'straightness', 'range_med']

def relabel(t, pos, min_matched, min_inband):
    inb = t['matched'] + t['contra']
    if t['matched'] >= min_matched and t['matched_frac'] is not None and t['matched_frac'] >= pos: return 'person'
    if t['matched'] <= 1 and inb >= min_inband: return 'clutter'
    return None

def rule_based(t):
    # tools/radar_classify_n6.classify, applied to track-level stand-ins:
    #   |v| < STATIC_V -> static ; extent > VEH_EXTENT & many pts -> vehicle ; else pedestrian
    if t['v_abs_mean'] < 0.25 and t['ground_speed_med'] < 0.25: return 'clutter'
    if t['extent_med'] > 1.5 and t['npts_mean'] >= 6: return 'clutter'      # 'vehicle' -> not a person here
    return 'person'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos', type=float, default=0.2); ap.add_argument('--min-matched', type=int, default=8)
    ap.add_argument('--min-inband', type=int, default=15)
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, '*_labelled.json'))):
        for t in json.load(open(p)):
            y = relabel(t, a.pos, a.min_matched, a.min_inband)
            if y: rows.append((t['session'], y, t))
    sess = sorted({r[0] for r in rows})
    print('label rule: person = matched>=%d & frac>=%.2f ; clutter = matched<=1 & inband>=%d' % (a.min_matched, a.pos, a.min_inband))
    for s in sess:
        ys = [r[1] for r in rows if r[0] == s]; print('  %-16s person %3d  clutter %3d' % (s, ys.count('person'), ys.count('clutter')))
    X = np.array([[r[2][f] for f in FEATS] for r in rows], float); y = np.array([r[1] == 'person' for r in rows]); S = np.array([r[0] for r in rows])
    X = np.nan_to_num(X)
    # leave-one-session-out
    pred = np.zeros_like(y); prob = np.zeros(len(y))
    for s in sess:
        tr, te = S != s, S == s
        if y[tr].sum() == 0 or (~y[tr]).sum() == 0: pred[te] = False; continue
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced', random_state=0).fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te]); prob[te] = clf.predict_proba(X[te])[:, 1]
    def report(name, yhat):
        cm = confusion_matrix(y, yhat, labels=[True, False]); tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); acc = (tp + tn) / len(y)
        print('%-22s acc %.2f  person: precision %.2f recall %.2f  (tp %d fn %d fp %d tn %d)' % (name, acc, prec, rec, tp, fn, fp, tn))
        return dict(acc=acc, precision=prec, recall=rec, tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn))
    print('\nn tracks %d (person %d, clutter %d) — leave-one-session-out:' % (len(y), y.sum(), (~y).sum()))
    r_rf = report('RandomForest (LOSO)', pred)
    r_rule = report('rule-based classify_n6', np.array([rule_based(r[2]) == 'person' for r in rows]))
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced', random_state=0).fit(X, y)
    imp = sorted(zip(clf.feature_importances_, FEATS), reverse=True)
    print('\nfeature importance:'); [print('  %-18s %.3f' % (f, w)) for w, f in imp]
    pickle.dump({'model': clf, 'features': FEATS, 'label_rule': vars(a), 'sessions': sess}, open(os.path.join(OUT, 'gate5_model.pkl'), 'wb'))
    json.dump({'n': int(len(y)), 'n_person': int(y.sum()), 'sessions': sess, 'rf_loso': r_rf, 'rule_based': r_rule,
               'importance': [(f, float(w)) for w, f in imp], 'label_rule': vars(a)}, open(os.path.join(OUT, 'gate5_report.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
