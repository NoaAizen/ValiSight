#!/usr/bin/env python3
"""Cluster-level person/clutter classifier — first research-based training (2026-08-18).

Per radar frame: region-grow clustering (as tools/radar_classify_n6.cluster, eps 0.6 m),
then per cluster the features the literature points at:
  n_pts, v_mean, v_spread (micro-Doppler proxy, Abdu 2021), rcs = snr_max + 40 log10 R,
  extent (max pairwise distance), range, |displacement between consecutive frames| of the
  nearest cluster (tracker-delta anti-aliasing cue, Heuel & Rohling), and disp - |v|*dt.
Labels come from PHYSICS in a single-person session, not from a camera teacher:
  clutter = cluster whose (range,az) cell is occupied in >= 60% of ALL frames of the session
            (static map: furniture, walls, glass) AND |v_mean| < 0.15
  person  = cluster inside an exercise window whose |v_mean| >= 0.25 OR displacement >= 0.15 m,
            and NOT in a static cell
Everything else is unlabelled. Evaluation: leave-one-session-out; baseline = classify_n6 rules.
Usage: python3 perception/radar_ai/train_clusters.py captures/session_clean15 captures/session_dyn ...
"""
import sys, os, json, math, collections, pickle
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), '..', '..'); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import radar_classify_n6 as rc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
FEATS = ['n_pts', 'v_mean_abs', 'v_spread', 'rcs', 'extent', 'range', 'disp', 'disp_minus_v']

def load(sess):
    rad = [json.loads(l) for l in open(os.path.join(sess, 'radar.jsonl')) if l.strip()]
    fr = [json.loads(l) for l in open(os.path.join(sess, 'frames.jsonl')) if l.strip()]
    win = {}
    for f in fr:
        p = f.get('pose_id')
        if p: win.setdefault(p, [f['t_mono'], f['t_mono']]); win[p][1] = f['t_mono']
    return rad, win

def cell(x, y): r = math.hypot(x, y); return (round(r / 0.3), round(math.degrees(math.atan2(-y, x)) / 6))

def dataset(sess, exercise_prefixes=('C', 'D', 'E')):
    rad, win = load(sess)
    occ = collections.Counter(); N = len(rad)
    for r in rad:
        for c in {cell(p[0], p[1]) for p in r['points']}: occ[c] += 1
    static = {c for c, n in occ.items() if n >= 0.6 * N}
    ex = [(a, b) for k, (a, b) in win.items() if k[0] in exercise_prefixes]
    rows = []; prev = []
    for r in rad:
        pts = [(p[0], p[1], p[2], p[3], p[4]) for p in r['points']]
        groups = rc.cluster(pts, eps=0.6, min_pts=1)   # 5-tuples: cluster() only reads [0..2]
        cur = []
        for g in groups:
            g = list(g)
            xs = np.array([[pts_i[0], pts_i[1]] for pts_i in [pts[i] for i in g]]) if isinstance(g[0], int) else np.array([[q[0], q[1]] for q in g])
            # rc.cluster returns lists of point tuples (x,y,z,v); map snr back by identity of coords
            P = [q for q in g]
            snr = [q[4] for q in P]
            cx, cy = float(np.mean([q[0] for q in P])), float(np.mean([q[1] for q in P]))
            vs = np.array([q[3] for q in P]); rng = math.hypot(cx, cy)
            ext = max((math.hypot(a[0] - b[0], a[1] - b[1]) for a in P for b in P), default=0.0)
            disp = min((math.hypot(cx - px, cy - py) for px, py in prev), default=float('nan'))
            f = dict(n_pts=len(P), v_mean_abs=abs(float(vs.mean())), v_spread=float(vs.max() - vs.min()), rcs=float(max(snr) + 40 * math.log10(max(rng, .1))) if snr else 0.0,
                     extent=ext, range=rng, disp=disp, disp_minus_v=(disp - abs(float(vs.mean())) * 0.1) if disp == disp else float('nan'))
            in_static = cell(cx, cy) in static
            in_ex = any(a <= r['t_mono'] <= b for a, b in ex)
            moving = f['v_mean_abs'] >= 0.25 or (disp == disp and disp >= 0.15 and disp < 1.0)
            label = 'clutter' if (in_static and f['v_mean_abs'] < 0.15) else ('person' if (in_ex and moving and not in_static) else None)
            rule = rc.classify(rc.features([(q[0], q[1], q[2], q[3]) for q in P]))
            rows.append(dict(f, session=os.path.basename(sess.rstrip('/')), label=label, rule=rule, t=r['t_mono']))
            cur.append((cx, cy))
        prev = cur
    return rows

def main():
    rows = []
    for s in sys.argv[1:]: rows += dataset(s)
    lab = [r for r in rows if r['label']]
    S = np.array([r['session'] for r in lab]); y = np.array([r['label'] == 'person' for r in lab])
    X = np.nan_to_num(np.array([[r[f] for f in FEATS] for r in lab], float), nan=-1)
    for s in sorted(set(S)): print('%-18s person %5d  clutter %5d' % (s, (y[S == s]).sum(), (~y[S == s]).sum()))
    pred = np.zeros_like(y)
    for s in sorted(set(S)):
        tr, te = S != s, S == s
        if y[tr].sum() == 0 or (~y[tr]).sum() == 0: continue
        clf = RandomForestClassifier(200, min_samples_leaf=5, class_weight='balanced', random_state=0).fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
    def rep(name, yhat):
        cm = confusion_matrix(y, yhat, labels=[True, False]); tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        print('%-26s acc %.2f  person precision %.2f recall %.2f   (tp %d fn %d fp %d tn %d)' % (name, (tp + tn) / len(y), tp / max(tp + fp, 1), tp / max(tp + fn, 1), tp, fn, fp, tn))
    print('\nclusters labelled: %d (person %d, clutter %d) — leave-one-session-out' % (len(y), y.sum(), (~y).sum()))
    rep('RandomForest (LOSO)', pred)
    rule = np.array([str(r['rule']) == 'pedestrian' for r in lab]); rep('rule-based classify_n6', rule)
    clf = RandomForestClassifier(200, min_samples_leaf=5, class_weight='balanced', random_state=0).fit(X, y)
    print('\nfeature importance:'); [print('  %-14s %.3f' % (f, w)) for w, f in sorted(zip(clf.feature_importances_, FEATS), reverse=True)]
    out = os.path.join(ROOT, 'perception', 'out', 'radar_ai'); os.makedirs(out, exist_ok=True)
    pickle.dump({'model': clf, 'features': FEATS, 'sessions': sorted(set(S))}, open(os.path.join(out, 'cluster_model_v0.pkl'), 'wb'))
    json.dump(lab, open(os.path.join(out, 'cluster_dataset_v0.json'), 'w'))
    print('saved cluster_model_v0.pkl / cluster_dataset_v0.json (%d rows)' % len(lab))

if __name__ == '__main__':
    main()
