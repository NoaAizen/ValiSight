"""
Empirical range-baseline for radar reflectivity scoring (IWR1843).

Why: the old metric snr + 40*log10(R) assumed point-target R^-4 falloff, but
the mmWave demo only reports CFAR *detections* — points below the threshold
vanish instead of appearing weak, so the reported SNR hugs the CFAR floor at
every range (measured over 25k logged points: median 12-15 dB from 0.4 m to
12 m, correlation with 40log10(R) = -0.18). Adding 40log10(R) therefore
produced a distance meter, not a material meter: 96.8% of the old material
labels were reproducible from range alone.

Fix: score each point against what the sensor *typically* reports at that
range, measured from this rig's own session logs:

    score_db = signal - baseline(range)

where signal is snr + noise (absolute level, TLV 7) when the noise field is
available, else raw snr. A score of 0 dB means "as strong as the median
static return at that range"; metal glints sit well above, absorbers at/below.
Range-independent by construction — no propagation model needed.

Build/refresh the baseline table from all recorded sessions:

    python radar_calibration.py                 # scans logs/, writes
                                                # configs/refl_baseline.json
    python radar_calibration.py --logs D:\\other\\logs --out my_baseline.json

The Baseline class is MCU-friendly (pure Python + one small JSON table).
"""
import json
import math
import os

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "configs", "refl_baseline.json")
BIN_M = 0.5              # baseline range-bin width
MIN_BIN_PTS = 40         # bins with fewer samples are dropped (interpolated)
FALLBACK_SNR_DB = 14.0   # median CFAR-detection SNR measured over all logs;
                         # used when no baseline file exists


class Baseline(object):
    """Piecewise-linear expected-signal-vs-range curves ('snr' and 'abs')."""

    def __init__(self, curves=None, fallback_db=FALLBACK_SNR_DB):
        # curves: {"snr": [[r, db], ...] sorted by r, "abs": [[r, db], ...]}
        self.curves = {k: v for k, v in (curves or {}).items() if v}
        self.fallback_db = fallback_db

    @classmethod
    def load(cls, path=BASELINE_PATH, quiet=False):
        """Load a built baseline; falls back to a flat curve if missing."""
        try:
            with open(path, "r") as f:
                d = json.load(f)
            return cls(d.get("curves"), d.get("fallback_db", FALLBACK_SNR_DB))
        except (IOError, OSError, ValueError):
            if not quiet:
                print("radar_calibration: no baseline file (%s) — using flat "
                      "%.1f dB fallback. Run: python radar_calibration.py"
                      % (path, FALLBACK_SNR_DB))
            return cls()

    def expected(self, r, mode):
        """Interpolated typical signal (dB) at range r for 'snr'/'abs' mode."""
        curve = self.curves.get(mode)
        if not curve:
            return None
        if r <= curve[0][0]:
            return curve[0][1]
        if r >= curve[-1][0]:
            return curve[-1][1]
        for i in range(1, len(curve)):
            r1, v1 = curve[i]
            if r <= r1:
                r0, v0 = curve[i - 1]
                f = (r - r0) / max(r1 - r0, 1e-6)
                return v0 + f * (v1 - v0)
        return curve[-1][1]

    def score(self, r, snr_db, noise_db=None):
        """dB above the typical return at this range (range-normalized).

        Uses absolute signal (snr+noise) when the point carries the noise
        field AND an 'abs' curve was built; otherwise scores raw snr against
        the 'snr' curve (or the flat fallback)."""
        if snr_db is None:
            return None
        if noise_db is not None:
            exp = self.expected(r, "abs")
            if exp is not None:
                return snr_db + noise_db - exp
        exp = self.expected(r, "snr")
        return snr_db - (self.fallback_db if exp is None else exp)


# ---------------------------------------------------------------------------
# builder (PC only): scan logs/*/radar.jsonl -> baseline JSON
# ---------------------------------------------------------------------------

def _median(v):
    v = sorted(v)
    return v[len(v) // 2]


def _pct(v, p):
    v = sorted(v)
    return v[min(int(len(v) * p / 100.0), len(v) - 1)]


def _collect_points(logs_dir):
    """Yield (r, snr, noise_or_None) from every session's radar.jsonl."""
    import glob
    for path in sorted(glob.glob(os.path.join(logs_dir, "*", "radar.jsonl"))):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue                       # torn last line
                for p in rec.get("points", ()):
                    if len(p) < 5 or p[4] is None:
                        continue
                    r = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
                    noise = p[5] if len(p) > 5 and p[5] is not None else None
                    yield r, p[4], noise


def _build_curve(samples, bin_m=BIN_M, min_pts=MIN_BIN_PTS):
    """samples: [(r, db)] -> [[bin_center, median_db], ...] (sparse bins skipped)."""
    bins = {}
    for r, db in samples:
        bins.setdefault(int(r / bin_m), []).append(db)
    curve = []
    for b in sorted(bins):
        vals = bins[b]
        if len(vals) >= min_pts:
            curve.append([round((b + 0.5) * bin_m, 2),
                          round(_median(vals), 1)])
    return curve


def build_baseline(logs_dir, out_path=BASELINE_PATH):
    snr_samples, abs_samples = [], []
    for r, snr, noise in _collect_points(logs_dir):
        snr_samples.append((r, snr))
        if noise is not None:
            abs_samples.append((r, snr + noise))
    if not snr_samples:
        raise SystemExit("No points found under %s — record a session first."
                         % logs_dir)

    curves = {"snr": _build_curve(snr_samples)}
    if abs_samples:
        curves["abs"] = _build_curve(abs_samples)

    out = {
        "curves": curves,
        "fallback_db": round(_median([s for _, s in snr_samples]), 1),
        "n_points": {"snr": len(snr_samples), "abs": len(abs_samples)},
        "bin_m": BIN_M,
        "source": os.path.abspath(logs_dir),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # report: how flat are the normalized scores, and where do the strong
    # returns sit? (helps picking metal/fabric score thresholds)
    bl = Baseline(curves, out["fallback_db"])
    print("Baseline written: %s" % out_path)
    for mode, samples in (("snr", snr_samples), ("abs", abs_samples)):
        if not samples:
            print("  abs curve: no data yet — record new sessions (the noise "
                  "field is logged from now on), then rebuild.")
            continue
        scores = []
        for r, db in samples:
            e = bl.expected(r, mode)
            if e is not None:
                scores.append(db - e)
        print("  %-3s curve: %d bins from %d points | score p50 %+.1f  "
              "p90 %+.1f  p99 %+.1f dB"
              % (mode, len(curves.get(mode, [])), len(samples),
                 _median(scores), _pct(scores, 90), _pct(scores, 99)))
    print("Suggested thresholds: metal >= ~p99 score, fabric <= ~p50+2 dB "
          "(verify with a known metal/fabric object at 1-3 m).")
    return out


if __name__ == "__main__":
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):    # Hebrew paths on cp1252 consoles
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        description="Build the empirical reflectivity baseline from session logs")
    ap.add_argument("--logs", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs"),
        help="logs directory (default: ./logs)")
    ap.add_argument("--out", default=BASELINE_PATH,
                    help="output JSON path (default: configs/refl_baseline.json)")
    args = ap.parse_args()
    build_baseline(args.logs, args.out)
