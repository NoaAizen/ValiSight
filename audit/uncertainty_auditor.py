"""Read-only auditor for core/fusion/uncertainty.py.

Every check below runs an ACTUAL NUMERIC PROBE against the live module —
imported and called, never reasoned about from source. The auditor modifies
nothing; it prints one PASS/FAIL line per check and exits nonzero if any
check fails. Its teeth are proven by audit/uncertainty_mutation_test.py.

Run:  python3 audit/uncertainty_auditor.py
"""
import math
import os
import random
import sys

try:
    from core.fusion import uncertainty as unc
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.fusion import uncertainty as unc

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _est(score, sigma, sensor):
    return unc.SensorEstimate(score=score, sigma=sigma, sensor=sensor)


def _cluster(n=6, v_spread=0.8, v_abs=1.0, extent=0.9):
    return {"n": n, "v_spread": v_spread, "v_abs": v_abs, "extent": extent}


def _rand_pair(rng):
    a = _est(rng.random(), 0.05 + 3.0 * rng.random(), unc.THERMAL)
    b = _est(rng.random(), 0.05 + 3.0 * rng.random(), unc.RADAR)
    return a, b


@check
def inverse_variance_sigma_is_exact():
    """Fused sigma == sqrt(1/sum(1/s_i^2)) over many random pairs."""
    rng = random.Random(1843)
    for _ in range(500):
        a, b = _rand_pair(rng)
        f = unc.combine([a, b])
        want = math.sqrt(1.0 / (a.sigma ** -2 + b.sigma ** -2))
        assert abs(f.sigma - want) < 1e-9, \
            "sigma %.9f, inverse-variance says %.9f" % (f.sigma, want)


@check
def fused_sigma_never_exceeds_most_certain_sensor():
    rng = random.Random(7936)
    for _ in range(500):
        ests = [_est(rng.random(), 0.05 + 3.0 * rng.random(), s)
                for s in (unc.THERMAL, unc.RADAR, "extra")][:rng.randint(2, 3)]
        f = unc.combine(ests)
        assert f.sigma <= min(e.sigma for e in ests) + 1e-9, \
            "fusion made things LESS certain than the best sensor"


@check
def score_is_precision_weighted_not_a_plain_mean():
    rng = random.Random(355)
    for _ in range(500):
        a, b = _rand_pair(rng)
        f = unc.combine([a, b])
        pa, pb = a.sigma ** -2, b.sigma ** -2
        want = (pa * a.score + pb * b.score) / (pa + pb)
        assert abs(f.score - want) < 1e-9, \
            "score %.9f, precision-weighted mean says %.9f" % (f.score, want)
    # and the two formulas genuinely differ on an asymmetric pair
    f = unc.combine([_est(1.0, 0.1, unc.THERMAL), _est(0.0, 1.0, unc.RADAR)])
    assert abs(f.score - 0.5) > 0.3, "score is indistinguishable from a plain mean"


@check
def contributing_sensor_is_argmax_precision():
    rng = random.Random(573)
    for _ in range(500):
        a, b = _rand_pair(rng)
        if abs(a.sigma - b.sigma) < 1e-6:
            continue
        f = unc.combine([a, b])
        want = a.sensor if a.sigma < b.sigma else b.sensor
        assert f.contributing_sensor == want, \
            "contributing=%s but %s has higher precision" % (f.contributing_sensor, want)


@check
def raising_a_sensors_sigma_strictly_lowers_its_weight():
    """Monotonicity, probed black-box: with scores 1 and 0 the fused score IS
    sensor A's weight fraction; it must strictly fall as A's sigma rises."""
    other = _est(0.0, 0.5, unc.RADAR)
    prev = None
    for sigma in (0.1, 0.2, 0.4, 0.8, 1.6, 3.2):
        w = unc.combine([_est(1.0, sigma, unc.THERMAL), other]).score
        assert prev is None or w < prev - 1e-9, \
            "weight did not fall when sigma rose to %.2f" % sigma
        prev = w


@check
def clutter_high_vspread_is_penalised():
    clean = unc.radar_estimate(_cluster(n=8, v_spread=0.4, extent=1.0))
    smeared = unc.radar_estimate(_cluster(n=8, v_spread=2.5, extent=1.0))
    assert smeared.sigma > clean.sigma + 0.1, \
        "multipath smear (v_spread 2.5) not penalised over clean (0.4)"


@check
def compact_glint_is_penalised_2x_velocity_independent():
    """Same kinematics, same point count, only extent differs — so the margin
    cannot come from the static or sparsity terms."""
    body = unc.radar_estimate(_cluster(n=6, v_abs=1.0, extent=0.9))
    glint = unc.radar_estimate(_cluster(n=6, v_abs=1.0, extent=0.2))
    assert glint.sigma >= 2.0 * body.sigma, \
        "moving compact glint sigma %.3f < 2x body sigma %.3f" % (glint.sigma, body.sigma)


@check
def static_aliased_return_is_penalised():
    moving = unc.radar_estimate(_cluster(v_abs=1.0))
    static = unc.radar_estimate(_cluster(v_abs=0.05))
    assert static.sigma > moving.sigma + 0.2, \
        "static/aliased cluster is trusted as much as a moving one"


@check
def thermal_blind_defers_to_radar_without_dropping():
    th = unc.thermal_estimate(None)
    ra = unc.radar_estimate(_cluster(n=10))
    assert th.sigma > 5.0 * ra.sigma, "blind thermal is not marked uncertain"
    f = unc.combine([th, ra])
    assert f.contributing_sensor == unc.RADAR, \
        "blind thermal still owns the decision"
    assert f.score > 0.5, "track collapses when thermal goes blind"
    assert f.sigma < 1.0, "fusion of a good radar cluster stays uncertain"


@check
def outputs_bounded_and_finite_on_random_inputs():
    rng = random.Random(20260729)
    for _ in range(1000):
        box = None if rng.random() < 0.2 else {"confidence": rng.uniform(-0.5, 1.5)}
        cl = _cluster(n=rng.randint(2, 40), v_spread=rng.uniform(0.0, 4.0),
                      v_abs=rng.uniform(0.0, 3.0), extent=rng.uniform(0.05, 3.0))
        th, ra = unc.thermal_estimate(box), unc.radar_estimate(cl)
        for e in (th, ra):
            assert 0.0 <= e.score <= 1.0, "per-sensor score out of [0,1]"
            assert e.sigma > 0.0 and math.isfinite(e.sigma), "bad per-sensor sigma"
        f = unc.combine([th, ra])
        assert 0.0 <= f.score <= 1.0, "fused score out of [0,1]"
        assert f.sigma > 0.0 and math.isfinite(f.sigma), "bad fused sigma"
        assert f.contributing_sensor in (unc.THERMAL, unc.RADAR)


def main():
    failures = 0
    for fn in CHECKS:
        try:
            fn()
        except Exception as e:                       # noqa: BLE001 — a crash is a FAIL
            failures += 1
            print("FAIL  %-52s %s: %s" % (fn.__name__, type(e).__name__, e))
        else:
            print("PASS  %s" % fn.__name__)
    print("%d/%d checks passed" % (len(CHECKS) - failures, len(CHECKS)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
