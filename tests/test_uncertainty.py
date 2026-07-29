"""Uncertainty-guided fusion: the four demo beats, in numbers.

The module under test is pure (core/fusion/uncertainty.py), so unlike the
aliasing sweep nothing here needs a recording. Each test is one sensing
situation from the demo script — agreement in the open, thermal blind in the
dark, a carried metal glint, an aliased/static return — plus the raw
inverse-variance arithmetic those beats all stand on.

Run:  cd <project root> && python3 -m unittest discover -s tests

This file also runs standalone from a bare copy (the mutation harness copies
it next to a mutated module), so the usual `import _path` is optional here:
without it the classifier cross-check is skipped, everything else runs.
"""
import math
import os
import sys
import unittest

try:
    import _path                            # noqa: F401  (sys.path side effect)
    if _path.ROOT not in sys.path:
        sys.path.insert(0, _path.ROOT)
except ImportError:
    _path = None

from core.fusion import uncertainty as unc


def _body_cluster(n=6, v_spread=0.8, v_abs=1.0, extent=0.9):
    """A walking person as radar_classify_n6.features() reports one."""
    return {"n": n, "v_spread": v_spread, "v_abs": v_abs, "extent": extent}


class TestInverseVariance(unittest.TestCase):

    def test_two_equal_sensors_shrink_sigma_by_sqrt2(self):
        a = unc.SensorEstimate(score=0.2, sigma=0.4, sensor=unc.THERMAL)
        b = unc.SensorEstimate(score=0.8, sigma=0.4, sensor=unc.RADAR)
        f = unc.combine([a, b])
        self.assertAlmostEqual(f.sigma, 0.4 / math.sqrt(2.0), delta=1e-12)
        self.assertAlmostEqual(f.score, 0.5, delta=1e-12)

    def test_score_is_precision_weighted_not_a_plain_mean(self):
        a = unc.SensorEstimate(score=1.0, sigma=0.1, sensor=unc.THERMAL)
        b = unc.SensorEstimate(score=0.0, sigma=1.0, sensor=unc.RADAR)
        f = unc.combine([a, b])
        # weights 100:1, so the plain mean of 0.5 would be badly wrong
        self.assertAlmostEqual(f.score, 100.0 / 101.0, delta=1e-12)
        self.assertEqual(f.contributing_sensor, unc.THERMAL)


class TestDemoBeats(unittest.TestCase):

    def test_beat1_agreement_raises_confidence_and_certainty(self):
        """Open light: both sensors see the person and the fusion is more
        certain than either alone."""
        th = unc.thermal_estimate({"confidence": 0.9})
        ra = unc.radar_estimate(_body_cluster(n=12))
        f = unc.combine([th, ra])
        self.assertGreater(f.score, 0.9)
        self.assertLess(f.sigma, min(th.sigma, ra.sigma))

    def test_beat3_thermal_blind_defers_to_radar_without_dropping(self):
        """Dark + occlusion: no thermal box at all, a good moving cluster.
        The indicator flips to radar and the track must not collapse."""
        th = unc.thermal_estimate(None)
        ra = unc.radar_estimate(_body_cluster(n=10))
        f = unc.combine([th, ra])
        self.assertEqual(f.contributing_sensor, unc.RADAR)
        self.assertGreater(f.score, 0.5, "track collapsed on thermal blindness")
        self.assertLess(f.sigma, 1.0)

    def test_moving_metal_glint_is_distrusted_but_velocity_ignored(self):
        """Metal carried BY a walking person: same kinematics and point count
        as the body, compact extent. The glint must be distrusted on shape
        alone -- the 2x margin must not lean on sparsity or velocity."""
        body = unc.radar_estimate(_body_cluster(n=6, extent=0.9))
        glint = unc.radar_estimate(_body_cluster(n=6, extent=0.2))
        self.assertGreaterEqual(glint.sigma, 2.0 * body.sigma)
        # and the person decision rides on thermal, not on the glint
        f = unc.combine([unc.thermal_estimate({"confidence": 0.85}), glint])
        self.assertEqual(f.contributing_sensor, unc.THERMAL)

    def test_static_aliased_cluster_is_less_trusted_than_moving(self):
        moving = unc.radar_estimate(_body_cluster(v_abs=1.0))
        static = unc.radar_estimate(_body_cluster(v_abs=0.1))
        self.assertGreater(static.sigma, moving.sigma + 0.2)


class TestSeams(unittest.TestCase):

    def test_static_threshold_matches_the_radar_classifier(self):
        """RADAR_STATIC_V mirrors radar_classify_n6.STATIC_V by value, not by
        import; this is the drift alarm."""
        if _path is None:
            self.skipTest("bare copy: classifier source not present")
        import radar_classify_n6
        self.assertEqual(unc.RADAR_STATIC_V, radar_classify_n6.STATIC_V)

    def test_accessors_take_features_dict_aliases_and_objects(self):
        via_features = unc.radar_estimate(_body_cluster())
        via_aliases = unc.radar_estimate(
            {"n_points": 6, "v_spread": 0.8, "v_abs": 1.0, "extent_m": 0.9})

        class C:
            n, v_spread, v_abs, extent = 6, 0.8, 1.0, 0.9
        via_object = unc.radar_estimate(C())
        for e in (via_aliases, via_object):
            self.assertAlmostEqual(e.sigma, via_features.sigma, delta=1e-12)
            self.assertAlmostEqual(e.score, via_features.score, delta=1e-12)

        class B:
            confidence = 0.7
        self.assertAlmostEqual(unc.thermal_estimate(B()).score, 0.7, delta=1e-12)
        self.assertAlmostEqual(unc.thermal_estimate({"score": 0.7}).score, 0.7,
                               delta=1e-12)


if __name__ == "__main__":
    unittest.main()
