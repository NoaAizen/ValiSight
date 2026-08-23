#!/usr/bin/env python3
import unittest

from perception.decision import Evidence, decide


class DecisionTest(unittest.TestCase):
    def test_both_positive_raise_confidence(self):
        d = decide(Evidence(.8, 20), Evidence(.8, 30))
        self.assertEqual((d.label, d.provenance), ('PERSON', 'BOTH'))
        self.assertGreater(d.probability, .8)

    def test_thermal_only_degrades_explicitly(self):
        d = decide(Evidence(.85, 20), None)
        self.assertEqual((d.label, d.provenance), ('PERSON', 'THERMAL_ONLY'))
        self.assertIn('single-modality', d.caveats)
        self.assertIn('radar-missing', d.caveats)

    def test_radar_only_degrades_explicitly(self):
        d = decide(None, Evidence(.82, 20))
        self.assertEqual((d.label, d.provenance), ('PERSON', 'RADAR_ONLY'))

    def test_stale_evidence_is_not_fused(self):
        d = decide(Evidence(.99, 500), Evidence(.8, 20))
        self.assertEqual(d.provenance, 'RADAR_ONLY')
        self.assertIn('thermal-stale', d.caveats)

    def test_invalid_probabilities_fail_closed(self):
        d = decide(Evidence(1.2, 10), None)
        self.assertEqual((d.label, d.provenance), ('UNKNOWN', 'NONE'))
        self.assertIn('thermal-uncalibrated-range', d.caveats)

    def test_strong_contradiction_is_unknown(self):
        d = decide(Evidence(.9, 10), Evidence(.1, 10))
        self.assertEqual((d.label, d.provenance), ('UNKNOWN', 'BOTH'))
        self.assertIn('sensor-contradiction', d.caveats)

    def test_both_negative(self):
        d = decide(Evidence(.1, 10), Evidence(.2, 10))
        self.assertEqual(d.label, 'NO_PERSON')


if __name__ == '__main__':
    unittest.main()
