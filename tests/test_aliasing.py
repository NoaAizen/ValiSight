"""The fold sweep. This is the only offline check on the aliasing work.

A walked recording does not exist yet, so the alias detector cannot be validated
against real motion. What CAN be validated, and is, is the geometry and the fold
arithmetic: take the real bearing distribution out of a recorded session, impose
a known ego-velocity, fold the radial velocities onto the real 16-bin Doppler
grid exactly as the hardware would, and check that the estimator recovers the
velocity it was given.

That is a genuine regression test -- only the Doppler is synthetic; the bearings,
the point counts and the clustering are the rig's own. It is NOT a substitute for
a walked session: it has no micro-Doppler from limbs, no moving people to outvote
the world, and no multipath. Those are what `alias_unresolved` exists for.

Run:  cd <project root> && python3 -m unittest discover -s tests -t .
"""
import json
import math
import os
import unittest

import _path                                    # noqa: F401  (sys.path side effect)

import ego_velocity as ev
import radar_static as rs

SESSION = os.path.join(_path.RECORDINGS, "20260728_203833", "radar_frames.jsonl")
MAX_FRAMES = 300


def _load_geometry(limit=MAX_FRAMES):
    """Real conditioned frames from the recording, for their bearings only."""
    out = []
    if not os.path.exists(SESSION):
        return out
    with open(SESSION) as fh:
        for line in fh:
            if len(out) >= limit:
                break
            try:
                r = json.loads(line)
            except ValueError:
                continue
            pts = rs.condition(r.get("pts", []), r.get("snr"), r.get("noise"))
            if len(pts) >= 4:
                out.append(pts)
    return out


def _impose(pts, vx, vy, grid):
    """Overwrite each point's vr with what the hardware would have reported."""
    for p in pts:
        true_vr = -(p.u[0] * vx + p.u[1] * vy)
        folded = grid.wrap(true_vr)
        p.vr = round(folded / grid.bin_mps) * grid.bin_mps      # peak bin, no interp


class TestDopplerGrid(unittest.TestCase):

    def test_stock_grid_matches_measured_hardware(self):
        """0.121715 m/s and 16 levels, measured over 60144 real points."""
        g = rs.STOCK_GRID
        self.assertAlmostEqual(g.bin_mps, 0.121715, delta=1e-5)
        self.assertEqual(g.n_bins, 16)
        self.assertAlmostEqual(g.v_max, 0.973723, delta=1e-4)

    def test_wrap_is_an_involution_on_the_grid(self):
        g = rs.STOCK_GRID
        for k in range(-8, 8):
            v = k * g.bin_mps
            self.assertAlmostEqual(g.wrap(v), v, delta=1e-9)
            # one whole fold away must land back on the same value
            self.assertAlmostEqual(g.wrap(v + g.fold), v, delta=1e-9)
            self.assertAlmostEqual(g.wrap(v - g.fold), v, delta=1e-9)

    def test_infer_recovers_the_grid_from_values_alone(self):
        g = rs.STOCK_GRID
        vals = [k * g.bin_mps for k in range(-8, 8)]
        got = rs.infer_doppler_grid(vals)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got.bin_mps, g.bin_mps, delta=1e-9)
        self.assertEqual(got.n_bins, 16)

    def test_infer_refuses_values_off_a_single_grid(self):
        self.assertIsNone(rs.infer_doppler_grid([0.1, 0.15, 0.23]))


class TestFoldRecovery(unittest.TestCase):
    """The estimator must recover velocities the Doppler axis cannot represent."""

    @classmethod
    def setUpClass(cls):
        cls.frames = _load_geometry()
        cls.grid = rs.STOCK_GRID

    def setUp(self):
        if not self.frames:
            self.skipTest("no recorded session to take bearings from")

    def _sweep(self, vx, vy):
        errs, aliased, n = [], 0, 0
        for pts in self.frames:
            _impose(pts, vx, vy, self.grid)
            sol = ev.solve(pts)
            if sol.v is None:
                continue
            n += 1
            aliased += 1 if sol.aliased else 0
            errs.append(math.hypot(sol.v[0] - vx, sol.v[1] - vy))
        errs.sort()
        return errs[len(errs) // 2], (aliased / n if n else 0.0), n

    def test_below_the_fold_is_unaffected(self):
        """0.90 m/s needs no unwrap, and must not be flagged as one."""
        err, alias_frac, n = self._sweep(0.90, 0.0)
        self.assertLess(err, 0.12, "error at 0.90 m/s")
        self.assertLess(alias_frac, 0.05, "false alias rate below the fold")

    def test_walking_speed_past_the_fold_is_recovered(self):
        """1.4 m/s folds to -0.55. The old estimator reported that as truth."""
        err, alias_frac, n = self._sweep(1.40, 0.0)
        self.assertLess(err, 0.20, "1.4 m/s must be recovered, not reported folded")
        self.assertGreater(alias_frac, 0.80, "the wrap must be detected")

    def test_recovery_holds_with_a_lateral_component(self):
        err, _, _ = self._sweep(1.40, 0.40)
        self.assertLess(err, 0.20)

    def test_recovery_holds_well_past_the_fold(self):
        """2.0 m/s folds to 0.05 -- the old estimator called this stationary."""
        for vx in (2.00, 2.60):
            err, alias_frac, _ = self._sweep(vx, 0.0)
            self.assertLess(err, 0.25, "error at %.2f m/s" % vx)
            self.assertGreater(alias_frac, 0.70, "wrap detection at %.2f" % vx)


class TestNoFalseAliasOnRealData(unittest.TestCase):
    """The truth on the recorded session is v = 0. Every alias claim is false."""

    def test_false_alias_rate_stays_low(self):
        frames = _load_geometry(limit=600)
        if not frames:
            self.skipTest("no recorded session")
        aliased = n = 0
        for pts in frames:
            sol = ev.solve(pts)                 # untouched vr, i.e. real data
            if sol.v is None:
                continue
            n += 1
            aliased += 1 if sol.aliased else 0
        # Measured 0.2% with ALIAS_COST_MARGIN = 0.5. Loosening the margin to
        # 1.0 (lowest cost wins) takes this to 1.2%, which is the reason the
        # margin exists at all.
        self.assertLess(aliased / n, 0.02,
                        "false alias rate on static data (%d/%d)" % (aliased, n))


if __name__ == "__main__":
    unittest.main()
