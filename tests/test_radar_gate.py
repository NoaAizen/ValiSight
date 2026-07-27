"""Radar relevance gating: cut the manufactured points, keep the real cluster,
and measure the reduction — without silently dropping real targets.
"""
from core.radar_gate import gate_points


def _cluster(n=5, x0=3.0):
    return [(x0 + 0.1 * i, 0.3, 0.1, 1.0) for i in range(n)]


def test_gate_drops_isolated_ghost_keeps_real_cluster():
    pts = _cluster() + [(6.0, 3.0, 0.0, 0.0)]        # 5-point cluster + 1 ghost
    snr = [20.0] * 6
    noise = [5.0] * 6
    kept, rep = gate_points(pts, snr, noise)
    assert rep.n_in == 6 and rep.n_out == 5
    assert rep.dropped_isolated == 1                  # the lone return is a ghost
    assert rep.reduction_ratio == round(1 - 5 / 6, 3)


def test_weak_return_dropped_as_manufactured():
    pts = _cluster(n=2)                               # close enough to be neighbours
    snr = [2.0, 2.0]
    noise = [3.0, 3.0]                                # abs = 5 dB < MIN_ABS_DB (8)
    kept, rep = gate_points(pts, snr, noise)
    assert rep.dropped_weak == 2
    assert rep.n_out == 0


def test_isolated_single_point_dropped():
    kept, rep = gate_points([(3.0, 0.3, 0.1, 1.0)], snr=[30.0], noise=[5.0])
    assert rep.dropped_isolated == 1 and rep.n_out == 0


def test_out_of_fov_dropped():
    pts = _cluster(n=2) + [(15.0, 0.3, 0.1, 1.0)]    # third beyond R_MAX (9 m)
    kept, rep = gate_points(pts, snr=[30.0] * 3, noise=[5.0] * 3)
    assert rep.dropped_fov == 1
    assert rep.n_out == 2                             # the real pair survives


def test_weak_but_distant_point_is_spared():
    # a weak pair at ~6 m (beyond REFL_MAX_RANGE_M): the reflectivity gate must
    # NOT delete it — beyond ~4 m the level is not discriminative (survivor bias)
    pts = [(6.0, 0.3, 0.1, 1.0), (6.1, 0.3, 0.1, 1.0)]
    snr = [2.0, 2.0]
    noise = [3.0, 3.0]                                # abs = 5 dB < MIN_ABS_DB
    kept, rep = gate_points(pts, snr, noise)
    assert rep.dropped_weak == 0                      # spared: too far to gate on level
    assert rep.n_out == 2


def test_no_side_info_skips_reflectivity_gate():
    # without snr/noise the reflectivity gate cannot be applied honestly
    kept, rep = gate_points(_cluster())
    assert rep.dropped_weak == 0 and rep.n_out == 5


def test_empty_frame():
    kept, rep = gate_points([])
    assert kept == [] and rep.reduction_ratio == 0.0
