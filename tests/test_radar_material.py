"""radar_material: scoring, classification and mixed-cluster splitting."""
import math

from radar_calibration import Baseline
from radar_classify_n6 import classify_frame
from radar_material import (material, peak_score, point_scores,
                            annotate_clusters, split_mixed_clusters,
                            METAL_DB, FABRIC_DB)
from conftest import METAL_SNR, SOFT_SNR, MID_SNR


def pt(x, y, z, v=0.0, snr=None, noise=None):
    p = (x, y, z, v)
    if snr is not None:
        p += (snr,) if noise is None else (snr, noise)
    return p


# --- material() ----------------------------------------------------------------

def test_material_thresholds():
    assert material(METAL_DB) == "metal"
    assert material(FABRIC_DB) == "fabric"
    assert material((METAL_DB + FABRIC_DB) / 2) == "mid"
    assert material(None) == "unknown"


def test_material_range_gate():
    assert material(20.0, range_m=5.0) == "unknown"   # beyond max_range
    assert material(20.0, range_m=2.0) == "metal"


# --- peak_score() ----------------------------------------------------------------

def test_peak_score_uses_second_highest_with_enough_points():
    # one lone outlier must not flip a cluster to metal
    assert peak_score([1.0, 2.0, 3.0, 25.0]) == 3.0


def test_peak_score_max_for_few_points():
    assert peak_score([1.0, 25.0]) == 25.0
    assert peak_score([]) is None


# --- point_scores() with the flat fallback baseline ------------------------------

def test_point_scores_flat_baseline():
    scores = point_scores([pt(2, 0, 0, snr=20.0)], Baseline())
    assert scores == [6.0]                     # 20 - 14 fallback


def test_point_scores_skips_missing_snr():
    scores = point_scores([pt(2, 0, 0), pt(2, 0, 0, snr=None)], Baseline())
    assert scores == []


# --- split_mixed_clusters() -------------------------------------------------------

def mixed_cluster(glint_n=4, body_n=6, glint_spread=0.2, body_snr=SOFT_SNR,
                  rng=2.0):
    """One merged cluster: a compact metal glint core inside a soft body."""
    pts = []
    for i in range(glint_n):                   # glint core at (rng, 0.3)
        off = (i - glint_n / 2.0) / glint_n * glint_spread
        pts.append(pt(rng + off, 0.3 + off, 0.0, snr=METAL_SNR))
    for i in range(body_n):                    # body around (rng, -0.3)
        off = (i - body_n / 2.0) / body_n * 0.5
        pts.append(pt(rng + off, -0.3 - off, off, snr=body_snr))
    clusters = classify_frame(pts, include_points=True)
    assert len(clusters) == 1                  # they really did merge
    return annotate_clusters(clusters)


def test_split_separates_metal_core_from_soft_body():
    out = split_mixed_clusters(mixed_cluster())
    assert len(out) == 2
    mats = sorted(c["material"] for c in out)
    assert mats == ["fabric", "metal"]
    metal = next(c for c in out if c["material"] == "metal")
    body = next(c for c in out if c["material"] == "fabric")
    assert metal["n_points"] == 4 and body["n_points"] == 6


def test_split_leaves_far_clusters_alone():
    out = split_mixed_clusters(mixed_cluster(rng=6.0))    # beyond max_range
    assert len(out) == 1


def test_split_needs_enough_glint_points():
    out = split_mixed_clusters(mixed_cluster(glint_n=2))
    assert len(out) == 1


def test_split_needs_soft_body():
    # body at mid level (score +6 > fabric threshold) -> could be one solid
    # metal-ish object; do not split
    out = split_mixed_clusters(mixed_cluster(body_snr=MID_SNR))
    assert len(out) == 1


def test_split_rejects_scattered_glints():
    # glint points spread over > 2 * SPLIT_CORE_RADIUS_M -> speckle, no split
    out = split_mixed_clusters(mixed_cluster(glint_spread=2.0))
    assert len(out) == 1


def test_split_passthrough_without_points():
    c = {"label": "static", "range_m": 2.0, "material": "unknown"}
    assert split_mixed_clusters([c]) == [c]
