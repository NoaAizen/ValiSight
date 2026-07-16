"""radar_classify_n6: clustering, features and rule-based labels."""
from radar_classify_n6 import (cluster, features, classify, classify_frame,
                               cluster_dict, CLUSTER_EPS, STATIC_V)


def blob(x, y, z=0.0, v=0.0, n=6, spread=0.2, snr=None):
    """n points around (x, y, z) with doppler v."""
    pts = []
    for i in range(n):
        off = (i - n / 2.0) / n * spread
        p = (x + off, y - off, z + off / 2.0, v)
        if snr is not None:
            p += (snr,)
        pts.append(p)
    return pts


def test_cluster_merges_nearby_and_separates_far():
    groups = cluster(blob(2, 0) + blob(8, 0))
    assert len(groups) == 2


def test_cluster_drops_lone_points():
    assert cluster([(2.0, 0.0, 0.0, 0.0)]) == []          # min_pts = 2


def test_cluster_eps_boundary():
    # two pairs just over CLUSTER_EPS apart stay separate
    a = [(0.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0, 0.0)]
    b = [(0.1 + CLUSTER_EPS + 0.05, 0.0, 0.0, 0.0),
         (0.2 + CLUSTER_EPS + 0.05, 0.0, 0.0, 0.0)]
    assert len(cluster(a + b)) == 2


def test_classify_static_below_doppler_threshold():
    f = features(blob(3, 0, v=STATIC_V / 2))
    assert classify(f) == "static"


def test_classify_pedestrian_small_and_moving():
    f = features(blob(3, 0, v=1.0, n=5, spread=0.5))
    assert classify(f) == "pedestrian"


def test_classify_vehicle_large_many_points():
    pts = blob(5, 0, v=4.0, n=4, spread=0.1) + \
          blob(6.1, 1.1, v=4.0, n=4, spread=0.1)          # extent > 1.5 m
    f = features(pts)
    assert f["extent"] > 1.5 and f["n"] >= 6
    assert classify(f) == "vehicle"


def test_cluster_dict_fields_and_points_passthrough():
    pts = blob(3, 1, v=0.0, snr=20.0)
    c = cluster_dict(pts, include_points=True)
    assert set(c) >= {"label", "range_m", "doppler_mps", "extent_m",
                      "n_points", "centroid", "points"}
    assert c["n_points"] == len(pts)
    assert len(c["points"][0]) == 5           # extra snr field preserved
    assert cluster_dict(pts).get("points") is None


def test_classify_frame_end_to_end():
    out = classify_frame(blob(2, 0, v=0.0) + blob(6, -2, v=1.2, n=5))
    labels = sorted(c["label"] for c in out)
    assert labels == ["pedestrian", "static"]
