"""radar_tracker: confirmation, sticky person identity, materials, timeouts."""
import math

from radar_classify_n6 import classify_frame
from radar_material import annotate_clusters
from radar_tracker import (Tracker, Track, CONFIRM_HITS, MISS_TIMEOUT_S,
                           PERSON_TIMEOUT_S, CAM_PERSON_VOTES,
                           SIGNATURE_FIELDS)
from conftest import METAL_SNR, SOFT_SNR

DT = 1.0 / 15                       # camera-ish frame period


def make_cluster(x=2.0, y=0.0, v=0.0, n=6, snr=SOFT_SNR, spread=0.4):
    pts = []
    for i in range(n):
        off = (i - n / 2.0) / n * spread
        pts.append((x + off, y - off, off / 2.0, v, snr))
    clusters = annotate_clusters(classify_frame(pts, include_points=True))
    assert len(clusters) == 1
    return clusters[0]


def feed(tracker, cluster_fn, n_frames, t0=0.0):
    t = t0
    tracks = []
    for _ in range(n_frames):
        tracks = tracker.update([cluster_fn()], t)
        t += DT
    return tracks, t


def test_track_confirmed_after_enough_hits():
    tr = Tracker()
    tracks, _ = feed(tr, lambda: make_cluster(), CONFIRM_HITS - 1)
    assert tracks == []                        # still tentative
    tracks, _ = feed(tr, lambda: make_cluster(), 1,
                     t0=(CONFIRM_HITS - 1) * DT)
    assert len(tracks) == 1


def test_static_object_stays_static():
    tr = Tracker()
    tracks, _ = feed(tr, lambda: make_cluster(v=0.0), 30)
    assert tracks[0].as_cluster()["label"] == "static"
    assert not tracks[0].person_evidence()


def test_pedestrian_sticks_after_stopping():
    """A person who walked and then stood still must stay 'pedestrian' —
    the dark-scene fix: Doppler goes to 0 but identity is kept."""
    tr = Tracker()
    _, t = feed(tr, lambda: make_cluster(v=1.0), 15)          # walking
    tracks, _ = feed(tr, lambda: make_cluster(v=0.0), 60, t0=t)   # standing
    c = tracks[0].as_cluster()
    assert c["label"] == "pedestrian"
    # sanity: the raw doppler really is static-level now
    assert tracks[0].v_abs.mean < 0.25


def test_camera_votes_lock_person_identity():
    tr = Tracker()
    tracks, _ = feed(tr, lambda: make_cluster(v=0.0), 10)
    tid = tracks[0].id
    for _ in range(CAM_PERSON_VOTES):
        tr.note_camera(tid, "person")
    assert tracks[0].as_cluster()["label"] == "person"
    assert tracks[0].signature()["cam_label"] == "person"


def test_moving_metal_object_is_not_a_person():
    """A carried metal object moves like its carrier; without camera
    confirmation it must be displayed as 'object', not 'pedestrian'."""
    tr = Tracker()
    tracks, _ = feed(tr, lambda: make_cluster(v=1.0, snr=METAL_SNR), 10)
    c = tracks[0].as_cluster()
    assert c["material"] == "metal"
    assert c["label"] == "object"
    assert not tracks[0].person_evidence()


def test_track_material_p90_ignores_absorbed_glints():
    """A person track that absorbed a few metal glint points must not flip
    to 'metal' — p90 stays with the majority of samples."""
    tr = Tracker()
    _, t = feed(tr, lambda: make_cluster(v=0.0, snr=SOFT_SNR), 5)  # 30 soft
    tracks = tr.update([make_cluster(v=0.0, n=3, snr=METAL_SNR)], t)  # 3 glints
    assert tracks[0].material_class() != "metal"


def test_person_track_survives_longer_gaps():
    tr = Tracker()
    tracks, t = feed(tr, lambda: make_cluster(v=0.0), 10)
    tid = tracks[0].id
    for _ in range(CAM_PERSON_VOTES):
        tr.note_camera(tid, "person")
    # a gap longer than the normal timeout but within the person timeout
    gap = (MISS_TIMEOUT_S + PERSON_TIMEOUT_S) / 2.0
    tracks = tr.update([], t + gap)
    assert [tr_.id for tr_ in tracks] == [tid]
    # and past the person timeout it finally expires
    assert tr.update([], t + PERSON_TIMEOUT_S + 0.5) == []


def test_non_person_track_expires_at_normal_timeout():
    tr = Tracker()
    _, t = feed(tr, lambda: make_cluster(v=0.0), 10)
    assert tr.update([], t + MISS_TIMEOUT_S + 0.3) == []


def test_body_and_metal_keep_separate_tracks():
    """Material-aware association: two co-located clusters (soft body +
    metal glint) must feed two stable tracks, not swap into one."""
    tr = Tracker()
    t = 0.0
    for _ in range(12):
        body = make_cluster(x=2.0, y=-0.3, v=0.0, snr=SOFT_SNR)
        glint = make_cluster(x=2.0, y=0.3, v=0.0, n=4, snr=METAL_SNR)
        tracks = tr.update([body, glint], t)
        t += DT
    mats = sorted(tr_.material_class() for tr_ in tracks)
    assert len(tracks) == 2
    assert mats[-1] == "metal" and mats[0] != "metal"


def test_signature_vector_matches_field_order():
    tr = Tracker()
    tracks, _ = feed(tr, lambda: make_cluster(), 10)
    sig = tracks[0].signature()
    assert len(sig["vector"]) == len(SIGNATURE_FIELDS)
    assert abs(sig["vector"][0] - sig["features"]["range_m"]) < 1e-6
    assert abs(sum(sig["doppler_hist"]) - 1.0) < 0.01
