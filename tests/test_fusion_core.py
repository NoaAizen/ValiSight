"""fusion_core: geometry, matching, low-light and the full radar pipeline."""
import math

import numpy as np

from fusion_core import (CameraGeometry, ClusterMatcher, RadarPipeline,
                         LowLightEnhancer, DARK_CONF_THR)
from conftest import METAL_SNR, SOFT_SNR

SHAPE = (480, 640, 3)


def det(label="person", cx=320, bh=470, bw=100):
    return {"label": label, "conf": 0.9,
            "box": (int(cx - bw / 2), 5, bw, bh)}


def cluster(centroid, material="mid", rng=None):
    x, y, z = centroid
    return {"centroid": centroid, "material": material,
            "range_m": rng if rng is not None else
            round(math.sqrt(x * x + y * y + z * z), 2)}


# --- CameraGeometry -------------------------------------------------------------

def test_box_azimuth_center_and_side():
    geo = CameraGeometry(hfov_deg=60.0)
    az, rng = geo.box_azimuth_range((270, 5, 100, 470), "person", SHAPE)
    assert abs(az) < 0.01                       # centered box -> 0 deg
    az_r, _ = geo.box_azimuth_range((430, 5, 100, 470), "person", SHAPE)
    assert az_r > 10                            # right of centre -> +az


def test_pinhole_range_from_box_height():
    geo = CameraGeometry(hfov_deg=60.0)
    fx = geo.fx(640)
    bh = 470
    _, rng = geo.box_azimuth_range((270, 5, 100, bh), "person", SHAPE)
    assert abs(rng - 1.70 * fx / bh) < 0.01     # person real_h = 1.70


def test_yaw_offset_consistent_between_matching_and_projection():
    """The yaw sign must be identical for the matcher and the overlay —
    they used to disagree (+yaw in matching, -yaw in drawing)."""
    geo = CameraGeometry(hfov_deg=60.0, yaw_offset_deg=5.0)
    boresight = (3.0, 0.0, 0.0)                 # radar azimuth = 0
    assert abs(geo.cluster_azimuth_deg(boresight) - 5.0) < 1e-6
    px, py = geo.project(boresight, SHAPE)
    expected = 320 + geo.fx(640) * math.tan(math.radians(5.0))
    assert abs(px - expected) <= 1


def test_project_rejects_behind_and_outside_fov():
    geo = CameraGeometry(hfov_deg=60.0)
    assert geo.project((-1.0, 0.0, 0.0), SHAPE) is None      # behind
    assert geo.project((1.0, -5.0, 0.0), SHAPE) is None      # ~79 deg right


# --- ClusterMatcher -------------------------------------------------------------

def test_match_prefers_nearest_azimuth():
    geo = CameraGeometry()
    m = ClusterMatcher(geo)
    dets = [det(cx=320)]
    on_axis = cluster((2.0, 0.0, 0.0))
    off_axis = cluster((2.0, -0.35, 0.0))       # ~10 deg right
    assigned = m.match(dets, [off_axis, on_axis], SHAPE)
    assert assigned[0] is on_axis


def test_match_rejects_outside_gate():
    m = ClusterMatcher(CameraGeometry())
    far_left = cluster((2.0, 2.0, 0.0))         # ~45 deg left
    assert m.match([det()], [far_left], SHAPE) == [None]


def test_match_is_one_to_one():
    m = ClusterMatcher(CameraGeometry())
    dets = [det(cx=320), det(cx=320)]
    c = cluster((2.0, 0.0, 0.0))
    assigned = m.match(dets, [c], SHAPE)
    assert assigned.count(c) == 1 and assigned.count(None) == 1


def test_person_box_avoids_metal_glint():
    """Same azimuth + range: the person box must latch onto the body
    cluster, not the metal glint split off an object they are holding."""
    m = ClusterMatcher(CameraGeometry())
    body = cluster((2.0, -0.05, 0.0), material="fabric")
    glint = cluster((2.0, 0.05, 0.0), material="metal")
    assigned = m.match([det("person")], [glint, body], SHAPE)
    assert assigned[0] is body
    # a non-person box gets no such bias
    assigned = m.match([det("chair", bh=250)], [glint, body], SHAPE)
    assert assigned[0] is glint


# --- LowLightEnhancer -----------------------------------------------------------

def test_enhance_brightens_dark_frames_only():
    night = LowLightEnhancer()
    rng = np.random.RandomState(0)
    dark = rng.randint(0, 15, SHAPE).astype(np.uint8)
    out, gamma, mean_b = night.enhance(dark)
    assert gamma < 1.0
    assert out.mean() > dark.mean() * 3
    bright = np.full(SHAPE, 120, np.uint8)
    out, gamma, _ = night.enhance(bright)
    assert gamma == 1.0 and out is bright


def test_conf_threshold_relaxed_only_when_very_dark():
    night = LowLightEnhancer()
    assert night.conf_threshold(10.0, 0.35) == DARK_CONF_THR
    assert night.conf_threshold(40.0, 0.35) == 0.35
    assert night.color_checks_usable(40.0)
    assert not night.color_checks_usable(10.0)


# --- RadarPipeline (integration, offline) ----------------------------------------

def frame_points(v=0.0, snr=SOFT_SNR, x=2.0, n=6):
    pts = []
    for i in range(n):
        off = (i - n / 2.0) / n * 0.4
        pts.append((x + off, -off, off / 2.0, v, snr))
    return pts


def test_pipeline_person_walks_then_stands_still():
    """End-to-end person stickiness through the full pipeline."""
    pipe = RadarPipeline(CameraGeometry())
    t = 0.0
    for _ in range(15):                        # walking
        tcl = pipe.process(frame_points(v=1.0), t)
        t += 1.0 / 15
    assert any(c["label"] == "pedestrian" for c in tcl)
    for _ in range(60):                        # standing still
        tcl = pipe.process(frame_points(v=0.0), t)
        t += 1.0 / 15
    assert any(c["label"] == "pedestrian" for c in tcl)


def test_pipeline_splits_person_from_held_metal():
    pipe = RadarPipeline(CameraGeometry())
    t = 0.0
    for _ in range(12):
        pts = frame_points(v=0.0, snr=SOFT_SNR) + \
              [(2.0, 0.5 + i * 0.03, 0.0, 0.0, METAL_SNR) for i in range(4)]
        tcl = pipe.process(pts, t)
        t += 1.0 / 15
    mats = sorted(c["material"] for c in tcl)
    assert len(tcl) == 2 and mats[-1] == "metal" and mats[0] != "metal"


def test_pipeline_camera_votes_reach_the_tracker():
    pipe = RadarPipeline(CameraGeometry())
    t = 0.0
    for _ in range(10):
        tcl = pipe.process(frame_points(v=0.0), t)
        t += 1.0 / 15
    d = {"label": "person", "conf": 0.9, "box": (270, 5, 100, 470)}
    for _ in range(3):
        pipe.note_camera_matches([d], [tcl[0]])
    tcl = pipe.process(frame_points(v=0.0), t)
    assert tcl[0]["label"] == "person"
