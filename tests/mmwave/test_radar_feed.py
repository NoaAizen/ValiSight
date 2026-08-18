"""בדיקות ל-radar_feed — הממשק של יעל (סעיפים 1 ו-3 במסמך הדרישות).

כל בדיקה מסבירה בכותרת שלה איזו דרישה היא מוודאת.
הרצה:  python3 -m pytest tests/mmwave/test_radar_feed.py -v
"""
import math

import pytest

import radar_feed as rf
from radar_feed import (RadarFeed, SimulatedSource, ParsedFrameSource,
                        detection_from_point, point_from_detection,
                        estimate_ego_velocity, STATIC_V, FRAME_MS)
from mmwave_parser import parse_frame
from frame_builder import build_frame


# ---------------------------------------------------------------------------
# עזר: שעון ידני שאפשר להזיז
# ---------------------------------------------------------------------------

class ManualClock:
    def __init__(self, ms=0):
        self.ms = ms

    def __call__(self):
        return self.ms

    def advance(self, ms=FRAME_MS):
        self.ms += ms


REQUIRED_FIELDS = {'range_m', 'azimuth_deg', 'elevation_deg',
                   'velocity_mps', 'snr_db', 'timestamp_ms'}


# ---------------------------------------------------------------------------
# 1.1 radar_detections_all — שדות, מוסכמות, "הכל כולל סטטיים"
# ---------------------------------------------------------------------------

def test_detection_has_every_field_yael_asked_for():
    d = detection_from_point(1.0, 2.0, 0.1, -0.5, 153, 1234)
    assert REQUIRED_FIELDS <= set(d)
    assert d['timestamp_ms'] == 1234
    assert d['snr_db'] == pytest.approx(15.3)      # 0.1dB units -> dB


def test_azimuth_positive_is_right_elevation_positive_is_up():
    # TI: x ימינה, y קדימה, z למעלה
    right = detection_from_point(1.0, 1.0, 0.0, 0.0, None, 0)
    left = detection_from_point(-1.0, 1.0, 0.0, 0.0, None, 0)
    up = detection_from_point(0.0, 1.0, 1.0, 0.0, None, 0)
    assert right['azimuth_deg'] == pytest.approx(45.0)
    assert left['azimuth_deg'] == pytest.approx(-45.0)
    assert up['elevation_deg'] == pytest.approx(45.0)
    assert right['range_m'] == pytest.approx(math.sqrt(2))


def test_spherical_roundtrip_is_exact():
    for xyz in [(1.2, 3.4, -0.3), (-2.0, 0.5, 0.9), (0.0, 7.0, 0.0)]:
        d = detection_from_point(*xyz, 0.0, None, 0)
        back = point_from_detection(d)
        assert back == pytest.approx(xyz, abs=1e-9)


def test_missing_snr_stays_none_not_crash():
    d = detection_from_point(0.0, 1.0, 0.0, 0.0, None, 0)
    assert d['snr_db'] is None


def test_all_returns_including_static_walls_are_delivered():
    """הבקשה הקריטית: קירות (סטטיים) לא מסוננים."""
    clk = ManualClock()
    feed = RadarFeed(SimulatedSource(ego_speed_mps=0.0), clock_ms=clk)
    dets = feed.radar_detections_all()
    static = [d for d in dets if d['is_static']]
    assert len(dets) >= 16
    assert len(static) >= 16                      # כל הקירות שם


def test_is_static_means_static_in_world_even_when_rig_moves():
    """רכב ב-5 מ'/ש': קירות "מתקרבים" ברדאר אבל is_static=True;
    הולך הרגל is_static=False."""
    clk = ManualClock(100)                          # הולך רגל בתוך ה-FOV
    feed = RadarFeed(SimulatedSource(ego_speed_mps=5.0), clock_ms=clk)
    dets = feed.radar_detections_all()
    walls = [d for d in dets if abs(d['snr_db'] - 12.0) > 1e-6]
    ped = [d for d in dets if abs(d['snr_db'] - 12.0) < 1e-6]
    assert all(d['is_static'] for d in walls)
    assert all(abs(d['velocity_mps']) > STATIC_V for d in walls)  # ובכל זאת נעים ברדאר
    assert len(ped) == 1 and not ped[0]['is_static']


def test_timestamps_are_on_the_shared_clock():
    clk = ManualClock(5000)
    feed = RadarFeed(clock_ms=clk)
    dets = feed.radar_detections_all()
    assert feed.shared_clock_ms() == 5000
    assert all(d['timestamp_ms'] == 5000 for d in dets)


def test_returned_list_is_a_copy_yael_can_mutate():
    feed = RadarFeed(clock_ms=ManualClock())
    a = feed.radar_detections_all()
    a.clear()
    assert len(feed.radar_detections_all()) > 0


# ---------------------------------------------------------------------------
# 1.2 shared_clock_ms — שלם, מונוטוני
# ---------------------------------------------------------------------------

def test_shared_clock_is_int_and_monotonic_by_default():
    feed = RadarFeed()
    a = feed.shared_clock_ms()
    b = feed.shared_clock_ms()
    assert isinstance(a, int) and b >= a


# ---------------------------------------------------------------------------
# 1.3 radar_ego_velocity — מתוך ההחזרים הסטטיים
# ---------------------------------------------------------------------------

def _static_scene(vx, vy, n=20, noise=0.0):
    """קירות מסביב, rig נע ב-(vx, vy). מחזיר רשומות בפורמט של יעל."""
    dets = []
    for i in range(n):
        az = math.radians(-60 + 120 * i / (n - 1))
        r = 4.0 + (i % 5)
        x, y, z = r * math.sin(az), r * math.cos(az), 0.0
        v = -(vx * x + vy * y) / r + noise * ((i % 3) - 1)
        dets.append(detection_from_point(x, y, z, v, 200, 0))
    return dets


def test_ego_speed_recovered_exactly_from_clean_walls():
    ego = estimate_ego_velocity(_static_scene(0.0, 5.0))
    assert ego['speed_mps'] == pytest.approx(5.0, abs=1e-6)
    assert ego['static_count'] == 20
    assert ego['speed_sigma_mps'] == pytest.approx(0.0, abs=1e-6)


def test_ego_speed_sign_when_reversing():
    ego = estimate_ego_velocity(_static_scene(0.0, -2.0))
    assert ego['speed_mps'] == pytest.approx(-2.0, abs=1e-6)


def test_ego_estimate_rejects_moving_target_as_outlier():
    dets = _static_scene(0.0, 5.0)
    # אדם ב-3 מ' קדימה שרץ לכיוון הרדאר ב-2 מ'/ש' (מעל ה-gate)
    dets.append(detection_from_point(0.0, 3.0, 0.0, -5.0 - 2.0, 120, 0))
    ego = estimate_ego_velocity(dets)
    assert ego['speed_mps'] == pytest.approx(5.0, abs=1e-6)
    assert ego['static_count'] == 20                 # האדם הושלך


def test_ego_sigma_grows_with_noise():
    clean = estimate_ego_velocity(_static_scene(0.0, 5.0))
    noisy = estimate_ego_velocity(_static_scene(0.0, 5.0, noise=0.1))
    assert noisy['speed_sigma_mps'] > clean['speed_sigma_mps']
    assert noisy['speed_mps'] == pytest.approx(5.0, abs=0.15)


def test_ego_yaw_rate_only_when_lever_arm_given():
    # סיבוב 10°/ש' עם זרוע 1.5 מ' => מהירות הצידה 0.2618 מ'/ש'
    vx = math.radians(10.0) * 1.5
    dets = _static_scene(vx, 3.0)
    assert estimate_ego_velocity(dets)['yaw_rate_dps'] is None
    ego = estimate_ego_velocity(dets, lever_arm_m=1.5)
    assert ego['yaw_rate_dps'] == pytest.approx(10.0, abs=1e-6)


def test_ego_with_too_few_points_returns_none_not_garbage():
    ego = estimate_ego_velocity(_static_scene(0.0, 5.0, n=2))
    assert ego['speed_mps'] is None
    assert ego['static_count'] == 0


def test_feed_ego_matches_simulated_speed():
    clk = ManualClock()
    feed = RadarFeed(SimulatedSource(ego_speed_mps=4.0), clock_ms=clk)
    ego = feed.radar_ego_velocity()
    assert set(ego) >= {'speed_mps', 'yaw_rate_dps',
                        'speed_sigma_mps', 'static_count'}
    assert ego['speed_mps'] == pytest.approx(4.0, abs=0.05)


# ---------------------------------------------------------------------------
# 3.1 קצב ומדיניות שמיטה, 3.2 דגלי בריאות
# ---------------------------------------------------------------------------

def test_rate_and_drop_policy_are_declared():
    assert RadarFeed.RATE_HZ == 10.0
    assert 'returns []' in RadarFeed.DROP_POLICY


def test_health_flags_have_every_key():
    feed = RadarFeed(clock_ms=ManualClock())
    feed.radar_detections_all()
    f = feed.sensor_health_flags()
    assert set(f) == {'radar_ok', 'frame_gap', 'no_points',
                      'parser_desync', 'stale_ms'}
    assert f['radar_ok'] is True and f['frame_gap'] is False


def _parsed(frame_number, points=((0.0, 2.0, 0.0, 0.0),), **kw):
    side = [(150, 30)] * len(points)
    return parse_frame(build_frame(frame_number=frame_number,
                                   points=points, side_info=side, **kw))


def test_frame_gap_flag_rises_when_frame_number_jumps():
    clk = ManualClock()
    src = ParsedFrameSource()
    feed = RadarFeed(src, clock_ms=clk)
    src.push(_parsed(10), clk()); feed.radar_detections_all()
    clk.advance(); src.push(_parsed(11), clk()); feed.radar_detections_all()
    assert feed.sensor_health_flags()['frame_gap'] is False
    clk.advance(); src.push(_parsed(13), clk()); feed.radar_detections_all()  # 12 אבד
    assert feed.sensor_health_flags()['frame_gap'] is True
    clk.advance(); src.push(_parsed(14), clk()); feed.radar_detections_all()
    assert feed.sensor_health_flags()['frame_gap'] is False


def test_dropped_frame_gives_empty_list_and_clock_keeps_counting():
    """מדיניות שמיטה: אין פריים חדש => מחזירים את הפריים הקודם / ריק,
    לא NaN ולא חריגה; השעון ממשיך; stale גדל; אחרי 3 פריימים radar_ok יורד."""
    clk = ManualClock()
    src = ParsedFrameSource()
    feed = RadarFeed(src, clock_ms=clk)
    assert feed.radar_detections_all() == []             # עוד לא הגיע כלום
    assert feed.sensor_health_flags()['radar_ok'] is False
    src.push(_parsed(1), clk()); feed.radar_detections_all()
    assert feed.sensor_health_flags()['radar_ok'] is True
    for _ in range(4):
        clk.advance()
        feed.radar_detections_all()
    f = feed.sensor_health_flags()
    assert f['stale_ms'] == 4 * FRAME_MS
    assert f['radar_ok'] is False
    assert feed.shared_clock_ms() == 4 * FRAME_MS


def test_no_points_flag_on_empty_frame():
    clk = ManualClock()
    src = ParsedFrameSource()
    feed = RadarFeed(src, clock_ms=clk)
    src.push(_parsed(1, points=()), clk())
    assert feed.radar_detections_all() == []
    assert feed.sensor_health_flags()['no_points'] is True


def test_real_parser_frame_flows_end_to_end_with_yaels_field_names():
    """UART/replay -> mmwave_parser -> radar_feed -> יעל, בלי לאבד כלום."""
    clk = ManualClock(777)
    src = ParsedFrameSource()
    feed = RadarFeed(src, clock_ms=clk)
    pts = [(1.0, 1.0, 0.0, 0.3), (-2.0, 2.0, 0.5, -1.2)]
    src.push(_parsed(5, points=pts, pad_to_32=True), clk())
    dets = feed.radar_detections_all()
    assert len(dets) == 2
    assert dets[0]['azimuth_deg'] == pytest.approx(45.0)
    assert dets[1]['azimuth_deg'] == pytest.approx(-45.0)
    assert dets[0]['velocity_mps'] == pytest.approx(0.3, abs=1e-6)
    assert dets[0]['snr_db'] == pytest.approx(15.0)
    assert all(d['timestamp_ms'] == 777 for d in dets)
    assert feed.sensor_health_flags()['parser_desync'] is False


def test_simulation_is_deterministic():
    a = SimulatedSource().frame(300)
    b = SimulatedSource().frame(300)
    assert a == b
