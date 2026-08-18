"""בדיקות למתמטיקה הטהורה של yael_api — בלי חומרה. python3 -m pytest yael_api/tests -q"""
import math
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from yael_api.clock import N6ClockMap                                     # noqa: E402
from yael_api.imu import AttitudeEstimator, ImuTail, DEFAULT_R_RIG_IMU    # noqa: E402
from yael_api.thermal import (decode_thermal, gray_to_celsius, ffc_in_progress,
                              camera_geometry, ThermalTail, W, H)          # noqa: E402
from yael_api.rig import Rig                                              # noqa: E402


# ---------------------------------------------------------------- clock
def test_clock_map_min_filter_recovers_offset():
    m = N6ClockMap()
    offset = 123456.0
    # arrival latency varies 3..40 ms; the min-filter must land on the 3 ms one
    lat = [30, 12, 3, 40, 8, 25, 3, 15]
    for i, l in enumerate(lat):
        n6_ticks = 1000 * (10 * i)                # 10 ms steps in µs
        m.observe(n6_ticks, 0, int(offset + 10 * i + l))
    assert m.ready and abs(m.offset_ms - (offset + 3)) < 1e-6
    assert m.to_shared_ms(1000 * 100, 0) == int(offset + 100 + 3)


def test_clock_map_handles_tim2_half_us_ticks():
    m = N6ClockMap()
    m.observe(2000 * 50, 1, 1050)               # 50 ms in 0.5 µs ticks
    assert m.to_shared_ms(2000 * 60, 1) == 1060


# ---------------------------------------------------------------- attitude
def _feed_static(est, accel, gyro=(0, 0, 0), t0=0.0, seconds=1.0, hz=200):
    n = int(seconds * hz)
    for i in range(n):
        out = est.feed(t0 + i / hz, accel, gyro)
    return out, t0 + n / hz


def test_level_pose_is_zero_roll_pitch_and_static():
    est = AttitudeEstimator()
    out, _ = _feed_static(est, (1000.0, 0.0, 0.0))     # +X_imu up = level
    assert abs(out["roll_deg"]) < 1e-6 and abs(out["pitch_deg"]) < 1e-6
    assert out["is_static"] is True
    assert out["yaw_sigma_deg"] < 0.2


def test_pitch_sign_nose_up():
    # rig x (forward) = +Z_imu. Nose up 30°: gravity gets a component along -x_rig
    est = AttitudeEstimator()
    g = 1000.0
    fx, fz = -g * math.sin(math.radians(30)), g * math.cos(math.radians(30))
    accel = (fz, 0.0, fx)                                # imu X <- rig z, imu Z <- rig x
    out, _ = _feed_static(est, accel)
    assert abs(out["pitch_deg"] - 30.0) < 1e-6
    assert abs(out["roll_deg"]) < 1e-6


def test_roll_sign():
    est = AttitudeEstimator()
    g = 1000.0
    fy, fz = g * math.sin(math.radians(20)), g * math.cos(math.radians(20))
    out, _ = _feed_static(est, (fz, fy, 0.0))
    assert abs(out["roll_deg"] - 20.0) < 1e-6


def test_yaw_integrates_gyro_about_rig_up_and_sigma_grows():
    est = AttitudeEstimator()
    _, t = _feed_static(est, (1000.0, 0.0, 0.0), seconds=1.0)     # static, bias learned = 0
    # rotate about rig z (= imu X) at 10 dps for 2 s while moving (accel off-static)
    for i in range(400):
        out = est.feed(t + i / 200, (900.0, 300.0, 0.0), (10.0, 0.0, 0.0))
    assert abs(out["yaw_deg"] - 20.0) < 0.2, out["yaw_deg"]
    assert out["is_static"] is False
    s1 = out["yaw_sigma_deg"]
    est.reset_yaw(t + 2.0)
    out2 = est.feed(t + 2.005, (900.0, 300.0, 0.0), (10.0, 0.0, 0.0))
    assert abs(out2["yaw_deg"]) < 0.1 and out2["yaw_sigma_deg"] < s1


def test_gyro_bias_is_learned_while_static_and_removed():
    est = AttitudeEstimator()
    _, t = _feed_static(est, (1000.0, 0.0, 0.0), gyro=(0.5, 0.0, 0.0), seconds=2.0)
    assert abs(est.bias_dps[0] - 0.5) < 1e-6
    est.reset_yaw(t)            # before the bias was known, the first 0.5 s integrated it — expected
    for i in range(200):
        out = est.feed(t + i / 200, (900.0, 300.0, 0.0), (0.5, 0.0, 0.0))   # bias only, no real rotation
    assert abs(out["yaw_deg"]) < 1e-6


def test_imu_tail_reads_bridge_csv_and_maps_clock():
    d = tempfile.mkdtemp()
    m = N6ClockMap()
    with open(os.path.join(d, "imu.csv"), "w") as f:
        f.write("seq,ts_ticks,ts_src,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps,host_ms\n")
        for i in range(300):
            f.write("%d,%d,0,1000,0,0,0,0,0,%d\n" % (i, 5000 * i, 500000 + 5 * i + 4))
        f.write("300,1500000,0,1000,0,0,0,0,0,50")     # half-written line: no newline
    tail = ImuTail(d, m)
    raw = tail.imu_raw()
    assert tail.samples == 300
    assert raw["accel_mg"] == (1000.0, 0.0, 0.0)
    assert raw["timestamp_ms"] == 500000 + 5 * 299 + 4
    att = tail.imu_attitude()
    assert att["is_static"] is True and att["timestamp_ms"] == raw["timestamp_ms"]
    # the partial line is completed later and picked up
    with open(os.path.join(d, "imu.csv"), "a") as f:
        f.write("0000\n")
    tail.poll()
    assert tail.samples == 301


# ---------------------------------------------------------------- thermal
def _thermal_payload(fill=None, rows=None):
    px = bytearray(W * H)
    if fill is not None:
        for i in range(W * H):
            px[i] = fill
    if rows is not None:
        for y in range(H):
            for x in range(W):
                px[y * W + x] = rows(y, x)
    return struct.pack("<HHBB", W, H, 0, 0) + bytes(px)


def test_decode_and_celsius():
    w, h, g = decode_thermal(_thermal_payload(fill=0))
    assert (w, h, len(g)) == (W, H, W * H)
    assert gray_to_celsius(b"\x00\xff")[0] == 15.0 and abs(gray_to_celsius(b"\x00\xff")[1] - 45.0) < 1e-9


def test_ffc_flat_or_frozen():
    _, _, flat = decode_thermal(_thermal_payload(fill=100))
    _, _, scene = decode_thermal(_thermal_payload(rows=lambda y, x: (x * 3 + y) % 256))
    assert ffc_in_progress(flat) is True
    assert ffc_in_progress(scene) is False
    assert ffc_in_progress(scene, scene) is True          # frozen frame


def test_camera_geometry_from_measured_K():
    g = camera_geometry()
    assert g["width_px"] == 160 and g["height_px"] == 120
    assert 50 < g["hfov_deg"] < 58 and 38 < g["vfov_deg"] < 46, g


def test_thermal_tail_reads_frames_csv():
    d = tempfile.mkdtemp()
    m = N6ClockMap()
    open(os.path.join(d, "%010d_thermal.bin" % 7), "wb").write(
        _thermal_payload(rows=lambda y, x: (x + y) % 256))
    with open(os.path.join(d, "frames.csv"), "w") as f:
        f.write("seq,type,ts_ticks,ts_src,host_ms,len\n7,thermal,20000,0,1025,19206\n8,rgb,21000,0,1026,900\n")
    t = ThermalTail(d, m)
    fr = t.thermal_frame()
    assert fr["seq"] == 7 and fr["timestamp_ms"] == 1025 and fr["ffc_in_progress"] is False
    assert len(fr["temps_c"]) == W * H


def test_rig_without_hardware_reports_not_ok():
    d = tempfile.mkdtemp()
    rig = Rig(out_dir=d, radar=False)
    h = rig.sensor_health_flags()
    assert h["imu_ok"] is False and h["thermal_ok"] is False and h["radar_ok"] is False
    assert rig.gnss_status_and_security()["fix_type"] == "No Fix"
    assert rig.imu_attitude() is None and rig.thermal_frame() is None
    assert isinstance(rig.shared_clock_ms(), int)
