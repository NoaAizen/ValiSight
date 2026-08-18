"""Rig — נקודת הכניסה של יעל: כל הפונקציות מהמסמך שלה במקום אחד.

    from yael_api import Rig
    rig = Rig(out_dir="~/ValiSight_git/bridge/out")   # מה ש-bridge_rx כותב
    rig.shared_clock_ms()            # 1.2
    rig.radar_detections_all()       # 1.1   רדאר חי (UART) — כל ההחזרים
    rig.radar_ego_velocity()         # 1.3
    rig.gnss_pvt(); rig.gnss_status_and_security()     # 2.x  (אין מקלט -> No Fix)
    rig.sensor_health_flags()        # 3.2   לכל החיישנים
    rig.imu_attitude(); rig.imu_raw()                  # 4.x
    rig.camera_geometry(); rig.thermal_frame()         # 5.x
    Rig.RATES, Rig.DROP_POLICY       # 3.1

תנאי מוקדם: הגשר רץ (n6_bridge_start.py + bridge_rx -o out/) — Rig קורא רק
קבצים שהמקלט כותב, ולא נוגע ב-USB של ה-N6. הרדאר נפתח כאן ישירות (חוט רקע).
"""
import os

from .clock import shared_clock_ms, N6ClockMap
from .imu import ImuTail, DEFAULT_R_RIG_IMU
from .thermal import ThermalTail, camera_geometry
from .gnss import Gnss

STALE_IMU_MS = 200          # 40 samples at 200 Hz
STALE_THERMAL_MS = 700      # ~6 frames at 8.6 Hz; an FFC/re-init outage is longer


class Rig:
    RATES = {"imu_hz": 200.0, "thermal_hz": 8.6, "rgb_hz": 8.6, "radar_hz": 10.0,
             "gnss_hz": None}
    DROP_POLICY = (
        "every N6 record carries a global seq: a gap is counted (bridge stats gaps/lost), "
        "never silent; the missing sample is simply absent from imu.csv/frames.csv "
        "(timestamps jump, no NaN, no empty record). radar: a missing frame -> "
        "radar_detections_all() returns [] and frame_gap=True on the next frame; "
        "shared_clock_ms keeps counting. IMU ring overflow on the N6 is counted "
        "(imu_overflow in HELLO), never silently dropped.")

    def __init__(self, out_dir=None, radar=True, radar_kwargs=None,
                 R_rig_imu=DEFAULT_R_RIG_IMU, stats_path=None):
        self.out_dir = os.path.expanduser(out_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge", "out"))
        self.clock_map = N6ClockMap()
        self.imu = ImuTail(self.out_dir, self.clock_map, R_rig_imu)
        self.thermal = ThermalTail(self.out_dir, self.clock_map)
        self.gnss = Gnss()
        self.stats_path = stats_path
        self.radar = None
        if radar:
            from .radar import LiveRadar
            self.radar = LiveRadar(clock_ms=shared_clock_ms, **(radar_kwargs or {}))

    # 1.2
    def shared_clock_ms(self):
        return shared_clock_ms()

    # 1.1 / 1.3
    def radar_detections_all(self):
        return self.radar.radar_detections_all() if self.radar else []

    def radar_ego_velocity(self):
        return self.radar.radar_ego_velocity() if self.radar else None

    # 2.x
    def gnss_pvt(self):
        return self.gnss.gnss_pvt()

    def gnss_status_and_security(self):
        return self.gnss.gnss_status_and_security()

    # 4.x
    def imu_raw(self):
        return self.imu.imu_raw()

    def imu_attitude(self):
        return self.imu.imu_attitude()

    def reset_yaw(self):
        self.imu.reset_yaw()

    # 5.x
    def camera_geometry(self):
        return camera_geometry()

    def thermal_frame(self, celsius=True):
        return self.thermal.thermal_frame(celsius)

    # 3.2
    def sensor_health_flags(self):
        now = shared_clock_ms()
        self.imu.poll(); self.thermal.poll()
        imu_stale = (None if self.imu.last_host_ms is None else now - self.imu.last_host_ms)
        th_stale = (None if self.thermal.last is None else now - self.thermal.last["host_ms"])
        flags = {
            "timestamp_ms": now,
            "clock_map_ready": self.clock_map.ready,
            "imu_ok": imu_stale is not None and imu_stale <= STALE_IMU_MS,
            "imu_stale_ms": imu_stale,
            "thermal_ok": th_stale is not None and th_stale <= STALE_THERMAL_MS,
            "thermal_stale_ms": th_stale,
            "thermal_ffc": None,
            "radar_ok": False, "radar": None,
            "gnss_ok": False,
        }
        fr = self.thermal.thermal_frame(celsius=False)
        if fr is not None:
            flags["thermal_ffc"] = fr["ffc_in_progress"]
            if fr["ffc_in_progress"]:
                flags["thermal_ok"] = False           # a flat frame is not scene data
        if self.radar:
            r = self.radar.sensor_health_flags()
            flags["radar"] = r
            flags["radar_ok"] = bool(r.get("radar_ok"))
        flags["bridge"] = self._bridge_stats()
        return flags

    def _bridge_stats(self):
        """הסטטיסטיקה האחרונה של bridge_rx (gaps/lost/bad_crc) אם stats.txt נתון."""
        p = self.stats_path
        if not p or not os.path.exists(p):
            return None
        try:
            with open(p, "rb") as fh:
                fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 2048))
                lines = fh.read().decode(errors="replace").strip().splitlines()
        except OSError:
            return None
        for ln in reversed(lines):
            if "gaps" in ln and "bad_crc" in ln:
                import re
                d = {k: int(v) for k, v in re.findall(r"(gaps|lost|bad_crc|resyncs|sender_drops|imu_overflow)[ =(]+(\d+)", ln)}
                return d
        return None
