"""סעיף 4 — מה-IMU: imu_raw() ו-imu_attitude().

מקור: imu.csv ש-bridge_rx כותב (200 Hz, mg / mdps, חותמת N6 + host_ms).
ImuTail עוקב אחרי סוף הקובץ (כמו bridge_view — קורא רק מה שהמקלט כתב, לא
נוגע ב-USB). AttitudeEstimator הוא מתמטיקה טהורה בלי חומרה — נבדק בטסטים.

מוסכמת צירים (נמדד 18.8.2026, recordings/imu_holds_2026-08-18):
  המד-תאוצה מודד כוח-נגד: במנוחה הציר שמצביע *למעלה* קורא +1000 mg.
  לוח שטוח על השולחן: a = (+1002, 4, 15)  =>  +X של ה-IMU = למעלה.
  על הצד:              a = (−160, +985, −20) => +Y למעלה.
  על הפאה:             a = (+75, −19, −988)  => −Z למעלה (+Z למטה).
איך ה-IMU יושב ביחס למצלמה (איזה ציר קדימה) — זה בדיוק מה שהפותר של יעל
מוציא מההחזקות. עד אז R_RIG_IMU כאן הוא ברירת מחדל מוצהרת:
  rig z (למעלה) = +X_imu,  rig x (קדימה) = +Z_imu,  rig y (שמאלה) = +Y_imu.
כשיש פתרון — מעבירים R_rig_imu אמיתי ל-AttitudeEstimator, שאר הקוד לא משתנה.

roll/pitch מהכובד (לא נסחפים); yaw = אינטגרל הג'ירו סביב ציר-מעלה של ה-rig,
יחסי מאז reset_yaw() (או ההתחלה); yaw_sigma_deg גדל עם זמן-התנועה מאז האיפוס (בדומם לא מצטבר כלום).
is_static: |a| בטווח 60 mg מ-1000 וג'ירו < 3°/ש' לאורך 0.5 ש' — אותם ספים
כמו imu_holds.py ו-attitude.STATIONARY_GYRO_DPS של יעל.
"""
import math
import os

STATIC_ACC_TOL_MG = 60.0
STATIC_GYRO_DPS = 3.0
STATIC_WINDOW_S = 0.5
BIAS_SIGMA_DPS = 0.05          # bias instability we assume AFTER a static re-estimate
ARW_DPS_SQRT_S = 0.15          # angle random walk (deg/√s); measured std ~0.3 dps @200 Hz -> 0.3/√200*√... conservative
DEFAULT_R_RIG_IMU = ((0.0, 0.0, 1.0),      # rig x (forward) = +Z_imu
                     (0.0, 1.0, 0.0),      # rig y (left)    = +Y_imu
                     (1.0, 0.0, 0.0))      # rig z (up)      = +X_imu


def _mat_vec(R, v):
    return tuple(R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2] for i in range(3))


class AttitudeEstimator:
    """מתמטיקה טהורה: feed(sample) -> attitude(). ללא חומרה."""

    def __init__(self, R_rig_imu=DEFAULT_R_RIG_IMU):
        self.R = R_rig_imu
        self.yaw_deg = 0.0
        self.t_reset_s = None
        self.t_last_s = None
        self.bias_dps = [0.0, 0.0, 0.0]
        self.t_moving_s = 0.0          # integrated non-static time since reset: yaw drift only accrues while moving
        self._static_buf = []          # (t, |a|-1000 ok, gyro ok)
        self._static_since_s = None
        self._bias_acc = [0.0, 0.0, 0.0, 0]
        self.last = None

    def reset_yaw(self, t_s=None):
        self.yaw_deg = 0.0
        self.t_moving_s = 0.0
        self.t_reset_s = t_s if t_s is not None else self.t_last_s

    def feed(self, t_s, accel_mg, gyro_dps):
        ax, ay, az = accel_mg
        g_rig = _mat_vec(self.R, (ax, ay, az))
        w_rig = _mat_vec(self.R, tuple(gyro_dps[i] - self.bias_dps[i] for i in range(3)))
        # static detection over a window
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        ok = (abs(mag - 1000.0) <= STATIC_ACC_TOL_MG
              and max(abs(g) for g in gyro_dps) < STATIC_GYRO_DPS)
        self._static_buf.append((t_s, ok))
        while self._static_buf and t_s - self._static_buf[0][0] > STATIC_WINDOW_S:
            self._static_buf.pop(0)
        window_full = self._static_buf and (t_s - self._static_buf[0][0]) >= STATIC_WINDOW_S * 0.9
        is_static = bool(window_full and all(o for _, o in self._static_buf))
        # gyro bias: learn while static (mean of raw gyro), apply afterwards
        if is_static:
            for i in range(3):
                self._bias_acc[i] += gyro_dps[i]
            self._bias_acc[3] += 1
            if self._bias_acc[3] >= 100:       # 0.5 s @ 200 Hz
                self.bias_dps = [self._bias_acc[i] / self._bias_acc[3] for i in range(3)]
                self._bias_acc = [0.0, 0.0, 0.0, 0]
        else:
            self._bias_acc = [0.0, 0.0, 0.0, 0]
        # yaw integration about rig z (up)
        if self.t_last_s is not None and not is_static:
            dt = t_s - self.t_last_s
            if 0 < dt < 0.5:
                self.yaw_deg += w_rig[2] * dt
                self.t_moving_s += dt
        if self.t_reset_s is None:
            self.t_reset_s = t_s
        self.t_last_s = t_s
        # roll/pitch from gravity (rig: x fwd, y left, z up; +1g reads on the up axis)
        fx, fy, fz = g_rig
        pitch = math.degrees(math.atan2(-fx, math.sqrt(fy * fy + fz * fz)))   # nose up = +
        roll = math.degrees(math.atan2(fy, fz))                                # right side down = +? see README
        t_since = self.t_moving_s          # while static nothing is integrated, so nothing drifts
        sigma = math.sqrt((BIAS_SIGMA_DPS * t_since) ** 2 + (ARW_DPS_SQRT_S ** 2) * t_since)
        self.last = {"roll_deg": roll, "pitch_deg": pitch, "yaw_deg": self.yaw_deg,
                     "yaw_sigma_deg": sigma, "is_static": is_static,
                     "gravity_rig_mg": g_rig, "gyro_rig_dps": w_rig}
        return self.last


class ImuTail:
    """עוקב אחרי imu.csv של bridge_rx; מספק imu_raw() ו-imu_attitude()."""

    def __init__(self, out_dir, clock_map, R_rig_imu=DEFAULT_R_RIG_IMU):
        self.path = os.path.join(out_dir, "imu.csv")
        self.clock_map = clock_map
        self.est = AttitudeEstimator(R_rig_imu)
        self._pos = 0
        self._partial = b""
        self.last_raw = None
        self.samples = 0
        self.last_host_ms = None

    def poll(self):
        """קורא את השורות החדשות; מחזיר כמה נקלטו."""
        if not os.path.exists(self.path):
            return 0
        n = 0
        with open(self.path, "rb") as fh:
            fh.seek(self._pos)
            data = self._partial + fh.read()
            self._pos = fh.tell()
        lines = data.split(b"\n")
        self._partial = lines.pop()               # last piece may be half-written
        for ln in lines:
            f = ln.decode(errors="replace").split(",")
            if len(f) < 10 or not f[0].isdigit():
                continue                          # header or old 9-column file
            try:
                seq, ts, src = int(f[0]), int(f[1]), int(f[2])
                a = (float(f[3]), float(f[4]), float(f[5]))
                g = (float(f[6]) / 1000.0, float(f[7]) / 1000.0, float(f[8]) / 1000.0)
                host_ms = int(f[9])
            except ValueError:
                continue
            self.clock_map.observe(ts, src, host_ms)
            t_s = self.clock_map.ticks_to_ms(ts, src) / 1000.0
            att = self.est.feed(t_s, a, g)
            self.last_raw = {"accel_mg": a, "gyro_dps": g,
                             "timestamp_ms": self.clock_map.to_shared_ms(ts, src),
                             "seq": seq}
            self.last_att = att
            self.last_host_ms = host_ms
            self.samples += 1
            n += 1
        return n

    # --- 4.2 ------------------------------------------------------------
    def imu_raw(self):
        self.poll()
        return None if self.last_raw is None else dict(self.last_raw)

    # --- 4.1 ------------------------------------------------------------
    def imu_attitude(self):
        self.poll()
        if self.last_raw is None:
            return None
        a = self.est.last
        return {"roll_deg": round(a["roll_deg"], 2), "pitch_deg": round(a["pitch_deg"], 2),
                "yaw_deg": round(a["yaw_deg"], 2), "yaw_sigma_deg": round(a["yaw_sigma_deg"], 3),
                "is_static": a["is_static"], "timestamp_ms": self.last_raw["timestamp_ms"]}

    def reset_yaw(self):
        self.est.reset_yaw()
