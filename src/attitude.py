"""Orientation from the N6 IMU. Pure maths -- no serial, no hardware imports.

WHAT THIS CAN AND CANNOT GIVE YOU

It can give ATTITUDE. Roll and pitch come from gravity, which is an absolute
external reference the accelerometer measures directly, so they do not drift.
Measured on this board at rest: |a| = 1004 mg with sd 3.6 mg, which is about
0.2 degrees of angular noise. That number is good.

It cannot give POSITION, and no amount of filtering changes that. Position needs
acceleration integrated twice, so a constant acceleration bias b becomes a
position error of 0.5*b*t^2 -- it grows with the SQUARE of time and nothing in
the signal distinguishes it from real motion. Even assuming a very good 2 mg
residual bias after calibration:

    after  1 s   0.01 m
    after  5 s   0.25 m
    after 10 s   0.98 m
    after 60 s  35.32 m

A GTA-style minimap fed by this would show you drifting through walls within
seconds. Position needs an external reference: wheel odometry, GNSS, visual
odometry, or -- available on this rig -- radar range to static clutter.

YAW is in between. There is no magnetometer on this module (the `imu` module
exposes acceleration, angular rate, pitch, roll and temperature -- no heading),
so yaw can only be integrated from the gyro and it drifts linearly. Measured
Z-axis bias at rest is -0.43 deg/s, which is 26 deg/min. It is therefore
reported as RELATIVE heading with an explicit, growing uncertainty, and it is
honest to reset it often.
"""
import math

# Measured on this board, stationary, 5977 samples. Bias is removed by
# calibration at runtime; these are the residual uncertainties that survive it
# and drive the drift estimate.
GYRO_BIAS_UNCERTAINTY_DPS = 0.05      # what calibration cannot pin down
STATIONARY_GYRO_DPS = 3.0             # below this, treat as at rest
STATIONARY_ACCEL_TOL_MG = 60.0        # |a| this close to 1 g


def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v] if n > 1e-9 else [0.0, 0.0, 0.0], n


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class Attitude:
    """Roll/pitch against gravity, yaw integrated from the gyro.

    Everything is reported RELATIVE to a reference orientation captured by
    level(), so the maths never assumes how the board is bolted on. This board
    happens to sit with +X up (X reads 1003 mg at rest), which would put a
    naive aerospace pitch/roll straight into gimbal lock -- hence the reference
    frame rather than a fixed convention.
    """

    def __init__(self):
        self.ref = None            # unit gravity direction at the reference pose
        self.e1 = self.e2 = None   # two axes perpendicular to it
        self.roll = self.pitch = self.yaw = 0.0
        self.bias = [0.0, 0.0, 0.0]
        self.tilt = 0.0
        self.drift_deg = 0.0       # accumulated 1-sigma yaw uncertainty
        self.calibrated = False
        self.stationary = False
        self._cal = []
        self._last_us = None
        self._yaw_t = 0.0

    def level(self, accel_mg):
        """Adopt the current pose as the zero for roll, pitch and yaw."""
        d, _ = _norm(accel_mg)
        if d == [0.0, 0.0, 0.0]:
            return False
        # Any vector not parallel to d works to seed the perpendicular pair.
        seed = [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0]
        e1, _ = _norm(_cross(d, seed))
        self.ref, self.e1, self.e2 = d, e1, _cross(d, e1)
        self.roll = self.pitch = self.yaw = 0.0
        self.drift_deg = 0.0
        self._yaw_t = 0.0
        return True

    def update(self, accel_mg, gyro_mdps, mono_us):
        """Feed one sample. mono_us is the board axis; epoch changes must reset()."""
        dt = 0.0
        if self._last_us is not None:
            dt = max(0.0, (mono_us - self._last_us) / 1e6)
            if dt > 1.0:                     # a stall, or a new epoch: no integration
                dt = 0.0
        self._last_us = mono_us

        g = [v / 1000.0 for v in gyro_mdps]          # deg/s
        d, amag = _norm(accel_mg)
        gmag = math.sqrt(sum(c * c for c in g))
        self.stationary = (gmag < STATIONARY_GYRO_DPS and
                           abs(amag - 1000.0) < STATIONARY_ACCEL_TOL_MG)

        # Bias is only meaningful while genuinely at rest, so it is learned then
        # and frozen. Averaging through real motion would poison it.
        if self.stationary and not self.calibrated:
            self._cal.append(g)
            if len(self._cal) >= 40:
                n = float(len(self._cal))
                self.bias = [sum(s[i] for s in self._cal) / n for i in range(3)]
                self.calibrated = True
                self._cal = []

        if self.ref is None:
            self.level(accel_mg)
            return self

        # Roll and pitch: where gravity sits relative to the reference pose.
        # Absolute, bounded, drift-free.
        self.pitch = math.degrees(math.atan2(_dot(d, self.e1), _dot(d, self.ref)))
        self.roll = math.degrees(math.atan2(_dot(d, self.e2), _dot(d, self.ref)))
        self.tilt = math.degrees(math.acos(max(-1.0, min(1.0, _dot(d, self.ref)))))

        # Yaw: rotation about the current up axis, integrated. Drift-prone by
        # construction; the uncertainty is tracked alongside so the display can
        # say how much to trust it.
        wz = _dot([g[i] - self.bias[i] for i in range(3)], d)
        if dt > 0.0 and self.calibrated:
            self.yaw = (self.yaw + wz * dt) % 360.0
            self._yaw_t += dt
            self.drift_deg = GYRO_BIAS_UNCERTAINTY_DPS * self._yaw_t
        return self

    def reset_heading(self):
        self.yaw = 0.0
        self.drift_deg = 0.0
        self._yaw_t = 0.0

    def reset(self):
        """New epoch: the board clock restarted, so integration state is stale."""
        self._last_us = None

    def as_dict(self):
        return {"roll": round(self.roll, 2), "pitch": round(self.pitch, 2),
                "yaw": round(self.yaw, 2), "tilt": round(self.tilt, 2),
                "drift_deg": round(self.drift_deg, 1),
                "calibrated": self.calibrated, "stationary": self.stationary,
                "bias_dps": [round(b, 3) for b in self.bias],
                "heading_age_s": round(self._yaw_t, 1)}
