"""radar_feed — הממשק שדרכו יעל מקבלת את נתוני הרדאר.

ממומש לפי "רשימת דרישות מלאה — מיקומים, כיול, GNSS וסנכרון חומרה"
(עדכון 17.8.2026), סעיף 1 (צינור הרדאר) + סעיף 3 (קצבים, שמיטה, בריאות).
שאר הסעיפים (GNSS, IMU, מצלמה תרמית) הם באחריות אחרים ולא כאן.

הפונקציות שיעל קוראת (שמות ושדות בדיוק כמו במסמך שלה):

    feed = RadarFeed()                       # ברירת מחדל: סימולציה
    feed.radar_detections_all()   -> [ {range_m, azimuth_deg, elevation_deg,
                                        velocity_mps, snr_db, timestamp_ms,
                                        is_static}, ... ]      (סעיף 1.1)
    feed.shared_clock_ms()        -> int  (מילישניות, מונוטוני)  (סעיף 1.2)
    feed.radar_ego_velocity()     -> {speed_mps, yaw_rate_dps,
                                      speed_sigma_mps, static_count}
                                     (סעיף 1.3 — מחושב בתוכנה)
    feed.sensor_health_flags()    -> {radar_ok, frame_gap, no_points,
                                      parser_desync, stale_ms}  (סעיף 3.2)
    RadarFeed.RATE_HZ, RadarFeed.DROP_POLICY                    (סעיף 3.1)

חוזה חשוב (הבקשה הקריטית של יעל): radar_detections_all מחזירה את *כל*
ההחזרים של הפריים, גם הסטטיים (קירות). לא מסננים כלום. השדה is_static
הוא סימון עזר: "נייח בעולם" — כלומר המהירות שנמדדה מוסברת כולה
על ידי תנועת ה-rig עצמו (שארית < STATIC_V = 0.25, הערך המאומת ממשימה
11). כשה-rig עומד זה מתלכד עם |velocity_mps| < 0.25. יעל מחליטה מה
לעשות איתו.

מוסכמות (יעל שאלה "+ הוא ימינה או שמאלה"):
    azimuth_deg   > 0  = ימינה מציר הרדאר. נגזר מהצירים של TI:
                  x = ימינה, y = קדימה, z = למעלה  =>  az = atan2(x, y)
    elevation_deg > 0  = למעלה.
    velocity_mps  > 0  = מתרחק מהרדאר (מוסכמת TI לדופלר).
    range_m       = sqrt(x²+y²+z²), מטרים.
    snr_db        = ה-side-info של TI (יחידות 0.1dB) חלקי 10.
    timestamp_ms  = זמן הגעת הפריים בשעון המשותף (מילישניות).

מדיניות שמיטה (סעיף 3.1): פריים שאבד => radar_detections_all מחזירה
רשימה ריקה, השעון ממשיך (לא מדלג ולא NaN), והדגל frame_gap עולה
בפריים הבא לפי קפיצה ב-frame_number.

מקורות נתונים (מחליפים בלי לשנות את הקוד של יעל):
    SimulatedSource   — נתונים מומצאים (רכב נוסע + קירות + הולך רגל)
    ParsedFrameSource — מקבל פלט של mmwave_parser.parse_frame (UART/replay)
"""

import math
import time

STATIC_V = 0.25            # מ'/ש' — מתחת לזה ההחזר נחשב "נייח"
RADAR_HZ = 10.0
FRAME_MS = int(1000 / RADAR_HZ)


# ---------------------------------------------------------------------------
# המרות טהורות (בלי מצב) — קל לבדוק, קל להריץ גם על ה-N6
# ---------------------------------------------------------------------------

def detection_from_point(x, y, z, v, snr_0p1db, timestamp_ms):
    """נקודה אחת של TI (קרטזי, מטרים) -> רשומה בפורמט של יעל (ספרי)."""
    rng = math.sqrt(x * x + y * y + z * z)
    az = math.degrees(math.atan2(x, y))
    horiz = math.sqrt(x * x + y * y)
    el = math.degrees(math.atan2(z, horiz)) if horiz > 0 else 0.0
    snr_db = None if snr_0p1db is None else snr_0p1db / 10.0
    return {
        'range_m': rng,
        'azimuth_deg': az,
        'elevation_deg': el,
        'velocity_mps': v,
        'snr_db': snr_db,
        'timestamp_ms': timestamp_ms,
        'is_static': abs(v) < STATIC_V,
    }


def point_from_detection(d):
    """הפוך: רשומה של יעל -> (x, y, z). לבדיקות ולציור."""
    r = d['range_m']
    az = math.radians(d['azimuth_deg'])
    el = math.radians(d['elevation_deg'])
    return (r * math.cos(el) * math.sin(az),
            r * math.cos(el) * math.cos(az),
            r * math.sin(el))


def estimate_ego_velocity(dets, lever_arm_m=None, static_gate_mps=0.35,
                          rounds=3):
    """סעיף 1.3 — מהירות ה-rig מתוך ההחזרים הסטטיים (בתוכנה).

    הרעיון: קיר לא זז. אם הרדאר רואה קיר "מתקרב" ב-v, זה כי הרדאר
    זז. לנקודה סטטית בזווית az/el, המהירות הרדיאלית שהרדאר מודד היא
        v_r = -(vx*sin(az) + vy*cos(az)) * cos(el)
    כאשר (vx, vy) = מהירות הרדאר (ימינה, קדימה). עם הרבה נקודות פותרים
    least-squares ל-(vx, vy). נקודות נעות (אנשים) לא מקיימות את הנוסחה
    => מזוהות כחריגות ומושלכות בסיבובים חוזרים.

    yaw_rate_dps: אי אפשר לקבל קצב סיבוב מרדאר בודד בלי לדעת איפה הוא
    יושב יחסית לציר הסיבוב. אם נותנים lever_arm_m (מרחק הרדאר קדימה
    מציר הסיבוב), אז vx = omega * lever_arm ונגזר מזה. אחרת None —
    לא ממציאים מספר.

    מחזיר dict בשמות של יעל. אם אין מספיק נקודות: speed None, count 0.
    """
    rows = []
    for d in dets:
        if d.get('velocity_mps') is None:
            continue
        az = math.radians(d['azimuth_deg'])
        el = math.radians(d['elevation_deg'])
        c = math.cos(el)
        rows.append((-math.sin(az) * c, -math.cos(az) * c, d['velocity_mps']))

    used = rows
    vx = vy = 0.0
    for _ in range(rounds):
        if len(used) < 3:
            return {'speed_mps': None, 'yaw_rate_dps': None,
                    'speed_sigma_mps': None, 'static_count': 0}
        vx, vy, sigma_vy = _lstsq_2(used)
        resid = [(a * vx + b * vy - v) for a, b, v in used]
        keep = [r for r, e in zip(used, resid) if abs(e) < static_gate_mps]
        if len(keep) == len(used):
            break
        used = keep
    vx, vy, sigma_vy = _lstsq_2(used)
    yaw = None
    if lever_arm_m:
        yaw = math.degrees(vx / lever_arm_m)
    return {
        'speed_mps': vy,
        'yaw_rate_dps': yaw,
        'speed_sigma_mps': sigma_vy,
        'static_count': len(used),
        '_vx': vx,                 # פנימי: מהירות הצידה, למיון is_static
    }


def expected_static_velocity(d, vx, vy):
    """איזו מהירות רדיאלית היינו מצפים למדוד על נקודה נייחת-בעולם
    בזווית של d, כשה-rig זז ב-(vx ימינה, vy קדימה)."""
    az = math.radians(d['azimuth_deg'])
    el = math.radians(d['elevation_deg'])
    return -(vx * math.sin(az) + vy * math.cos(az)) * math.cos(el)


def mark_static(dets, ego):
    """מעדכן במקום את is_static לפי פיצוי תנועת ה-rig. אם אין אומדן
    (מעט מדי נקודות) נשארים עם |v| < STATIC_V הפשוט."""
    if not ego or ego.get('speed_mps') is None:
        return dets
    vx = ego.get('_vx', 0.0)
    vy = ego['speed_mps']
    for d in dets:
        if d.get('velocity_mps') is None:
            continue
        resid = d['velocity_mps'] - expected_static_velocity(d, vx, vy)
        d['is_static'] = abs(resid) < STATIC_V
    return dets


def _lstsq_2(rows):
    """פתרון least-squares ל-2 נעלמים בלי numpy (רץ גם על MicroPython).
    rows: (a, b, v) עם a*vx + b*vy ≈ v. מחזיר (vx, vy, sigma_vy)."""
    saa = sab = sbb = sav = sbv = 0.0
    for a, b, v in rows:
        saa += a * a; sab += a * b; sbb += b * b
        sav += a * v; sbv += b * v
    det = saa * sbb - sab * sab
    if abs(det) < 1e-9:                     # כל הנקודות באותה זווית
        vy = sbv / sbb if sbb > 0 else 0.0
        return 0.0, vy, None
    vx = (sbb * sav - sab * sbv) / det
    vy = (saa * sbv - sab * sav) / det
    n = len(rows)
    rss = sum((a * vx + b * vy - v) ** 2 for a, b, v in rows)
    sigma2 = rss / (n - 2) if n > 2 else 0.0
    sigma_vy = math.sqrt(sigma2 * saa / det)
    return vx, vy, sigma_vy


# ---------------------------------------------------------------------------
# מקורות נתונים
# ---------------------------------------------------------------------------

class SimulatedSource:
    """נתונים מומצאים, דטרמיניסטיים לחלוטין (אותו זמן -> אותו פריים).

    התרחיש (מותאם למסמך של יעל — rig על רכב):
      * הרכב נוסע קדימה במהירות ego_speed_mps (ברירת מחדל 5 = 18 קמ"ש)
      * שני קירות לאורך הדרך: החזרים סטטיים רבים (בעולם), שברדאר נראים
        עם מהירות רדיאלית שלילית = "מתקרבים"
      * הולך רגל אחד שחוצה — החזר נע אמיתי
    """

    def __init__(self, ego_speed_mps=5.0, walls=True, pedestrian=True,
                 lever_arm_m=1.5, yaw_rate_dps=0.0):
        self.ego_speed = ego_speed_mps
        self.walls = walls
        self.pedestrian = pedestrian
        self.lever_arm = lever_arm_m
        self.yaw_rate = yaw_rate_dps
        self.frame_number = 0

    def frame(self, t_ms):
        """מחזיר (frame_number, [ (x, y, z, v, snr_0p1db), ... ]),
        או None אם עוד לא הגיע פריים בכלל."""
        self.frame_number += 1
        t = t_ms / 1000.0
        pts = []
        vx = math.radians(self.yaw_rate) * self.lever_arm   # תנועה הצידה
        vy = self.ego_speed
        if self.walls:
            for side in (-3.0, +3.0):                    # קיר משמאל ומימין
                for i in range(8):
                    y = 2.0 + 2.5 * i + (0.7 * ((i * 7 + int(t * 3)) % 5) / 5)
                    x = side + 0.15 * (((i * 13) % 7) / 7 - 0.5)
                    z = -0.4 + 0.1 * (i % 3)
                    pts.append(self._static(x, y, z, vx, vy, snr=180 + 20 * (i % 3)))
        if self.pedestrian:
            ped_v = 1.5                                  # מ'/ש', מימין לשמאל
            px = 2.5 - ped_v * (t % 4.0)
            py = 5.0                                     # מרחק קבוע לצורך הדמו
            if -2.5 < px < 2.5:
                # מהירות רדיאלית = חלק תנועת הרכב + חלק תנועת ההולך
                x, y, z = px, py, -0.2
                r = math.sqrt(x * x + y * y)
                v_static = -(vx * x + vy * y) / r
                v_ped = (-ped_v * x) / r                  # ההולך זז רק ב-x
                pts.append((x, y, z, v_static + v_ped, 120))
        return self.frame_number, pts

    @staticmethod
    def _static(x, y, z, vx, vy, snr):
        r = math.sqrt(x * x + y * y + z * z)
        v = -(vx * x + vy * y) / r
        return (x, y, z, v, snr)


class ParsedFrameSource:
    """מקור אמיתי: מוזנים אליו פריימים שכבר עברו mmwave_parser.parse_frame
    (מ-UART חי או מ-replay). מי שקורא לרדאר עושה:
        src.push(parsed, timestamp_ms)
    ו-RadarFeed מושך ממנו את האחרון."""

    def __init__(self):
        self._latest = None      # (frame_number, points, timestamp_ms, desync)

    def push(self, parsed, timestamp_ms):
        pts = [(p['x'], p['y'], p['z'], p['v'], p['snr'])
               for p in parsed['points']]
        desync = parsed.get('length_mode') not in ('payload', 'includes_header')
        self._latest = (parsed['header']['frame_number'], pts,
                        timestamp_ms, desync)

    def frame(self, t_ms):
        if self._latest is None:
            return None                       # עוד לא הגיע שום פריים
        return self._latest[0], self._latest[1]

    def latest_timestamp_ms(self):
        return None if self._latest is None else self._latest[2]

    def desync(self):
        return bool(self._latest and self._latest[3])


# ---------------------------------------------------------------------------
# הממשק של יעל
# ---------------------------------------------------------------------------

class RadarFeed:
    RATE_HZ = RADAR_HZ
    DROP_POLICY = ('missing frame -> radar_detections_all() returns [] ; '
                   'shared_clock_ms keeps counting ; frame_gap flag set')

    def __init__(self, source=None, clock_ms=None, lever_arm_m=None):
        """
        source     — SimulatedSource (ברירת מחדל) או ParsedFrameSource
        clock_ms   — פונקציה שמחזירה את השעון המשותף במילישניות.
                     ברירת מחדל: שעון מונוטוני של המחשב. כשה-N6 יספק
                     שעון משותף אמיתי, מעבירים כאן פונקציה שקוראת אותו.
        lever_arm_m — מרחק הרדאר קדימה מציר הסיבוב של הרכב (למהירות
                     סיבוב). None = לא מחשבים yaw_rate.
        """
        self._source = source or SimulatedSource()
        self._clock = clock_ms or (lambda: time.monotonic_ns() // 1_000_000)
        self._lever_arm = lever_arm_m
        self._last_frame_number = None
        self._last_frame_ms = None
        self._last_dets = []
        self._ego = None
        self._flags = {'radar_ok': False, 'frame_gap': False,
                       'no_points': True, 'parser_desync': False,
                       'stale_ms': None}

    # --- 1.2 -----------------------------------------------------------
    def shared_clock_ms(self):
        """מונה מונוטוני יחיד, מילישניות, שכל החיישנים חותמים בו."""
        return int(self._clock())

    # --- 1.1 -----------------------------------------------------------
    def radar_detections_all(self):
        """כל ההחזרים של הפריים האחרון — כולל סטטיים. אין סינון."""
        now = self.shared_clock_ms()
        got = self._source.frame(now)
        if got is None:                       # המקור עוד לא קיבל כלום
            self._flags['radar_ok'] = False
            return []
        frame_no, pts = got
        if frame_no == self._last_frame_number:
            # אין פריים חדש מאז הפעם הקודמת: מחזירים את הקיים ומעדכנים stale
            self._flags['stale_ms'] = (None if self._last_frame_ms is None
                                       else now - self._last_frame_ms)
            self._flags['radar_ok'] = self._radar_ok()
            return list(self._last_dets)

        gap = (self._last_frame_number is not None
               and frame_no != self._last_frame_number + 1)
        ts = now
        if hasattr(self._source, 'latest_timestamp_ms'):
            ts = self._source.latest_timestamp_ms() or now
        dets = [detection_from_point(x, y, z, v, snr, ts)
                for (x, y, z, v, snr) in pts]
        self._ego = estimate_ego_velocity(dets, self._lever_arm)
        mark_static(dets, self._ego)

        self._last_frame_number = frame_no
        self._last_frame_ms = ts
        self._last_dets = dets
        self._flags.update({
            'frame_gap': bool(gap),
            'no_points': len(dets) == 0,
            'parser_desync': bool(getattr(self._source, 'desync', lambda: False)()),
            'stale_ms': now - ts,
        })
        self._flags['radar_ok'] = self._radar_ok()
        return list(dets)

    # --- 1.3 -----------------------------------------------------------
    def radar_ego_velocity(self):
        """מהירות ה-rig מתוך ההחזרים הסטטיים של הפריים האחרון."""
        if self._ego is None:
            self.radar_detections_all()
        return dict(self._ego)

    # --- 3.2 -----------------------------------------------------------
    def sensor_health_flags(self):
        """דגלי תקינות של הרדאר לפריים האחרון."""
        if self._last_frame_ms is not None:
            self._flags['stale_ms'] = self.shared_clock_ms() - self._last_frame_ms
            self._flags['radar_ok'] = self._radar_ok()
        return dict(self._flags)

    def _radar_ok(self):
        f = self._flags
        stale = f['stale_ms']
        return (not f['parser_desync']
                and stale is not None and stale <= 3 * FRAME_MS)


# ---------------------------------------------------------------------------
# הדגמה: python3 radar_feed.py
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t = [0]
    feed = RadarFeed(clock_ms=lambda: t[0], lever_arm_m=1.5)
    for _ in range(3):
        dets = feed.radar_detections_all()
        ego = feed.radar_ego_velocity()
        print(f"t={feed.shared_clock_ms()} ms  {len(dets)} החזרים  "
              f"(סטטיים: {sum(d['is_static'] for d in dets)})  "
              f"ego speed={ego['speed_mps']:.2f} m/s ±{ego['speed_sigma_mps']:.2f} "
              f"מתוך {ego['static_count']} נק'  health={feed.sensor_health_flags()}")
        for d in dets[:3]:
            print("   ", {k: (round(v, 2) if isinstance(v, float) else v)
                          for k, v in d.items()})
        t[0] += FRAME_MS
