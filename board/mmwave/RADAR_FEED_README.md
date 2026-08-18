# radar_feed — הממשק של יעל לרדאר

מימוש סעיפים 1 ו-3 מתוך "רשימת דרישות מלאה" (17.8.2026).
קובץ: `board/mmwave/radar_feed.py` · טסטים: `tests/mmwave/test_radar_feed.py`

## איך משתמשים

```python
from radar_feed import RadarFeed
feed = RadarFeed()                       # סימולציה (ברירת מחדל)

dets = feed.radar_detections_all()       # 1.1 — כל ההחזרים, כולל קירות
now  = feed.shared_clock_ms()            # 1.2 — שעון משותף, מילישניות
ego  = feed.radar_ego_velocity()         # 1.3 — מהירות ה-rig מהקירות
ok   = feed.sensor_health_flags()        # 3.2 — דגלי תקינות
```

## מה חוזר מ-radar_detections_all()

רשימה; לכל החזר: `range_m, azimuth_deg, elevation_deg, velocity_mps, snr_db, timestamp_ms`
ובנוסף `is_static` (עזר).

| שדה | מוסכמה |
|---|---|
| `azimuth_deg` | **+ = ימינה** (TI: x ימינה, y קדימה ⇒ `atan2(x, y)`) |
| `elevation_deg` | + = למעלה |
| `velocity_mps` | + = מתרחק מהרדאר (דופלר, TI) |
| `snr_db` | דציבלים (side-info של TI ÷ 10); `None` אם לא הגיע |
| `timestamp_ms` | זמן הגעת הפריים בשעון המשותף |
| `is_static` | "נייח **בעולם**": השארית אחרי פיצוי תנועת ה-rig < 0.25 מ'/ש'. כשה-rig עומד = פשוט \|v\| < 0.25 |

**לא מסננים כלום.** קירות חוזרים תמיד. אם `is_static` לא מתאים לך — התעלמי ממנו.

## radar_ego_velocity()

`{speed_mps, yaw_rate_dps, speed_sigma_mps, static_count}` — least-squares על ההחזרים
הסטטיים, עם השלכת חריגות (אנשים). `yaw_rate_dps` הוא `None` אלא אם נתנו
`RadarFeed(lever_arm_m=...)` (מרחק הרדאר קדימה מציר הסיבוב) — בלי זה אי-אפשר
לגזור קצב סיבוב מרדאר בודד, ולא ממציאים.

## 3.1 קצב ושמיטה

- `RadarFeed.RATE_HZ = 10`
- פריים אבד ⇒ `radar_detections_all()` מחזירה `[]` (או את הפריים הקודם אם עדיין "טרי"),
  לא NaN ולא חריגה. השעון ממשיך. `frame_gap=True` בפריים הבא. אחרי 3 פריימים בלי
  נתונים `radar_ok=False`.

## 3.2 sensor_health_flags()

`radar_ok, frame_gap, no_points, parser_desync, stale_ms`

## מה עוד לא אמיתי

- **מקור הנתונים** הוא סימולציה. הצינור האמיתי (`ParsedFrameSource` + `mmwave_parser`)
  קיים ונבדק בטסטים, אבל עדיין לא מחובר ל-UART חי.
- **השעון המשותף** הוא כרגע השעון של ה-Jetson. כשה-N6 ייתן שעון אמיתי, מעבירים
  `RadarFeed(clock_ms=<פונקציה>)` — הקוד של יעל לא משתנה.
