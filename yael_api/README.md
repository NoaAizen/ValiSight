# yael_api — כל מה שיעל ביקשה, במקום אחד (על הג'ייסון)

מימוש "רשימת דרישות מלאה — מיקומים, כיול, GNSS וסנכרון חומרה" (17.8.2026).
**אומת חי 18.8.2026** עם N6 + Lepton + IWR1843 אמיתיים: רדאר, IMU ותרמי על שעון משותף אחד.

```python
import sys; sys.path.insert(0, "/home/valisigth/ValiSight_git")
from yael_api import Rig
rig = Rig(stats_path="/home/valisigth/ValiSight_git/bridge/stats.txt")   # out_dir ברירת מחדל: bridge/out
```
תנאי מוקדם: הגשר רץ (`cd bridge && python3 n6_bridge_start.py && ./bridge_rx /dev/serial/by-id/usb-MicroPython* -o out/ > stats.txt &`).
הדגמה: `python3 -m yael_api.demo --seconds 10 --stats bridge/stats.txt`.
בדיקות: `python3 -m pytest yael_api/tests tests/mmwave -q` (13 + 72 עוברות) ו-`make -C bridge test`.

## מיפוי מסמך → קריאה

| סעיף | קריאה | מקור אמיתי | מצב |
|---|---|---|---|
| 1.1 | `rig.radar_detections_all()` | IWR1843 UART 921600, `mmwave_parser` → `RadarFeed` | ✅ חי. **כל** ההחזרים, כולל קירות (`clutterRemoval 0` ב-cfg). שדות: `range_m, azimuth_deg (+ = ימינה), elevation_deg (+ = למעלה), velocity_mps (+ = מתרחק), snr_db, timestamp_ms, is_static` |
| 1.2 | `rig.shared_clock_ms()` | `CLOCK_MONOTONIC` של הג'ייסון (ms) | ✅ רדאר נחתם בו ישירות; N6 (IMU/תרמי) ממופה אליו ב-`N6ClockMap` (מסנן-מינימום על host_ms−n6_ticks; דיוק ~5 ms) |
| 1.3 | `rig.radar_ego_velocity()` | least-squares על ההחזרים הסטטיים | ✅ `speed_mps, speed_sigma_mps, static_count`; `yaw_rate_dps=None` בלי `lever_arm_m` (לא ממציאים) |
| 2.1/2.2 | `rig.gnss_pvt()`, `rig.gnss_status_and_security()` | — אין מקלט | ⛔ מחזיר `None` / `fix_type="No Fix"` בכנות. הממשק מוכן למקלט |
| 3.1 | `Rig.RATES`, `Rig.DROP_POLICY` | נמדד חי | ✅ ראה טבלה למטה |
| 3.2 | `rig.sensor_health_flags()` | כל החיישנים | ✅ `imu_ok/imu_stale_ms, thermal_ok/thermal_stale_ms/thermal_ffc, radar_ok + radar{frame_gap, frames_lost_total, stale_ms, parser_desync, link}, bridge{gaps,lost,bad_crc,...}, clock_map_ready` |
| 4.1 | `rig.imu_attitude()` | imu.csv של הגשר, 200 Hz | ✅ `roll_deg, pitch_deg` מהכובד; `yaw_deg` יחסי (ג'ירו, bias נלמד בדומם); `yaw_sigma_deg` גדל עם זמן-תנועה מאז `rig.reset_yaw()`; `is_static` (60 mg / 3°/ש' / 0.5 ש' — הספים שלך); `timestamp_ms` |
| 4.2 | `rig.imu_raw()` | אותו מקור | ✅ `accel_mg, gyro_dps, timestamp_ms, seq` |
| 5.1 | `rig.camera_geometry()` | `cfg/calib_thermal_rgb.json` (K_th מדוד, rms 0.43 px) | ✅ 160×120, **hfov 53.4°, vfov 41.2°**, fx/fy/cx/cy/dist |
| 5.2 | `rig.thermal_frame()` | פריימים של הגשר | ✅ `pixels_gray8` (160×120), `temps_c` (0..255 = 15..45 °C, צעד 0.12 °C), `timestamp_ms`, **`ffc_in_progress` מהחיישן עצמו** (`ffc_source="sensor"`; N6 קורא SYS_FFC_STATUS לכל פריים), `seq` |
| 6 | `initial_fix()` | — | שלך (`map_anchor.py`) |

## 3.1 קצבים ומדיניות שמיטה (נמדד 18.8.2026)

| חיישן | קצב | חותמת זמן | כשפריים אובד |
|---|---|---|---|
| IMU | **200 Hz** (0 אובדן ב-190 ש') | N6 ticks_us → שעון משותף | הדגימה פשוט חסרה ב-`imu.csv` (חותמת קופצת). גלישת טבעת ב-N6 נספרת (`imu_overflow`), לא נעלמת |
| תרמי | **8.6 Hz** (מקסימום הלפטון) | N6 → משותף | פריים חסר = `seq` קופץ, נספר ב-`gaps/lost`; בזמן FFC/re-init `thermal_ok=False` |
| RGB | 8.6 Hz | N6 → משותף | כנ"ל |
| רדאר | **10 Hz** | הגעה ב-UART, שעון משותף | `radar_detections_all()` מחזירה `[]` (או את הפריים האחרון אם טרי), `frame_gap=True` בפריים הבא, `frames_lost_total` מצטבר; אחרי 3 פריימים בלי נתונים `radar_ok=False` |
| GNSS | — | — | אין מקלט |

אף מקום לא מחזיר NaN ולא ממציא ערך; כל אובדן נספר.

## מוסכמות שצריך לדעת

- **צירי ה-IMU** (נמדד מההחזקות): המד-תאוצה מודד כוח-נגד — הציר שמצביע *למעלה* קורא +1000 mg. שטוח: +X למעלה; על הצד: +Y למעלה; על הפאה: −Z למעלה. איך זה יושב מול המצלמה — הפותר שלך מוציא מ-`bridge/recordings/imu_holds_2026-08-18/holds.json`. עד אז `imu.DEFAULT_R_RIG_IMU` (up=+X, forward=+Z) — מוצהר, ניתן להחלפה: `Rig(R_rig_imu=...)`.
- **סימני roll/pitch**: מסגרת rig = x קדימה, y שמאלה, z למעלה. `pitch` חיובי = אף למעלה; `roll = atan2(f_y, f_z)` (חיובי = הצד השמאלי עולה). אם תרצי הפוך — זה סימן אחד.
- **`is_static` של הרדאר** = "נייח בעולם" אחרי פיצוי תנועת ה-rig (שארית < 0.25 מ'/ש'); כשה-rig עומד = |v| < 0.25.
- **FFC**: הדגל מגיע מהלפטון (bit0 בבית הדגלים של כותרת התמונה בגשר; N6 קורא `LEP_CID_SYS_FFC_STATUS`, 3.9 ms לפריים, קצב לא נפגע: 8.8 Hz). **אומת על FFC טבעי 18.8:** 19 פריימים / 2.3 ש' מסומנים; בזמן FFC הלפטון מקפיא את הפריים האחרון (עם רעש) — זיהוי מהתמונה מפספס — ואחרי FFC הטמפרטורות קופצות ~2.4 °C. בזמן FFC `thermal_ok=False`. אם הדגל לא ידוע (שולח ישן) יש נפילה לזיהוי מהתמונה, החלש.
- **דיוק השעון המשותף** בין N6 לרדאר: ~5 ms (latency USB מינימלית). ב-5 מ'/ש' זה 2.5 ס"מ.
- הפריים התרמי גולמי — 14 השורות המתות של הלפטון הזה לא מתוקנות כאן (זה בצד של חגי, `lepton_fix`).

## קבצים
`clock.py` שעון משותף ומיפוי N6 · `radar.py` רדאר חי + דחיפת cfg · `imu.py` attitude · `thermal.py` פריים+FFC+גאומטריה · `gnss.py` · `rig.py` הפאסאדה · `demo.py` · `tests/`
