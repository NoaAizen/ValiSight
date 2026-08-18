# ליעל — מה מוכן לך על הג'ייסון (18.8.2026)

עברתי על "רשימת דרישות מלאה — מיקומים, כיול, GNSS וסנכרון חומרה" סעיף-סעיף. **הכול חוץ מ-GNSS קיים, רץ, ואומת עם החומרה האמיתית** (IWR1843 + לוח N6 + Lepton). הקוד: ענף `IMU` ב-`ValiSight_git`, תיקייה `yael_api/`. שמות הפונקציות והשדות — בדיוק כמו במסמך שלך.

## איך מתחילים (3 פקודות)

```bash
cd ~/ValiSight_git/bridge
python3 n6_bridge_start.py                                              # מפעיל את השולח על ה-N6
./bridge_rx /dev/serial/by-id/usb-MicroPython* -o out/ > stats.txt &    # המקלט כותב את הנתונים
cd .. && python3 -m yael_api.demo --seconds 10 --stats bridge/stats.txt # רואים הכול חי
```

ובקוד שלך:
```python
import sys; sys.path.insert(0, "/home/valisigth/ValiSight_git")
from yael_api import Rig
rig = Rig(stats_path="/home/valisigth/ValiSight_git/bridge/stats.txt")
```
`Rig` קורא רק את הקבצים שהמקלט כותב (לא נוגע ב-USB של ה-N6), ופותח את הרדאר בעצמו (חוט רקע). הרדאר מקבל cfg אוטומטית אם הוא שותק אחרי הדלקה.

## סעיף 1 — הרדאר

### 1.1 `rig.radar_detections_all()` — הבקשה הקריטית שלך
כל ההחזרים של הפריים האחרון, **כולל הסטטיים**. שום סינון. ב-cfg שלנו `clutterRemoval 0`, כך שגם הרדאר עצמו לא מסיר קירות. דוגמה אמיתית מהיום (המתקן עומד מול קיר):
```python
{'range_m': 3.25, 'azimuth_deg': 10.9, 'elevation_deg': 7.5, 'velocity_mps': 0.0,
 'snr_db': 15.3, 'timestamp_ms': 33233346, 'is_static': True}
{'range_m': 6.35, 'azimuth_deg': 20.5, 'elevation_deg': 11.3, 'velocity_mps': 0.0,
 'snr_db': 14.5, 'timestamp_ms': 33233346, 'is_static': True}
{'range_m': 11.2, 'azimuth_deg': 3.6, 'elevation_deg': 8.5, 'velocity_mps': 0.0,
 'snr_db': 11.7, 'timestamp_ms': 33233346, 'is_static': True}
```
| שדה | מוסכמה |
|---|---|
| `azimuth_deg` | **+ = ימינה** (שאלת — זו התשובה). נגזר מצירי TI: x ימינה, y קדימה |
| `elevation_deg` | + = למעלה |
| `velocity_mps` | דופלר, + = מתרחק מהרדאר |
| `snr_db` | דציבלים; `None` אם הרדאר לא שלח |
| `timestamp_ms` | זמן הגעת הפריים, בשעון המשותף |
| `is_static` | עזר: "נייח בעולם" — המהירות מוסברת כולה ע"י תנועת ה-rig (שארית < 0.25 מ'/ש'). לא מסנן כלום; אם לא מתאים לך — התעלמי |

קצב: 10 Hz. אם פריים אבד — הרשימה ריקה (או הפריים הקודם אם הוא עדיין טרי), השעון ממשיך, `frame_gap=True` בדגלים.

### 1.2 `rig.shared_clock_ms()` — שעון משותף
מספר שלם, מילישניות, מונוטוני (לא קופץ אם מישהו מכוון שעון). זה `CLOCK_MONOTONIC` של הג'ייסון. **כל** החותמות שאת מקבלת — רדאר, IMU, תרמי — הן בשעון הזה:
- הרדאר נחתם ברגע שהפריים נסגר על ה-UART.
- ה-N6 (IMU + מצלמות) חותם בשעון שלו; המקלט רושם גם את זמן ההגעה למחשב, ומסנן-מינימום על ההפרש נותן את ההיסט. דיוק שנמדד: **~5 ms**. ב-5 מ'/ש' זה 2.5 ס"מ — הרבה מתחת ל-200 ms שציינת.

### 1.3 `rig.radar_ego_velocity()`
```python
{'speed_mps': 0.0, 'yaw_rate_dps': None, 'speed_sigma_mps': 0.0, 'static_count': 4}
```
ריבועים-פחותים על ההחזרים הסטטיים, עם השלכת חריגות (אנשים). `yaw_rate_dps` יהיה `None` עד שניתן `Rig(radar_kwargs={"lever_arm_m": X})` — מרחק הרדאר קדימה מציר הסיבוב של הרכב. בלי זה אי-אפשר לגזור קצב סיבוב מרדאר בודד, ואני לא ממציאה.

## סעיף 2 — GNSS
אין עדיין מקלט על ה-rig. הממשק קיים כדי שהקוד שלך ירוץ, והוא **לא ממציא מיקום**:
```python
rig.gnss_pvt()                    # -> None
rig.gnss_status_and_security()    # -> {'fix_type': 'No Fix', 'jamming_indicator': False,
                                  #     'spoofing_indicator': False, 'receiver_present': False}
```
כשיגיע מקלט — ממלאים `gnss.py`, הקריאות שלך לא משתנות. דגלי jamming/spoofing דורשים מקלט שמדווח עליהם (u-blox M8/M9).

## סעיף 3 — קצבים, שמיטה, בריאות

### 3.1 `Rig.RATES`, `Rig.DROP_POLICY` (נמדד היום)
| חיישן | קצב | כשפריים/דגימה אובדים |
|---|---|---|
| IMU | **200 Hz** (0 אובדן ב-190 ש') | הדגימה פשוט חסרה — החותמת קופצת. גלישת חוצץ על ה-N6 **נספרת** (`imu_overflow`) |
| תרמי | 8.6–8.8 Hz (מקסימום הלפטון) | `seq` קופץ, נספר ב-`gaps/lost`; בזמן FFC או אתחול-מחדש `thermal_ok=False` |
| RGB | 8.6 Hz | כנ"ל |
| רדאר | 10 Hz | רשימה ריקה + `frame_gap=True`; אחרי 3 פריימים בלי נתונים `radar_ok=False` |
| GNSS | — | אין מקלט |

**אף מקום לא מחזיר NaN ולא ממציא ערך.** לכל הודעה מה-N6 יש מספר רץ גלובלי; אובדן נספר, לא נעלם.

### 3.2 `rig.sensor_health_flags()`
```python
{'timestamp_ms': 33234422, 'clock_map_ready': True,
 'imu_ok': True, 'imu_stale_ms': 30,
 'thermal_ok': True, 'thermal_stale_ms': 42, 'thermal_ffc': False,
 'radar_ok': True, 'radar': {'radar_ok': True, 'frame_gap': False, 'frames_lost_total': 0,
                             'stale_ms': 94, 'parser_desync': False, 'link': 'ok', ...},
 'gnss_ok': False,
 'bridge': {'gaps': 0, 'lost': 0, 'bad_crc': 0, 'resyncs': 0, 'sender_drops': 0, 'imu_overflow': 1019}}
```
(`imu_overflow` הגבוה הוא מהשניות הראשונות, לפני שהמקלט נפתח — לא מאובדן בזמן ריצה.)

## סעיף 4 — IMU

### 4.1 `rig.imu_attitude()`
```python
{'roll_deg': 0.17, 'pitch_deg': -0.0, 'yaw_deg': -0.12, 'yaw_sigma_deg': 0.103,
 'is_static': True, 'timestamp_ms': 33233016}
```
- `roll_deg`, `pitch_deg` — מהכובד, לא נסחפים.
- `yaw_deg` — אינטגרל הג'ירו סביב ציר-מעלה, **יחסי** מאז `rig.reset_yaw()` (או ההתחלה). הטיית הג'ירו נלמדת אוטומטית בזמן דומם ומוסרת.
- `yaw_sigma_deg` — גדל עם זמן-**התנועה** מאז האיפוס (בדומם לא מצטבר כלום, כי לא מאנטגרלים). קבועים: 0.05°/ש' אי-ודאות הטיה + random-walk 0.15°/√ש'. שני מספרים אחד ליד השני ב-`imu.py`, קל לכוונן.
- `is_static` — |a| בטווח 60 mg מ-1000 וג'ירו < 3°/ש' לאורך 0.5 ש'. **אותם ספים כמו ב-`attitude.STATIONARY_GYRO_DPS` שלך.**

**מוסכמת צירים — נמדדה היום מהחזקות:** המד-תאוצה מודד כוח-נגד, כלומר הציר שמצביע **למעלה** קורא +1000 mg.
| תנוחת הלוח | קריאה (mg) | מסקנה |
|---|---|---|
| שטוח על השולחן | (+1002, 4, 15) | +X למעלה |
| על הצד | (−160, +985, −20) | +Y למעלה |
| על הפאה | (+75, −19, −988) | −Z למעלה |

איך זה יושב מול המצלמה (איזה ציר קדימה) — זה בדיוק מה שהפותר שלך מוציא מההחזקות. עד אז roll/pitch מחושבים לפי מסגרת מוצהרת: `up = +X_imu, forward = +Z_imu, left = +Y_imu` (`imu.DEFAULT_R_RIG_IMU`). כשיש לך את הסיבוב האמיתי: `Rig(R_rig_imu=R)` — 3×3, ושאר הקוד לא משתנה. סימנים: pitch חיובי = אף למעלה; roll = atan2(f_y, f_z). אם את רוצה הפוך — זה סימן אחד.

### 4.2 `rig.imu_raw()`
```python
{'accel_mg': (1002.0, 3.0, 0.0), 'gyro_dps': (-0.21, -0.98, -0.07), 'timestamp_ms': 33233016, 'seq': 6479}
```

### הקלטת ההחזקות שלך
`bridge/recordings/imu_holds_2026-08-18/` — `imu.csv` (200 Hz), `holds.json` בצורת `RigOrientation` שלך (`camera_gravity=null` כמוסכם), ותיקיית `frames/<hold>/` עם הפריימים התרמיים וה-RGB של כל החזקה. 8 החזקות ב-3 תנוחות, הפרשים עד 100° (הכלי אומר "OK for the solver"). שתי הסתייגויות: ההחזקות הנטויות קצרות (2–6 ש'), ולוח הכיול **לא** היה בפריים — אז זה טוב לצירים ולפותר ה-IMU, לא לכיול מצלמה↔IMU סופי. סבב עם לוח — כשנוח לך; `bridge/FOR_YAEL.md` מתאר את הפרוטוקול.

## סעיף 5 — המצלמה התרמית

### 5.1 `rig.camera_geometry()`
```python
{'width_px': 160, 'height_px': 120, 'hfov_deg': 53.4, 'vfov_deg': 41.2,
 'fx': 159.27, 'fy': 159.59, 'cx': 79.5, 'cy': 59.5, 'dist': [...], 'source': 'calib.json K_th (measured)'}
```
מהכיול המדוד של חגי (rms 0.43 px), לא מדף-הנתונים (שאומר 57°). הקובץ: `yael_api/cfg/calib_thermal_rgb.json`, יש שם גם K של ה-RGB והסיבוב ביניהן.

### 5.2 `rig.thermal_frame()`
```python
{'width': 160, 'height': 120, 'pixels_gray8': b'...', 'temps_c': [26.4, 26.5, ...],
 'range_c': (15.0, 45.0), 'timestamp_ms': 33233326, 'seq': 6525,
 'ffc_in_progress': False, 'ffc_source': 'sensor', 'mean_c': 26.5, 'std_gray': 18.4}
```
- הפיקסלים מגיעים מהלפטון במצב מדידה, ממופים ליניארית: 0..255 = 15..45 °C (צעד 0.12 °C). `temps_c` הוא ההמרה.
- **`ffc_in_progress` — מהחיישן עצמו.** השולח על ה-N6 קורא את מצב ה-FFC מהלפטון לכל פריים (`SYS_FFC_STATUS`, 3.9 ms, הקצב לא נפגע) ומעביר אותו בכותרת. **אימתתי היום על FFC טבעי:** 19 פריימים / 2.3 ש' מסומנים. וגילוי חשוב: בזמן FFC הלפטון **מקפיא את הפריים האחרון** (עם רעש קטן) — התמונה נראית לגמרי רגילה, אז זיהוי מהתמונה היה מפספס. ואחרי ה-FFC הטמפרטורות קופצות ב-~2.4 °C (כיול-מחדש). בזמן FFC גם `thermal_ok=False`. את צדקת שזה קריטי.
- הפריים גולמי: 14 השורות המתות של הלפטון הזה לא מתוקנות כאן (התיקון של חגי, `lepton_fix`).

## סעיף 6 — `initial_fix()`
שלך (`map_anchor.py`). לא נגעתי.

## מה אני צריכה ממך
1. **את הסיבוב IMU↔מצלמה** מהפותר שלך, כשיצא — ואני מכניסה אותו כברירת מחדל.
2. **מרחק הרדאר קדימה מציר הסיבוב** של הרכב (מטרים) — בשביל `yaw_rate_dps`.
3. אם הסימנים של roll/pitch/yaw לא נוחים לך — תגידי מה את רוצה, זה שינוי של שורה.
4. אם החוזה של איזה שדה לא מתאים לפילטר שלך — תגידי לפני שאת מתפתלת סביבו.

## מה בטוח
101 בדיקות עוברות בלי חומרה (`python3 -m pytest yael_api/tests tests/mmwave -q` + `make -C bridge test`), וכל מספר במסמך הזה נלקח מריצה חיה היום.

— נעה
