# mapinit — גאומטריה, כיול וסחיפה

מיקום יחסית למפה, כיול ה-rig, וכמה מהר הכל מתדרדר בין תיקונים. הכלל שכל הקוד
כאן בנוי סביבו: **כשלון חייב להיות רועש.** כל שלב מחזיר את מה שנמדד מול מה
שציפינו, ושום פונקציה לא מחזירה ערך סביר במקום לזרוק — כי גובה שגוי בעשרים מטר
נראה בדיוק כמו שגיאת כיול, ושולח מי שמדבג לכיוון הלא נכון.

---

## התקנה

הקוד שטוח, אז **שתי הדרכים עובדות**. בלי התקנה בכלל:

```python
import sys; sys.path.insert(0, "/path/to/valisigth-IMU")
import mapinit
```

או עם:

```bash
pip install -e /path/to/valisigth-IMU
```

**אין תלויות חובה, וזו הכוונה.** ליבת הוולידציה וכל `mapinit.nav` — סחיפה,
שחזור כיוון, Allan deviation — הן אריתמטיקה טהורה ורצות בלי `numpy`, בלי
`pyproj` ובלי `rasterio`. יש טסט שמריץ מפרש נקי ומוודא את זה, כדי שזה לא יישחק
בטעות.

לחצאים הכבדים, לפי מה שצריך:

```bash
pip install -e ".[calibration]"   # פותר IMU→מצלמה                 (numpy)
pip install -e ".[geo]"           # שטח, בניינים, גאואיד, חיזוי פריים
pip install -e ".[all,test]"      # הכל, ועוד הטסטים
```

נגיעה בשם שדורש extra שלא מותקן אומרת **איזה extra** — לא איזה מודול צד-שלישי
שלא ביקשת:

```
mapinit.GeoidModel needs the 'geo' extra, which is not installed.
Install it with: pip install 'mapinit[geo]'
```

`requirements.txt` מחזיק את הגרסאות המדויקות שהריפו מפותח ונבדק מולן. הגבולות
ב-`pyproject.toml` רחבים יותר בכוונה — הסביבה של צרכן היא לא שלנו לנעוץ.

---

## מה ש-pip לא יכול לספק

התחום הגאו-י דורש את גריד EGM2008, כ-77MB:

```bash
curl -o "$(python -c 'import pyproj;print(pyproj.datadir.get_user_data_dir())')/us_nga_egm08_25.tif" \
     https://cdn.proj.org/us_nga_egm08_25.tif
```

בלעדיו PROJ **לא זורק** — הוא נופל בשקט לטרנספורם ballpark שמחזיר את הגובה
כמו שהוא, וזה בישראל הפרש של כ-20 מטר. לכן `GeoidModel` מסרב להיבנות במקום
לעבוד, ו-`MapInitializer.run()` מחזיר דוח שבו שלב ה-geoid נכשל עם פקודת ההורדה
בתוך הודעת השגיאה. גריד `.pgm` של GeographicLib **לא** נקרא על ידי PROJ.

---

## מה יש כאן

הקוד מתעד את עצמו. אין רשימה שצריך לתחזק בנפרד:

```bash
python -c "import mapinit; print(mapinit.describe())"
python -c "import mapinit; print(mapinit.describe('navigation'))"
```

77 שמות בשישה תחומים, לכל אחד שורה מה הוא ושורה למה הוא טוב. `mapinit/api.py`
הוא **מקור המשטח ולא תיאור שלו**: `__all__` וטבלת הייבוא העצל נבנים ממנו, אז שם
לא יכול להיות ציבורי בלי תיאור — ושם רשום שהפסיק להתקיים מפיל טסט במקום לשרוד
כתיעוד של משהו שנמחק.

| תחום | מה בו | צריך |
|---|---|---|
| `health` | ולידציה, שלבים, דוחות | — |
| `navigation` | סחיפה, כיוון, זיהוי קבועים | — |
| `calibration` | הגאומטריה הקפואה של ה-rig | numpy |
| `geo` | דאטום אנכי, שטח, בניינים | geo |
| `view` | מה אמור להיות בפריים | geo |
| `map` | אתחול מיקום מול המפה | geo |

---

## דוגמאות

### health — ולידציה שאומרת מה נמדד

```python
from mapinit import Check

print(Check.in_range("clock_skew", 4.8, 0.0, 20.0, unit="ms"))
# [PASS] clock_skew: 4.800 ms is within [0.000, 20.000] ms
```

`Check` במקום `bool`: דוח כשלון מראה איזו הנחה נשברה, לא רק שמשהו נשבר.

### navigation — כמה זמן קיבוע מחזיק

```python
from mapinit import DeadReckoner, SpeedAiding

aided = DeadReckoner(aiding=SpeedAiding(speed_mps=5.0, sigma_speed_mps=0.05))
print(aided.horizon_for(1.0))     # שניות עד מטר של שגיאה
print(aided.budget_at(60.0))      # והפירוק לפי מקור
```

הפירוק הוא העיקר. "35 מטר" לא ניתן לפעולה; "35 מטר, מתוכם 34 דליפת gravity
דרך הטיה שאיש לא מדד" אומר בדיוק לאן ללכת.

### navigation — קבועים מדודים במקום מונחים

```python
from pathlib import Path
from mapinit import DeadReckoner, identify, identification_checks, load_recording

recording = load_recording(Path("bridge/recordings/imu_holds_2026-08-18"))
model = identify(recording)
for check in identification_checks(recording, model):
    print(check)                  # מה ההקלטה באמת תומכת בו, ומה נשאר חסם עליון
print(DeadReckoner(model).budget_at(60.0))
```

מה שהקלטה דוממת לא יכולה להכריע נשאר מסומן `assumed`, והבדיקות **נכשלות ברעש**
במקום לתת לחסם עליון להיות מצוטט כמדידה.

### navigation — שחזור כיוון מפריים

```python
from mapinit import HeadingMatcher, bearings_from_columns, bearings_from_view

observed = bearings_from_columns([31.0, 78.0, 122.0], fx=159.27, cx=79.5)
predicted = bearings_from_view(view)          # מ-ViewPredictor, למטה

fix = HeadingMatcher().match(observed, predicted, hfov_deg=53.4)
if fix.accepted:
    use(fix.heading_deg, fix.sigma_deg)
```

**לבדוק `accepted`, לא רק `heading_deg`.** רחוב של חזיתות דומות מייצר כמה פסגות
שוות, והפסגה הגבוהה שם היא רעש. כיוון שגוי ובטוח גרוע מכיוון לא ידוע, כי כל מה
שבמורד הזרם מכפיל אותו בטווח.

**והעמודות חייבות להיות מתוקנות-עיוות לפני שהן נכנסות.** ה-`k1 = −0.366` של
העדשה הזו מזיז פיקסל פינתי בכ-10 עמודות.

### calibration — הסיבוב IMU→מצלמה

```python
from mapinit import RigOrientation, solve_imu_camera

orientations = [
    RigOrientation("flat", imu_accel_mg=(1002., 3., 14.), camera_up=(0., -1., 0.)),
    RigOrientation("tilt", imu_accel_mg=(-157., 985., -18.), camera_up=(0., -0.17, 0.98)),
]
solution = solve_imu_camera(orientations)
print(solution.observability)     # כמה חזק מפוענח הציר החלש
```

**לקרוא `observability` לפני שסומכים.** שני כיווני כובד במרחק 25 מעלות מפענחים
את הציר החלש פי עשרה פחות טוב משני כיוונים במרחק 90.

### view — מה המפה אומרת שאמור להיות בפריים

```python
from mapinit import BuildingLayer, DemSampler, ViewPredictor

predictor = ViewPredictor(
    buildings=BuildingLayer.from_geojson(paths.overture_path),
    dem=DemSampler(paths.glo30_path),
)
view = predictor.predict(31.7683, 35.2137, heading_deg=40.0)

for edge in view.visible_edges:
    column = edge.pixel_column(width_px=160, hfov_deg=53.4)
```

הפינות תלויות רק **איפה** המצלמה עומדת, לא לאן היא מסתכלת, אז
`sweep_headings()` עושה את העבודה היקרה פעם אחת למעגל שלם — זה מה שהופך חיפוש
כיוון על כל 360 המעלות לזול.

### navigation — מיקום וכיוון מקירות (רדאר מול המפה)

```python
from mapinit import WallMatcher, PosePrior, static_returns, BuildingLayer

walls = static_returns(rig.radar_detections_all())      # סטטיים, 0.5–40 מ'
prior = PosePrior(31.7683, 35.2137, heading_deg=40.0,
                  sigma_position_m=5.0, sigma_heading_deg=5.0, source="manual_pin")
fix = WallMatcher().match(walls, BuildingLayer.from_geojson(paths.overture_path), prior)
if fix.accepted:
    use(fix.latitude, fix.longitude, fix.heading_deg, fix.sigma_east_m, fix.sigma_heading_deg)
```

**לקרוא `accepted`, ואז `ambiguous`.** קיר ישר אחד קובע את המרחק ממנו ואת
הכיוון, ולא כלום לאורכו — הפיקס אז מסומן `ambiguous` עם שם הציר, ולא מצוטט
כאילו נמדד. `on_boundary` אומר שה-prior גרוע מהסיגמה שהוצהרה: החיפוש הוא 3σ
ולא יותר, **בכוונה**, כדי שרחוב דומה במרחק 40 מ' לא ינצח.

דרך הצינור: `MapInitializer(lat, lon, pose_observations=PoseObservations(prior, walls))`
ושלב `pose_init` רץ ומפרסם `InitialPose` (גובה מה-DEM, לא מהרדאר).

### map — אתחול מיקום מול המפה

```python
from mapinit import MapInitializer

init = MapInitializer(latitude=31.7683, longitude=35.2137)
report = init.run()               # אתחול שמור, כל בדיקה מדווחת
paths = init.priors()             # שכבות ה-DEM והוקטור לנקודה הזו
```

הבנייה זולה ולא נוגעת בדיסק. `run()` **לא זורק** כשמשהו חסר — הוא מחזיר דוח
שאומר איזה שלב נכשל, על סמך מה, ומה לעשות.

---

## מוסכמות שחייבות להתאים

הדברים שנשברים בשקט אם לא מיישרים אותם:

- **מסגרת rig / רדאר:** x קדימה, y שמאלה, z מעלה
- **מסגרת מצלמה:** x ימינה, y מטה, z קדימה
- **המאיץ במנוחה מודד `up`, לא `down`.** הציר שמצביע לשמיים קורא ‎+1000 mg.
  `RigOrientation` דורש את שני הווקטורים באותו מובן, ובמסגרת המצלמה שלמעלה
  `camera_up` של rig מפולס הוא **`(0, −1, 0)`**. העברת `(0, +1, 0)` כי "y זה
  למעלה" נותנת רוטציה שגויה ב-180 מעלות **עם residual נקי** — כלומר שום דבר
  במורד הזרם לא יתפוס אותה.
- **אזימוט:** חיובי הוא ימינה מהבורסייט, כמו שה-rig מדווח.
- **גבהים:** אורתומטריים, מול EGM2008. GNSS מדווח אליפסואידי; ההמרה היא
  `GeoidModel`, לא קבוע.
- **אופטיקה:** `fx=159.27`, `fy=159.59`, `cx=79.5`, `cy=59.5`, HFOV ‎53.4°.
  אלה נמדדו על היחידה הזו. דף-הנתונים אומר ‎57×44 והוא לא נכון כאן.

---

## טסטים

```bash
pip install -e ".[all,test]"
pytest
```

האיסוף מוגבל ל-`tests/` בכוונה: `board/` מחזיק קוד on-target מהענפים האחרים,
והטסט שלו מריץ בינארי C לא מקומפל.

---

## מה שלא מיוצא, ולמה

עוזרים פנימיים של הפותרים, תת-מחלקות ה-stage, וכל מה שמתחיל בקו תחתון. חשיפה
שלהם מקפיאה את הפירוק הנוכחי כחוזה, והופכת כל refactor לשינוי שובר בשני ריפואים
אחרים. כשצרכן צריך אחד מהם — מוסיפים אותו ל-`api.py` עם סיבה, וזו שיחה קטנה
יותר מלבטל ייצוא אחר כך.
