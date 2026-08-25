# ValiSight v0.4.0 — סיכום שינויים מ־v0.3.0

## פרטי הגרסה

| פריט | ערך |
|---|---|
| גרסת בסיס | `v0.3.0` |
| גרסה חדשה | `v0.4.0` |
| טווח ההשוואה | `v0.3.0..v0.4.0` |
| תאריך `v0.3.0` | 19 באוגוסט 2026 |
| תאריך `v0.4.0` | 23 באוגוסט 2026 |
| commits חדשים | 3 |
| קבצים שהשתנו | 56 — מתוכם 38 חדשים ו־18 מעודכנים |
| diff כולל | 12,402 שורות נוספו, 138 הוסרו |

> חלק גדול מנפח ה־diff מגיע ממחברות Colab ומסשן הוולידציה `session_walk`
> הכולל וידאו, קבצי radar/thermal ורשומות JSONL; הוא אינו כולו קוד מוצר.

## תקציר מנהלים

גרסה `v0.4.0` הופכת את העבודה שהייתה ב־`v0.3.0` סביב כיול
Radar↔RGB ותחילת Radar‑AI לצינור עבודה רחב יותר:

1. הקלטת סשן מסונכרן עם provenance מלא יותר.
2. תיוג אוטומטי באמצעות teacher ניתן לבחירה.
3. ייצוא dataset ב־schema חדש הכולל thermal, נקודות radar ו־heatmaps אופציונליים.
4. אימון שני student models נפרדים — thermal ו־radar — ב־Colab או ב־PyTorch מקומי.
5. baseline שקוף לקבלת החלטה משותפת משתי המודאליות.
6. שדרוג תצוגת ה־live, בחירת מודלי YOLO וכיול ההקרנה של ה־radar.
7. הוספת תהליך מאומת לסנכרון סשנים ל־NAS.

הגרסה מוסיפה את **תשתית האימון**, אך אינה כוללת checkpoint מאומן (`.pt`),
TensorRT engine או תוצאות accuracy סופיות בתוך ה־tag.

## 1. צינור נתונים ואימון students

נוסף flow מתועד מקצה לקצה:

```text
live.py --record
        ↓
run_teacher.py + build_dataset.py
        ↓
export_shards.py (schema v2)
        ↓
Google Drive / Colab
        ↓
thermal_student.pt + radar_student.pt
```

### ייצוא dataset חדש

- נוסף `perception/export/export_shards.py` לייצוא סשנים ל־NPZ shards.
- ה־split ל־train/validation נעשה לפי **סשנים שלמים**, כדי למנוע זליגת
  פריימים כמעט זהים בין train ל־validation.
- לכל פריים ניתן לשמור:
  - thermal גולמי בגודל 160×120;
  - RGB בגודל 640×400, עם אפשרות להסירו באמצעות `--no-rgb`;
  - עד 64 נקודות radar, ממוינות לפי SNR;
  - `dt_ms` והיסטוריה זמנית;
  - teacher boxes ו־thermal grade-A boxes;
  - Range–Azimuth, Range–Doppler ו־Range Profile, אם ה־TLVs קיימים בסשן.
- נוסף label תלת־מצבי לכל student:
  - `1` — positive;
  - `0` — negative שאומת ידנית;
  - `-1` — unknown, שאינו נכנס ל־loss או למדדים מפוקחים.
- סשן שלילי חייב להיות מסומן מפורשות עם `--verified-negative`; היעדר boxes
  מה־RGB teacher לבדו אינו נחשב הוכחה שאין אדם בסצנה.
- כל export כולל snapshot חתום ב־SHA256 של קוד ה־loader והמודלים ששימש אותו,
  כדי לאפשר שחזור provenance של checkpoint עתידי.

### טעינת נתונים ואימון

- נוסף `perception/student_data.py` לטעינת manifest/shards, אימות provenance,
  בניית רצפים זמניים, scaling של thermal ובניית targets.
- נוסף `perception/students.py` עם:
  - `ThermalStudent` — encoder מרחבי מעל thermal והיסטוריה של שלושה פריימים;
  - `RadarStudent` — שילוב של point cloud עם RA/RD/Range Profile אופציונליים;
  - detection head לעד שמונה אובייקטים, עם Hungarian matching;
  - loss, evaluation ו־training loop.
- נוסף `perception/train_students.py` לאימון thermal, radar או שניהם.
- נוספו ablations ללא שינוי קוד:
  - ביטול ערוצי thermal נגזרים;
  - ביטול משפחת radar שלמה: points/RA/RD/RP;
  - ביטול ערוצים ספציפיים בתוך משפחה.
- checkpoints עתידיים כוללים manifest hash, רשימות סשני train/val, metrics,
  קונפיגורציית אימון, ערוצים פעילים ופרטי המודל.
- נוספו מחברות Colab:
  - `train_students_colab.ipynb` — נקודת הכניסה העדכנית;
  - `train_colab.ipynb` ו־`train_colab_v3.ipynb` — איטרציות נוספות של flow האימון.

### baseline לקבלת החלטה

נוסף `perception/decision.py` עם baseline מפורש ולא־נלמד:

- מסנן evidence חסר, לא תקין או ישן מדי;
- מחזיר `PERSON`, `NO_PERSON` או `UNKNOWN`;
- מתעד provenance: שתי המודאליות, thermal בלבד, radar בלבד או ללא evidence;
- משלב הסתברויות מכוילות באמצעות log-odds כאשר שני החיישנים זמינים;
- מחזיר `UNKNOWN` במקרה של סתירה חזקה בין החיישנים במקום לייצר ביטחון שגוי.

זהו baseline להשוואה מול fusion נלמד עתידי, לא שכבת fusion מאומנת.

## 2. Thermal: הקלטה, scaling ותיוג

- פורמט ההקלטה עודכן ל־`schema_version: 2`.
- `recorder.py` מזהה ומתעד thermal frames מסוג:
  - `uint8` — 19,200 bytes;
  - `uint16_le` — 38,400 bytes;
  - `opaque` עבור גודל לא מוכר.
- `perception/dataset.py` קורא גם `uint8` וגם little-endian `uint16`, בודק
  התאמה בין dtype לאורך הפריים ונכשל במפורש על metadata לא עקבי.
- `live.py` רושם ב־`meta.json` את חלון הטמפרטורה ואת הסקאלה:
  `tmin`, `tmax`, `c_per_lsb`, `thermal_dtype`, `thermal_encoding`,
  `thermal_counts_max` ו־Lepton gain.
- ה־auto-label מנרמל delta תרמי ליחידות `uint8_equivalent`, כך שספי התיוג
  הישנים נשארים בעלי אותה משמעות גם עבור מקור עם עומק גדול יותר.
- סשן `uint16` ללא `thermal_counts_max` נדחה במקום להתפרש בסקאלה שגויה.
- ה־teacher תומך כעת בבחירת `yolov10n`, `yolov8n` או `yolo11n`, ומתעד איזה
  model/backend שימש לתיוג.

## 3. Radar: heatmaps, parsing וקונפיגורציה

### קונפיגורציית איסוף חדשה

נוסף `radar/configs/radar_people_ra.cfg`, המבוסס על `radar_people.cfg` ומיועד
לסשני איסוף heatmap:

- קצב ירד מ־10 FPS ל־5 FPS כדי לעמוד בתקציב UART;
- Range–Azimuth heatmap ו־Range Profile מופעלים;
- Range–Doppler נשאר כבוי בקונפיגורציה זו בגלל נפח השידור;
- הקונפיגורציה מיועדת ל־**איסוף נתוני אימון**, לא להחלפת מצב ההפעלה הרגיל.

נוסף `calibData 0 0 0` לפני `sensorStart` בחמש קונפיגורציות radar קיימות,
כדי לבטל שימוש אפשרי בנתוני calibration שמורים שאינם שייכים לקונפיגורציה.

### parser ו־heatmaps

- `radar/mmwave.py` יודע כעת לפרש:
  - TLV 2/3 — Range/Noise Profile;
  - TLV 4 — Range–Azimuth complex heatmap;
  - TLV 5 — Range–Doppler heatmap.
- נשמר במפורש סדר `imaginary, real` של TLV 4 כדי למנוע היפוך ציר azimuth.
- Range–Doppler מפורש עם 64 Doppler bins עבור `radar_people`, ולא עם ההנחה
  הישנה של 16 bins.
- גודל frame מרבי גדל מ־16/24 KB ל־48 KB וה־buffer גדל ל־256 KB, כדי לא
  לדחות heatmap frames תקינים או לאבד bytes בזמן השהיה קצרה בצד ה־host.
- נוסף `perception/export/radar_heatmaps.py` לפענוח, סיכום והפקת RA image.

## 4. Live viewer וחוויית מפעיל

- נוספה תצוגת `operator`:
  - שומרת את ה־luminance של המצלמה הנראית;
  - משתמשת בעיקר בצבע מהשכבה התרמית;
  - מתאימה את המשקל לפי contrast מקומי;
  - מבצעת feathering בגבול הכיסוי התרמי במקום חיתוך מלבני חד.
- ברירת המחדל של `run_live.sh` השתנתה מ־`visible` ל־`operator`.
- משקל thermal ברירת המחדל עלה מ־50 ל־60.
- `HTTP` ניתן כעת לדריסה באמצעות environment variable.
- metadata של סשן כולל SHA256 של warp LUT, radar calibration ו־detector engine.

חשוב: צבעי `operator` מיועדים לצפייה נוחה ואינם סקאלת טמפרטורה. מדידה עדיין
נעשית מה־thermal data וה־probe, לא מהפיקסל המרונדר.

## 5. הקרנת Radar וכיול

- `run_live.sh` טוען אוטומטית את
  `calib-artifacts/radar_rgb_2026-08-18.json` כאשר הוא קיים.
- `radar_overlay.py` תומך כעת בפתרון מלא של `R`, `t`, מטריצת מצלמה `K`
  ו־distortion, במקום להסתמך רק על yaw/translation מקורבים.
- distortion מוחל על ההקרנה כאשר הוא קיים, עם guard נגד fold של פולינום
  Brown–Conrady מחוץ לטווח המונוטוני.
- מנגנון `/set` הישן נשאר fallback בלבד; הוא אינו דורס את ה־extrinsic הפתור.

### ולידציית הליכה

נוספו `tools/calib/walk_validate.py` וסשן הראיות `tools/captures/session_walk`.
התוצאות המתועדות:

- 567 פריימים נמדדו;
- שגיאה זוויתית מוחלטת חציונית: **1.84°** — יעד ≤2° הושג;
- שגיאה חתומה חציונית: **+0.03°** — אין הצדקה לתיקון yaw גלובלי;
- 87.3% מההקרנות בתוך רוחב הגוף מול יעד של 90%; הפער מיוחס בעיקר ל־ghosts
  ובלבול מטרות בצד ימין של הסצנה, ולא לשגיאת extrinsic גלובלית.

## 6. מודלי detection ו־TensorRT

- `live.py` תומך ב־`--detect-model yolov10n|yolov8n|yolo11n`.
- נוסף `--detect-engine` לדריסת engine path.
- נוסף `tools/export_ultralytics_onnx.py` לייצוא YOLOv8/YOLO11 ל־ONNX סטטי
  עם NMS כלול.
- `trt_detect.py` הפך מ־runtime ייעודי ל־YOLOv10n ל־runtime רב־מודלי.
- נוספה ולידציה קשיחה לחוזה TensorRT:
  - input סטטי `[1,3,H,W]`;
  - output יחיד `[1,max_det,6]` עם NMS;
  - דחיית export דינמי או raw prediction tensor;
  - תמיכה ב־FP16/FP32 בהתאם ל־dtype האמיתי של engine.
- שחרור זיכרון CUDA הפך idempotent ובטוח יותר.

TensorRT engines נשארים תלויי GPU וגרסת TensorRT ויש לבנות אותם על מחשב
היעד; הם אינם כלולים ב־Git.

## 7. סנכרון סשנים ל־NAS

נוסף `tools/sync_session.sh`:

- מאתר את כתובת ה־NAS הזמינה לפי סדר Ethernet, Wi‑Fi ואז Tailscale;
- מעתיק סשנים רק לאחר סיום ההקלטה;
- מבצע verification מלא עם checksum;
- יוצר `.synced.ok` רק לאחר אימות;
- מסמן `DIED_MIDRECORDING` כאשר `meta.json` קיים ללא `closed_wall`;
- אינו מוחק את המקור המקומי.

נוספו גם marker files עבור סשנים שכבר סונכרנו.

## 8. בדיקות ותיעוד

נוספו או הורחבו בדיקות עבור:

- parser וייצוג heatmaps;
- label states ו־verified negatives;
- dtype ו־scaling של thermal;
- בניית temporal histories;
- student models ו־PyTorch forward path;
- baseline decision והטיפול ב־stale/contradicting evidence;
- חוזי input/output של detector models;
- תצוגת `operator` ונתיבי live חדשים.

נוסף תיעוד מפורט תחת:

- `perception/export/README.md` — flow מלא מאיסוף עד אימון;
- `tools/README.md` — מבנה כלי ה־live, מודלי detector ותהליך בניית engine;
- `tools/captures/session_walk/VALIDATION.md` — תוצאות ולידציית ההקרנה.

## שינויי תאימות והערות לשדרוג

1. **יש לייצא מחדש shards ישנים.** מחברת האימון העדכנית דורשת schema v2,
   label masks תלת־מצביים ו־thermal dtype מפורש.
2. **סשני `fused` ישנים אינם מתאימים לאימון.** כאשר השכבה התרמית אפויה בתוך
   הווידאו ואין `thermal_off`, ה־exporter מדלג על הסשן.
3. **Negative דורש אימות ידני.** אין להפוך “ה־RGB teacher לא מצא אדם” ל־negative.
4. **`uint16` דורש metadata.** יש להצהיר על encoding/scale ו־counts maximum;
   אחרת auto-labeling נכשל בכוונה.
5. **חוזה TensorRT השתנה והוקשח.** YOLOv8/YOLO11 חייבים export סטטי עם NMS
   ו־output `[1,max_det,6]`.
6. **ברירת המחדל הוויזואלית השתנתה.** `run_live.sh` עולה כעת ב־`operator` ולא
   ב־`visible`; לכיול pixel picking יש להעביר `--view visible` במפורש.
7. **קונפיגורציית RA היא מצב איסוף.** אין להשתמש בה כברירת מחדל תפעולית.

## מגבלות ידועות ב־v0.4.0

- ה־thermal החי עדיין מוקלט בדרך כלל כ־8-bit לינארי בתוך `[TMIN,TMAX]`, לא
  כ־RAW14 אמיתי. תמיכת הקריאה ב־`uint16_le` אינה משנה לבדה את firmware החיישן.
- ה־radar מספק TLVs ונקודות דרך UART, לא raw ADC, complex IQ או radar cube מלא.
- מהירות radar עדיין folded במחזור 1.298 m/s ויש להתייחס אליה בהתאם באימון.
- אין בגרסה תוצאות benchmark סופיות ל־student models ואין model checkpoint
  שמוכן לפריסה.
- Fusion נלמד בין thermal ל־radar טרם נוסף; קיים baseline שקוף בלבד.

## ה־commits שנכללו

| Commit | תאריך | תיאור |
|---|---|---|
| `3e2a351` | 2026-08-19 | כיול/ולידציית הליכה, סשן ראיות ושיפורי radar overlay |
| `9880e23` | 2026-08-19 | תשתית Colab/export, heatmaps, קונפיגורציית RA וסנכרון NAS |
| `0c272c6` | 2026-08-23 | אימון students, decision baseline ושדרוגי live/detection |

## פקודת שחזור ההשוואה

```bash
git diff --stat v0.3.0..v0.4.0
git log --oneline --no-merges v0.3.0..v0.4.0
```
