# מדריך הפעלת האימון (thermal + radar students)

המסלול המלא: הקלטה על הריג → תיוג אוטומטי (D1) → ייצוא shards → אימון ב-Colab
(או מקומית) → checkpoints. כל הפקודות רצות מ-`~/thermal-fusion/openmv-n6`.

```
הקלטה (run_live.sh --record) → run_teacher.py → build_dataset.py
      → export_shards.py → העלאה ל-Drive → train_students_colab.ipynb → *.pt
```

## 0. תנאים מקדימים

- הריג מחובר (N6 + IWR1843), הרדאר עם `radar_people.cfg` (זה ה-default של
  run_live.sh — לא לשנות `RADAR_CFG` בהקלטת אימון).
- הדיטקטור על GPU (TensorRT). אם `run_live.sh` מדפיס
  `RUNNING WITHOUT DETECTION` או `falling back to yolov4-tiny` — לעצור ולתקן
  לפני שמקליטים; ההקלטה עצמה תקינה, אבל שלב ה-teacher ירוץ לאט פי 10.

## 1. הקלטת סשן לאימון

```sh
cd ~/thermal-fusion/openmv-n6/tools
./run_live.sh --view visible --range 0:60 --record captures/<שם-סשן>
```

- **`--view visible` — חובה.** ה-teacher מסרב לכל סשן שאינו visible: בתצוגה
  fused שכבת התרמי צרובה בפיקסלים שהדיטקטור רואה (ככה נפסלו walk1–4).
- **`--range 0:60` — מומלץ לאימון.** מצמצם את חלון הטמפרטורה מ־‎-10..140
  (‎0.59 °C/LSB) ל־‎0.24 °C/LSB. הטווח נרשם ב-meta.json ומגיע עד ל-manifest
  (`c_per_lsb`).
- לסיום: `Ctrl+C` או `./run_live.sh --stop`.

מה שנשמר בתיקיית הסשן: `session.mp4` (וידאו נקי מ-overlay), `frames.jsonl`
(אינדקס + זמנים), `thermal.bin` (התרמי הרדיומטרי הגולמי), `radar.bin/jsonl`,
`meta.json`.

אילו סשנים שווה להקליט (הפערים של משימת 2–15 מ'): חדר חשוך (המורה RGB עיוור),
הליכות 7–15 מ' (אדם בגודל 10–20 px תרמי), שני אנשים חוצים, התקרבות/התרחקות על
ציר המבט (מהירות רדיאלית), וסצנות ריקות עם עומס חם (מסכים, רדיאטורים).

## 2. תיוג אוטומטי — שרשרת D1

```sh
cd ~/thermal-fusion/openmv-n6
python3 perception/autolabel/run_teacher.py captures/<שם-סשן>
python3 perception/autolabel/build_dataset.py
```

- `run_teacher.py` מקבל את הסשן כארגומנט **פוזיציונלי** (נתיב, לא `--session`).
  אופציות: `--model yolov10n|yolov8n|yolo11n`, `--conf 0.30`, `--limit N`.
  רץ ישירות על ה-Jetson, בלי venv (ה-teacher של TensorRT לא צריך).
- `build_dataset.py` סורק את **כל** קבצי `*_teacher.jsonl` שתחת
  `perception/out/autolabel/` ובונה מחדש את ה-COCO התרמי ואת דירוגי המסלולים
  (A ≥ 60 / B / REJ < 30). אופציות: `--min-track 5`, `--sheet` (גיליון תמונות
  לבדיקה בעין).

## 3. ייצוא shards

```sh
python3 perception/export/export_shards.py \
    --sessions <sess1> <sess2> ... \
    --val <סשן-מוחזק-בצד> \
    --out-name v2
```

- הפלט: `perception/out/gexport/v2/` — `manifest.json` + קבצי `.npz` +
  snapshot של הקוד תחת `code/` (חתום SHA256, כדי שכל checkpoint יהיה קשיר
  לקוד שיצר אותו).
- **ה-val הוא סשנים שלמים, לא פריימים** — פריימים עוקבים הם כמעט-כפילויות,
  ופיצול ברמת פריים מנפח כל מדד.
- סשן ריק משמש כ-negative **רק** אחרי שבן אדם וידא שהוא באמת ריק:
  `--verified-negative <sess>`. לא לסמן ככה רק כי המורה לא מצא קופסאות —
  חושך/ערפל/הסתרה הם בדיוק המקומות שבהם זה מרעיל תוויות.
- shards ישנים מגרסת v1 חייבים ייצוא מחדש (v2 מוסיף מסכות תוויות tri-state
  ו-dtype תרמי מוצהר; המחברת המעודכנת דורשת אותם).
- `--no-rgb` משמיט את ה-RGB אם צריך לחסוך נפח.

## 4. העלאה ל-Google Drive

להעלות את התיקייה `perception/out/gexport/v2/` אל:

```
MyDrive/thermal-fusion/gexport/v2
```

גרירה בדפדפן או rclone. **לא דרך GitHub** — מאות MB. (v1 היה ‎1.2GB / 47
shards לקנה מידה.)

## 5. אימון ב-Colab (המסלול הרגיל)

1. לפתוח ב-Colab את `train_students_colab.ipynb` **מאותה תיקיית Drive**
   (`gexport/v2/`) — המחברת מיוצאת יחד עם ה-shards.
2. Runtime → Change runtime type → **GPU**.
3. Run all.

ה-checkpoints נכתבים ל-`gexport/v2/models/`:
`thermal_student.pt` ו-`radar_student.pt`, כל אחד עם המטריקות, ה-config,
ו-SHA256 של ה-manifest שאימן אותו.

## 6. אימון מקומי (אלטרנטיבה)

אותו entrypoint רץ בכל מקום שיש בו PyTorch תואם:

```sh
PYTHONPATH=perception/out/gexport/v2/code \
python3 -m perception.train_students \
    --data perception/out/gexport/v2 --student both --epochs 50
```

דגלים עיקריים: `--student thermal|radar|both`, `--epochs 50`,
`--batch-size 32`, `--learning-rate 1e-3`, `--device auto|cpu|cuda`,
`--out DIR` (ברירת מחדל `DATA/models`).

> **אזהרה ל-Jetson:** אין `pip install torch` רגיל — כל התקנת pip שגוררת
> numpy 2.x שוברת את cv2 של apt (numpy נעול על 1.21). רק wheel של NVIDIA
> ל-Jetson, או פשוט Colab.

### אבלציות (בלי לערוך פייתון)

```sh
# תרמי בלי המשפחה הטמפורלית
... --disable-thermal temporal_diff,temporal_rate,previous_frame,previous_frame_2

# רדאר בלי Range-Azimuth, וענן נקודות בלי SNR/noise
... --disable-radar-family ra \
    --disable-radar-channel points:snr,noise
```

משפחות רדאר: `points`, `ra`, `rd`, `rp` — רק מה שה-UART באמת פלט קיים
ב-shards; אבלציה של משפחה שלא קיימת נכשלת בקול.

## מלכודות ידועות

| מלכודת | תוצאה |
|---|---|
| הקלטה ב-view שאינו visible | `run_teacher.py` מסרב לסשן כולו |
| `RADAR_CFG` אחר בהקלטה | ‎10 Hz default עושה aliasing ופוסל את ה-sha |
| קריאת טמפרטורה מה-mp4 | ה-mp4 הוא lossy 8-bit; המדידה היא `thermal.bin` בלבד |
| val ברמת פריים | מספרים מנופחים — תמיד סשנים שלמים |
| `--verified-negative` בלי בדיקה אנושית | תוויות מורעלות בדיוק בתנאים הקשים |
| shards v1 במחברת החדשה | חסרות מסכות tri-state — לייצא מחדש |
