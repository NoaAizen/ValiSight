# חוזה החיבור מול ענף `yael`

החיבור הוא API פנימי של Python. שני החלקים רצים באותו מאגר ובאותו תהליך, ולכן
אין כאן שרת HTTP, פורט נוסף או סכמת retry שאינם מוסיפים מידע. ענף יעל אחראי על
חישובי המפה; ענף ה־fusion אחראי על המרת התוצאה לחוזה JSON יציב דרך
`perception.map_api`.

החוזה נבדק מול `origin/yael` בקומיט `fce2918` (2026-08-25), בריצה מלאה על
הג'טסון עם גריד EGM2008 ומטמון ה-priors של ירושלים: `ok=true`.

## מה צריך לקבל מיעל

החבילה `mapinit` צריכה להיות זמינה ב־`PYTHONPATH`, יחד עם תלויות הריצה שלה
(`pyproj`, ‏`rasterio`, ו־`numpy` כאשר מפעילים את כיול ה־IMU), ולייצא:

```python
from mapinit import MapInitializer

initializer = MapInitializer(
    latitude=31.7683,
    longitude=35.2137,
    repo_dir=None,                    # אופציונלי
    expected_geoid_range=(19, 20.5), # אופציונלי
)
report = initializer.run(fail_fast=True)
```

זהו הממשק היחיד שקוד ה־fusion צורך. הוא אינו מייבא את `InitContext`, את שלבי
הצנרת או את מחלקות ה־geo הפנימיות.

`report` חייב לחשוף:

- `ok: bool`
- `results: list[StageResult]`
- `summary() -> str`

כל `StageResult` חייב לחשוף `stage`, ‏`status`, ‏`checks`, ‏`data`, ‏`error`.
ערכי `status` הם רק `ok`, ‏`failed`, ‏`skipped`. כל check חושף לפחות `name`,
‏`passed`, ‏`detail`; השדות המספריים `measured`, ‏`expected`, ‏`unit` אופציונליים.

מפתחות הפלט שהמתאם צורך מתוך `StageResult.data`:

| stage | מפתחות |
|---|---|
| `geoid` | `undulation_m`, `transform` |
| `priors` | `glo30_path`, `overture_path`, `provider`, `ground_elevation_m`, `surface_elevation_m`, `dem_posting_m`, `ego_altitude_prior` |

`ego_altitude_prior`, כאשר קיים, חושף `orthometric_m`, ‏`ellipsoidal_m`,
‏`geoid_undulation_m`, ‏`sigma_m`. נתיב חסר או grid חסר הם כשל stage מדווח —
לא fallback לגובה אפס.

בנוסף לתלויות Python נדרשים קובץ EGM2008 ואריחי GLO-30/Overture תחת ספריית
הנתונים של `mapinit`. תלות Python חסרה מדווחת כ־`MapAPIUnavailable` עם שמה;
היא בעיית deployment ולא תוצאת מיפוי.

## מה הקוד שלנו מחזיר

```python
from perception.map_api import MapInitRequest, MapInitializationAPI

result = MapInitializationAPI().initialize(
    MapInitRequest(
        latitude=31.7683,
        longitude=35.2137,
        expected_geoid_range_m=(19.0, 20.5),
    )
)
```

`result` מכיל primitives בלבד ולכן ניתן לשמור אותו ב־JSON או להחזיר אותו
מ־endpoint של `live.py`. השדה `schema_version` הוא `1.0`; שינוי שם או משמעות של
שדה מחייב גרסה חדשה. `ok=false` הוא תוצאת אתחול תקינה שמדווחת כשל במפה;
`MapAPIContractError` מצביע על שבירת החוזה בין הענפים.

להרצה ידנית, מתוך `openmv-n6/` ולאחר שהחבילה של יעל נמצאת ב־path:

```sh
PYTHONPATH=.. python3 -m perception.map_api \
  --lat 31.7683 --lon 35.2137 --expect-geoid 19 20.5
```

ולבדיקת המתאם ללא DEM או geoid grid:

```sh
python3 -m pytest -q perception/tests/test_map_api.py
```

## איך זה רץ על הג'טסון (2026-08-25)

הענף של יעל **לא ממוזג** — הוא נשאר checkout נפרד ו-`perception.map_api` מוצא
אותו לבד, לפי הסדר: `--mapinit-dir`, המשתנה `VALISIGHT_MAPINIT`, ואז
`<repo>/mapinit/` או `ValiSight_yael/` לצד הריפו (`~/ValiSight_yael` על הג'טסון).

מה שצריך להיות קיים פעם אחת במכונה:

```sh
git worktree add ~/ValiSight_yael origin/yael             # הקוד של יעל
pip install --user pyproj==3.7.1 rasterio==1.4.4 'attrs>=23'   # התלויות (attrs: של אובונטו ישן מדי)
curl -o ~/.local/share/proj/us_nga_egm08_25.tif https://cdn.proj.org/us_nga_egm08_25.tif
cd ~/ValiSight_yael && python3 tools/fetch_priors.py --lat 31.7683 --lon 35.2137 --radius 2000
```

ואז ב-`live.py`:

```sh
./run_live.sh --map 31.7683,35.2137 --map-geoid 19 20.5
curl -s localhost:8088/map | python3 -m json.tool      # החוזה (schema 1.0) כמו שהוא
curl -s localhost:8088/health                            # שורת "map": ok / warn / fail
```

האתחול רץ ב-thread נפרד ולא מעכב את הווידאו. ב-`/health` הטקסט אומר איזה תיקון
צריך: `stage failed: geoid` = חסר הגריד, `stage failed: priors` = אין מטמון
לאזור הזה, `deployment: ...` = החבילה או תלות שלה חסרות.

מה שיעל עוד לא מימשה, ומדווח `skipped`: `calibration` (צריך אילוצי כיול)
ו-`pose_init` (פתרון מיקום מול המפה). הדרישות שלה מהצד שלנו — ב-`DRISHOT.md`
על הענף שלה: `radar_detections_all()` כולל החזרים סטטיים, `yaw_sigma_deg`,
`camera_geometry()` עם המספרים המדודים, ו-`initial_fix()`.

## מיקום מהקירות (pose_init v0, 2026-08-25)

`mapinit` פותר עכשיו מיקום+כיוון מהחזרי-קיר סטטיים של הרדאר מול מתארי הבניינים
(`mapinit.nav.walls.WallMatcher`). דרך החוזה:

```python
MapInitRequest(latitude, longitude,
               heading_prior_deg=40.0,            # מצפני; אין לו מקור אחר — GPS/נעיצה
               sigma_position_m=5.0, sigma_heading_deg=5.0,   # כנים! החיפוש הוא 3σ ולא יותר
               wall_returns=((range_m, azimuth_deg), ...))    # + = ימינה, סטטיים, 0.5–40 מ'
```

התשובה מקבלת בלוק `pose`: `latitude_deg, longitude_deg, height_m, heading_deg`,
הסיגמות, ו-**`accepted / ambiguous / ambiguity_axis / on_boundary`**. לקרוא את
`accepted` ולא רק את המיקום: קיר ישר אחד לא קובע את הציר לאורכו (`ambiguous`
עם שם הציר), ו-`on_boundary` אומר שה-prior גרוע ממה שהצהרת.

CLI: `python3 -m perception.map_api --lat .. --lon .. --heading 40 --walls radar.jsonl`

ב-`live.py`: `--map-heading DEG --map-sigma M DEG` (או `/set?heading=`), `/mapfix`
פותר מההחזרים העדכניים (~5 ש'), `/topdown` נותן את קווי הבניינים וההחזרים במטרים
מזרח/צפון, וב-`/ui` יש `nav` — אופק ניווט מ-`DeadReckoner` (שניות עד מטר סחיפה,
עם/בלי עזרת מהירות מהרדאר). הכרטיס בדף מצייר את זה: **ההחזרים (ענבר) חייבים לשבת על
הקירות (אפור)** — במעבדה הם לא, כי קירות פנימיים לא במפה, והפותר מסרב בהתאם.
