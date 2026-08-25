# חוזה החיבור מול ענף `yael`

החיבור הוא API פנימי של Python. שני החלקים רצים באותו מאגר ובאותו תהליך, ולכן
אין כאן שרת HTTP, פורט נוסף או סכמת retry שאינם מוסיפים מידע. ענף יעל אחראי על
חישובי המפה; ענף ה־fusion אחראי על המרת התוצאה לחוזה JSON יציב דרך
`perception.map_api`.

החוזה נבדק מול `origin/yael` בקומיט
`a54b7df309122528067d7c5ea45e1bfc4d134346`.

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
