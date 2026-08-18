"""yael_api — הממשק שדרכו יעל מקבלת את כל החיישנים על הג'ייסון.

ממומש לפי "רשימת דרישות מלאה — מיקומים, כיול, GNSS וסנכרון חומרה" (17.8.2026).
נקודת הכניסה: Rig (rig.py). כל השמות והשדות כמו במסמך של יעל.
"""
from .rig import Rig                      # noqa: F401
from .clock import shared_clock_ms        # noqa: F401
