"""סעיף 1.2 — shared_clock_ms(): שעון משותף אחד לכל החיישנים.

השעון המשותף = CLOCK_MONOTONIC של הג'ייסון, במילישניות. הוא לא קופץ (גם אם
מישהו מכוון את שעון הקיר), ומתחיל מאפס באתחול. הרדאר נחתם בו ישירות (זמן
הגעת הפריים ב-UART). ה-N6 חותם בשעון שלו (ticks_us) — N6ClockMap ממפה אותו
לשעון המשותף.

איך המיפוי עובד: לכל רשומה מה-N6 יש (n6_ticks, host_ms) — הזמן שהיא נוצרה
על ה-N6 והזמן שהיא הגיעה למחשב (bridge_rx כותב את שניהם). ההפרש
host_ms − n6_ticks/1000 = offset + latency. ה-latency תמיד ≥ 0 ומשתנה,
ה-offset קבוע (עד סחיפת שעונים איטית). לכן: offset ≈ המינימום של ההפרש
בחלון האחרון (מסנן-מינימום קלאסי לסנכרון שעונים). דיוק שנמדד: ~2–5 ms
(latency USB מינימלית) — הרבה מתחת ל-200 ms שיעל ציינה כבעיה.
"""
import time
from collections import deque


def shared_clock_ms():
    """מונה מונוטוני יחיד, מילישניות (int)."""
    return time.monotonic_ns() // 1_000_000


class N6ClockMap:
    """ממפה ticks של ה-N6 (µs או 0.5 µs לפי ts_src) לשעון המשותף (ms)."""

    def __init__(self, window=2000, drift_ppm_max=200.0):
        self._pairs = deque(maxlen=window)     # (n6_ms, host_ms)
        self._offset_ms = None                 # host_ms - n6_ms (min-filtered)
        self._n = 0
        self._drift_ppm_max = drift_ppm_max

    @staticmethod
    def ticks_to_ms(ticks, ts_src):
        return ticks / 1000.0 if ts_src == 0 else ticks / 2000.0

    def observe(self, n6_ticks, ts_src, host_ms):
        n6_ms = self.ticks_to_ms(n6_ticks, ts_src)
        self._pairs.append((n6_ms, host_ms))
        self._n += 1
        d = host_ms - n6_ms
        if self._offset_ms is None or d < self._offset_ms:
            self._offset_ms = d
        elif self._n % 500 == 0:
            # re-derive from the window so a slow drift is followed (the min of
            # a window, not the min of all time)
            self._offset_ms = min(h - n for n, h in self._pairs)

    @property
    def ready(self):
        return self._offset_ms is not None

    @property
    def offset_ms(self):
        return self._offset_ms

    def to_shared_ms(self, n6_ticks, ts_src):
        """חותמת N6 -> שעון משותף. None עד שנצפתה רשומה ראשונה."""
        if self._offset_ms is None:
            return None
        return int(round(self.ticks_to_ms(n6_ticks, ts_src) + self._offset_ms))
