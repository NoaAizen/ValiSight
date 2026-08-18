"""סעיף 2 — GNSS. אין עדיין מקלט על ה-rig (18.8.2026). הממשק קיים כדי שהקוד
של יעל ירוץ; הוא מחזיר בכנות "אין קליטה" ולא ממציא מיקום.

כשיגיע מקלט (NMEA/UBX דרך USB או gpsd), מממשים GnssSource.poll() שממלא
self.pvt / self.status — הקריאות של יעל לא משתנות. הדגלים jamming/spoofing
דורשים מקלט שמדווח עליהם (u-blox M8/M9 עם UBX-MON-RF / UBX-SEC-SIG).
"""


class Gnss:
    available = False

    def gnss_pvt(self):
        """2.1 — None כשאין פתרון ניווט (אין מקלט / No Fix)."""
        return None

    def gnss_status_and_security(self):
        """2.2 — סטטוס ואבטחה."""
        return {"fix_type": "No Fix", "jamming_indicator": False,
                "spoofing_indicator": False, "receiver_present": False}
