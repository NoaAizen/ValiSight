"""סעיף 5 — מהמצלמה התרמית: camera_geometry() ו-thermal_frame().

מקור: הפריימים ש-bridge_rx כותב (<seq>_thermal.bin) + frames.csv (seq, חותמת
N6, host_ms). הפריים מגיע מה-N6 כ-160×120 GRAY8 אחרי מיפוי ליניארי של
טמפרטורה: הלפטון במצב מדידה (רדיומטרי), IOCTL_LEPTON_SET_RANGE(15, 45) —
כלומר 0..255 = 15..45 °C, צעד 0.118 °C. thermal_frame() מחזיר גם את
הבתים הגולמיים וגם מערך °C.

FFC (כיול-מסך של הלפטון, "flat-field correction"): התריס נסגר ל-~1–1.5 ש',
התמונה הופכת אחידה. **הדגל מגיע מהחיישן עצמו**: השולח על ה-N6 קורא
LEP_CID_SYS_FFC_STATUS (I2C, 3.9 ms) לכל פריים ושם אותו בבית הדגלים של כותרת
התמונה (bit0 = FFC רץ, bit1 = נקרא בהצלחה). bridge_rx כותב אותו בעמודת flags
של frames.csv. אם הדגל לא ידוע (שולח ישן / קריאה נכשלה) — נופלים לזיהוי
מהתמונה (סטיית תקן נמוכה מאוד או פריים קפוא). **הזיהוי מהתמונה חלש**: ב-FFC
טבעי שנמדד 18.8 (19 פריימים, 2.3 ש') הלפטון החזיק את הפריים האחרון עם רעש
קטן — סטיית התקן לא זזה והבתים לא היו זהים — הניחוש פספס לגמרי. אחרי ה-FFC
הממוצע קפץ ב-~20 רמות (≈2.4 °C): כיול-מחדש אמיתי, לא תנועה. הדגל מהחיישן
הוא הדרך היחידה האמינה.

בנוסף: הלפטון הזה נושא 14 שורות מתות (בלי מידע סצנה) — ראה
valisight-lepton-dead-rows. הפריים כאן גולמי; התיקון הוא בצד של חגי (lepton_fix).
"""
import glob
import json
import math
import os
import struct

_HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(_HERE, "cfg", "calib_thermal_rgb.json")
MIN_C, MAX_C = 15.0, 45.0            # == n6_bridge_tx.main MIN_C/MAX_C
W, H = 160, 120
FFC_STD_MAX = 2.5                    # gray levels (~0.3 °C): below this the frame is "flat"
_IMG = "<HHBB"                       # == bridge_protocol._IMG


def camera_geometry(calib_path=CALIB_PATH):
    """5.1 — hfov/vfov וממדי הפיקסלים, מתוך הכיול המדוד (calib.json של חגי,
    K_th, rms 0.43 px) — לא מהדף-נתונים."""
    K = None
    if os.path.exists(calib_path):
        d = json.load(open(calib_path))
        K = d.get("K_th")
    if K:
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        src = "calib.json K_th (measured)"
    else:
        # Lepton 3.5 datasheet: 57° H, 71° diagonal
        fx = fy = (W / 2) / math.tan(math.radians(57 / 2))
        cx, cy = W / 2 - 0.5, H / 2 - 0.5
        src = "Lepton 3.5 datasheet (no calib file)"
    return {"width_px": W, "height_px": H,
            "hfov_deg": round(math.degrees(2 * math.atan((W / 2) / fx)), 2),
            "vfov_deg": round(math.degrees(2 * math.atan((H / 2) / fy)), 2),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "dist": (d.get("dist_th") if K else None),
            "source": src}


def decode_thermal(payload):
    """payload של רשומת THERMAL -> (w, h, bytes של פיקסלים). ValueError אם לא GRAY8."""
    if len(payload) < 6:
        raise ValueError("short thermal payload")
    w, h, fmt, _ = struct.unpack_from(_IMG, payload, 0)
    if fmt != 0 or len(payload) != 6 + w * h:
        raise ValueError("thermal payload is not GRAY8 %dx%d" % (w, h))
    return w, h, payload[6:]


def gray_to_celsius(gray_bytes):
    scale = (MAX_C - MIN_C) / 255.0
    return [MIN_C + b * scale for b in gray_bytes]


def frame_stats(gray_bytes):
    n = len(gray_bytes)
    if n == 0:
        return 0.0, 0.0
    mean = sum(gray_bytes) / n
    var = sum((b - mean) ** 2 for b in gray_bytes) / n
    return mean, math.sqrt(var)


IMG_FLAG_FFC, IMG_FLAG_FFC_KNOWN = 1, 2       # == bridge_protocol.IMG_FLAG_*


def ffc_in_progress(gray_bytes, prev_gray_bytes=None, flags=0):
    """הדגל מהחיישן אם ידוע; אחרת: פריים "שטוח" (תריס סגור) או קפוא זהה לקודם."""
    if flags & IMG_FLAG_FFC_KNOWN:
        return bool(flags & IMG_FLAG_FFC)
    if prev_gray_bytes is not None and gray_bytes == prev_gray_bytes:
        return True
    _, sd = frame_stats(gray_bytes)
    return sd < FFC_STD_MAX


class ThermalTail:
    """עוקב אחרי frames.csv של bridge_rx ומחזיר את הפריים התרמי האחרון."""

    def __init__(self, out_dir, clock_map):
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, "frames.csv")
        self.clock_map = clock_map
        self._pos = 0
        self._partial = b""
        self.last = None            # dict(seq, ts_ticks, ts_src, host_ms, len)
        self.frames = 0
        self._prev_gray = None
        self._last_gray = None
        self._last_seq_decoded = None

    def poll(self):
        if not os.path.exists(self.path):
            return 0
        n = 0
        with open(self.path, "rb") as fh:
            fh.seek(self._pos)
            data = self._partial + fh.read()
            self._pos = fh.tell()
        lines = data.split(b"\n")
        self._partial = lines.pop()
        for ln in lines:
            f = ln.decode(errors="replace").split(",")
            if len(f) < 6 or not f[0].isdigit() or f[1] != "thermal":
                continue
            try:
                rec = {"seq": int(f[0]), "ts_ticks": int(f[2]), "ts_src": int(f[3]),
                       "host_ms": int(f[4]), "len": int(f[5]),
                       "flags": int(f[6]) if len(f) > 6 and f[6].strip().isdigit() else 0}
            except ValueError:
                continue
            self.clock_map.observe(rec["ts_ticks"], rec["ts_src"], rec["host_ms"])
            self.last = rec
            self.frames += 1
            n += 1
        return n

    def _read_latest(self):
        if self.last is None:
            return None
        if self._last_seq_decoded == self.last["seq"]:
            return self._last_gray
        p = os.path.join(self.out_dir, "%010d_thermal.bin" % self.last["seq"])
        try:
            payload = open(p, "rb").read()
            _, _, gray = decode_thermal(payload)
        except (OSError, ValueError):
            return None
        self._prev_gray, self._last_gray = self._last_gray, gray
        self._last_seq_decoded = self.last["seq"]
        return gray

    # --- 5.2 ------------------------------------------------------------
    def thermal_frame(self, celsius=True):
        """הפריים התרמי האחרון: pixels (bytes 160×120), temps_c (רשימה שטוחה,
        אופציונלי), timestamp_ms בשעון המשותף, ffc_in_progress, seq."""
        self.poll()
        gray = self._read_latest()
        if gray is None:
            return None
        ts = self.clock_map.to_shared_ms(self.last["ts_ticks"], self.last["ts_src"])
        mean, sd = frame_stats(gray)
        return {"width": W, "height": H, "pixels_gray8": gray,
                "temps_c": gray_to_celsius(gray) if celsius else None,
                "range_c": (MIN_C, MAX_C),
                "timestamp_ms": ts, "seq": self.last["seq"],
                "ffc_in_progress": ffc_in_progress(gray, self._prev_gray, self.last["flags"]),
                "ffc_source": "sensor" if self.last["flags"] & IMG_FLAG_FFC_KNOWN else "image",
                "mean_c": round(MIN_C + mean * (MAX_C - MIN_C) / 255.0, 2),
                "std_gray": round(sd, 2)}
