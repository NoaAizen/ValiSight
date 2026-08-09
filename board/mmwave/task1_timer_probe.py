# task1_timer_probe.py — משימה 1: מיפוי ה-timer-ים של ה-N6, ביחד עם נעה.
#
# מה הסקריפט עונה:
#   א. לכל timer — האם הוא קיים, מה התדר שלו, והאם המונה 16 או 32 ביט.
#   ב. לכל פין (P4–P9) — אילו צירופי (timer, channel) מקבלים input capture.
#   ג. האם אפשר להחזיק *שני* ערוצי IC על אותו timer בו-זמנית (VSYNC + SYNC_OUT).
#   ד. האם הצירוף שבחרנו שורד גם כשהמצלמה רצה (בדיקת התנגשות).
#
# איך מריצים: לחבר את ה-N6 ל-USB, לפתוח את OpenMV IDE, להדביק את הקובץ
# וללחוץ על החץ הירוק. לא צריך שום חיווט — הסקריפט שואל את השבב "מה יש לך",
# הוא עוד לא מודד אותות אמיתיים.
#
# מה רושמים בסוף: את הפלט של ארבעת החלקים בתוך NOA_TASK1_SESSION.md.

import pyb

# הפינים שבבדיקה קודמת (28/07) נמצאו כיחידים עם alt-function של timer.
# שאר הפינים על ה-header פשוט לא מחוברים לשום timer — אין טעם לבדוק אותם.
PINS = ['P4', 'P5', 'P6', 'P7', 'P8', 'P9']

# מספרי timer סבירים ב-STM32N6. מה שלא קיים בפירמוור פשוט יזרוק שגיאה — וזה בסדר,
# גם "לא קיים" זו תשובה שאנחנו רושמים.
TIMERS = [1, 2, 3, 4, 5, 8, 12, 13, 14, 15, 16, 17]

print("=" * 60)
print("part A: timer inventory — width and clock")
print("=" * 60)
# הטריק לרוחב המונה: מבקשים period של 32 ביט (0xFFFFFFFF).
# ל-timer של 16 ביט אין איפה לשמור מספר כזה — הוא ייחתך ל-0xFFFF.
# קוראים בחזרה את מה שנשמר בפועל, וכך החומרה בעצמה מסגירה את הרוחב.
for t in TIMERS:
    try:
        tim = pyb.Timer(t, prescaler=0, period=0xFFFFFFFF)
        real = tim.period()
        width = 32 if real > 0xFFFF else 16
        print("TIM%-2d  width=%2d-bit  period_readback=0x%X  source=%d Hz"
              % (t, width, real, tim.source_freq()))
        tim.deinit()
    except Exception as e:
        print("TIM%-2d  -- %s" % (t, e))

print()
print("=" * 60)
print("part B: which (timer, channel) gives IC on each pin")
print("=" * 60)
# פשוט מנסים הכל: לכל פין, לכל timer, לכל ערוץ 1-4 — האם ערוץ IC נתפס?
# צירוף לא חוקי נכשל מיד עם ValueError, אז הניסוי בטוח.
for pin_name in PINS:
    found = []
    for t in TIMERS:
        for ch in (1, 2, 3, 4):
            tim = None
            try:
                tim = pyb.Timer(t, prescaler=399, period=0xFFFF)
                tim.channel(ch, pyb.Timer.IC, pin=pyb.Pin(pin_name),
                            polarity=pyb.Timer.RISING)
                found.append("TIM%d_CH%d" % (t, ch))
            except Exception:
                pass
            if tim:
                try:
                    tim.deinit()
                except Exception:
                    pass
    print("%s: %s" % (pin_name, ", ".join(found) if found else "no IC route"))

print()
print("=" * 60)
print("part C: two IC channels on ONE timer, simultaneously")
print("=" * 60)
# למה חשוב ששני האותות יהיו על אותו timer? כי אז שתי חותמות הזמן נקראות
# מאותו מונה בדיוק — אפס סטייה בין השעונים. שני timer-ים נפרדים = שני
# שעונים שצריך ליישר, וזו בעיה שעדיף פשוט לא לייצר.
#
# על סמך בדיקת 28/07: P4 ו-P5 שניהם על TIM2 (הערוצים המדויקים יתגלו בחלק ב';
# אם חלק ב' הראה ערוצים אחרים — לעדכן כאן את CH_P4/CH_P5 לפי הפלט).
CH_P4 = 3   # <-- לעדכן לפי הפלט של חלק ב'
CH_P5 = 4   # <-- לעדכן לפי הפלט של חלק ב'
try:
    # prescaler=199 על שעון 400MHz: כל טיק = 200/400MHz = 500 ננו-שניות.
    tim2 = pyb.Timer(2, prescaler=199, period=0x3FFFFFFF)
    ch_a = tim2.channel(CH_P4, pyb.Timer.IC, pin=pyb.Pin('P4'),
                        polarity=pyb.Timer.RISING)
    ch_b = tim2.channel(CH_P5, pyb.Timer.IC, pin=pyb.Pin('P5'),
                        polarity=pyb.Timer.RISING)
    print("SUCCESS: TIM2 holds both IC channels at once")
    print("  P4 -> %s" % ch_a)
    print("  P5 -> %s" % ch_b)
    print("  counter now = %d (free-running)" % tim2.counter())
    # ניסוי כיף (רשות): לגעת בחוט מ-3V3 אל P4 — capture() יקפוץ לערך
    # המונה של רגע הנגיעה. ככה נראה input capture בעיניים:
    # >>> ch_a.capture()
    tim2.deinit()
except Exception as e:
    print("FAILED: %s" % e)
    print("  -> if this fails, update CH_P4/CH_P5 from part B output")

print()
print("=" * 60)
print("part D: does the camera steal our timer?")
print("=" * 60)
# הפירמוור של המצלמה משתמש בעצמו ב-timer-ים מסוימים. אם TIM2 תפוס על-ידי
# המצלמה — עדיף לגלות עכשיו ולא באמצע האינטגרציה. מאתחלים את החיישן,
# ואז מנסים שוב לתפוס את שני הערוצים.
try:
    import csi
    cam = csi.CSI()          # ברירת המחדל של המודול; אם ה-LEPTON מחובר,
    cam.reset()              # זה יאתחל אותו (זוכרים את המלכודת: csi.LEPTON
                             # הוא ה-sensor type, לא פרמטר שמנחשים).
    print("camera initialized")
except Exception as e:
    print("camera init failed (%s) — running part D without it" % e)

try:
    tim2 = pyb.Timer(2, prescaler=199, period=0x3FFFFFFF)
    tim2.channel(CH_P4, pyb.Timer.IC, pin=pyb.Pin('P4'),
                 polarity=pyb.Timer.RISING)
    tim2.channel(CH_P5, pyb.Timer.IC, pin=pyb.Pin('P5'),
                 polarity=pyb.Timer.RISING)
    print("SUCCESS: TIM2 dual-IC still works with the camera up")
    tim2.deinit()
except Exception as e:
    print("CONFLICT: %s" % e)
    print("  -> TIM2 is taken by the camera pipeline; fall back to the")
    print("     other pins/timers found in part B and re-run part C there")

print()
print("done. paste the full output into NOA_TASK1_SESSION.md")
