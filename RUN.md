# VailSight — איך מפעילים את המערכת

מדריך הפעלה מעשי. כל הפקודות מהשורש של הריפו.

## מיפוי החומרה
| רכיב | חיבור | הערה |
|---|---|---|
| ראדר IWR1843 | COM7 + COM8 (CONFIG + DATA) | מזוהה אוטומטית |
| OpenMV N6 (תרמי Lepton + RGB) | COM12 | VID 0x37C5; מריץ תוכנית **אחת** בכל רגע |
| מצלמת USB | cam 0 | webcam רגילה |
| DCA1000 (ADC גולמי, לדופק) | NIC 192.168.33.30 + אתרנט | **לא מחובר כרגע** |

---

## 0. חשוב: שחזור ה-N6 ממצב radar-bridge
אם ה-N6 תקוע במצב הגשר (המצלמה התרמית לא עולה), כי `main.py` מריץ את הגשר:
1. **reset פיזי** ל-N6 (כפתור reset או נתק/חבר USB).
2. תוך ~3 שניות (חלון ה-grace):
   ```
   python -m mpremote connect COM12 rm :main.py
   ```
   אחרי זה כלי ה-mpremote (תרמי וכו') עובדים שוב.
   (אם הגרסה עם `--restore` כבר מותקנת: `python bridge_radar_to_n6.py --restore`.)

---

## 1. בדיקות (בלי חומרה)
```
python -m pytest -q            # 133 בדיקות — הליבה הטהורה + החוזים
```

## 2. ראדר + מצלמה — סריקה חיה עם fusion
דורש: ראדר (COM7/8) + מצלמת USB. עצמאי.
```
python live_radar_camera.py
# q = יציאה, s = צילום מסך.  שומר סשן ל-logs/session_*/
```

## 3. תרמי + RGB (black-hot) על ה-N6
דורש: N6 פנוי (ראה §0). 
```
python view_thermal_rgb.py                 # black-hot כברירת מחדל
python view_thermal_rgb.py --palette inferno
# m = החלפת פלטה,  q/Esc = יציאה.  התרמי נעול ~8.7fps (חומרת Lepton)
```

## 4. עיבוד ראדר על ה-N6 — **בלי חיווט** (גשר PC)
הראדר מדבר ל-PC, ה-PC מזין ל-N6, ה-N6 מעבד (parse+gate+classify) ומחזיר.

**התקנה חד-פעמית** (אחרי §0, כשה-N6 פנוי):
```
python -m mpremote connect COM12 cp iwr1843_uart.py :iwr1843_uart.py
python -m mpremote connect COM12 cp radar_classify_n6.py :radar_classify_n6.py
python -m mpremote connect COM12 cp core/radar_gate/gate.py :radar_gate.py
python -m mpremote connect COM12 cp n6_radar_bridge.py :main.py
python -m mpremote connect COM12 reset
```
**הרצה:**
```
python bridge_radar_to_n6.py --synthetic        # בדיקה בלי ראדר
python bridge_radar_to_n6.py --radar-port COM7  # ראדר חי דרך הגשר
python bridge_radar_to_n6.py --restore          # לשחרר את ה-N6 בחזרה
```

## 5. סיווג ראדר על ה-N6 — עם חוט ישיר (edge אמיתי)
דורש: חיווט DATA UART של הראדר ל-RX של UART ב-N6 + GND משותף (3.3V!).
```
python view_radar_n6.py --cfg-port COM8    # מציג bird's-eye של אשכולות מה-N6
```

## 6. נוד fusion מאוחד על ה-N6 — תרמי + ראדר מסונן, מתוזמן
דורש: חיווט DATA UART ל-N6 (כמו §5). מזווג ראדר↔תרמי לפי חותמת זמן.
```
python view_fusion_n6.py --cfg-port COM8
```

## 7. זיהוי דופק (vital signs) — דורש DCA1000
דורש: DCA1000 מחובר (NIC 192.168.33.30 + אתרנט + `lvdsStreamCfg`).
```
python radar_vitals_live.py --seconds 25        # נבדק יושב, ~0.3-1מ', חזה מול הראדר
python radar_vitals_live.py --from-bin cap.bin  # ניתוח הקלטה בלי חומרה
```

על ה-**Jetson** (2026-07: הצינור מאומת אופליין ב-`tests/test_vitals_adapter.py`;
חי — ברגע שה-DCA1000 מחובר):
1. ‏DCA1000 על מחבר ה-LVDS ‏(60-pin) של ה-IWR1843BOOST; מתג SW2.5 = SW_CONFIG.
2. אתרנט DCA1000 → ‏Jetson. ה-NIC כבר משמש את ה-LAN, אז או מתג קטן, או IP משני:
   `sudo ip addr add 192.168.33.30/24 dev enP8p1s0`
3. `python3 radar_vitals_live.py --seconds 25` — איתור הפורטים אוטומטי
   ‏(‏`/dev/ttyACM*`; אם המספור זז — `--cfg-port /dev/ttyACM1`).
4. הראדר מודאלי: קונפיג ה-vitals מחליף את קונפיג הזיהוי — לא בו-זמנית עם ‏fusion.
   הצינור מסרב ביושר ("no reliable pulse") כשהנבדק זז או ה-SNR נמוך — זה פיצ'ר.

---

## כלל הזהב ל-N6
ה-N6 מריץ **תוכנית אחת**:
- כלי mpremote (§3, §5, §6) דורשים ש-`main.py` **לא** יהיה מותקן (ראה §0).
- הגשר (§4) דורש ש-`main.py` = `n6_radar_bridge.py`.
לכן מעבר בין מצבים = reset + rm/cp של `main.py`.
