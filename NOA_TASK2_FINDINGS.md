# משימה 2 — ממצאי מחקר-שולחן (SPRUIM4B) לפני המעבדה

מקור: TI SPRUIM4B — xWR1843BOOST EVM User's Guide, Rev. B. נבדק 09/08/2026.
**הערה:** הלוח שבריג (לפי צילומי ה-WhatsApp מ-05/08) מסומן **IWR1843BOOST**, לא AWR — pinout זהה.
אלה תשובות מהמסמכים; ה-Gate של משימה 2 עדיין דורש אימות בסקופ.

## 2א — SYNC_OUT: כן, יש פין פיזי

| אות | מיקום | מקור |
|---|---|---|
| **SYNC_OUT** | **J6 פין 18** (BoosterPack) | Table 2, עמ' 6 |
| SYNC_IN | J6 פין 9, וגם J1 פין 16 (60-pin HD) | Table 2–3 |

- אין במדריך שום מתג/ג'אמפר/נגד 0Ω שנדרש להפעלתו — מופיע כזמין ישירות.
- ⚠️ צד firmware: SYNC_OUT חייב להיות מאופשר בקונפיגורציית ה-chirp
  (ב-mmWave Studio: checkbox "SyncOut Master Dis"). לוודא שהקונפיג שלנו מפעיל אותו.
- אימות בסקופ: פולס נקי 3.3V, פעם בפריים, על J6-18 מול GND.

## 2ב — ניתוב UART: מקביל, בלי מתג בחירה

ה-UART מגיע גם ל-headers וגם ל-XDS110/USB **במקביל** (Figure 3, עמ' 5).
אין מתג שמנתב לאחד או לשני:
- S1 = SOP/boot בלבד (101=flash, 001=functional, 011=debug — Table 4, עמ' 13)
- S2 = בחירת SPI מול CAN בלבד (עמ' 8, 15)

| אות | מיקום | הערה |
|---|---|---|
| UART1 TX from radar (CFG @115200) | **J6 פין 5** | בטבלה כתוב "RS232RX" — שגיאת דפוס; הסוגריים "(Tx from AWR device)" קובעים |
| UART1 RX into radar | **J6 פין 7** | לא להזרים אליו כשה-XDS110 מחובר — שני drivers על קו אחד |
| **MSS LOGGER = DATA UART @921600** | **J5 פין 9** | זה הפין שמתחבר ל-N6 UART RX |
| BSS LOGGER | J5 פין 13 | |
| DSS LOGGER | J5 פין 4 | |
| UART1 גם על 60-pin | J1 פינים 55/57 | |

- **האזנה (tap) על TX לא מנתקת את ה-USB** — אפשר להשאיר את ה-USB לצריבה/קונפיגורציה
  ולחבר את ה-N6 במקביל. בדיוק מה שרצינו.
- לאימות סופי של נגדים טוריים צריך את חבילת ה-schematic של TI
  ("AWR1843BOOST Schematic, Assembly Files, and BOM" מדף המוצר; קיימת גם גרסת IWR).

## חשמל ולוגיקה (משימה 3)

- הזנה: **5V ≥2.5A**, barrel jack 2.1mm center-positive (P6). ספק נפרד — לא מה-N6/USB.
- אחרי חיבור מתח: ללחוץ NRST (SW2) פעם אחת.
- headers ב-**3.3V** (J6-1 = 3V3, J6-2 = 5V). בלי level shifter.
- ⚠️ **ה-I/O אינם failsafe**: אסור להזרים אותות אל הרדאר לפני ש-PGOOD גבוה
  (PGOOD = J6 פין 14 / J1 פין 13). בסדר ההדלקה: קודם הרדאר, אחר כך אותות.

## טבלת החיווט המעודכנת (משלים את משימה 3 ב-NOA.md)

| מה | מאיפה (רדאר) | לאן (N6) | הערות |
|---|---|---|---|
| DATA_TX (MSS logger) | **J5 פין 9** | UART RX | 921600 8N1 |
| SYNC_OUT | **J6 פין 18** | GPIO עם input capture | פולס לכל פריים; לוודא מאופשר ב-cfg |
| GND | כל פין GND | כל פין GND | חובה |
| VSYNC תרמי | מהמצלמה | GPIO שני עם input capture | |
