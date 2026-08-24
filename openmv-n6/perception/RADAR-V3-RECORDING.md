# Radar v3 recording plan — 2026-08-24

Goal: training data that makes the RADAR student stand on its own
(darkness-independent by physics; what it lacks is range coverage, static
people, and real negatives). Light does NOT matter for any session except the
dark demo — record wherever there is 2–15 m of space, in daylight.

Labels come from the thermal student
(`perception/autolabel/thermal_student_teacher.py`), so the person must be
inside the thermal↔RGB overlap to get labeled — walk the center of the frame,
not the extreme edges.

## The one command

```bash
RADAR_CFG=radar_people.cfg ./run_live.sh \
    --record ../captures/radar3-<name> --range 0:60 --students
```

Do NOT add `--radar` or `--warp` - run_live.sh passes both itself with the
correct USB-resolved port and LUT; a bare `--radar` after it OVERRIDES the
port with a default that may be the board (this exact collision killed the
first walkdepth1 attempt).

Non-negotiable: `radar_people.cfg` and `--range 0:60` — the exact config the
students were trained on. A different chirp config shifts the point statistics
and silently poisons v3. `--students` is optional but recommended: the orange
(thermal) boxes on screen are the labels-to-be — if the orange box is on the
person, the frame will label correctly.

## Sessions — big space, any light

| # | name | what happens | duration |
|---|------|--------------|----------|
| 1 | radar3-walkdepth1 | one person walks away/toward: 2 m → 15 m → 2 m, slow, twice | ~3 min |
| 2 | radar3-walkcross1 | one person crosses left↔right at ~3 m, ~6 m, ~10 m (2 passes each) | ~3 min |
| 3 | radar3-stand1 | person stands STILL 45–60 s at each of ~3 / 6 / 9 / 12 m (the no-doppler hard case — breathe, don't move) | ~4 min |
| 4 | radar3-sit1 | person sits / crouches ~45 s at ~3 / 6 / 9 m (different radar cross-section) | ~3 min |
| 5 | radar3-multi1 | two people: one static, one walking, at different ranges; swap roles halfway | ~3 min |
| 6 | radar3-negative1 | space completely EMPTY of people — operator out of the field of view the whole time | ~3 min |

## Sessions — dark demo (regular room, night, lights off)

| # | name | what happens | duration |
|---|------|--------------|----------|
| 7 | radar3-dark1 | person walks + stands still; RGB should be near-black | ~2 min |
| 8 | radar3-dark-negative1 | empty dark room | ~2 min |

## Rules

- One session = one scenario. Do not mix "empty" and "person" in one
  recording — verified-negative is declared per WHOLE session at export.
- Do not power-cycle the radar between sessions (voids the cfg stamp).
- Close each session properly (Ctrl+C once, wait for the recorder to close).
- After the last session: `tools/sync_session.sh radar3-*` to the NAS.
- Person 2 for multi1: Noa (on-site).

## After recording (Claude runs these)

1. `thermal_student_teacher.py` over every radar3-* session
2. spot-check the label overlay on a few frames
3. `build_dataset.py` → gate
4. `export_shards.py --out-name v3` with `--verified-negative
   radar3-negative1 radar3-dark-negative1`, val TBD (hold out one motion
   session, e.g. radar3-walkcross1)
5. Colab: retrain radar student (`STUDENT='radar'`) on v3
6. validate_students_trt.py + live `--students` demo in the dark room
