# Walk validation — radar→thermal via the 2026-08-18 extrinsic (2026-08-19)

~4 min walk, 2→8 m + diagonals. Tool: `tools/calib/walk_validate.py` (warm-blob
center in the LUT-registered thermal plane vs nearest moving radar cluster,
gate ±100 px, time-bracketed pairing). Per-frame records: `walk_validate.jsonl`.

## Headline numbers
- 567 frames measured (234 no warm blob, 1027 no moving radar return — person
  standing/turning, 279 radar motion elsewhere only).
- **|error| median 1.84° (target ≤2° — MET), signed median +0.03°** → no global
  yaw correction is warranted.
- **Inside body extent 87.3% (target ≥90% — formally missed).**

## Where the misses live, and why they are not the extrinsic
- Left half: |med| 1.36°, inside 93%. Range 1.5–3 m: |med| 1.59°, inside 96%.
- Signed median per body-az bin is within ±1° from −25° to +10°.
- Right flank (+10..+20°, range ~3.5 m) is **multimodal**: a lobe at ~0° (the
  person) plus lobes at −7..−9° and +5..+7° — two extra moving targets, the
  documented glass ghosts of this room (plus a second person near the desk at
  session start). Target confusion for the tracking layer (D2 ghost veto),
  not an angular error of the calibration.

## Verdict
The radar↔RGB↔thermal chain is validated to the ≤2° budget where the target is
unambiguous. Do NOT touch /set. The inside-% shortfall is scene/detection,
to be re-tested outdoors or after the D2 tracker's ghost handling.
