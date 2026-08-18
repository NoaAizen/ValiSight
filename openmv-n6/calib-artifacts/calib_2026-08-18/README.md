# Radar <-> RGB calibration day, 2026-08-18 (Noa)

Session: `session_fit/` (live.py, --view visible, radar_people.cfg NEW phase table sha b3e4a561...)
- session.mp4 was NOT finalized (writer killed before release); `session_broken_nomoov.mp4` = original,
  `session_raw.m4v` = mdat, `session_fixed.m4v` = with VOL header (decodes, 78129 frames vs 78144 in frames.jsonl),
  `session.mp4` = re-encoded indexed copy (remux.py) if present.
- `stations_windows.json`  pose_id -> RGB frame window (last run per id)
- `stations_radar.json`    per station: radar median (x,y,z project frame), range/az/el/SNR, gate, frames
- `station_frames/`        3 PNGs per station (start/mid/end of hold), from session_fixed.m4v
- `picks.json`             manual pixel picks (Noa, picker/serve.py on :8090)
- `corr_fit.json` (13: F01-F08,F10-F14) / `corr_holdout.json` (6: V01-V06)   excluded: F09 (no radar), F15 (static lamp)
- `radar_rgb_2026-08-18_uonly.json`  reduced solution (yaw+tx from u only), per-station residuals
- `phase_table_run*.json`  phase-table measurements (run4 adopted), `radar_people.cfg.before-2026-08-18`, `.sha256`

Result: fit u-median 5.9 px (0.64 deg), holdout u-median 13.4 px (1.47 deg), max 27 px.  V3 gate (<=0.72 deg) NOT met.
Formulation: reduced (radar z untrusted: el reads ~3x true; low targets filtered by aoaFov +-30 el).
Ambiguity: yaw +2.6 deg & tx +127 mm  vs  yaw 0.1 deg & tx 0 (mechanical) - equivalent over 2-4 m. Needs 6-8 m stations
(bigger reflector) or a caliper measurement of the lateral camera-radar offset to break.
Radar az is quantised (~1.8 deg FFT bins) -> +-8 px floor per station.
Room: office lobby with glass doors (5.6 m) and mirror wall (left); static clutter at 4.1 m (couch leg / lamp) - avoid 4.0-4.2 m side stations.

## Round 2 (big 50 cm trihedral, session_fit2) and FINAL
- G01-G09 fit stations 5.3-8.6 m (G06 flagged near glass door). W01/W02 validation attempts failed (no return) - skipped.
- session_fit2/session.mp4 unfinalized too -> session_fixed.m4v (40681/40694 frames). station_frames/G*.png, picks in picks.json.
- corr_fit_all.json (22) + corr_holdout.json (6) -> radar_rgb_2026-08-18.json (also copied to calib-artifacts/).
- With 2-8.6 m the yaw/tx degeneracy is broken: tx = +24 mm (~0, matches "camera directly above radar"), yaw = -0.06 +- 0.24 deg.
- FINAL (yaw-only, mechanical t = [0, 74.9, 8.8] mm): fit median 7.1 px (0.77 deg), holdout median 15.4 px (1.68 deg), max 25.6 px.
  V3 gate (0.72 deg) NOT met; the floor is the radar's ~1.8 deg azimuth quantisation + reflector on CFAR threshold.

## Stage B (pitch from known-height stations)
- pitch = -4.31 deg solved from v of F10-F14 + G01-G09 (vertex at antenna height, radar z := 0). Without it v was off by ~39 px.
- fit v-median 2.6 px (max 8); holdout V01-V05 v-median 10.6 px, +18 px at 2.5 m (residual grows with 1/range -> ~3 cm t_y/height mismatch at the near stations, or the small-TCR vertex height was not exactly 841 there).
- Final file updated: calib-artifacts/radar_rgb_2026-08-18.json (R includes yaw -0.06 & pitch -4.31; t = [0, 74.9, 8.8] mm).

## Dynamic validation (session_dyn, D1-D5, one walking person)
- yolov4-tiny person box (CPU) vs projected SNR-weighted radar centroid of moving points, nearest radar frame (<0.15 s).
- 67 matched frames: |du| median 9.5 px (1.03 deg), p90 25 px, signed median -5.8 px. Radial walks (D4/D5): 2-11 px; lateral walks (D1-D3): 16-21 px (walk speed x 0.1 s timing = 10-30 px, not calibration).
- Standing person (D5 first 6 s at 3 m): NOT detected by the radar at all (static = clutter with this cfg); appears immediately when moving. Config/algorithm decision, not calibration.
- dyn_validation.json holds per-frame rows.

## Standing-person test (session_repeat S1/S2)
- S1 radar_people.cfg (CFAR 15 dB range & doppler): person standing still at 3.1 m present in 12% of frames (only when swaying). 9 pts/frame.
- S2 radar_people_cfar10_TEST.cfg (CFAR 10 dB both): present in 84% of frames (static + breathing/sway components, SNR 9-28). 18 pts/frame (more static clutter).
- Decision for Hagai (perception): threshold trade-off; the calibration is unaffected (geometry only). Original cfg re-sent afterwards.

## PASS D - repeatability after radar USB power cycle (session_repeat2/3, small TCR, R1-R4)
- radar re-enumerated on the same ports (ACM1/ACM2); cfg re-sent (same sha). live.py's radar reader does NOT survive a replug -> restart live.py.
- projected radar (day's calibration) vs picked vertex: |du| median 6.8 px (0.75 deg), max 13.6 px -> same level as the fit itself. Phase table + extrinsics stable.
- dv -10..-15 px (morning V stations: +5..+18): reflector stand height differed (~10 cm) - not a calibration change. repeatability_passD.json.
- 50 cm reflector: at 5 m centre it returned NOTHING after the power cycle (morning: 15 dB) -> the big trihedral is unreliable (plates likely not 90 deg). Do not use it as-is.

## Two-people test (session_multi, E1-E4; both people swaying/walking, still = invisible)
- E1 1 m apart @4 m: 2 clusters in 44% of frames (14 deg apart ~ angular resolution). E2 one behind the other (2.5/6 m): rear seen weakly (SNR 11-12, shadowed) - range separates. E3 +-1.5 m @4 m: 43%. E4 crossing (3 & 5 m): 2 moving targets at distinct ranges in 39% of frames.
- Implication for perception: associate on TRACKS over time, not per frame. session.mp4 unfinalized (same recovery recipe) - not recovered, radar.jsonl + frames.jsonl are the data.

## Left for later (decided 2026-08-18 evening)
- 15 m walking-person validation: do it in a more open space (this lobby: glass at 5.6/7.5 m, far wall 15 m).
- PASS E (unbolt/rebolt) only if/when the rig is going to be disassembled.
- Live view recipe with today's calibration: live.py --radar-calib calib_2026-08-18/live_radar_K.json, then
  /set?yaw=0.063&pitch=4.314&roll=0&tx=0&ty=74.9&tz=8.8  (overlay convention; derived from radar_rgb_2026-08-18.json)
