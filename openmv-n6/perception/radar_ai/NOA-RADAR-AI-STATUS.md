# Radar AI (Noa) — status 2026-08-18 evening

Goal: the person/clutter classifier for radar tracks (gate 5 GBM in Hagai's PERCEPTION plan) = the
"trained model upgrade path" of tools/radar_classify_n6.py.

## Pipeline as it stands (all runnable on the Jetson)
1. `perception/radar_ai/run_channel_b.py captures/<sess>`  -> static map -> clusters -> tracks -> per-track features
   (`out/radar_ai/<sess>_radar_tracks.json`).  session_dyn: 127 tracks >=2 s, D2 gate 75% (bar 80%).
2. `perception/autolabel/run_teacher.py captures/<sess>`   -> person boxes per video frame. NEW: CPU yolov4-tiny
   fallback when the TensorRT yolov10n engine is absent (it IS absent on this Jetson: no ~/archive/radar/models/yolov10n.onnx).
3. `perception/autolabel/build_dataset.py`                  -> graded person tracks (session_dyn: 29 tracks, 100% grade A).
4. `perception/run_associate.py captures/<sess>`            -> u-only Hungarian radar<->box; session_dyn du median 8.4 px,
   D3 gate 77.6% (bar 90%).  Uses calib-artifacts/T_camera_radar.json = TODAY's calibration (yaw -0.06, pitch -4.31,
   t=[0,74.9,8.8] mm; old file kept as T_camera_radar_2026-08-10.json), project.py VALID_RANGE_M = 2.0-8.6.
5. NEW `perception/radar_ai/label_tracks.py captures/<sess>` -> per-track label from association counts
   (`out/radar_ai/<sess>_labelled.json`). session_dyn: 96 tracks: clutter 59, ambiguous 19, unknown 17, person 1.

## What blocks a useful classifier right now (in order)
- Teacher recall: yolov4-tiny misses many frames -> person tracks get matched_frac 0.1-0.36 (labels noisy). Fix: get yolov10n.onnx
  onto the Jetson and build the engine (BUILD_CMD in tools/trt_detect.py), or lower thresholds + require matched>=20.
- Feature v_spread_med == 0 for every track: clusters are ~1 point (npts_mean ~1). radar_people.cfg peakGrouping/CFAR 15 dB
  starves the point cloud -> the micro-Doppler feature is dead. Same lever as the standing-person finding (CFAR 10 dB test cfg
  gave 2x points). Decide with Hagai (cfg change = new sha, re-validate calibration quickly (PASS D style)).
- Sessions: session_dyn/session_multi are lobby chaos (people everywhere). Clean sessions (one person, empty room) will make
  labels crisp. Reflector-only sessions (session_fit/fit2) are pure clutter negatives.
- Then: train GBM/RandomForest on features (v_abs_mean, v_spread, v_p90_10, rcs_mean/max, extent, npts, ground_speed,
  straightness, range) with leave-one-session-out; compare against tools/radar_classify_n6.classify (rule-based).

## Data available today
- captures/session_dyn (D1-D5 walking/standing person), captures/session_multi (E1-E4 two people),
  calib_2026-08-18/session_fit, session_fit2 (reflector holds = static), session_repeat* (standing person S1/S2, R1-R4).

## First training attempt (2026-08-18 evening) — negative result, documented
- `perception/radar_ai/train_gate5.py`: LOSO over session_dyn + session_multi (208 tracks: 59 person / 149 clutter by
  matched>=8 & frac>=0.2 vs matched<=1 & inband>=15). RF/GBM/LogReg: person precision 0.06-0.32, recall 0.03-0.44.
  Rule-based classify_n6 baseline: recall 0.85, precision 0.30.
- Root cause is the LABELS: 112/149 'clutter' tracks move at walking speed (undetected/out-of-box people in a crowded lobby);
  feature medians person vs clutter are indistinguishable (speed 0.84 vs 0.70, Doppler 0.00 vs 0.01, npts ~1 both).
  Also npts~1 & Doppler~0 per track = point cloud starved by radar_people.cfg (CFAR 15 dB / peak grouping).
- Do NOT use gate5_model.pkl. Next: yolov10n engine, denser-cloud cfg (with Hagai), clean single-person sessions, then retrain.

## Second attempt, same evening — cluster-level, physics labels (train_clusters.py) — FIRST POSITIVE NUMBERS
- Labels from an empty room, no camera: person = moving cluster (|v|>=0.25 or displacement>=0.15 m) inside an exercise
  window & not in a static cell; clutter = cell occupied >=60% of the session & |v|<0.15. Sessions: session_clean15
  (office, one person; C1-C3 CFAR15, C5-C6 CFAR10 test cfg) + session_dyn.
- 29,257 clusters (5,108 person / 24,149 clutter), LOSO: RandomForest precision 1.00 recall 0.73 vs rule-based
  classify_n6 precision 1.00 recall 0.45. Top features: displacement (0.37), displacement-minus-Doppler (0.33), |v| (0.11),
  range, rcs. v_spread ~0 importance (dead at CFAR 15).
- CFAR 15 vs 10 on the SAME walking person (C2 vs C5): frames with target 40% -> 60%, moving pts/frame 2.4 -> 6.1,
  micro-Doppler spread 0.16 -> 0.93 m/s. => CFAR 10 dB makes the literature's key feature usable. Hagai's call.
- Caveats: label balance is per-session degenerate (clean15 ~all person, dyn ~all clutter); need 2-3 clean CFAR-10 sessions.
- Artifacts: out/radar_ai/cluster_model_v0.pkl, cluster_dataset_v0.json. Original cfg restored on the radar.
