# core/timesync

Pure clock discipline for multi-sensor fusion. Standard library only, clock
injected, fully testable off-hardware. This is the module that makes fusion
timestamps honest (CLAUDE.md / realtime-soc-auditor: deterministic replay is the
top priority).

## The two failures it prevents

1. **Arrival time used as sample time.** Radar frames cross a UART with ms-scale,
   load-dependent latency; the moment the N6 finishes reading a frame is not when
   the scene was measured. `sensor_time_from_frame(frame_no, period, anchor)`
   reconstructs the sample time from the **frame index** (the radar's own regular
   cadence), anchored to the first frame — so it does not inherit UART jitter,
   and it stays correct across dropped frames because a dropped frame makes
   `frame_no` jump and the timestamp self-corrects. (A local counter would
   silently compress time across drops — never use one.)

2. **Mixing time sources.** `MonoClock` wraps one injected monotonic tick source;
   wall-clock is never used for a delta.

## Pairing two async streams

`SyncBuffer(slop_s, depth)` matches stream A (radar) and B (thermal) by
reconstructed sample time within `slop_s`, off a bounded deque with an explicit
drop-oldest policy, and **counts every drop**.

**Honest numbers (from `configs/iwr1843_live.cfg`, not the corpus):** the radar
frameCfg period is 100 ms → ~10 Hz vs the Lepton's ~8.7 Hz → ratio ~1.15:1, so
~13% of radar frames go unpaired in steady state (not an error — the steady
state). The honest slop floor at 10 Hz is **~50–60 ms**; tighter drops most
pairs and can't be fixed without interpolation.

## Known limits (documented, not hidden)

- **Clock drift:** the radar crystal vs the N6 tick differ by tens of ppm → tens
  of ms/hour. For long captures, estimate the true period by a robust linear fit
  of arrival-time vs `frame_no` over a rolling window, and re-anchor slowly
  (never on a single jittery arrival).
- **`frame_no` reset** on `sensorStart`/reconfigure → detect and re-anchor.
- **The real ceiling is upstream:** `fusion/radar.py` aggregates points over a
  **0.5 s** window, which is 10× the sync slop — so 0.5 s, not 50 ms, is the
  actual fusion time resolution today. Refining timestamps below the window is
  moot until the window is shrunk or sub-stamped. Don't claim tighter than 0.5 s.
