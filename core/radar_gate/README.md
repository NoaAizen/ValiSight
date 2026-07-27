# core/radar_gate

Pure, cheap radar point relevance gating — drop the manufactured returns, keep
the relevant ones, and report the reduction. Runs on the N6 (stdlib `math` only).

## Why

The IWR1843 point cloud is ~half fictitious (corpus: fict% ~59% for the point
cloud — quote it with its missed% 93.6, it's directional not absolute). Ghosts
cost the N6 compute + USB bandwidth and pollute clustering. Cutting them
on-device before clustering/streaming is the efficiency win; the
`ReductionReport` measures it.

## The gates (cheapest first)

1. **Absolute reflectivity** — drop points whose `snr + noise` (the ABSOLUTE
   level that follows the range equation, per `iwr1843_uart`; NOT `snr` alone,
   which hugs the floor and is ~constant) is below `MIN_ABS_DB`. **Applied only
   within `REFL_MAX_RANGE_M` (~4 m)** — beyond that the censored return keeps
   1–2 dB of dynamic range, so gating on level would delete real distant weak
   targets (survivor bias).
2. **FOV** — drop points outside the relevant range/azimuth window
   (mirrors `cfarFovCfg`/`aoaFovCfg`).
3. **Isolated-point rejection** — drop points with no neighbour within the
   cluster epsilon (1.2 m); a lone return is almost always a sidelobe/multipath
   ghost.

## Honesty — what this does NOT do (validated against the physics corpus)

- It does **not** raise the CFAR threshold or use `clutterRemoval` as a ghost
  filter — those are snake oil here: multipath/sidelobe ghosts carry real SNR,
  and CFAR already keeps <1% of the cube, so you hit missed% before you cut
  them. `clutterRemoval` removes static returns (which fusion needs), not ghosts.
- It does **not** reject on Doppler — with ±0.65 m/s unambiguous velocity and
  aliasing, Doppler is not trustworthy enough to drop points on.
- Every gate trades against missed% (a distant pedestrian is 1–2 points). Gates
  are deliberately conservative and the report makes the cut visible; **quote
  the reduction with the fact that aggressive gating raises missed%.**

## Recommended tuning order (from the physics review)

Geometric gates first (range/angle FOV — nearly free), then the near-range
absolute-level gate, then min-points/isolation (modest). Statistical gates must
be reported with their missed% cost; geometric gates are close to free.

## Adding a gate

Keep it pure (stdlib math), name thresholds as UPPER_CASE constants (the
physics-lint blocks inline magic numbers), and add a positive + a
real-failure-mode negative test to `tests/test_radar_gate.py`.
