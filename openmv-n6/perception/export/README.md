# Data collection → Google Colab training

The full loop, from a live rig to a trained student and back:

```
 rig (live.py --record)          Jetson                      Google
┌────────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│ session dir:       │   │ D1 teacher chain:   │   │ Drive: gexport/v1/   │
│  session.mp4       │──►│  run_teacher.py     │──►│  manifest.json       │
│  thermal.bin       │   │  build_dataset.py   │   │  <sess>-NNN.npz      │
│  frames.jsonl      │   │ then:               │   │        │             │
│  radar.jsonl       │   │  export_shards.py   │   │  train_colab.ipynb   │
└────────────────────┘   └─────────────────────┘   │  → student.pt        │
                                                   └──────────────────────┘
```

## Step by step

1. **Record** a session with live.py as usual (`--record <name>`). Nothing about
   recording changes for training; the exporter refuses what it cannot use
   (fused-view sessions, unclean frames) rather than asking the recorder to care.
2. **Label** it with the D1 chain:
   ```
   python3 perception/autolabel/run_teacher.py --session <name>
   python3 perception/autolabel/build_dataset.py
   ```
3. **Export**:
   ```
   python3 perception/export/export_shards.py \
       --sessions <name> [...] --val <heldout-session> --out-name v1
   ```
   Output lands in `perception/out/gexport/v1/`. The val split is whole
   sessions on purpose — consecutive frames are near-duplicates, and a
   frame-level split inflates every number it touches.
4. **Upload** `gexport/v1/` to Google Drive at
   `MyDrive/thermal-fusion/gexport/v1` (drive.google.com drag-and-drop, or
   rclone). GitHub is not the transport — shards are hundreds of MB.
5. **Train**: open `train_colab.ipynb` in Colab (upload it once, or open from
   Drive), set the runtime to GPU, run all. The last cell is the ablation
   sweep — the measurement this whole design exists for.

## What is (and is not) in a shard

Per frame: raw thermal 160×120 u8, RGB 640×400 u8 (drop with `--no-rgb`),
radar points (64×6, SNR-ranked, `n_radar` real), `dt_ms` to the previous
stored frame, teacher boxes (RGB px, with conf) and grade-A boxes (thermal
px, with delta). Derived channels — gradients, |∇T|, temporal diff,
Δ-from-background — are built in the notebook from the raw plane, never
stored: they are recomputable, and keeping one derivation point is what makes
the ablation honest.

## Honest limits of this rig (do not design around more than it gives)

- **Thermal is 8-bit over [TMIN, TMAX], not 14-bit.** The board runs the
  Lepton in measurement mode; `c_per_lsb = (tmax−tmin)/255`. At the default
  −10..140 °C window that is 0.59 °C/LSB — ~12× coarser than the sensor's
  50 mK NEDT. The cheap win when a session is FOR TRAINING: narrow
  `SET_RANGE` (e.g. 0..60 °C → 0.24 °C/LSB) and record tmin/tmax into
  meta.json so the manifest carries `c_per_lsb`. True RAW14 off the board is
  firmware work (OpenMV Lepton driver), a separate project.
- **Radar is the point cloud, nothing rawer.** UART TLVs give
  (x, y, z, v, snr, noise); range-Doppler maps and raw ADC need a DCA1000
  capture path this rig does not have. v is folded at 1.298 m/s — the
  notebook uses |v| only.
- **walk1–4 cannot feed training** — they recorded the fused view, the
  thermal overlay is baked into the pixels. The exporter skips them by
  construction (no `thermal_off` ⇒ no export).

## What to record next (the dataset gaps that matter for 2–15 m)

Sessions that make the radar and temporal channels *earn* their ablation
score: dark rooms (RGB teacher blind — label via the thermal COCO only),
7–15 m walks where a person is 10–20 thermal px, two people crossing,
approach/retreat along the boresight (radial v), and empty-scene negatives
with warm clutter (monitors, radiators).
