# Data collection → Google Colab training

The full loop, from a live rig to a trained student and back:

```
 rig (live.py --record)          Jetson                      Google
┌────────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│ session dir:       │   │ D1 teacher chain:   │   │ Drive: gexport/v2/   │
│  session.mp4       │──►│  run_teacher.py     │──►│  manifest.json       │
│  thermal.bin       │   │  build_dataset.py   │   │  <sess>-NNN.npz      │
│  frames.jsonl      │   │ then:               │   │        │             │
│  radar.jsonl       │   │  export_shards.py   │   │  train_students_     │
└────────────────────┘   └─────────────────────┘   │  colab.ipynb → *.pt  │
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
       --sessions <name> [...] --val <heldout-session> --out-name v2
   ```
   Output lands in `perception/out/gexport/v2/`. The val split is whole
   sessions on purpose — consecutive frames are near-duplicates, and a
   frame-level split inflates every number it touches.
   Re-export old v1 shards before training: v2 carries tri-state label masks
   and an explicit thermal dtype, and the updated notebook requires them.

   A session may be used as a negative only after a person checked that it is
   empty. Mark such a session explicitly:
   ```
   python3 perception/export/export_shards.py \
       --sessions empty-room-01 people-01 \
       --verified-negative empty-room-01 \
       --val people-01 --out-name v2
   ```
   Do not use `--verified-negative` merely because the RGB teacher produced
   no boxes; fog, darkness and occlusion are exactly where that assumption
   creates poisoned labels.
4. **Upload** `gexport/v2/` to Google Drive at
   `MyDrive/thermal-fusion/gexport/v2` (drive.google.com drag-and-drop, or
   rclone). GitHub is not the transport — shards are hundreds of MB.
5. **Train**: open the exported `train_students_colab.ipynb` from the same
   Drive directory, set the runtime to GPU, and run all. Every export includes
   a SHA256-stamped snapshot under `code/perception/`, so the checkpoint can
   always be tied to the loader/model code that produced it. Checkpoints are
   written to `gexport/v2/models/`.

   The same entrypoint works outside Colab when compatible PyTorch is present:

   ```sh
   PYTHONPATH=perception/out/gexport/v2/code \
   python3 -m perception.train_students \
       --data perception/out/gexport/v2 --student both --epochs 50
   ```

   Ablations do not require editing Python:

   ```sh
   # Thermal without the temporal family
   ... --disable-thermal temporal_diff,temporal_rate,previous_frame,previous_frame_2

   # Radar without RA, and point cloud without SNR/noise
   ... --disable-radar-family ra \
       --disable-radar-channel points:snr,noise
   ```

## What is (and is not) in a shard

Per frame: raw thermal 160×120 (`uint8` or little-endian `uint16`, declared
per session), RGB 640×400 u8 (drop with `--no-rgb`), radar points (64×6,
SNR-ranked, `n_radar` real), `dt_ms` to the previous stored frame, teacher
boxes (RGB px, with conf) and grade-A boxes (thermal px, with delta). Each
student also gets a label state: `1` positive, `0` independently verified
negative, and `-1` unknown. Unknown frames are excluded from supervised losses
and metrics. Optional UART TLVs add Range–Azimuth, Range–Doppler and Range
Profile arrays plus explicit validity masks. Derived channels — gradients,
second derivatives, local variance, persistence and temporal rate — are built
on the GPU from the raw plane, never stored. Keeping one derivation point is
what makes the ablation honest.

## Honest limits of this rig (do not design around more than it gives)

- **Current live capture is 8-bit over [TMIN, TMAX], not RAW14.** The board runs the
  Lepton in measurement mode; `c_per_lsb = (tmax−tmin)/255`. At the default
  −10..140 °C window that is 0.59 °C/LSB — ~12× coarser than the sensor's
  50 mK NEDT. The cheap win when a session is FOR TRAINING: narrow
  `SET_RANGE` (e.g. 0..60 °C → 0.24 °C/LSB) and record tmin/tmax into
  meta.json so the manifest carries `c_per_lsb`. True RAW14 off the board is
  firmware work (OpenMV Lepton driver), a separate project. The reader and
  recorder now preserve `uint16_le` frames without offset corruption, but a
  16-bit session still needs a declared encoding/scale before it can be mixed
  with calibrated temperature data. Auto-label deltas are normalized back to
  8-bit-equivalent counts so the existing grade thresholds remain meaningful.
- **Radar training uses only what the UART actually emitted.** The base stream
  is (x, y, z, v, snr, noise). When enabled in the chirp configuration, UART
  TLV 2/4/5 additionally provide Range Profile, Range–Azimuth and
  Range–Doppler. Raw ADC, complex IQ and the full radar cube are still absent;
  those require a DCA1000-class capture path. Velocity is folded at 1.298 m/s,
  so its ablation must justify whether it helps.
- **walk1–4 cannot feed training** — they recorded the fused view, the
  thermal overlay is baked into the pixels. The exporter skips them by
  construction (no `thermal_off` ⇒ no export).

## What to record next (the dataset gaps that matter for 2–15 m)

Sessions that make the radar and temporal channels *earn* their ablation
score: dark rooms (RGB teacher blind — label via the thermal COCO only),
7–15 m walks where a person is 10–20 thermal px, two people crossing,
approach/retreat along the boresight (radial v), and empty-scene negatives
with warm clutter (monitors, radiators).
