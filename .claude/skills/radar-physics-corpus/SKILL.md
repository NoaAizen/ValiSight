---
name: radar-physics-corpus
description: Distilled findings from the mmWave radar and thermal literature relevant to this project — HawkEye, Radatron, EchoFusion, RadarHD, ColoRadar, RaDICaL, K-Radar, milliEgo — plus the measured results of our own point-cloud-vs-heatmap experiment. Load when reasoning about radar signal processing, sensor selection, angular resolution, specularity, multipath, CFAR, fusion architecture, or what any modality can honestly claim.
---

# Radar / thermal research corpus

Facts with sources. If a claim here has no source, it is our own measurement
and says so. **Do not invent numbers. If it is not here and not measured, say
you don't know.**

---

## 1. The measured result that drives this project

Our experiment (`run_comparison.py`, averaged over 12 vehicle orientations,
identical physical scene through every path):

| representation | hyp/ray | missed% | fict% |
|---|---|---|---|
| HawkEye rig, heatmap + top-8 | 1.28 | 42.2 | 45.6 |
| HawkEye rig, point cloud + top-8 | 0.06 | 84.8 | 19.0 |
| IWR1843, heatmap + top-8 | 4.95 | 50.7 | 73.7 |
| IWR1843, point cloud + top-8 | 0.05 | 93.6 | 59.2 |
| Altos V1, heatmap + top-8 | 1.31 | 35.9 | 49.5 |
| Altos V1, point cloud + top-8 | 0.16 | 74.9 | 38.5 |

**Conclusions, in order of importance:**

1. HawkEye's top-m skip cannot be fed from a TLV point cloud. ~95% of rays
   arrive with zero hypotheses.
2. **The bottleneck is the front-end, not the array.** Altos V1 with a point
   cloud still collapses (0.16 hyp/ray) despite 1.4° azimuth resolution. CFAR
   + peak-grouping discards the information before it reaches the UART.
   Upgrading the antenna does not fix this. **Raw ADC access does.**
3. `med err` looks *better* for point clouds. This is survivor bias: the cloud
   only survives on strong retro-reflective corners, which are accurate by
   nature. Never quote `med err` without `missed%`.
4. IWR1843's `fict%` of 73.7 with a heatmap is not a bug — it is the absence
   of elevation. 2 elevation channels ⇒ 57.3° PSF ⇒ energy smears over the
   whole elevation axis.

**Caveat that must accompany any citation of these numbers:** the synthesiser
is a model, not a measurement. The specular lobe (12°), PSF, and ground-bounce
coefficient (0.3) are plausible but uncalibrated. Absolute values will move
against real hardware. The *ratios between rows* are far more robust — every
row runs on the identical scene. Quote the ~20-30× gap, not `0.05`.

---

## 2. HawkEye — Guan, Madani, Jog, Gupta, Hassanieh, CVPR 2020

Custom 60 GHz synthetic 2D array. cGAN recovers a 2D depth map from a 3D
mmWave heatmap.

- Grid 64×32×96 (φ, θ, ρ) → output 256×128 depth map.
- Range resolution 10 cm (3.3× worse than LiDAR); **angular 5° (50× worse)**.
  Sub-degree angular would need a **9 m antenna array**.
- The wide 2D sinc PSF eliminates almost all high-frequency perceptual content
  such as object boundaries. That is why heatmaps look like blobs.
- **Eq. 1 skip:** `x_2D(φ,θ) = argmax_r x_3D(φ,θ,r)`; bare argmax is unstable,
  so they take m=8 largest, power-ordered, concatenated at decoder layer 6.
  Non-differentiable, input only.
- Loss: `L_GAN + λ1·L1 + λp·L_perceptual(VGG)`.
- Trained 170 epochs on 3000 synthetic, fine-tuned 60 epochs on **100 real
  clear-weather** images. **Never trained on fog.**
- Results in fog: ranging 50 cm, orientation error **29°**, fictitious 2.5%,
  surface missed 15.4%.
- Failure cases: (i) correct bbox, front/back confused; (ii) orientation wrong
  under specularity; (iii) **degrades with multiple cars**.
- 60 GHz was forced by FCC unlicensed spectrum; they expect 77 GHz to do better
  because water attenuates 60 GHz more.
- **Footnote 1: "Other modalities such as thermal imaging also fail in dense
  fog"** (cites Beier & Gemperlein 2004). This constrains what we may claim.

**Project stance:** the GAN is a hallucination engine and must never enter a
perception pipeline — 29° median orientation error is unusable for path
planning, and failure (i) is the network filling specular holes from a learned
prior. **The skip is the part worth stealing.** It transfers *measured* depth
instead of inventing it, and is architecture-agnostic.

---

## 3. Radatron — Madani, Guan, Ahmed, Gupta, Hassanieh, ECCV 2022

Direct successor. Cascaded MIMO radar as a **stand-alone** sensor.

- **5 cm range, 1.2° angular** — 10× finer than other public datasets.
- 92.6% AP50, **56.3% AP75** in 2D bbox detection (+8% / +15.9% over prior art).
- Motivation: full arrays are prohibitively expensive in cost, power and form
  factor; cascaded MIMO is the cheap alternative.

**Why this matters here:** 1.2° ≈ our Altos V1 model (1.4°). **AP75 = 56.3% is
the realistic ceiling for radar-alone** — best resolution available, deep
learning, clear weather, and precise localisation still halves. Use AP75, never
AP50, as the reference for any "radar-alone sufficiency" claim.

---

## 4. EchoFusion — Liu, Wang, Wang, Zhang, NeurIPS 2023

**Independent confirmation of our finding #2.** They state that point cloud
generation algorithms already drop weak signals to reduce false targets, which
may be suboptimal for deep fusion. They skip the radar signal processing
pipeline entirely: BEV queries sample raw spectrum features. Result: surpasses
all existing methods on RADIal, approaching LiDAR.

Corroborating: CFAR-class detectors typically retain **<1% of the data cube**.

Code: `tusen-ai/EchoFusion`.

---

## 5. Datasets

| dataset | radar | gives you | why we care |
|---|---|---|---|
| **ColoRadar** (Kramer, IJRR 2022) | **AWR1843BOOST** | **raw ADC + dense 3D heatmap + sparse point cloud**, 3D LiDAR, IMU, GT pose. 52 seqs, ~2 h, indoor/outdoor/mine | **Same silicon as our IWR1843, and all three representations from one sensor.** The only way to validate our simulation on real data for free. API: `azinke/coloradar`, renders heatmap from ADC with/without sidelobes |
| **RadarHD** (Prabhakara, ICRA 2023) | single-chip | ~40k raw I/Q + LiDAR pairs, 67 trajectories, Docker + pretrained | Closest published work to our projection module |
| **RaDICaL** (Illinois) | 3TX/4RX 77 GHz | raw ADC + RGB + depth | **Elevation TX disabled** — no height. Raw ADC costs **up to 325 Mbps** |
| **Radatron** | cascaded MIMO | 5 cm / 1.2° | Altos-class reference |
| **K-Radar** | 4D | RAD tensor + 3D annotations, various weather | Weather robustness |
| **View-of-Delft** | 3+1D | VRU classification | Pedestrian focus |
| **TIDEP-01012** (Xiangyu Gao) | **12TX/16RX, 192 virtual** | raw ADC, **1.35° az / 19° el** | Independently validates our `ALTOS_V1` config (1.4°/18°) |

---

## 6. Thermal / LWIR — what may honestly be claimed

Our sensor is a **Lepton 3.5, LWIR 8–14 µm, NETD ~50 mK**.

- **Strong:** dry air, dust, sand, smoke, long-range outdoor. LWIR is more
  stable in dry, dusty and sandy conditions because long-wave radiation is less
  affected by dust scattering.
- **Weak:** dense fog, high humidity. Cooled **MWIR** achieves longer detection
  range than LWIR in dense fog and humid environments.
- **Worst:** **maritime aerosols always give the lowest detection range,
  independent of climate model**, because their mean particle radii are largest.
  The Israeli coastal plain is exactly this.
- **Thermal crossover** is not an obscurant problem: transmission is fine,
  contrast is simply zero. No lens or NETD fixes it.

**Correct claim:** "radar for fog; thermal for smoke, dust, sand and darkness;
fusion for robustness." **False claim:** "thermal + radar covers all degraded
visibility."

---

## 7. Prior art and IP

Patents exist in the map-prior/static-subtraction space. "No commercial product
implements this" and "nobody has claimed it" are different statements.

- US 10845806 — Autonomous vehicle control using **prior radar space map**
- US 12174641 — Vehicle localization based on radar detections (occupancy grid
  reference map built over repeated runs)
- US 12429574 — Static scene mapping using radar
- US 10338208 — Object detection in multiple radars (static clutter mitigation)

These appear to build the map **from prior runs**, whereas our layer registers
against an **external offline 3D terrain map + satellite imagery**. That may be
a real distinction. **We are not lawyers and must not assess validity or
infringement.** Flag for FTO review; do not reason about it further.

Open alternative: Gao, Roy & Zhang, *Static Background Removal in Vehicular
Radar* — 4D imaging then filtering in the azimuth–elevation–Doppler domain.

Also relevant: milliEgo (point-cloud, AWR1843, needs retraining outdoors),
CFEAR, 4DRadarSLAM, ERASOR, OdomBeyondVision (mmWave + LWIR + IMU + GT).

---

## 8. Standing engineering rules

- **DLSS-style frame generation must never enter perception.** Hallucinated
  pixels plus latency. Neural super-resolution and optical-flow temporal
  upsampling are legitimate; frame *generation* is not.
- Late fusion ≫ deep feature fusion on SWaP-C. `ApproximateTimeSynchronizer`
  is a sound choice.
- Adding thermal to radar costs negligible SWaP-C for a large robustness gain.
- Clustering epsilon: 0.8 m fragmented wide vehicles; **1.2 m** resolved it.
- The OpenMV Multispectral module is **platform-locked** to the OpenMV N6.
