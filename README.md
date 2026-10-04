# ValiSight

ValiSight is a team-developed multi-sensor perception and navigation system that combines data from radar, IMU, thermal cameras, and visible cameras.

The project focuses on sensor integration, calibration, synchronization, AI-based perception, sensor fusion, tracking, pose initialization, and navigation-related processing.

Different parts of the system were developed by several team members and across multiple development branches.

## Project Overview

ValiSight combines information from several sensors and processing components in order to build a richer representation of the environment.

The system includes multiple processing paths rather than routing every sensor through a single component.

Main development areas include:

- Radar data acquisition and processing
- Radar calibration
- IMU integration
- Sensor synchronization and timestamp handling
- Thermal and visible camera processing
- AI-based person detection
- Radar-based classification
- Sensor fusion
- Multi-sensor tracking
- Pose initialization
- Navigation-related processing
- Integration between sensor data and higher-level algorithmic components

## AI and Perception

The project includes AI-based perception for person detection and tracking.

The perception layer includes:

- Vision-based person detection using YOLO models
- Thermal-based person detection
- Radar-based classification
- Trained models for thermal and radar channels
- Fusion between detections from different sensors
- Multi-sensor tracking
- Dataset generation and autolabeling tools
- Model training and evaluation workflows

The system combines information from multiple sensors while keeping track of the source of each detection.

## Sensor Integration

A major part of the project is the integration between different sensing components.

The system works with:

- Radar (TI IWR1843 mmWave)
- IMU
- Thermal camera
- Visible camera
- OpenMV N6
- NVIDIA Jetson

Sensor data is synchronized, calibrated, and passed between different processing components depending on the role of each sensor.

Not all sensor data is routed directly through the Jetson. Different components communicate and process data through separate paths before selected information is used for higher-level processing.

## My Contribution

ValiSight was developed collaboratively by several developers. My work focused on the sensor layer: bringing up the radar and IMU, moving sensor data reliably between boards on a shared clock, calibrating the radar, and connecting this data to the team's fusion and navigation components.

- **mmWave radar bring-up and parsing:** a TLV frame parser for the TI IWR1843 radar that runs under MicroPython on the OpenMV N6 and on the host, a raw UART check tool in C, and radar configuration profiles.
- **Radar tests and validation:** pytest suites for frame synchronization, TLV headers, edge cases and physical invariants, plus a recorded radar session used as a fixture for deterministic replay tests.
- **N6 to Jetson bridge:** a binary protocol with sequence numbers, timestamps and CRC-32 that streams thermal, RGB and IMU data over USB, with a MicroPython sender on the N6 and a C receiver on the Jetson. I traced a recurring stall to garbage collection on the N6 and fixed it with a zero-copy sender, reaching 200 Hz IMU sampling and a 190 s live run with no lost frames.
- **IMU and thermal integration:** IMU sampling into a ring buffer, recorded IMU sessions, and thermal camera status flags (FFC) passed through to consumers.
- **Sensor synchronization:** unwrapped timestamps and a shared host clock, exposed through a sensor API for Yael's components, together with recording tools that capture radar, IMU and thermal data on one clock.
- **Radar calibration:** re-measured radar phase calibration and a Radar/RGB extrinsic calibration from 22 fit and 6 holdout stations, validated on a walking person.
- **Radar AI experiments:** track-level and cluster-level radar classifiers evaluated with leave-one-session-out testing, including a documented negative result.
- **Map and navigation integration:** connected Yael's map layer to the live viewer, with API endpoints and tests.
- **Pose initialization (v0):** estimates position and heading by matching static radar wall returns against building footprints, with unit tests.

Other parts of the system were developed by other team members. The fusion and perception pipeline on the `Open-mv-n6---fusion` branches was developed mainly by Moshe Ashush, and the map initialization and navigation layer mainly by Yael.

## My Work in the Repository

Development is organized in branches by component and integration stage. These branches contain my work:

- [`noa`](https://github.com/NoaAizen/ValiSight/tree/noa): mmWave radar bring-up, the TLV parser, radar configuration and the radar test suite.
- [`IMU`](https://github.com/NoaAizen/ValiSight/tree/IMU): the N6 to Jetson bridge (C receiver, MicroPython sender, protocol and tests), IMU and thermal integration, and the shared-clock sensor API.
- [`noa-radar-calib-ai-2026-08-18`](https://github.com/NoaAizen/ValiSight/tree/noa-radar-calib-ai-2026-08-18): Radar/RGB calibration data and results, and the radar AI experiments.
- [`noa-yael-mapinit-2026-08-25`](https://github.com/NoaAizen/ValiSight/tree/noa-yael-mapinit-2026-08-25): integration of Yael's map layer into the live viewer, and the radar, IMU and thermal recording tools.
- [`noa-pose-init-v0`](https://github.com/NoaAizen/ValiSight/tree/noa-pose-init-v0): pose initialization from radar wall returns, built on Yael's map initialization code.

## Repository Structure

The `main` branch is the project overview. Each development branch holds a specific component, experiment, or integration stage:

- `Open-mv-n6---fusion` and `Open-mv-n6---fusion-work`: the fusion and perception pipeline
- `noa`, `IMU`, `noa-radar-calib-ai-2026-08-18`, `noa-yael-mapinit-2026-08-25`, `noa-pose-init-v0`: my work, described above
- `yael`: map initialization and navigation
- `hagai`: earlier Jetson-side fusion, thermal and radar development

## Technologies and Concepts

The project includes work with:

- Python, MicroPython and C
- Embedded systems
- Radar
- IMU
- Thermal imaging
- Computer vision
- YOLO
- OpenMV N6
- NVIDIA Jetson
- Sensor synchronization
- Timestamp handling
- Calibration
- Sensor fusion
- Multi-sensor tracking
- Pose initialization
- Navigation
- AI model training
- Dataset generation
- Real-time system concepts
