# ValiSight

ValiSight is a team-developed multi-sensor perception and navigation system that combines data from radar, IMU, thermal cameras, and visible cameras.

The project focuses on sensor integration, calibration, synchronization, AI-based perception, sensor fusion, tracking, pose initialization, and navigation-related processing.

Different parts of the system were developed by several team members and across multiple development branches.

## Project Overview

ValiSight combines information from several sensors and processing components in order to build a richer representation of the environment.

The system includes multiple processing paths rather than routing every sensor through a single component.

Main development areas include:

• Radar data acquisition and processing  
• Radar calibration  
• IMU integration  
• Sensor synchronization and timestamp handling  
• Thermal and visible camera processing  
• AI-based person detection  
• Radar-based classification  
• Sensor fusion  
• Multi-sensor tracking  
• Pose initialization  
• Navigation-related processing  
• Integration between sensor data and higher-level algorithmic components

## AI and Perception

The project includes AI-based perception for person detection and tracking.

The perception layer includes:

• Vision-based person detection using YOLO models  
• Thermal-based person detection  
• Radar-based classification  
• Trained models for thermal and radar channels  
• Fusion between detections from different sensors  
• Multi-sensor tracking  
• Dataset generation and autolabeling tools  
• Model training and evaluation workflows

The system combines information from multiple sensors while keeping track of the source of each detection.

## Sensor Integration

A major part of the project is the integration between different sensing components.

The system works with:

• Radar  
• IMU  
• Thermal camera  
• Visible camera  
• OpenMV N6  
• NVIDIA Jetson

Sensor data is synchronized, calibrated, and passed between different processing components depending on the role of each sensor.

Not all sensor data is routed directly through the Jetson. Different components communicate and process data through separate paths before selected information is used for higher-level processing.

## My Contribution

ValiSight was developed collaboratively by several developers.

My main contributions focused on sensor integration and connecting the radar and IMU parts of the system with other project components.

I worked on:

• Radar integration and data processing  
• Radar calibration  
• IMU integration  
• Sensor synchronization and timestamp handling  
• Connecting radar data with components developed by Yael  
• Integration between sensor data and higher-level fusion and navigation components  
• Pose initialization and navigation-related work

My role focused mainly on connecting the sensor layer with other parts of the system and helping integrate the radar and IMU data into the wider perception and navigation flow.

Other parts of the system, including additional perception and AI development, were developed collaboratively by other members of the team.

## Repository Structure

The project is currently organized across several development branches.

Examples include:

• `IMU`  
• `Open-mv-n6---fusion`  
• `Open-mv-n6---fusion-work`  
• `noa`  
• `noa-pose-init-v0`  
• `noa-radar-calib-ai-2026-08-18`  
• `noa-yael-mapinit-2026-08-25`  
• `yael`

Each branch represents a different development area, experiment, or stage of the system.

The `main` branch currently serves as the project overview and documentation entry point.

A future cleanup will consolidate the relevant working components into a clearer final structure.

## Technologies and Concepts

The project includes work with:

• Python  
• Embedded systems  
• Radar  
• IMU  
• Thermal imaging  
• Computer vision  
• YOLO  
• OpenMV N6  
• NVIDIA Jetson  
• Sensor synchronization  
• Timestamp handling  
• Calibration  
• Sensor fusion  
• Multi-sensor tracking  
• Pose initialization  
• Navigation  
• AI model training  
• Dataset generation  
• Real-time system concepts

## Project Status

The repository currently reflects an active development history across multiple branches.

The code has not yet been fully consolidated into `main`.

The branches represent different development stages and system components, and a future cleanup will organize the relevant stable parts into a clearer final repository structure.
