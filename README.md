
# CynLr Dual Robot System

This repository contains a dual-robot simulation and vision-based tracking system using UR5 and Franka Panda robots in PyBullet.

### Report
A detailed report is available: [View the report (PDF)](report.pdf)

### Example Videos
Videos demonstrating the system can be found in the `videos/` directory:
- 2_iterations.mp4
- 4_iterations.mp4


## Overview
- **Robot A (UR5):** Executes pick-and-place and 3D trajectory following.
- **Robot B (Franka Panda):** Tracks objects using a virtual camera and vision-based control.
- **Simulation:** Both robots operate in a shared environment, with synchronized data collection and visualization.

## Features
- Modular controllers for each robot
- Vision-based tracking and servoing
- 3D trajectory generation and execution
- Data logging and trajectory visualization
- Overlay and error plots for performance analysis

## Getting Started

### Prerequisites
- Python 3.7+
- [PyBullet](https://pybullet.org/)
- numpy
- matplotlib
- opencv-python

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Simulation
```bash
cd scripts
python main.py
```

### Directory Structure
- `scripts/` — Main simulation and robot control code
- `urdf/` — Robot model files (UR5, Panda)
- `meshes/` — Meshes for robot visualization
- `graphs/` — Output plots and overlays
- `videos/` — Example and output videos
- `requirements.txt` — Python dependencies

## License
This project is released under the MIT License. See `LICENSE` for details.
