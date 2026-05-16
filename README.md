
# Dual Robot System

This repository contains a dual-robot simulation and vision-based 3D end-effector tracking system using UR5 and Franka Panda robots in PyBullet.

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
python main.py --control-mode pid
```

The original visual-servoing PID baseline is frozen under:
```text
scripts/baselines/pid/
```

The active simulator now supports swappable Franka tracking policies:
```bash
cd scripts
python main.py --control-mode pid
python main.py --control-mode rl --rl-model-path ../models/ppo_franka_tracker.zip
```

### Stage 1 RL: Ground-Truth PPO Tracking
Train a PPO policy using PyBullet ground-truth target pose:
```bash
cd scripts
python train_rl_tracker.py --timesteps 200000 --save-path ../models/ppo_franka_tracker
```

The reward separates X-axis tracking from Y-Z plane tracking:
```bash
python train_rl_tracker.py \
  --timesteps 2000000 \
  --position-x-reward-weight 80 \
  --position-yz-reward-weight 50 \
  --velocity-x-reward-weight 1.0 \
  --velocity-yz-reward-weight 0.5
```

TensorBoard logs include `tracking/error_x_relative_m`, `tracking/error_x_world_m`,
`tracking/error_yz_m`, `tracking/velocity_error_x_mps`, and
`tracking/velocity_error_yz_mps`. The X reward uses relative displacement:
`(Franka_x - Franka_x_initial) - (Target_x - Target_x_initial)`, matching the
PID baseline's X-error convention.

For curriculum training, first train only the Y-Z plane, then resume with full
3D control:
```bash
python train_rl_tracker.py \
  --timesteps 1000000 \
  --action-mode yz \
  --save-path ../models/ppo_franka_tracker_yz

python train_rl_tracker.py \
  --timesteps 2000000 \
  --action-mode xyz \
  --load-path ../models/ppo_franka_tracker_yz.zip \
  --save-path ../models/ppo_franka_tracker_xyz
```

To train from compact vision features instead of ground-truth observations:
```bash
python train_rl_tracker.py \
  --timesteps 1000000 \
  --observation-mode vision \
  --action-mode yz \
  --save-path ../models/ppo_franka_tracker_vision_yz
```

Vision observations use pixel error, depth error, detection confidence features,
Franka velocity, previous action, and trajectory phase. The reward and metrics
still use ground-truth target pose for clean simulation training.

Evaluate the trained policy in the RL environment:
```bash
cd scripts
python evaluate_rl_tracker.py --model-path ../models/ppo_franka_tracker.zip
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
