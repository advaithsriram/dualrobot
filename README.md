# Learning-Based Dual-Robot 3D End-Effector Tracking

This project implements a simulated dual-robot tracking system in PyBullet. A UR5 robot picks up a red target object and moves it along a time-varying 3D trajectory. A Franka Panda then tracks that moving target using either a visual-servoing PD baseline or a PPO reinforcement-learning policy.

The core task is to continuously and smoothly track the end-effector trajectory while keeping position error low.

The final plots and videos can be found in `graphs/` and `videos/`.

Please find the technical report for this project here:
[`Technical_Report_Humanoid_Challenge.pdf`](Technical_Report_Humanoid_Challenge.pdf).

## Task Summary

| Requirement | Implementation |
| --- | --- |
| Standard simulated arm | UR5 and Franka Panda in PyBullet |
| Time-varying Cartesian trajectory | Alternating Circle and Lissajous/figure-eight motion in Y-Z, with sinusoidal X motion |
| RL | PPO policy controls Franka Cartesian displacement commands |
| Smooth, stable motion | Velocity matching, action magnitude penalty, and action-change penalty |
| Source of uncertainty | Optional visual pixel noise, depth noise, and detection dropout, evaluated after training |
| Evaluation | RMSE and MAE along X, Y, and Z axes |

## System Overview

- **Robot A: UR5**
  - Picks up the red cube target.
  - Executes the predefined 3D trajectory.
  - Provides the moving target that Robot B must follow.

- **Robot B: Franka Panda**
  - Tracks the UR5-held target.
  - Can run the PD baseline or the trained PPO policy.
  - Uses IK to convert Cartesian displacement commands into joint targets.

The robots are separated along the world X axis. For this reason, X tracking is evaluated using **relative displacement**, not absolute world X position. This matches the PD baseline: the Franka should reproduce the UR5 target's sinusoidal X motion while preserving the initial distance between the two robots.

## Repository Structure

```text
scripts/
  main.py                         # Dual-robot simulation entry point
  robotA.py                       # UR5 setup, pick-and-place, trajectory generation
  robotB.py                       # Franka setup and wrist camera
  vision_processor.py             # Red-object detector
  train_rl_tracker.py             # PPO training entry point
  evaluate_rl_tracker.py          # PPO evaluation entry point
  controllers/
    franka_policies.py            # PD and PPO policy wrappers
  rl/
    tracking_env.py               # Gymnasium/PyBullet RL environment
  baselines/pd/                   # Frozen PD baseline snapshot

models/
  curriculum_model_rl_yz.zip      # Midway curriculum policy: Y-Z tracking only
  final_model_rl_xyz.zip          # Final policy: full XYZ tracking

runs/                             # TensorBoard logs
graphs/                           # Generated plots
videos/                           # Example/output videos
urdf/, meshes/                    # Robot assets
```

## Installation

Create and activate the conda environment:

```bash
conda create -n 3d_end python=3.11
conda activate 3d_end
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies are `pybullet`, `gymnasium`, `stable-baselines3`, `opencv-python`, `numpy`, `matplotlib`, and `tensorboard`.

All commands below are run from the `scripts/` directory:

```bash
cd scripts
```

## Quick Start

Run the PD baseline:

```bash
python main.py --control-mode pd
```

Run the final RL policy in the full dual-robot simulator:

```bash
python main.py \
  --control-mode rl \
  --rl-model-path ../models/final_model_rl_xyz.zip \
  --action-mode xyz
```

Evaluate the final RL policy:

```bash
python evaluate_rl_tracker.py \
  --model-path ../models/final_model_rl_xyz.zip \
  --action-mode xyz
```

Evaluate the Y-Z curriculum policy:

```bash
python evaluate_rl_tracker.py \
  --model-path ../models/curriculum_model_rl_yz.zip \
  --action-mode yz
```

## Target Trajectory

After pickup, the UR5 target follows two alternating patterns in the Y-Z plane: a circle and a Lissajous/figure-eight curve. Both are combined with sinusoidal motion along X.

For the circular phase:

```text
x(t) = x0 + A_x sin(wt)
y(t) = y0 + r cos(t)
z(t) = z0 + r sin(t)
```

For the Lissajous phase:

```text
x(t) = x0 + A_x sin(wt)
y(t) = y0 + A_y sin(t + pi/2)
z(t) = z0 - A_z sin(2(t + pi/2))
```

The UR5 trajectory is precomputed with PyBullet inverse kinematics before execution. This keeps the target motion repeatable and gives the Franka a consistent moving trajectory to track.

## Controllers

### PD Baseline

The PD baseline uses the Franka wrist camera to detect the red target object. HSV color thresholding finds the target centroid in the RGB image, and the depth image provides the target depth. Pixel/depth error is converted to a Cartesian correction, then IK converts the Cartesian target into Franka joint commands.

The PD baseline parameters were tuned and preserved under:

```text
scripts/baselines/pd/
```

### PPO RL Controller

The RL controller uses PPO from Stable-Baselines3. The policy maps compact vision features to a Cartesian end-effector displacement command:

```text
vision features -> PPO policy -> Cartesian displacement -> IK -> joint targets
```

The trained policy is used by `main.py` through:

```bash
python main.py \
  --control-mode rl \
  --rl-model-path ../models/final_model_rl_xyz.zip \
  --action-mode xyz
```

## RL Environment

The Gymnasium-compatible environment is implemented in:

```text
scripts/rl/tracking_env.py
```

By default, training and evaluation run headless with PyBullet `DIRECT`. Use `--render` for visual debugging.

## Observation Space

The PPO policy uses compact vision features rather than raw images or ground-truth target positions.

Observation dimension: **17**

```text
normalized pixel x error    1
normalized pixel y error    1
depth error                 1
delta pixel x error         1
delta pixel y error         1
delta depth error           1
detected flag               1
normalized blob area        1
normalized depth            1
Franka end-effector velocity 3
previous action             3
trajectory phase sin/cos    2
```

The ground truth is used only inside simulation for reward calculation and evaluation metrics.

## Action Space

The policy outputs a continuous 3D Cartesian displacement command:

```text
action = [dx, dy, dz], each in [-1, 1]
```

The command is scaled by:

```text
action_scale = 0.02 m
```

Then applied as:

```text
target_ee_position = current_ee_position + action_scale * action
```

The environment supports three modes:

| Mode | Meaning |
| --- | --- |
| `yz` | X action is masked; train/evaluate Y-Z tracking only |
| `x` | Y/Z actions are masked; train/evaluate relative X tracking only |
| `xyz` | Full 3D control |

## Reward Design

The reward is axis-separated so Y-Z tracking and X tracking can be shaped independently.

Planar Y-Z error:

```text
e_yz = sqrt((target_y - ee_y)^2 + (target_z - ee_z)^2)
```

Relative X error:

```text
e_x_rel =
  (target_x - target_x_initial)
  -
  (franka_x - franka_x_initial)
```

This X metric rewards the Franka for matching the target's sinusoidal X displacement while maintaining the robot-to-robot spacing.

Velocity errors are split similarly:

```text
v_x_error  = abs(target_vx - ee_vx)
v_yz_error = sqrt((target_vy - ee_vy)^2 + (target_vz - ee_vz)^2)
```

Reward:

```text
r =
  - w_x_pos  * e_x_rel^2
  - w_yz_pos * e_yz^2
  - w_x_vel  * v_x_error^2
  - w_yz_vel * v_yz_error^2
  - 0.05 * ||action||^2
  - 0.20 * ||action - previous_action||^2
```

Additional terms:

```text
+1.0 if active tracking error < 0.02 m
-5.0 if active tracking error > 0.45 m
```

Default weights:

```text
position_x_reward_weight  = 80.0
position_yz_reward_weight = 50.0
velocity_x_reward_weight  = 1.0
velocity_yz_reward_weight = 0.5
```

## Curriculum Learning

The final model was trained using a curriculum:

1. **Y-Z tracking first:** train with `--action-mode yz` so the Franka learns accurate planar visual tracking.
2. **Full XYZ fine-tuning:** initialize from the Y-Z policy and enable `--action-mode xyz` so the policy learns relative X tracking while preserving Y-Z behavior.

The repository keeps two models:

| Model | Purpose |
| --- | --- |
| `models/curriculum_model_rl_yz.zip` | Midway curriculum model for Y-Z tracking |
| `models/final_model_rl_xyz.zip` | Final model for full 3D tracking |

Train the Y-Z stage:

```bash
python train_rl_tracker.py \
  --timesteps 1000000 \
  --action-mode yz \
  --save-path ../models/curriculum_model_rl_yz \
  --print-every 25000
```

Fine-tune for XYZ:

```bash
python train_rl_tracker.py \
  --timesteps 1000000 \
  --action-mode xyz \
  --load-path ../models/curriculum_model_rl_yz.zip \
  --save-path ../models/final_model_rl_xyz \
  --position-x-reward-weight 80 \
  --position-yz-reward-weight 150 \
  --velocity-x-reward-weight 1.0 \
  --velocity-yz-reward-weight 1.0 \
  --print-every 25000
```

## Uncertainty and Robustness

The environment can inject uncertainty into the visual observations:

| Profile | Pixel Noise | Depth Noise | Detection Dropout |
| --- | ---: | ---: | ---: |
| `none` | 0 px | 0 m | 0 |
| `mild` | 2 px | 0.01 m | 0.05 |
| `moderate` | 4 px | 0.02 m | 0.10 |

`--noise-profile mild` adds Gaussian pixel noise with standard deviation `2 px`, Gaussian depth noise with standard deviation `0.01 m`, and a 5% detection dropout probability.

Training and evaluating with visual uncertainty:

```bash
python train_rl_tracker.py \
  --timesteps 300000 \
  --action-mode xyz \
  --load-path ../models/final_model_rl_xyz.zip \
  --save-path ../models/final_model_rl_xyz_noisy \
  --noise-profile mild \
  --print-every 25000
```

```bash
python evaluate_rl_tracker.py \
  --model-path ../models/final_model_rl_xyz.zip \
  --action-mode xyz \
  --noise-profile mild
```

## Evaluation Metrics

- **X-relative RMSE/MAE:** sinusoidal X tracking while maintaining inter-robot spacing.
- **Y RMSE/MAE:** horizontal tracking error in the Y-Z plane.
- **Z RMSE/MAE:** vertical tracking error in the Y-Z plane.

## License

This project is released under the MIT License. See `LICENSE` for details.
