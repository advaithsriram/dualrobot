# Learning-Based Dual-Robot End-Effector Tracking

This project implements a simulated dual-robot tracking system in PyBullet. A UR5 robot carries a red target object along a time-varying 3D Cartesian trajectory, while a Franka Panda learns to track that moving target smoothly over time. The project includes a frozen visual-servoing PID baseline and a reinforcement-learning pipeline based on PPO.

The core objective is not simply to reach a static point. The Franka must continuously track a moving end-effector trajectory with low position error and smooth motion.

## Task Summary

**Requirement:** build a system that learns to control a robotic arm to track a desired end-effector trajectory.

This implementation satisfies the requirements as follows:

| Requirement | Implementation |
| --- | --- |
| Standard simulated arm | Franka Panda and UR5 in PyBullet |
| Time-varying Cartesian trajectory | UR5 target follows circle and Lissajous/figure-eight trajectories in Y-Z, with sinusoidal X motion |
| RL as core component | PPO policy controls the Franka end-effector |
| Smooth, stable tracking | Action magnitude, action-change, and velocity-error penalties |
| Source of uncertainty | Optional perception noise, depth noise, and detection dropout in vision-based RL |
| Evaluation | RMSE and MAE for 3D, relative X, and Y-Z tracking errors |

## System Overview

The simulation contains two robots:

- **Robot A: UR5**
  - Picks up a red cube target.
  - Executes precomputed trajectories while holding the cube.
  - Acts as the moving target generator.

- **Robot B: Franka Panda**
  - Tracks the UR5-held target.
  - Can run either the frozen PID baseline or a learned PPO policy.
  - Uses IK to convert learned Cartesian displacement commands into joint targets.

The robot bases are separated in world X. Therefore, X tracking is evaluated using relative displacement rather than absolute world position. This matches the original PID baseline and measures whether the Franka reproduces the UR5 target's X motion while preserving the initial distance between robots.

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
    franka_policies.py            # PID and PPO policy wrappers
  rl/
    tracking_env.py               # Gymnasium/PyBullet RL environment
  baselines/pid/                  # Frozen PID baseline snapshot

urdf/                             # UR5 and Panda models
meshes/                           # Robot meshes
graphs/                           # Generated plots
videos/                           # Example videos
models/                           # Trained PPO models/checkpoints
runs/                             # TensorBoard logs
```

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:

- `pybullet`
- `gymnasium`
- `stable-baselines3`
- `opencv-python`
- `numpy`
- `matplotlib`
- `tensorboard`

## Running the PID Baseline

The original visual-servoing baseline is preserved under:

```text
scripts/baselines/pid/
```

Run the active simulator with PID control:

```bash
cd scripts
python main.py --control-mode pid
```

The PID controller uses red-object image detection and depth from the Franka wrist camera. It maps pixel and depth errors to Cartesian end-effector corrections, then applies IK and position control.

## Running a Trained RL Policy

For a vision-based PPO policy trained on Y-Z tracking:

```bash
cd scripts
python main.py \
  --control-mode rl \
  --rl-model-path ../models/ppo_franka_tracker_vision_yz.zip \
  --observation-mode vision \
  --action-mode yz
```

For full 3D tracking:

```bash
python main.py \
  --control-mode rl \
  --rl-model-path ../models/ppo_franka_tracker_vision_xyz.zip \
  --observation-mode vision \
  --action-mode xyz
```

## Target Trajectory Representation

The UR5 target trajectory is precomputed using inverse kinematics before execution. The target object is attached to the UR5 end-effector and follows one of the following paths.

### Circular Trajectory in Y-Z

```text
x(t) = x0 + A_x sin(wt)
y(t) = y0 + r cos(t)
z(t) = z0 + r sin(t)
```

### Lissajous / Figure-Eight Trajectory

```text
x(t) = x0 + A_x sin(wt)
y(t) = y0 + A_y sin(t + pi/2)
z(t) = z0 - A_z sin(2(t + pi/2))
```

The X component is sinusoidal. In evaluation, the X metric compares relative displacement, not absolute world position, because the robots are physically separated.

## RL Environment

The RL environment is implemented in:

```text
scripts/rl/tracking_env.py
```

It is a Gymnasium-compatible PyBullet environment. By default, it runs headless using `p.DIRECT`. Use `--render` for visual debugging.

### Action Space

The policy outputs a continuous 3D Cartesian displacement command:

```text
action = [dx, dy, dz], each in [-1, 1]
```

The command is scaled by:

```text
action_scale = 0.02 m
```

Then the action is applied as a Cartesian end-effector target:

```text
target_ee_position = current_ee_position + action_scale * action
```

PyBullet inverse kinematics converts this Cartesian target into Franka joint targets.

The environment supports three action modes:

| Mode | Meaning |
| --- | --- |
| `yz` | X action is masked; train planar Y-Z tracking only |
| `x` | Y/Z actions are masked; train relative X tracking only |
| `xyz` | Full 3D control |

This makes curriculum learning possible.

## Observation Space

The PPO policy uses compact vision features rather than raw images or ground-truth target positions.

Dimension: **17**

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
Franka EE velocity          3
previous action             3
trajectory phase sin/cos    2
```

The policy does **not** observe the ground-truth target position. Ground truth is used only for reward and evaluation metrics during simulation training.

## Reward Design

The reward is axis-separated so that planar tracking and X tracking can be shaped independently.

Let:

```text
e_yz = sqrt((target_y - ee_y)^2 + (target_z - ee_z)^2)
```

For X, the project uses relative displacement error, matching the PID baseline:

```text
e_x_rel =
  (target_x - target_x_initial)
  -
  (franka_x - franka_x_initial)
```

This measures whether the Franka reproduces the target's X motion while preserving the initial robot-to-robot distance.

Velocity errors are also split:

```text
v_x_error  = abs(target_vx - ee_vx)
v_yz_error = sqrt((target_vy - ee_vy)^2 + (target_vz - ee_vz)^2)
```

The reward is:

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

For full 3D fine-tuning, Y-Z can be protected with higher Y-Z weights:

```bash
--position-x-reward-weight 80 \
--position-yz-reward-weight 150 \
--velocity-x-reward-weight 1.0 \
--velocity-yz-reward-weight 1.0
```

## Uncertainty and Robustness

The vision-based environment supports uncertainty injection:

```bash
--vision-pixel-noise-std 2.0
--vision-depth-noise-std 0.01
--vision-dropout-prob 0.05
```

These options perturb the compact visual feedback rather than the reward. This allows training policies that are robust to noisy image detections, noisy depth values, and temporary target loss.

Train with visual uncertainty:

```bash
python train_rl_tracker.py \
  --timesteps 300000 \
  --observation-mode vision \
  --action-mode xyz \
  --load-path ../models/ppo_franka_tracker_vision_xyz.zip \
  --save-path ../models/ppo_franka_tracker_vision_xyz_noisy \
  --vision-pixel-noise-std 2.0 \
  --vision-depth-noise-std 0.01 \
  --vision-dropout-prob 0.05 \
  --print-every 25000
```

Evaluate robustness under the same noise:

```bash
python evaluate_rl_tracker.py \
  --model-path ../models/ppo_franka_tracker_vision_xyz.zip \
  --observation-mode vision \
  --action-mode xyz \
  --vision-pixel-noise-std 2.0 \
  --vision-depth-noise-std 0.01 \
  --vision-dropout-prob 0.05
```

## Training Procedure

The most successful approach is curriculum learning.

### 1. Train Y-Z Tracking First

```bash
cd scripts
python train_rl_tracker.py \
  --timesteps 1000000 \
  --observation-mode vision \
  --action-mode yz \
  --save-path ../models/ppo_franka_tracker_vision_yz \
  --print-every 25000
```

This stage learns accurate planar tracking from visual pixel feedback.

### 2. Fine-Tune for Full 3D Tracking

```bash
python train_rl_tracker.py \
  --timesteps 1000000 \
  --observation-mode vision \
  --action-mode xyz \
  --load-path ../models/ppo_franka_tracker_vision_yz.zip \
  --save-path ../models/ppo_franka_tracker_vision_xyz \
  --position-x-reward-weight 80 \
  --position-yz-reward-weight 150 \
  --velocity-x-reward-weight 1.0 \
  --velocity-yz-reward-weight 1.0 \
  --print-every 25000
```

### 3. Optional X-Only Phase

If relative X tracking remains poor, train an X-focused policy phase:

```bash
python train_rl_tracker.py \
  --timesteps 500000 \
  --observation-mode vision \
  --action-mode x \
  --load-path ../models/ppo_franka_tracker_vision_xyz.zip \
  --save-path ../models/ppo_franka_tracker_vision_x \
  --position-x-reward-weight 180 \
  --velocity-x-reward-weight 2.0 \
  --print-every 25000
```

Then resume full `xyz` training from that checkpoint.

## Checkpointing and Resuming

Training saves checkpoints every 200,000 timesteps by default:

```text
models/checkpoints/
```

Manual options:

```bash
--checkpoint-every 200000
--checkpoint-dir ../models/checkpoints
--checkpoint-name vision_xyz_relx
```

If training is interrupted with `Ctrl+C`, the latest model is saved as:

```text
<save-path>_interrupted.zip
```

Resume from a checkpoint:

```bash
python train_rl_tracker.py \
  --timesteps 500000 \
  --observation-mode vision \
  --action-mode xyz \
  --load-path ../models/checkpoints/vision_xyz_relx_1200000_steps.zip \
  --save-path ../models/ppo_franka_tracker_vision_xyz_resume
```

## TensorBoard

Start TensorBoard from the `scripts/` directory:

```bash
tensorboard --logdir ../runs
```

Useful scalars:

```text
tracking/error_m
tracking/error_x_relative_m
tracking/error_x_world_m
tracking/error_yz_m
tracking/velocity_error_x_mps
tracking/velocity_error_yz_mps
tracking/step_reward
vision/detected
vision/area_norm
rollout/ep_rew_mean
```

For training quality, use `tracking/error_x_relative_m`, not `tracking/error_x_world_m`.

## Evaluation

Evaluate a trained model:

```bash
cd scripts
python evaluate_rl_tracker.py \
  --model-path ../models/ppo_franka_tracker_vision_yz.zip \
  --observation-mode vision \
  --action-mode yz
```

Full 3D evaluation:

```bash
python evaluate_rl_tracker.py \
  --model-path ../models/ppo_franka_tracker_vision_xyz.zip \
  --observation-mode vision \
  --action-mode xyz
```

The evaluator reports:

```text
3D RMSE / MAE
X-relative RMSE / MAE
X-world RMSE / MAE
Y-Z RMSE / MAE
```

The primary metrics are:

- **Y-Z RMSE/MAE** for planar trajectory tracking.
- **X-relative RMSE/MAE** for sinusoidal X motion tracking.
- **3D RMSE/MAE** using relative X and direct Y/Z errors.

## Example Results

Add final plots and videos here.

### PID Baseline

The PID baseline uses fixed visual-servoing gains. It is useful as a non-learning comparison and as a sanity check for the camera/detector pipeline.

Suggested results to include:

```text
Y-Z RMSE:
X-relative RMSE:
3D RMSE:
```

### RL: Vision-Based Y-Z Tracking

Example result from vision-based Y-Z tracking:

```text
Y-Z RMSE: 0.00658 m
Y-Z MAE:  0.00494 m
```

This demonstrates that the learned PPO controller can track the moving target in the Y-Z plane with sub-centimeter average error.

Include:

- Y-Z overlay plot.
- Y and Z absolute error over time.
- Video of the Franka tracking the UR5-held target.

### RL: Full 3D Tracking

Full 3D tracking is trained by fine-tuning the Y-Z model with `--action-mode xyz`. The key metric is X-relative error, because the Franka should preserve the robot-to-robot spacing while reproducing the target's sinusoidal X displacement.

Suggested results to include:

```text
X-relative RMSE:
Y-Z RMSE:
3D RMSE:
```

## Notes on PID vs RL

The PID controller is a useful baseline, but it relies on hand-tuned gains:

```text
pixel/depth error -> Cartesian displacement -> IK -> joint targets
```

The RL controller learns this mapping from interaction:

```text
vision features -> PPO policy -> Cartesian displacement -> IK -> joint targets
```

The PID baseline was frozen to avoid changing the comparison target while developing the RL pipeline. The active code supports both policies through:

```bash
python main.py --control-mode pid
python main.py --control-mode rl --rl-model-path <model.zip>
```

## Limitations and Future Work

- Full 3D tracking is more difficult than Y-Z tracking because X/depth feedback is weaker and must preserve the inter-robot distance.
- The current RL policy controls position only; orientation tracking is not yet included.
- The policy uses IK and position control rather than torque-level control.
- Raw image RL is not used; the current approach uses compact vision features for sample efficiency.
- Future work could freeze the successful Y-Z policy and train a separate X/depth policy, then combine both controllers.

## Quick Command Reference

Train vision Y-Z:

```bash
python train_rl_tracker.py --observation-mode vision --action-mode yz
```

Train full vision XYZ from Y-Z:

```bash
python train_rl_tracker.py \
  --observation-mode vision \
  --action-mode xyz \
  --load-path ../models/ppo_franka_tracker_vision_yz.zip
```

Evaluate:

```bash
python evaluate_rl_tracker.py \
  --model-path ../models/ppo_franka_tracker_vision_yz.zip \
  --observation-mode vision \
  --action-mode yz
```

Run trained model in the dual-robot simulator:

```bash
python main.py \
  --control-mode rl \
  --rl-model-path ../models/checkpoints/ppo_franka_tracker_vision_xyz_relx2_2200000_steps.zip \
  --observation-mode vision \
  --action-mode xyz
```
