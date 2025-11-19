# CynLr Assignment

## UR5 URDF Link:
https://github.com/ElectronicElephant/pybullet_ur5_robotiq/tree/robotflow/urdf

- Meshes in the same parent tree

## Franka Panda URDF Link:
https://github.com/PaulPauls/franka_emika_panda_pybullet/blob/master/panda_robot/model_description/panda_with_gripper.urdf


## Why PyBullet over other simulators?
PyBullet (without sleep): Very fast, runs at maximum CPU speed. Good for running many iterations quickly for learning/testing.
CoppeliaSim: Generally slower but has better visualization, more realistic physics options, and better sensors/cameras. It's more feature-rich but computationally heavier.

## How to run:
cd /Users/advaithsriram/cynlr-dualrobot/scripts && /opt/miniconda3/envs/cynlr/bin/python main.py

### Update:
Nov 13th Thursday: Circle working, UR5 URDF running and on a table. Able to move in a circular trajectory. Lissajous curve not working.

## For Reference:
initial_positions = [0, -90, 90, 180, -90, 0] #works
cube_position = [0.8, -0.3, table_top_z + cube_size/2 + 0.001] #works


# PD Gains:
Proportional Gains (P):
- self.pixel_to_meter_x = 0.0008 # X axis 
- self.pixel_to_meter_y = 0.0008 # Y axis
- self.depth_to_meter_z = 0.12 # Z axis

Derivative Gains (D):
- self.pixel_to_meter_x_d = 0.0006 # X axis
- self.pixel_to_meter_y_d = 0.0006 # Y axis
- self.depth_to_meter_z_d = 0.025 # Z axis

# V2:

## V2 GAINS

PD Gains:
X-axis:
- P-Gain: 0.0008
- D-Gain: 0.0004
Y-axis:
- P-Gain: 0.0008
- D-Gain: 0.0004
Z-axis (depth):
- P-Gain: 0.35
- D-Gain: 0.04

Low-pass filter alpha:
- Alpha: 0.5

## V2 RESULTS (MSE, MAE, RMSE)

Tracking Error Metrics:
- X axis:   MAE=0.01354  MSE=0.00025  RMSE=0.01592
- Y axis:   MAE=0.01289  MSE=0.00021  RMSE=0.01450
- Z axis:   MAE=0.01590  MSE=0.00042  RMSE=0.02058 


## End effector frame to world frame:
The PD gains are applied in the robot’s end-effector (camera) frame, which is oriented differently from the world frame. Due to the robot’s configuration and camera mounting, the mapping between camera frame axes and world frame axes is not one-to-one:

The Z axis in the camera frame (depth) primarily affects the X position in the world frame.
The Y axis in the camera frame affects the Z position in the world frame.
The X axis in the camera frame affects the Y position in the world frame.
This means that control actions computed for a given axis in the camera frame result in movement along a different axis in the world frame. This mapping is handled by transforming the desired displacement from the camera frame to the world frame using the end-effector’s rotation matrix.

# Error Metrics without Depth
Tracking Error Metrics:
X axis:  MAE=0.05364 RMSE=0.06199
Y axis:  MAE=0.01526 RMSE=0.01699
Z axis:  MAE=0.01642 RMSE=0.01886