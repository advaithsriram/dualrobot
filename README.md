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
