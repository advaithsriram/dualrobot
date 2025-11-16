import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from trajectory import precompute_trajectory_waypoints, execute_trajectory, alternate_trajectories

# Trajectory selection flags
USE_CIRCLE = False      # Set to True for circular trajectory
USE_LISSAJOUS = True  # Set to True for Lissajous (figure-8) trajectory

# 1️⃣ Connect to the physics server
p.connect(p.GUI)  # use p.DIRECT for headless mode
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 2️⃣ Setup the world
p.resetSimulation()
p.setGravity(0, 0, -9.81)
plane = p.loadURDF("plane.urdf")

# Create a table
table_height = 0.8  # 50cm tall table (cube/cuboid height)
table_width = 0.8  # 0.8m x 0.8m square top
table_depth = 0.8

# Table as a white cube/cuboid resting on the ground
table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[table_width/2, table_depth/2, table_height/2])
table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[table_width/2, table_depth/2, table_height/2], rgbaColor=[1, 1, 1, 1])
table = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_collision, baseVisualShapeIndex=table_visual, basePosition=[0, 0, table_height/2])

# 3️⃣ Load the UR5 - use absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Add the project root to search path so PyBullet can find meshes
p.setAdditionalSearchPath(project_root)

ur5_path = os.path.join(project_root, "urdf", "ur5.urdf")
start_pos = [0, 0, table_height]  # Place UR5 on top of table
start_orientation = p.getQuaternionFromEuler([0, 0, 0])

ur5 = p.loadURDF(ur5_path, start_pos, start_orientation, useFixedBase=True)

# 4️⃣ Inspect the joints
num_joints = p.getNumJoints(ur5)
print(f"UR5 has {num_joints} joints:")
for i in range(num_joints):
    info = p.getJointInfo(ur5, i)
    print(f"Joint {i}: {info[1].decode('utf-8')}")

# 5️⃣ Move the robot: Joint-space control
# Forward-facing position: end-effector pointing forward (perpendicular to vertical plane)
target_positions = [0, -np.pi/2, np.pi/2, -np.pi, -np.pi/2, 0]  # forward-facing pose
print("Moving to initial position...")
for step in range(480):  # 2 seconds at 240 Hz
    for joint_index in range(6):  # first 6 joints are revolute
        p.setJointMotorControl2(
            bodyUniqueId=ur5,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_positions[joint_index],
            force=500
        )
    p.stepSimulation()
    # time.sleep(1. / 240.)  # Comment out for faster simulation

print("Starting trajectory motion...")
print(f"USE_CIRCLE = {USE_CIRCLE}, USE_LISSAJOUS = {USE_LISSAJOUS}")

# Pre-compute the trajectory waypoints once (no drift, very fast!)
if USE_CIRCLE:
    trajectory_type = 'circle'
    waypoints, center = precompute_trajectory_waypoints(
        robot_id=ur5,
        trajectory_type='circle',
        radius=0.15,
        num_points=360  # More points = smoother, more accurate circle
    )
    draw_color = [0, 0, 1]  # Blue
    speed_factor = 2.0  # Circle can handle more speed
    print("→ Executing circular trajectory for 20 seconds...")
    
elif USE_LISSAJOUS:
    trajectory_type = 'lissajous'
    waypoints, center = precompute_trajectory_waypoints(
        robot_id=ur5,
        trajectory_type='lissajous',
        amplitude_y=0.15,      # Horizontal amplitude
        amplitude_z=0.075,     # Half vertical amplitude (2:1 ratio for proper figure-8)
        num_points=720  # Enough points for smooth figure-8
    )
    draw_color = [1, 0, 0]  # Red
    speed_factor = 2.0  # Same speed as circle
    print("→ Executing Lissajous trajectory for 20 seconds...")

# Execute the trajectory repeatedly for 30 seconds
import time as time_module
start_time = time_module.time()
duration = 30

while time_module.time() - start_time < duration:
    execute_trajectory(
        robot_id=ur5,
        waypoints=waypoints,
        draw_color=draw_color,
        center_pos=center,
        draw_trace=True,
        speed_factor=speed_factor  # Use trajectory-specific speed factor
    )

print("\nMotion complete!")

# 6️⃣ Disconnect
p.disconnect()