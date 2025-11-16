import pybullet as p
import pybullet_data
import time
import numpy as np
from pick_place_simple import create_graspable_cube, pick_and_place_demo

# 1️⃣ Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 2️⃣ Load environment
plane = p.loadURDF("plane.urdf")

# Load table (white cube)
table_height = 0.4
table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, table_height/2])
table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, table_height/2], 
                                    rgbaColor=[1, 1, 1, 1])
table = p.createMultiBody(baseMass=0, 
                          baseCollisionShapeIndex=table_collision,
                          baseVisualShapeIndex=table_visual,
                          basePosition=[0.5, 0, table_height/2])

# 3️⃣ Load UR5 robot on top of table (with buffer)
ur5 = p.loadURDF("../urdf/ur5.urdf", 
                 basePosition=[0.5, 0, table_height + 0.02],  # 2cm buffer as you found
                 baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                 useFixedBase=True)

print(f"UR5 loaded with {p.getNumJoints(ur5)} joints")

# Set initial joint positions (stable upright pose)
initial_positions = [0, -1.57, 1.57, -1.57, -1.57, 0]
for i in range(6):
    p.resetJointState(ur5, i, initial_positions[i])
    
# Enable position control to hold against gravity
for i in range(6):
    p.setJointMotorControl2(
        bodyIndex=ur5,
        jointIndex=i,
        controlMode=p.POSITION_CONTROL,
        targetPosition=initial_positions[i],
        force=5000,
        maxVelocity=2.0,
        positionGain=0.5
    )

# 4️⃣ Create a small cube to pick up
cube_size = 0.04  # 4cm cube
table_top_z = table_height + 0.02  # Account for robot base buffer
cube_position = [0.5, 0.3, table_top_z + cube_size/2 + 0.02]  # In front, to the side
cube = create_graspable_cube(
    position=cube_position,
    size=cube_size,
    color=[1, 0, 0, 1],  # Red
    mass=0.05  # 50 grams
)

print(f"\nCube created at position: {cube_position}")
print(f"Robot base is at (0.5, 0, {table_height + 0.02})")

# Let physics settle
for _ in range(100):
    p.stepSimulation()
    # time.sleep(1./240.)

# Get cube position after settling
cube_pos, _ = p.getBasePositionAndOrientation(cube)
print(f"Cube settled at: {cube_pos}")

# Check end-effector starting position
num_joints = p.getNumJoints(ur5)
ee_link = num_joints - 1
ee_state = p.getLinkState(ur5, ee_link)
print(f"End-effector starting position: {ee_state[0]}\n")

# 5️⃣ Define pick and place positions
pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2]]
place_pos = [0.55, -0.15, table_top_z + cube_size/2 + 0.02]

print(f"Pick position: {pick_pos}")
print(f"Place position: {place_pos}\n")

# 6️⃣ Execute pick and place with simple IK
time.sleep(1)
success = pick_and_place_demo(
    robot_id=ur5,
    object_id=cube,
    pick_pos=pick_pos,
    place_pos=place_pos,
    hover_height=0.08
)

if success:
    print("\n✓ Demo completed successfully!")
else:
    print("\n✗ Demo failed!")

# Keep simulation running
print("\nSimulation running. Close the window to exit.")
while True:
    p.stepSimulation()
    # time.sleep(1./240.)
