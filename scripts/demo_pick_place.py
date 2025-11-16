import pybullet as p
import pybullet_data
import time
import numpy as np
from pick_place import create_graspable_cube, pick_and_place_demo

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

# 3️⃣ Load UR5 robot on top of table
ur5 = p.loadURDF("../urdf/ur5.urdf", 
                 basePosition=[0.5, 0, table_height + 0.02],
                 baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                 useFixedBase=True)

print(f"UR5 loaded with {p.getNumJoints(ur5)} joints")

# # Disable default velocity control (important!)
# for i in range(6):
#     p.setJointMotorControl2(
#         bodyIndex=ur5,
#         jointIndex=i,
#         controlMode=p.VELOCITY_CONTROL,
#         force=0  # Disable default motor
#     )

# Set initial joint positions (stable upright pose)
initial_positions = [0, -1.57, 1.57, -1.57, -1.57, 0]
for i in range(6):
    p.resetJointState(ur5, i, initial_positions[i])

print("\nInitial joint positions set:")
for i in range(p.getNumJoints(ur5)):
    print(i, p.getJointInfo(ur5, i)[1])

# Enable strong position control to hold against gravity
for i in range(6):
    p.setJointMotorControl2(
        bodyIndex=ur5,
        jointIndex=i,
        controlMode=p.POSITION_CONTROL,
        targetPosition=initial_positions[i],
        force=5000,  # Much stronger force to overcome gravity
        maxVelocity=2.0,
        positionGain=0.5  # Higher gain for better tracking
    )

# 4️⃣ Create a small cube to pick up
# Place it on the table at a reachable location
# Robot is at (0.5, 0, 0.4), end-effector naturally reaches forward and to sides
cube_size = 0.04  # 4cm cube
table_top_z = table_height  # Table top is at z=0.4
# Place cube in front-right of robot - easier to reach without collision
cube_position = [0.8, -0.3, table_top_z + cube_size/2 + 0.001]  # Directly in front, to the side
cube = create_graspable_cube(
    position=cube_position,
    size=cube_size,
    color=[1, 0, 0, 1],  # Red
    mass=0.05  # 50 grams
)

print(f"\nCube created at position: {cube_position}")
# print(f"Table top is at z={table_top_z}")
# print(f"Robot base is at (0.5, 0, {table_height})")
print(f"Cube is {np.linalg.norm(np.array([0.65, 0.15]) - np.array([0.5, 0])):.3f}m from robot base (XY plane)")

# Let physics settle
for _ in range(100):
    p.stepSimulation()
    # time.sleep(1./240.)

# Check end-effector starting position
num_joints = p.getNumJoints(ur5)
ee_link = num_joints - 1
ee_state = p.getLinkState(ur5, ee_link)
print(f"\nEnd-effector starting position: {ee_state[0]}")

# # Move to a safe reachable position first to test workspace
# print("\nMoving to test position to verify reachability...")
# test_pos = [0.5, 0.3, 0.5]  # In front of robot, slightly to the side
# test_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

# joint_positions = p.calculateInverseKinematics(
#     ur5,
#     ee_link,
#     test_pos,
#     test_orn,
#     maxNumIterations=100,
#     residualThreshold=1e-5
# )

# # Apply the position
# for i in range(6):
#     p.setJointMotorControl2(
#         bodyIndex=ur5,
#         jointIndex=i,
#         controlMode=p.POSITION_CONTROL,
#         targetPosition=joint_positions[i],
#         maxVelocity=2.0,
#         force=500,
#         positionGain=0.3
#     )

# # Wait for movement
# for _ in range(240):
#     p.stepSimulation()
#     time.sleep(1./240.)

# ee_state = p.getLinkState(ur5, ee_link)
# print(f"End-effector reached position: {ee_state[0]}")
# print(f"This is where the cube should be placed (approximately)\n")

# 5️⃣ Define pick and place positions
# Get the cube's current position (after settling)
cube_pos, _ = p.getBasePositionAndOrientation(cube)
print(f"Cube settled at: {cube_pos}")

# Pick position: slightly above the cube (approach from top)
pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.02]

# Place position: move it to another location on the table
place_pos = [0.55, -0.15, table_top_z + cube_size/2 + 0.001]

print(f"\nPick position: {pick_pos}")
print(f"Place position: {place_pos}")

# 6️⃣ Execute pick and place with collision avoidance
time.sleep(1)  # Pause before starting
success = pick_and_place_demo(
    robot_id=ur5,
    object_id=cube,
    pick_pos=pick_pos,
    place_pos=place_pos,
    obstacle_ids=[table, plane],  # Avoid colliding with table and ground
    hover_height=0.05  # Lower hover height for better reachability
)

if success:
    print("\n✓ Demo completed successfully!")
    print("The cube should now be at the new position.")
else:
    print("\n✗ Demo failed!")

# Keep simulation running
print("\nSimulation running. Close the window to exit.")
while True:
    p.stepSimulation()
    # time.sleep(1./240.)
