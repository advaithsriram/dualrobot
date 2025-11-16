import pybullet as p
import pybullet_data
import time
import numpy as np
from pick_place import create_graspable_cube, attach_object_to_robot, detach_object, move_to_position


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

# 3️⃣ Load UR5 robot on top of table (with 2cm buffer)
ur5 = p.loadURDF("../urdf/ur5.urdf", 
                 basePosition=[0.5, 0, table_height + 0.02],
                 baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                 useFixedBase=True)

print(f"UR5 loaded with {p.getNumJoints(ur5)} joints")

# Set initial joint positions (stable upright pose)
initial_positions = [0, -90, 90, 180, -90, 0]
#convert deg2rad
initial_positions = [np.deg2rad(pos) for pos in initial_positions]
for i in range(6):
    p.resetJointState(ur5, i, initial_positions[i])

# Enable strong position control to hold against gravity
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
cube_position = [0.8, -0.3, table_top_z + cube_size/2 + 0.001]
cube = create_graspable_cube(
    position=cube_position,
    size=cube_size,
    color=[1, 0, 0, 1],  # Red
    mass=0.05  # 50 grams
)

print(f"\nCube created at position: {cube_position}")
print(f"Robot base at (0.5, 0, {table_height + 0.02})")

# Let physics settle
for _ in range(100):
    p.stepSimulation()

# Get initial end-effector position and orientation
num_joints = p.getNumJoints(ur5)
ee_link = num_joints - 1
initial_ee_state = p.getLinkState(ur5, ee_link)
initial_ee_pos = initial_ee_state[0]
initial_ee_orn = initial_ee_state[1]  # Save initial orientation
print(f"Initial end-effector position: {initial_ee_pos}")
print(f"Initial end-effector orientation: {initial_ee_orn}")

# Get cube position after settling
cube_pos, _ = p.getBasePositionAndOrientation(cube)
print(f"Cube settled at: {cube_pos}")


# 5️⃣ Pick and return to initial position demo
print("\n=== Pick and Return to Initial Position Demo ===\n")




hover_height = 0.05
pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2]]
hover_pick_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + hover_height]

# Step 1: Move to hover above cube (pointing down)
print("1. Moving to hover position above cube...")
if not move_to_position(ur5, hover_pick_pos, obstacle_ids=[table, plane], speed_factor=1.0):
    print("✗ Failed to reach hover position!")
else:
    time.sleep(0.5)
    
    # Step 2: Move down to pick (pointing down)
    print("2. Moving down to pick up cube...")
    if not move_to_position(ur5, pick_pos, obstacle_ids=[table, plane], speed_factor=0.5):
        print("✗ Failed to reach pick position!")
    else:
        time.sleep(0.5)
        
        # Step 3: Grasp the cube
        print("3. Grasping cube...")
        constraint_id = attach_object_to_robot(ur5, cube, ee_link)
        time.sleep(0.5)
        
        # Step 4: Lift up (pointing down)
        print("4. Lifting cube...")
        if not move_to_position(ur5, hover_pick_pos, obstacle_ids=[table, plane], speed_factor=0.5):
            print("✗ Failed to lift!")
        else:
            time.sleep(0.5)
            
            # Step 5: Return to initial position with cube (use initial orientation)
            print("5. Returning to initial position with cube...")
            if not move_to_position(ur5, initial_ee_pos, target_orn=initial_ee_orn, obstacle_ids=[table, plane], speed_factor=1.0):
                print("✗ Failed to return to initial position!")
            else:
                print("\n✓ Successfully picked up cube and returned to initial position!")
                time.sleep(0.5)
                
                # Keep simulation running with slower update rate for visualization
                print("\nSimulation running. Close window to exit.")
                while True:
                    p.stepSimulation()
                    time.sleep(1./240.)  # 240 Hz update rate 

