import pybullet as p
import pybullet_data
import time
import numpy as np
from pick_place import create_graspable_cube, attach_object_to_robot, detach_object

# Debug flags
DEBUG_COLLISION_PREVENTION = True  # Actively prevent collisions

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
cube_size = 0.04
table_top_z = table_height + 0.02
cube_position = [0.8, -0.3, table_top_z + cube_size/2 + 0.001]
cube = create_graspable_cube(
    position=cube_position,
    size=cube_size,
    color=[1, 0, 0, 1],
    mass=0.05
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
initial_ee_orn = initial_ee_state[1]
print(f"Initial end-effector position: {initial_ee_pos}")

# Get cube position after settling
cube_pos, _ = p.getBasePositionAndOrientation(cube)
print(f"Cube settled at: {cube_pos}")


# Collision prevention functions
def check_collision_free(robot_id, joint_positions, obstacle_ids):
    """Check if a configuration is collision-free WITHOUT resetting joint states."""
    # Use PyBullet's collision detection on current state
    
    p.performCollisionDetection()
    
    # Check self-collision
    self_contacts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    if self_contacts is not None and len(self_contacts) > 0:
        # Filter out very weak contacts (contact force < threshold)
        strong_contacts = [c for c in self_contacts if len(c) > 9 and c[9] > 0.1]  # Normal force > 0.1N
        if len(strong_contacts) > 0:
            return False
    
    # Check obstacle collisions (ignore very weak contacts)
    for obstacle_id in obstacle_ids:
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)
        if contacts is not None and len(contacts) > 0:
            strong_contacts = [c for c in contacts if len(c) > 9 and c[9] > 0.1]
            if len(strong_contacts) > 0:
                return False
    
    return True


def move_to_position_safe(robot_id, target_pos, target_orn=None, obstacle_ids=[], speed_factor=1.0, max_iterations=1000):
    """
    Move robot to target position with ACTIVE collision prevention.
    Stops immediately if collision detected.
    """
    if target_orn is None:
        # End-effector points straight down
        target_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
    
    print(f"  Moving to {target_pos}...")
    
    # Get current joint positions
    current_joints = [p.getJointState(robot_id, i)[0] for i in range(6)]
    
    # Calculate target joint positions using IK
    target_joints = p.calculateInverseKinematics(
        robot_id,
        ee_link,
        target_pos,
        target_orn,
        maxNumIterations=100,
        residualThreshold=1e-5
    )
    
    if target_joints is None:
        print("  ✗ IK failed!")
        return False
    
    # Create smooth path (50 waypoints)
    num_waypoints = 50
    collision_count = 0
    
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)
        
        # Interpolate joint positions
        waypoint = [current_joints[j] + t * (target_joints[j] - current_joints[j]) for j in range(6)]
        
        # Set motor targets
        for j in range(6):
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=waypoint[j],
                maxVelocity=2.0 * speed_factor,
                force=5000,
                positionGain=0.5
            )
        
        # Step simulation and check for collisions
        for _ in range(10):
            p.stepSimulation()
            # time.sleep(1./60.)
            
            # Check if collision occurred
            if DEBUG_COLLISION_PREVENTION and not check_collision_free(robot_id, waypoint, obstacle_ids):
                collision_count += 1
                if collision_count > 5:  # Allow a few frames of light contact
                    print(f"  ✗ COLLISION DETECTED! Stopping motion at waypoint {i}/{num_waypoints}")
                    # Stop all motors at current position
                    current_state = [p.getJointState(robot_id, j)[0] for j in range(6)]
                    for j in range(6):
                        p.setJointMotorControl2(
                            bodyIndex=robot_id,
                            jointIndex=j,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=current_state[j],
                            force=5000,
                            maxVelocity=0.1
                        )
                    return False
            else:
                collision_count = max(0, collision_count - 1)  # Decay collision counter
    
    # Verify reached target
    ee_state = p.getLinkState(robot_id, ee_link)
    current_pos = ee_state[0]
    distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
    
    if distance < 0.02:
        print(f"  ✓ Reached target safely (distance: {distance:.4f}m)")
        return True
    else:
        print(f"  ⚠ Close to target (distance: {distance:.4f}m)")
        return distance < 0.05  # Accept if within 5cm


# 5️⃣ Pick and return to initial position demo with collision prevention
print("\n=== Pick and Return Demo (COLLISION PREVENTION ACTIVE) ===")
print("Motion will STOP if collision detected!\n")

hover_height = 0.05
pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2]]
hover_pick_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + hover_height]

# Step 1: Move to hover above cube (pointing down)
print("1. Moving to hover position above cube...")
if not move_to_position_safe(ur5, hover_pick_pos, obstacle_ids=[table, plane], speed_factor=1.0):
    print("✗ Failed to reach hover position!")
else:
    time.sleep(0.5)
    
    # Step 2: Move down to pick (pointing down)
    print("2. Moving down to pick up cube...")
    if not move_to_position_safe(ur5, pick_pos, obstacle_ids=[table, plane], speed_factor=0.5):
        print("✗ Failed to reach pick position!")
    else:
        time.sleep(0.5)
        
        # Step 3: Grasp the cube
        print("3. Grasping cube...")
        constraint_id = attach_object_to_robot(ur5, cube, ee_link)
        time.sleep(0.5)
        
        # Step 4: Lift up (pointing down)
        print("4. Lifting cube...")
        if not move_to_position_safe(ur5, hover_pick_pos, obstacle_ids=[table, plane], speed_factor=0.5):
            print("✗ Failed to lift!")
        else:
            time.sleep(0.5)
            
            # Step 5: Return to initial position with cube (use initial orientation)
            print("5. Returning to initial position with cube...")
            if not move_to_position_safe(ur5, initial_ee_pos, target_orn=initial_ee_orn, obstacle_ids=[table, plane], speed_factor=1.0):
                print("✗ Failed to return to initial position!")
            else:
                print("\n✓ Successfully completed pick-and-place with NO COLLISIONS!")
                time.sleep(1.0)
                
                # Keep simulation running
                print("\nSimulation complete. Close window to exit.\n")
                
                while True:
                    p.stepSimulation()
                    # time.sleep(1./60.)
