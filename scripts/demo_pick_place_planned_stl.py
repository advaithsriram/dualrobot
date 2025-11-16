import pybullet as p
import pybullet_data
import time
import numpy as np
from pick_place import create_graspable_cube, attach_object_to_robot, detach_object

# Debug flags
DEBUG_TRAJECTORY_PLANNING = True
USE_DETAILED_COLLISION_MESHES = True

# 1️⃣ Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

print("\n=== Collision-Free Trajectory Planning with STL Meshes ===")
print("This validates entire trajectories BEFORE execution")
print("If goal configuration causes collisions, it's detected upfront\n")

# 2️⃣ Load environment
plane = p.loadURDF("plane.urdf")

# Load table
table_height = 0.4
table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, table_height/2])
table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, table_height/2], 
                                    rgbaColor=[1, 1, 1, 1])
table = p.createMultiBody(baseMass=0, 
                        baseCollisionShapeIndex=table_collision,
                        baseVisualShapeIndex=table_visual,
                        basePosition=[0.5, 0, table_height/2])

# 3️⃣ Load UR5 robot with detailed collision meshes
ur5 = p.loadURDF(
    "../urdf/ur5.urdf", 
    basePosition=[0.5, 0, table_height + 0.02],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE | 
          p.URDF_USE_SELF_COLLISION |
          p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
)

print(f"UR5 loaded with {p.getNumJoints(ur5)} joints")
print("Using detailed STL mesh collision detection\n")

# Set initial joint positions (stable upright pose)
initial_positions = [0, -90, 90, 180, -90, 0]
initial_positions = [np.deg2rad(pos) for pos in initial_positions]
for i in range(6):
    p.resetJointState(ur5, i, initial_positions[i])

# Enable strong position control
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
cube_position = [0.1, 0.1, table_top_z + cube_size/2 + 0.001]
cube = create_graspable_cube(
    position=cube_position,
    size=cube_size,
    color=[1, 0, 0, 1],
    mass=0.05
)

print(f"Cube created at position: {cube_position}")
print(f"Robot base at (0.5, 0, {table_height + 0.02})\n")

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
print(f"Cube settled at: {cube_pos}\n")


# === Trajectory Planning Functions ===

def check_configuration_collision_free(robot_id, joint_positions, obstacle_ids, force_threshold=0.1):
    """
    Check if a specific joint configuration causes collisions.
    Uses current state without moving motors.
    
    Args:
        robot_id: Robot body ID
        joint_positions: Joint configuration to test
        obstacle_ids: List of obstacle IDs
        force_threshold: Minimum force to consider as collision
    
    Returns:
        tuple: (is_collision_free, collision_info)
    """
    # Save current state
    num_joints = p.getNumJoints(robot_id)
    saved_states = [p.getJointState(robot_id, i)[0] for i in range(num_joints)]
    
    # Temporarily set to test configuration (doesn't affect motors)
    for i in range(min(len(joint_positions), num_joints)):
        p.resetJointState(robot_id, i, joint_positions[i], targetVelocity=0)
    
    # Perform collision detection
    p.performCollisionDetection()
    
    collision_info = {"self_collision": False, "obstacle_collision": False, "details": []}
    
    # Check self-collision
    self_contacts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    if self_contacts is not None and len(self_contacts) > 0:
        for contact in self_contacts:
            if len(contact) > 9 and contact[9] > force_threshold:
                collision_info["self_collision"] = True
                link_a = contact[3]
                link_b = contact[4]
                force = contact[9]
                collision_info["details"].append(f"Self-collision: link{link_a} <-> link{link_b} (force={force:.3f}N)")
    
    # Check obstacle collisions
    for obstacle_id in obstacle_ids:
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)
        if contacts is not None and len(contacts) > 0:
            for contact in contacts:
                if len(contact) > 9 and contact[9] > force_threshold:
                    collision_info["obstacle_collision"] = True
                    link_a = contact[3]
                    force = contact[9]
                    collision_info["details"].append(f"Obstacle collision: link{link_a} (force={force:.3f}N)")
    
    # Restore original state
    for i in range(num_joints):
        p.resetJointState(robot_id, i, saved_states[i], targetVelocity=0)
    
    # Re-establish motor control at original positions
    for i in range(6):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=saved_states[i],
            force=5000,
            maxVelocity=2.0,
            positionGain=0.5
        )
    
    is_free = not (collision_info["self_collision"] or collision_info["obstacle_collision"])
    return is_free, collision_info


def plan_trajectory(robot_id, target_pos, target_orn, obstacle_ids, num_waypoints=50):
    """
    Plan and validate a complete trajectory before execution.
    
    Args:
        robot_id: Robot body ID
        target_pos: Target end-effector position [x, y, z]
        target_orn: Target orientation (quaternion)
        obstacle_ids: List of obstacle IDs to check
        num_waypoints: Number of waypoints in trajectory
    
    Returns:
        tuple: (success, trajectory, collision_waypoint_index)
            - success: True if entire path is collision-free
            - trajectory: List of joint configurations
            - collision_waypoint_index: Index of first collision (-1 if none)
    """
    if DEBUG_TRAJECTORY_PLANNING:
        print(f"    🔍 Planning trajectory to {target_pos}...")
    
    # Get current joint positions
    current_joints = [p.getJointState(robot_id, i)[0] for i in range(6)]
    
    # Calculate IK for target
    target_joints = p.calculateInverseKinematics(
        robot_id,
        ee_link,
        target_pos,
        target_orn,
        maxNumIterations=100,
        residualThreshold=1e-5
    )
    
    if target_joints is None:
        if DEBUG_TRAJECTORY_PLANNING:
            print(f"    ✗ IK failed for target position")
        return False, [], -1
    
    target_joints = list(target_joints[:6])
    
    # First check: Is the goal configuration itself collision-free?
    is_free, info = check_configuration_collision_free(robot_id, target_joints, obstacle_ids)
    if not is_free:
        if DEBUG_TRAJECTORY_PLANNING:
            print(f"    ✗ Goal configuration causes collision!")
            for detail in info["details"]:
                print(f"      - {detail}")
        return False, [], num_waypoints - 1  # Collision at goal
    
    # Generate trajectory waypoints
    trajectory = []
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)
        waypoint = [current_joints[j] + t * (target_joints[j] - current_joints[j]) for j in range(6)]
        trajectory.append(waypoint)
    
    # Validate entire trajectory
    if DEBUG_TRAJECTORY_PLANNING:
        print(f"    🔍 Validating {num_waypoints} waypoints...")
    
    for idx, waypoint in enumerate(trajectory):
        is_free, info = check_configuration_collision_free(robot_id, waypoint, obstacle_ids)
        if not is_free:
            if DEBUG_TRAJECTORY_PLANNING:
                print(f"    ✗ Collision detected at waypoint {idx}/{num_waypoints}!")
                for detail in info["details"]:
                    print(f"      - {detail}")
            return False, trajectory[:idx], idx  # Return partial safe trajectory
    
    if DEBUG_TRAJECTORY_PLANNING:
        print(f"    ✓ Trajectory validated - all {num_waypoints} waypoints collision-free!")
    
    return True, trajectory, -1


def execute_trajectory(robot_id, trajectory, speed_factor=1.0):
    """
    Execute a validated collision-free trajectory.
    
    Args:
        robot_id: Robot body ID
        trajectory: List of joint configurations (already validated)
        speed_factor: Speed multiplier
    
    Returns:
        bool: True if executed successfully
    """
    if not trajectory:
        return False
    
    for i, waypoint in enumerate(trajectory):
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
        
        # Step simulation
        for _ in range(10):
            p.stepSimulation()
            # time.sleep(1./240.)  # Faster execution since we know it's safe
    
    return True


def move_to_position_planned(robot_id, target_pos, target_orn=None, obstacle_ids=[], speed_factor=1.0):
    """
    Move robot to target position with pre-validated trajectory planning.
    Checks entire path BEFORE moving.
    
    Returns:
        bool: True if successfully moved, False if planning failed
    """
    if target_orn is None:
        target_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
    
    print(f"  Planning motion to {target_pos}...")
    
    # Plan trajectory (validates BEFORE execution)
    success, trajectory, collision_idx = plan_trajectory(
        robot_id, 
        target_pos, 
        target_orn, 
        obstacle_ids,
        num_waypoints=50
    )
    
    if not success:
        print(f"  ✗ Planning FAILED - collision would occur at waypoint {collision_idx}")
        print(f"  ⚠️  Cannot reach target without collision! Try different position.")
        return False
    
    # Execute the validated trajectory
    print(f"  ✓ Plan validated - executing motion...")
    execute_trajectory(robot_id, trajectory, speed_factor)
    
    # Verify reached target
    ee_state = p.getLinkState(robot_id, ee_link)
    current_pos = ee_state[0]
    distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
    
    if distance < 0.02:
        print(f"  ✓ Reached target successfully (error: {distance:.4f}m)")
        return True
    else:
        print(f"  ⚠ Close to target (error: {distance:.4f}m)")
        return distance < 0.05


# === Pick and Place Demo ===

print("=== Pick and Return Demo (PRE-VALIDATED TRAJECTORY PLANNING) ===")
print("Each motion is fully validated before execution\n")

hover_height = 0.05
pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2]]
hover_pick_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + hover_height]

# Step 1: Move to hover above cube
print("1. Moving to hover position above cube...")
if not move_to_position_planned(ur5, hover_pick_pos, obstacle_ids=[table, plane], speed_factor=1.0):
    print("✗ Failed to plan path to hover position!")
    print("⚠️  Try adjusting the cube or target position\n")
else:
    time.sleep(0.5)
    
    # Step 2: Move down to pick
    print("\n2. Moving down to pick up cube...")
    if not move_to_position_planned(ur5, pick_pos, obstacle_ids=[table, plane], speed_factor=0.5):
        print("✗ Failed to plan path to pick position!")
    else:
        time.sleep(0.5)
        
        # Step 3: Grasp the cube
        print("\n3. Grasping cube...")
        constraint_id = attach_object_to_robot(ur5, cube, ee_link)
        time.sleep(0.5)
        
        # Step 4: Lift up
        print("\n4. Lifting cube...")
        if not move_to_position_planned(ur5, hover_pick_pos, obstacle_ids=[table, plane], speed_factor=0.5):
            print("✗ Failed to plan lift path!")
        else:
            time.sleep(0.5)
            
            # Step 5: Return to initial position with cube
            print("\n5. Returning to initial position with cube...")
            if not move_to_position_planned(ur5, initial_ee_pos, target_orn=initial_ee_orn, 
                                          obstacle_ids=[table, plane], speed_factor=1.0):
                print("✗ Failed to plan return path!")
            else:
                print("\n✓✓✓ Successfully completed with pre-validated trajectory planning!")
                print("All motions were checked for collisions BEFORE execution\n")
                time.sleep(1.0)
                
                # Keep simulation running
                print("Simulation complete. Close window to exit.\n")
                
                while True:
                    p.stepSimulation()
                    time.sleep(1./60.)
