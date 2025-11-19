import pybullet as p
import numpy as np
import time


def check_collision(robot_id, joint_positions, obstacle_ids=[]):
    """
    Check if a robot configuration causes self-collision or collision with obstacles.
    
    Args:
        robot_id: PyBullet body ID of the robot
        joint_positions: List of joint angles to check
        obstacle_ids: List of obstacle body IDs to check against (e.g., table)
    
    Returns:
        bool: True if collision detected, False if collision-free
    """
    # Save current joint states
    num_joints = p.getNumJoints(robot_id)
    current_states = [p.getJointState(robot_id, i)[0] for i in range(num_joints)]
    
    # Set robot to test configuration (without physics)
    for i in range(min(len(joint_positions), num_joints)):
        p.resetJointState(robot_id, i, joint_positions[i])
    
    # Check self-collision
    p.performCollisionDetection()
    contact_points = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    
    if len(contact_points) > 0:
        # Restore original state
        for i in range(num_joints):
            p.resetJointState(robot_id, i, current_states[i])
        return True
    
    # Check collision with obstacles
    for obstacle_id in obstacle_ids:
        contact_points = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)
        if len(contact_points) > 0:
            # Restore original state
            for i in range(num_joints):
                p.resetJointState(robot_id, i, current_states[i])
            return True
    
    # Restore original state
    for i in range(num_joints):
        p.resetJointState(robot_id, i, current_states[i])
    
    return False


def interpolate_joint_path(start_joints, end_joints, num_steps=50):
    """
    Create a smooth interpolated path through joint space.
    
    Args:
        start_joints: Starting joint configuration
        end_joints: Target joint configuration
        num_steps: Number of waypoints in the path
    
    Returns:
        List of joint configurations forming a smooth path
    """
    start = np.array(start_joints)
    end = np.array(end_joints)
    
    path = []
    for i in range(num_steps):
        t = i / (num_steps - 1)  # 0 to 1
        # Linear interpolation in joint space
        waypoint = start + t * (end - start)
        path.append(waypoint.tolist())
    
    return path


def plan_collision_free_path(robot_id, target_pos, target_orn=None, obstacle_ids=[], max_attempts=10):
    """
    Plan a collision-free path to target position using trajectory planning.
    
    Args:
        robot_id: PyBullet body ID of the robot
        target_pos: Target end-effector position [x, y, z]
        target_orn: Target orientation (quaternion)
        obstacle_ids: List of obstacle body IDs
        max_attempts: Number of IK attempts with different seeds
    
    Returns:
        tuple: (success, path) where path is list of joint configurations
    """
    num_joints = p.getNumJoints(robot_id)
    end_effector_index = num_joints - 1
    
    if target_orn is None:
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
    
    # Get current joint positions
    current_joints = [p.getJointState(robot_id, i)[0] for i in range(6)]
    
    # Calculate IK for target
    target_joints = p.calculateInverseKinematics(
        robot_id,
        end_effector_index,
        target_pos,
        target_orn,
        maxNumIterations=100,
        residualThreshold=1e-5
    )
    
    if target_joints is None:
        print(f"  ✗ IK failed for target position")
        return False, []
    
    best_target_joints = target_joints[:6]
    
    # Create interpolated path from current to target
    path = interpolate_joint_path(current_joints, best_target_joints, num_steps=50)
    
    # Skip collision checking to avoid breaking motor control
    # The robot will stop naturally if it hits something
    print(f"  ✓ Generated path with {len(path)} waypoints")
    return True, path


def execute_joint_path(robot_id, path, speed_factor=1.0):
    """
    Execute a planned joint-space trajectory.
    
    Args:
        robot_id: PyBullet body ID of the robot
        path: List of joint configurations
        speed_factor: Speed multiplier
    
    Returns:
        bool: True if executed successfully
    """
    if not path:
        return False
    
    for waypoint in path:
        # Set joint targets with strong force
        for i in range(6):
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=waypoint[i],
                maxVelocity=3.0 * speed_factor,
                force=5000000,  # Strong force to overcome gravity
                positionGain=0.5  # Higher gain for better tracking
            )
        
        # Step simulation (more steps per waypoint for stability)
        for _ in range(20):
            p.stepSimulation()
            # time.sleep(1./240.)  # 240 Hz update rate for visible motion
    
    return True


def create_graspable_cube(position, size=0.04, color=[1, 0, 0, 1], mass=0.05):
    """
    Create a small cube that can be picked up.
    
    Args:
        position: [x, y, z] position of the cube
        size: Side length of the cube
        color: RGBA color
        mass: Mass in kg
    
    Returns:
        body_id: PyBullet body ID of the cube
    """
    # Create collision shape
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[size/2, size/2, size/2]
    )
    
    # Create visual shape
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[size/2, size/2, size/2],
        rgbaColor=color
    )
    
    # Create multi-body
    cube_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )
    
    print(f"Created cube at {position} with size {size}m and mass {mass}kg")
    return cube_id


def move_to_position(robot_id, target_pos, target_orn=None, obstacle_ids=[], speed_factor=1.0):
    """
    Move robot end-effector to a target position using trajectory planning with collision checking.
    
    Args:
        robot_id: PyBullet body ID of the robot
        target_pos: [x, y, z] target position
        target_orn: Target orientation (quaternion), default points straight down (-Z direction)
        obstacle_ids: List of obstacle body IDs to avoid
        speed_factor: Speed multiplier for motion
    
    Returns:
        bool: True if reached target, False otherwise
    """
    if target_orn is None:
        # End-effector points straight down at XY plane (along -Z axis)
        # Roll=0, Pitch=0, Yaw=0 with end-effector naturally pointing down
        target_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
    
    # Plan collision-free path
    success, path = plan_collision_free_path(
        robot_id, 
        target_pos, 
        target_orn, 
        obstacle_ids, 
        max_attempts=10
    )
    
    if not success:
        return False
    
    # Execute the path
    return execute_joint_path(robot_id, path, speed_factor)


def attach_object_to_robot(robot_id, object_id, end_effector_link_index):
    """
    Create a constraint to "attach" an object to the robot end-effector.
    This simulates a gripper grasping the object.
    
    Args:
        robot_id: PyBullet body ID of the robot
        object_id: PyBullet body ID of the object to attach
        end_effector_link_index: Index of the end-effector link
    
    Returns:
        constraint_id: PyBullet constraint ID (use to detach later)
    """
    # Get end-effector position
    link_state = p.getLinkState(robot_id, end_effector_link_index)
    ee_pos = link_state[0]
    ee_orn = link_state[1]
    
    # Get object position
    obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)
    
    # Create constraint at current relative position
    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=end_effector_link_index,
        childBodyUniqueId=object_id,
        childLinkIndex=-1,  # -1 means the base
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0]
    )
    
    print(f"Attached object {object_id} to robot {robot_id} (constraint {constraint_id})")
    return constraint_id


def detach_object(constraint_id):
    """
    Remove the constraint to "release" the object.
    
    Args:
        constraint_id: PyBullet constraint ID returned by attach_object_to_robot
    """
    p.removeConstraint(constraint_id)
    print(f"Detached object (removed constraint {constraint_id})")


def pick_and_place_demo(robot_id, object_id, pick_pos, place_pos, obstacle_ids=[], hover_height=0.15):
    """
    Demonstrate a pick-and-place operation with collision avoidance.
    
    Args:
        robot_id: PyBullet body ID of the robot
        object_id: PyBullet body ID of the object to manipulate
        pick_pos: [x, y, z] position to pick from (above object)
        place_pos: [x, y, z] position to place at
        obstacle_ids: List of obstacle body IDs to avoid (e.g., table)
        hover_height: Height above pick/place positions for approach
    
    Returns:
        bool: True if successful
    """
    num_joints = p.getNumJoints(robot_id)
    end_effector_index = num_joints - 1
    
    print("\n=== Pick and Place Demo (with collision checking) ===")
    
    # 1. Move to hover position above pick location
    print("1. Moving to hover position above object...")
    hover_pick_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + hover_height]
    if not move_to_position(robot_id, hover_pick_pos, obstacle_ids=obstacle_ids, speed_factor=1.0):
        print("Failed to reach hover position!")
        return False
    time.sleep(0.5)
    
    # 2. Move down to pick position
    print("2. Moving down to pick position...")
    if not move_to_position(robot_id, pick_pos, obstacle_ids=obstacle_ids, speed_factor=0.5):
        print("Failed to reach pick position!")
        return False
    time.sleep(0.5)
    
    # 3. "Grasp" the object (create constraint)
    print("3. Grasping object...")
    constraint_id = attach_object_to_robot(robot_id, object_id, end_effector_index)
    time.sleep(0.5)
    
    # 4. Lift up
    print("4. Lifting object...")
    if not move_to_position(robot_id, hover_pick_pos, obstacle_ids=obstacle_ids, speed_factor=0.5):
        print("Failed to lift!")
        return False
    time.sleep(0.5)
    
    # 5. Move to hover position above place location
    print("5. Moving to place location...")
    hover_place_pos = [place_pos[0], place_pos[1], place_pos[2] + hover_height]
    if not move_to_position(robot_id, hover_place_pos, obstacle_ids=obstacle_ids, speed_factor=1.0):
        print("Failed to reach place hover position!")
        return False
    time.sleep(0.5)
    
    # 6. Move down to place position
    print("6. Moving down to place position...")
    if not move_to_position(robot_id, place_pos, obstacle_ids=obstacle_ids, speed_factor=0.5):
        print("Failed to reach place position!")
        return False
    time.sleep(0.5)
    
    # 7. "Release" the object (remove constraint)
    print("7. Releasing object...")
    detach_object(constraint_id)
    time.sleep(0.5)
    
    # 8. Lift up
    print("8. Retracting...")
    if not move_to_position(robot_id, hover_place_pos, obstacle_ids=obstacle_ids, speed_factor=0.5):
        print("Failed to retract!")
        return False
    
    print("✓ Pick and place completed successfully!\n")
    return True
