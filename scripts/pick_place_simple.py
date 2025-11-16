import pybullet as p
import numpy as np
import time


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
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[size/2, size/2, size/2]
    )
    
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[size/2, size/2, size/2],
        rgbaColor=color
    )
    
    cube_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )
    
    print(f"Created cube at {position} with size {size}m and mass {mass}kg")
    return cube_id


def move_to_position_ik(robot_id, target_pos, target_orn=None, max_iterations=500, tolerance=0.01):
    """
    Move robot end-effector to target using continuous IK (like trajectory code).
    
    Args:
        robot_id: PyBullet body ID of the robot
        target_pos: [x, y, z] target position
        target_orn: Target orientation (quaternion)
        max_iterations: Maximum simulation steps
        tolerance: Distance tolerance in meters
    
    Returns:
        bool: True if reached target
    """
    num_joints = p.getNumJoints(robot_id)
    end_effector_index = num_joints - 1
    
    if target_orn is None:
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
    
    print(f"  Moving to {target_pos}...")
    
    for iteration in range(max_iterations):
        # Compute IK at every step (continuous IK like your trajectory code)
        joint_positions = p.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5,
            # UR5 joint limits: all joints can rotate ±360 degrees (±2π radians)
            lowerLimits=[-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi],
            upperLimits=[2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi],
            jointRanges=[4*np.pi, 4*np.pi, 4*np.pi, 4*np.pi, 4*np.pi, 4*np.pi]
        )
        
        # Set motor control for all joints
        for i in range(6):
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                maxVelocity=2.0,
                force=5000,
                positionGain=0.5
            )
        
        # Step simulation
        p.stepSimulation()
        time.sleep(1./240.)
        
        # Check if reached target (every 10 iterations for efficiency)
        if iteration % 10 == 0:
            link_state = p.getLinkState(robot_id, end_effector_index)
            current_pos = link_state[0]
            distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
            
            if distance < tolerance:
                print(f"  ✓ Reached target (distance: {distance:.4f}m, iterations: {iteration})")
                return True
    
    # Check final distance
    link_state = p.getLinkState(robot_id, end_effector_index)
    current_pos = link_state[0]
    distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
    print(f"  ⚠ Timeout. Final distance: {distance:.4f}m")
    
    # Accept if close enough
    if distance < tolerance * 2:
        return True
    
    return False


def attach_object_to_robot(robot_id, object_id, end_effector_link_index):
    """
    Create a constraint to "attach" an object to the robot end-effector.
    
    Args:
        robot_id: PyBullet body ID of the robot
        object_id: PyBullet body ID of the object to attach
        end_effector_link_index: Index of the end-effector link
    
    Returns:
        constraint_id: PyBullet constraint ID
    """
    link_state = p.getLinkState(robot_id, end_effector_link_index)
    ee_pos = link_state[0]
    ee_orn = link_state[1]
    
    obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)
    
    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=end_effector_link_index,
        childBodyUniqueId=object_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0]
    )
    
    print(f"Attached object {object_id} to robot")
    return constraint_id


def detach_object(constraint_id):
    """
    Remove the constraint to "release" the object.
    
    Args:
        constraint_id: PyBullet constraint ID
    """
    p.removeConstraint(constraint_id)
    print(f"Detached object")


def pick_and_place_demo(robot_id, object_id, pick_pos, place_pos, hover_height=0.05):
    """
    Demonstrate pick-and-place using simple continuous IK.
    
    Args:
        robot_id: PyBullet body ID of the robot
        object_id: PyBullet body ID of the object
        pick_pos: [x, y, z] pick position
        place_pos: [x, y, z] place position
        hover_height: Hover height above positions
    
    Returns:
        bool: True if successful
    """
    num_joints = p.getNumJoints(robot_id)
    end_effector_index = num_joints - 1
    
    print("\n=== Pick and Place Demo (Simple IK) ===")
    
    # 1. Hover above pick
    print("1. Moving to hover above object...")
    hover_pick_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + hover_height]
    if not move_to_position_ik(robot_id, hover_pick_pos, max_iterations=500):
        print("Failed to reach hover position!")
        return False
    time.sleep(0.5)
    
    # 2. Move down to pick
    print("2. Moving down to pick...")
    if not move_to_position_ik(robot_id, pick_pos, max_iterations=300):
        print("Failed to reach pick position!")
        return False
    time.sleep(0.5)
    
    # 3. Grasp
    print("3. Grasping object...")
    constraint_id = attach_object_to_robot(robot_id, object_id, end_effector_index)
    time.sleep(0.5)
    
    # 4. Lift
    print("4. Lifting object...")
    if not move_to_position_ik(robot_id, hover_pick_pos, max_iterations=300):
        print("Failed to lift!")
        return False
    time.sleep(0.5)
    
    # 5. Move to place hover
    print("5. Moving to place location...")
    hover_place_pos = [place_pos[0], place_pos[1], place_pos[2] + hover_height]
    if not move_to_position_ik(robot_id, hover_place_pos, max_iterations=500):
        print("Failed to reach place hover!")
        return False
    time.sleep(0.5)
    
    # 6. Move down to place
    print("6. Moving down to place...")
    if not move_to_position_ik(robot_id, place_pos, max_iterations=300):
        print("Failed to reach place position!")
        return False
    time.sleep(0.5)
    
    # 7. Release
    print("7. Releasing object...")
    detach_object(constraint_id)
    time.sleep(0.5)
    
    # 8. Retract
    print("8. Retracting...")
    if not move_to_position_ik(robot_id, hover_place_pos, max_iterations=300):
        print("Failed to retract!")
        return False
    
    print("✓ Pick and place completed successfully!\n")
    return True
