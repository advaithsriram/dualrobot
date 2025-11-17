"""
Franka Panda Robot (Robot B) - Vision-Based Tracking

This script loads and controls the Franka Panda robot with a virtual camera
mounted on the end-effector. The camera will be used to track objects from
the UR5 robot (robotA) using visual information.

Flow:
1. Setup environment (table, Franka robot with camera)
2. Initialize robot to home configuration
3. Attach virtual camera to end-effector
4. Display camera feed in real-time
5. Once working, will track UR5's object using only visual information
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Environment parameters
TABLE_HEIGHT = 0.4  # Height of table in meters

# Franka Panda initial joint configuration (radians)
# Joint order: [base, shoulder, elbow, forearm_roll, wrist1, wrist2, wrist3]
INITIAL_JOINT_ANGLES = [0, -45, 0, -135, 0, 180, 90]  # degrees, 7 DOF
#convert to radians
INITIAL_JOINT_ANGLES = [np.deg2rad(angle) for angle in INITIAL_JOINT_ANGLES]

# Camera parameters
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FOV = 60  # Field of view in degrees
CAMERA_NEAR = 0.01
CAMERA_FAR = 5.0

# Debug flags
DEBUG_COLLISION_PREVENTION = True
USE_MESH_COLLISION = False
DEBUG_BOOL = False

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_franka_robot(physics_client, base_position=[0.5, 0, 0.42], base_orientation=[0, 0, np.pi]):
    """
    Setup Franka Panda robot with table and camera in an existing PyBullet environment.
    
    Args:
        physics_client: Existing PyBullet physics client ID
        base_position: [x, y, z] position for robot base
        base_orientation: [roll, pitch, yaw] orientation in radians
    
    Returns:
        tuple: (table_id, panda_id, ee_link_index)
    """
    
    print("\n" + "="*70)
    print("FRANKA PANDA ROBOT (ROBOT B) - SETUP")
    print("="*70)
    print("Loading Franka Panda with collision meshes\n")
    
    # Create table (white box) at specified position
    table_collision = p.createCollisionShape(
        p.GEOM_BOX, 
        halfExtents=[0.5, 0.5, TABLE_HEIGHT/2],
        physicsClientId=physics_client
    )
    table_visual = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=[0.5, 0.5, TABLE_HEIGHT/2], 
        rgbaColor=[1, 1, 1, 1],
        physicsClientId=physics_client
    )
    table_base_pos = [base_position[0], base_position[1], TABLE_HEIGHT/2]
    table = p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=table_collision,
        baseVisualShapeIndex=table_visual,
        basePosition=table_base_pos,
        physicsClientId=physics_client
    )
    
    # Load Franka Panda robot (self-collision disabled for performance)
    panda = p.loadURDF(
        "../urdf/panda.urdf", 
        basePosition=base_position,
        baseOrientation=p.getQuaternionFromEuler(base_orientation),
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=physics_client
    )
    
    # Print robot info
    num_joints = p.getNumJoints(panda, physicsClientId=physics_client)
    print(f"✓ Franka Panda loaded")
    print(f"  - Table position: {table_base_pos}")
    print(f"  - Robot base: {base_position}")
    print(f"  - Robot orientation: {[round(np.rad2deg(a), 1) for a in base_orientation]} (degrees)")
    print(f"  - Number of joints: {num_joints}")
    print(f"  - Self-collision checking disabled (performance optimized)\n")
    
    # Initialize robot to home position
    ee_link = initialize_robot(panda, physics_client)
    
    return table, panda, ee_link


def initialize_robot(robot_id, physics_client):
    """Set robot to initial joint configuration with strong motor control."""
    
    num_joints = p.getNumJoints(robot_id, physicsClientId=physics_client)
    
    # Find controllable joints (revolute joints for Panda arm)
    controllable_joints = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i, physicsClientId=physics_client)
        joint_type = joint_info[2]
        if joint_type == p.JOINT_REVOLUTE:
            controllable_joints.append(i)
    
    print(f"Controllable joints: {controllable_joints}")
    print(f"Initial positions: {[round(np.rad2deg(a), 1) for a in INITIAL_JOINT_ANGLES]} (degrees)\n")
    
    # Set joint states for the 7 DOF arm
    for i, joint_idx in enumerate(controllable_joints[:7]):
        if i < len(INITIAL_JOINT_ANGLES):
            p.resetJointState(robot_id, joint_idx, INITIAL_JOINT_ANGLES[i], physicsClientId=physics_client)
    
    # Enable strong position control
    for i, joint_idx in enumerate(controllable_joints[:7]):
        if i < len(INITIAL_JOINT_ANGLES):
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=INITIAL_JOINT_ANGLES[i],
                force=5000,
                maxVelocity=10.0,
                positionGain=0.5,
                physicsClientId=physics_client
            )
    
    # Let physics settle
    for _ in range(100):
        p.stepSimulation(physicsClientId=physics_client)
    
    # Find end-effector link (usually the last link before gripper)
    ee_link = 7  # For Panda, link 7 is typically the flange
    ee_state = p.getLinkState(robot_id, ee_link, physicsClientId=physics_client)
    
    print(f"✓ Robot initialized")
    print(f"  - Initial joint config: {[round(np.rad2deg(a), 1) for a in INITIAL_JOINT_ANGLES]} (degrees)")
    print(f"  - End-effector link: {ee_link}")
    print(f"  - End-effector position: [{ee_state[0][0]:.3f}, {ee_state[0][1]:.3f}, {ee_state[0][2]:.3f}]\n")
    
    return ee_link


def get_camera_image(robot_id, ee_link):
    """Capture image from camera attached to end-effector."""
    
    # Get end-effector state
    ee_state = p.getLinkState(robot_id, ee_link)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    # Convert quaternion to rotation matrix to get camera orientation
    rot_matrix = p.getMatrixFromQuaternion(ee_orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    
    # Camera points along the end-effector's local Z-axis (forward)
    # Adjust offset to position camera at end-effector tip
    camera_forward = rot_matrix.dot([0, 0, 1])
    camera_up = rot_matrix.dot([0, -1, 0])
    
    # Position camera forward from end-effector (far enough to not see robot)
    camera_offset = 0.15  # 15cm offset to avoid seeing robot parts
    camera_pos = np.array(ee_pos) + camera_forward * camera_offset
    target_pos = camera_pos + camera_forward * 1.0  # Look 1m ahead
    
    if DEBUG_BOOL:
        # Draw camera coordinate frame (X, Y, Z axes)
        line_length = 0.5  # 50cm lines for each axis
        
        # X-axis (red) - camera right
        camera_right = rot_matrix.dot([1, 0, 0])
        line_end_x = camera_pos + camera_right * line_length
        p.addUserDebugLine(
            lineFromXYZ=camera_pos,
            lineToXYZ=line_end_x,
            lineColorRGB=[1, 0, 0],  # Red for X
            lineWidth=3,
            lifeTime=0.5
        )
        
        # Y-axis (green) - camera up (negated for camera convention)
        camera_up_axis = rot_matrix.dot([0, -1, 0])
        line_end_y = camera_pos + camera_up_axis * line_length
        p.addUserDebugLine(
            lineFromXYZ=camera_pos,
            lineToXYZ=line_end_y,
            lineColorRGB=[0, 1, 0],  # Green for Y
            lineWidth=3,
            lifeTime=0.5
        )
        
        # Z-axis (blue) - camera forward (viewing direction)
        line_end_z = camera_pos + camera_forward * line_length
        p.addUserDebugLine(
            lineFromXYZ=camera_pos,
            lineToXYZ=line_end_z,
            lineColorRGB=[0, 0, 1],  # Blue for Z
            lineWidth=3,
            lifeTime=0.5
        )

    # Compute view matrix
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=camera_up
    )
    
    # Compute projection matrix
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=CAMERA_FOV,
        aspect=CAMERA_WIDTH / CAMERA_HEIGHT,
        nearVal=CAMERA_NEAR,
        farVal=CAMERA_FAR
    )
    
    # Get camera image (only RGB needed, disable depth/segmentation preview)
    _, _, rgb_img, _, _ = p.getCameraImage(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_NO_SEGMENTATION_MASK  # Disable segmentation computation
    )
    
    # Convert to numpy array (RGB only)
    rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 4)
    rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
    
    return rgb_array


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function for standalone testing."""
    
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Hide depth and segmentation windows (only show RGB)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    
    # Load ground plane
    plane = p.loadURDF("plane.urdf")
    
    # Setup Franka robot
    table, panda, ee_link = setup_franka_robot(
        physics_client,
        base_position=[0.5, 0, TABLE_HEIGHT + 0.02],
        base_orientation=[0, 0, 0]
    )
    
    print("="*70)
    print("FRANKA PANDA WITH CAMERA READY")
    print("="*70)
    print("Environment loaded successfully.")
    print("Robot is in home position (stationary).")
    print("Virtual camera attached to end-effector.")
    print(f"Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"Camera FOV: {CAMERA_FOV}°")
    print("\nCamera view displayed in PyBullet's Synthetic Camera panels.")
    print("Press Ctrl+C to exit.\n")
    
    # Keep simulation running with camera feed
    frame_count = 0
    try:
        while True:
            p.stepSimulation()
            
            # Update camera image every few frames
            if frame_count % 2 == 0:
                get_camera_image(panda, ee_link)
            
            frame_count += 1
            # time.sleep(1./240.)
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    # Disconnect PyBullet
    p.disconnect()


if __name__ == "__main__":
    main()
