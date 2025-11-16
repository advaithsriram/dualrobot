import pybullet as p
import numpy as np
import time

# Global debug flag to control trajectory visualization
DEBUG_DRAW_TRAJECTORY = True

def precompute_trajectory_waypoints(robot_id, trajectory_type='circle', radius=0.15, amplitude_y=0.15, amplitude_z=0.075, num_points=500):
    """
    Pre-compute all joint positions for a trajectory using IK once.
    This eliminates drift and is much faster than computing IK in real-time.
    
    Args:
        robot_id: PyBullet body ID of the robot
        trajectory_type: 'circle' or 'lissajous'
        radius: Radius for circle
        amplitude_y, amplitude_z: Amplitudes for Lissajous
        num_points: Number of waypoints in the trajectory
    
    Returns:
        List of joint positions (waypoints) for the complete trajectory
    """
    num_joints = p.getNumJoints(robot_id)
    end_effector_index = num_joints - 1
    
    # Get current end-effector position as center
    link_state = p.getLinkState(robot_id, end_effector_index)
    center_pos = link_state[0]
    
    print(f"Pre-computing {num_points} waypoints for {trajectory_type} trajectory...")
    start_time = time.time()
    
    waypoints = []
    
    for step in range(num_points):
        t = (step / num_points) * 2 * np.pi
        
        if trajectory_type == 'circle':
            # Circle on Y-Z plane
            x_offset = 0
            y_offset = radius * np.cos(t)
            z_offset = radius * np.sin(t)
        else:  # lissajous
            # Figure-8 on Y-Z plane
            x_offset = 0
            y_offset = amplitude_y * np.sin(t)
            z_offset = amplitude_z * np.sin(2 * t + np.pi/2)
        
        target_pos = [
            center_pos[0] + x_offset,
            center_pos[1] + y_offset,
            center_pos[2] + z_offset
        ]
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        # Compute IK once for this waypoint
        joint_positions = p.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Store only the 6 revolute joint positions
        waypoints.append(joint_positions[:6])
    
    elapsed = time.time() - start_time
    print(f"Pre-computation complete in {elapsed:.2f}s")
    
    return waypoints, center_pos


def execute_trajectory(robot_id, waypoints, draw_color=[0, 0, 1], center_pos=None, draw_trace=True, speed_factor=1.0):
    """
    Execute a pre-computed trajectory by following the waypoints.
    This is MUCH faster and has NO drift since we're just replaying joint positions.
    
    Args:
        robot_id: PyBullet body ID of the robot
        waypoints: List of joint positions computed by precompute_trajectory_waypoints
        draw_color: RGB color for trajectory visualization
        center_pos: Center position for trajectory (for visualization)
        draw_trace: Whether to draw the trajectory trace
        speed_factor: Speed multiplier (e.g., 2.0 = 2x faster, 0.5 = half speed)
    """
    num_joints = p.getNumJoints(robot_id)
    end_effector_index = num_joints - 1
    
    prev_pos = None
    
    # Skip waypoints based on speed factor to move faster
    step_size = max(1, int(speed_factor))
    
    for i in range(0, len(waypoints), step_size):
        joint_positions = waypoints[i]
        
        # Apply pre-computed joint positions with higher position gain for accuracy
        for joint_index in range(6):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[joint_index],
                force=500,
                maxVelocity=5.0 * speed_factor,  # Scale velocity with speed factor
                positionGain=0.3  # Higher gain = more accurate tracking (default 0.1)
            )
        
        p.stepSimulation()
        
        # Draw trajectory trace (only every few steps for performance)
        if draw_trace and DEBUG_DRAW_TRAJECTORY and i % max(1, step_size) == 0:
            link_state = p.getLinkState(robot_id, end_effector_index)
            current_pos = link_state[0]
            
            if prev_pos is not None:
                p.addUserDebugLine(prev_pos, current_pos, lineColorRGB=draw_color, lineWidth=2, lifeTime=0)
            
            prev_pos = current_pos

def move_circular_trajectory(robot_id, radius=0.15, draw_color=[0, 0, 1], center_pos=None, num_points=240):
    """
    Move the robot end-effector in a circular pattern on the vertical plane using inverse kinematics.
    Completes exactly one full circle (0 to 2π).
    
    Args:
        robot_id: PyBullet body ID of the robot
        radius: Radius of the circle
        draw_color: RGB color for trajectory line (default: blue)
        center_pos: Center position for the trajectory (if None, uses current position)
        num_points: Number of points to sample along the circle (higher = smoother)
    """
    # Get the end-effector link index (last link)
    num_joints = p.getNumJoints(robot_id)
    end_effector_index = num_joints - 1
    
    # Get current end-effector position as the center point if not provided
    if center_pos is None:
        link_state = p.getLinkState(robot_id, end_effector_index)
        center_pos = link_state[0]
    
    prev_pos = None  # Track previous position for drawing lines
    
    # Complete one full circle: t goes from 0 to 2π
    for step in range(num_points):
        t = (step / num_points) * 2 * np.pi  # 0 to 2π for one complete circle
        
        # Circular trajectory on vertical plane (Y-Z plane):
        # y(t) = R * cos(t)  - horizontal component
        # z(t) = R * sin(t)  - vertical component
        x_offset = 0  # No forward/back motion - stay on vertical plane
        y_offset = radius * np.cos(t)  # Horizontal circle motion
        z_offset = radius * np.sin(t)  # Vertical circle motion
        
        target_pos = [
            center_pos[0] + x_offset,
            center_pos[1] + y_offset,
            center_pos[2] + z_offset
        ]
        # Point end-effector toward the vertical plane (forward in +X direction)
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])  # Point forward, perpendicular to Y-Z plane
        
        # Inverse kinematics to get joint positions
        joint_positions = p.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Apply joint positions (only to revolute joints)
        for joint_index in range(6):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[joint_index],
                force=500
            )
        
        p.stepSimulation()
        
        # Draw the trajectory trace if debug flag is enabled
        if DEBUG_DRAW_TRAJECTORY:
            link_state = p.getLinkState(robot_id, end_effector_index)
            current_pos = link_state[0]
            
            if prev_pos is not None:
                # Draw a line between previous and current position
                p.addUserDebugLine(prev_pos, current_pos, lineColorRGB=draw_color, lineWidth=2, lifeTime=0)
            
            prev_pos = current_pos

def move_lissajous_curve(robot_id, amplitude_x=0.1, amplitude_y=0.1, amplitude_z=0.05, freq_ratio=2.0, draw_color=[1, 0, 0], center_pos=None, num_points=480):
    """
    Move the robot end-effector in a Lissajous curve pattern using inverse kinematics.
    Completes exactly one full figure-8 cycle (0 to 2π).
    
    Lissajous curve: A parametric curve formed by two perpendicular sinusoidal motions.
    - freq_ratio = 2.0 creates a figure-8 (infinity symbol ∞)
    - freq_ratio = 1.0 with phase shift creates a circle/ellipse
    
    Args:
        robot_id: PyBullet body ID of the robot
        amplitude_x, amplitude_y, amplitude_z: Size of the curve in each direction
        freq_ratio: Frequency ratio between x and y motions (2.0 = figure-8)
        draw_color: RGB color for trajectory line (default: red)
        center_pos: Center position for the trajectory (if None, uses current position)
        num_points: Number of points to sample along the curve (higher = smoother)
    """
    # Get the end-effector link index (last link)
    num_joints = p.getNumJoints(robot_id)
    end_effector_index = num_joints - 1
    
    # Get current end-effector position as the center point if not provided
    if center_pos is None:
        link_state = p.getLinkState(robot_id, end_effector_index)
        center_pos = link_state[0]
    
    prev_pos = None  # Track previous position for drawing lines
    
    # Complete one full figure-8: t goes from 0 to 2π
    for step in range(num_points):
        t = (step / num_points) * 2 * np.pi  # 0 to 2π for one complete cycle
        
        # Lissajous parametric equations on vertical plane (Y-Z plane):
        # For vertical figure-8: z oscillates 2x faster than y with 90° phase shift
        # y(t) = A * sin(t)            - horizontal on vertical plane (slow)
        # z(t) = B * sin(2*t + π/2)    - vertical motion (fast with phase shift)
        x_offset = 0  # No forward/back motion - stay on vertical plane
        y_offset = amplitude_y * np.sin(t)  # Side-to-side (horizontal, slower)
        z_offset = amplitude_z * np.sin(freq_ratio * t + np.pi/2)  # Up and down with phase shift
        
        target_pos = [
            center_pos[0] + x_offset,
            center_pos[1] + y_offset,
            center_pos[2] + z_offset
        ]
        # Point end-effector toward the vertical plane (forward in +X direction)
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])  # Point forward, perpendicular to Y-Z plane
        
        # Inverse kinematics to get joint positions
        joint_positions = p.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Apply joint positions (only to revolute joints)
        for joint_index in range(6):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[joint_index],
                force=500
            )
        
        p.stepSimulation()
        
        # Draw the trajectory trace if debug flag is enabled
        if DEBUG_DRAW_TRAJECTORY:
            link_state = p.getLinkState(robot_id, end_effector_index)
            current_pos = link_state[0]
            
            if prev_pos is not None:
                # Draw a line between previous and current position
                p.addUserDebugLine(prev_pos, current_pos, lineColorRGB=draw_color, lineWidth=2, lifeTime=0)
            
            prev_pos = current_pos

def alternate_trajectories(robot_id, num_cycles=5, diameter=0.3):
    """
    Alternate between circular and Lissajous trajectories for specified number of cycles.
    Both trajectories use the same center position for proper alignment.
    
    Args:
        robot_id: PyBullet body ID of the robot
        num_cycles: Number of times to alternate (each cycle = circle + lissajous)
        diameter: Diameter/width for both trajectories
    """
    radius = diameter / 2
    
    # Get the end-effector link index
    num_joints = p.getNumJoints(robot_id)
    end_effector_index = num_joints - 1
    
    # Get initial center position - both trajectories will use this same center
    link_state = p.getLinkState(robot_id, end_effector_index)
    center_pos = link_state[0]
    
    print(f"Trajectory center position: {center_pos}")
    
    for cycle in range(num_cycles):
        print(f"\nCycle {cycle + 1}/{num_cycles}")
        
        # Circular trajectory (blue) - completes one full circle
        print("  → Tracing circular trajectory (blue) - one full circle...")
        move_circular_trajectory(
            robot_id=robot_id,
            radius=radius,
            draw_color=[0, 0, 1],  # Blue
            center_pos=center_pos   # Use same center
        )
        
        # Lissajous trajectory (red) - completes one full figure-8
        print("  → Tracing Lissajous trajectory (red) - one full figure-8...")
        move_lissajous_curve(
            robot_id=robot_id,
            amplitude_x=0.0,
            amplitude_y=radius,      # Use radius for width
            amplitude_z=radius / 3 * 2,  # Half radius for height (flatter figure-8)
            freq_ratio=2.0,
            draw_color=[1, 0, 0],   # Red
            center_pos=center_pos    # Use same center
        )
