"""
Main entry point for dual robot system.

Coordinates:
- Robot A (UR5): Pick-and-place with 3D trajectory execution
- Robot B (Franka Panda): Vision-based tracking with camera

Robots are positioned 1.5m apart on the X-axis to avoid collision.
Franka is rotated 180° to face the UR5.
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import multiprocessing as mp
import os

# Import robot modules
import robotB
import vision_processor

# ============================================================================
# ROBOT CONTROLLERS (Independent control policies)
# ============================================================================

class RobotAController:
    """Controller for UR5 robot - handles trajectory execution"""
    
    def __init__(self, robot_id, ee_link, waypoints_circle, waypoints_lissajous, 
                 duration_circle, duration_lissajous):
        self.robot_id = robot_id
        self.ee_link = ee_link
        self.waypoints_circle = waypoints_circle
        self.waypoints_lissajous = waypoints_lissajous
        self.duration_circle = duration_circle
        self.duration_lissajous = duration_lissajous
        
        # State tracking
        self.current_trajectory = 'circle'
        self.waypoint_index = 0
        self.prev_pos = None
        self.cycle_count = 0
        
        # Timing
        self.sim_hz = 240.0
        self.frame_counter = 0
        
    def get_current_waypoints(self):
        """Get current trajectory waypoints and parameters"""
        if self.current_trajectory == 'circle' and self.waypoints_circle:
            return self.waypoints_circle, self.duration_circle, "circular trajectory", [0, 0, 1]
        elif self.current_trajectory == 'lissajous' and self.waypoints_lissajous:
            return self.waypoints_lissajous, self.duration_lissajous, "lissajous trajectory (figure-8)", [1, 0, 0]
        return None, None, None, None
    
    def control_step(self):
        """Execute one control step for UR5"""
        waypoints, duration, traj_name, color = self.get_current_waypoints()
        
        if waypoints is None:
            return
        
        # Check if trajectory is complete
        if self.waypoint_index >= len(waypoints):
            print(f"✓ {traj_name} completed")
            self.waypoint_index = 0
            self.prev_pos = None
            
            # Switch trajectories
            if self.current_trajectory == 'circle':
                self.current_trajectory = 'lissajous'
            else:
                self.current_trajectory = 'circle'
                self.cycle_count += 1
            return
        
        # Get current waypoint
        waypoint = waypoints[self.waypoint_index]
        
        # Set joint targets (optimized force/gain values)
        for j in range(6):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=waypoint[j],
                maxVelocity=10.0,
                force=500,
                positionGain=0.3
            )
        
        # Advance waypoint at proper timing
        time_per_waypoint = duration / len(waypoints)
        frames_per_waypoint = int(time_per_waypoint * self.sim_hz)
        
        if self.frame_counter % frames_per_waypoint == 0 and self.frame_counter > 0:
            # Store data for plotting (only query link state when needed)
            import robotA
            if robotA.PLOT_GRAPHS and ("circular" in traj_name or "lissajous" in traj_name):
                ee_state = p.getLinkState(self.robot_id, self.ee_link)
                current_pos = ee_state[0]
                robotA.trajectory_data.append([
                    current_pos[0], current_pos[1], current_pos[2]
                ])
            
            # Draw trajectory trace every 3 waypoints (reduce debug line overhead)
            if self.waypoint_index % 3 == 0:
                ee_state = p.getLinkState(self.robot_id, self.ee_link)
                current_pos = ee_state[0]
                if self.prev_pos is not None:
                    p.addUserDebugLine(self.prev_pos, current_pos, lineColorRGB=color, lineWidth=2, lifeTime=5.0)
                self.prev_pos = current_pos
            
            self.waypoint_index += 1
        
        self.frame_counter += 1


class RobotBController:
    """Controller for Franka Panda robot - handles vision-based tracking"""
    
    def __init__(self, robot_id, ee_link, image_queue=None, result_queue=None):
        self.robot_id = robot_id
        self.ee_link = ee_link
        self.frame_counter = 0
        
        # Vision processing queues
        self.image_queue = image_queue
        self.result_queue = result_queue
        self.latest_detection = None
        
        # Control parameters
        self.camera_width = robotB.CAMERA_WIDTH
        self.camera_height = robotB.CAMERA_HEIGHT
        
        # Cartesian visual servoing gains (pixels to meters mapping)
        # These map pixel errors to Cartesian displacement
        self.pixel_to_meter_x = 0.0006  # ~0.3mm per pixel horizontal
        self.pixel_to_meter_y = 0.0006  # ~0.3mm per pixel vertical

        self.pixel_to_meter_x_d = 0.0005 # D-Gain for X 
        self.pixel_to_meter_y_d = 0.0005 # D-Gain for Y

        self.last_error_x_pixels = 0.0
        self.last_error_y_pixels = 0.0
        
        # Depth control based on depth camera
        self.target_depth = None  # Will be set from first detection
        self.depth_to_meter_z = 0.12 # Gain for depth control (reduced for smoother tracking)
        self.depth_to_meter_z_d = 0.02 # D-Gain for Depth
        self.last_error_depth = 0.0
        
        # Deadband and filtering parameters
        self.pixel_deadband = 5.0  # Don't move if error < 5 pixels
        # self.area_deadband = 20.0  # Don't move in Z if area error < 20 px²
        self.filter_alpha = 0.5  # Low-pass filter: 0=no filter, 1=no smoothing
        self.filtered_error_x = 0.0
        self.filtered_error_y = 0.0
        self.filtered_error_depth = 0.0
        
        # Get controllable joints
        self.controllable_joints = []
        num_joints = p.getNumJoints(robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                self.controllable_joints.append(i)
        
        # Get initial end-effector position and orientation
        ee_state = p.getLinkState(robot_id, ee_link)
        self.target_ee_pos = list(ee_state[0])
        self.initial_ee_orn = ee_state[1]  # Store initial orientation to prevent drift
        
        # Trajectory data collection for plotting
        self.trajectory_data = []
    
    def control_step(self):
        """Execute one control step for Franka - update camera and track object"""
        
        # Update camera and send to vision processor every 8 frames (30 Hz)
        if self.frame_counter % 8 == 0:
            rgb_image, depth_array = robotB.get_camera_image(self.robot_id, self.ee_link)

            # Send both RGB and depth to vision processor (non-blocking)
            if self.image_queue is not None:
                try:
                    self.image_queue.put_nowait((rgb_image, depth_array))
                except:
                    pass  # Queue full, skip this frame
        
        # Get latest detection result (non-blocking)
        if self.result_queue is not None:
            try:
                while not self.result_queue.empty():
                    self.latest_detection = self.result_queue.get_nowait()
            except:
                pass
        
        # Cartesian visual servoing control
        if self.latest_detection is not None and self.latest_detection['detected']:
            pixel_x = self.latest_detection['pixel_x']
            pixel_y = self.latest_detection['pixel_y']
            # area = self.latest_detection['area']
            depth = self.latest_detection.get('depth', None)

            # Set target depth from first detection (desired distance)
            if self.target_depth is None and depth is not None:
                self.target_depth = depth
                print(f"[Tracking] Target depth set to {depth:.4f} m (initial distance locked)")

            # Compute pixel error from image center
            error_x_pixels = pixel_x - self.camera_width / 2   # Positive = object right of center
            error_y_pixels = pixel_y - self.camera_height / 2  # Positive = object below center

            # Compute depth error for Z control
            error_depth = 0.0
            if depth is not None and self.target_depth is not None:
                error_depth = depth - self.target_depth  # Positive = object farther away

            # Apply deadband (prevent micro-corrections)
            if abs(error_x_pixels) < self.pixel_deadband:
                error_x_pixels = 0.0
            if abs(error_y_pixels) < self.pixel_deadband:
                error_y_pixels = 0.0
            if abs(error_depth) < 0.002:  # 2mm deadband for depth
                error_depth = 0.0

            # Apply low-pass filter (exponential moving average)
            self.filtered_error_x = self.filter_alpha * error_x_pixels + (1 - self.filter_alpha) * self.filtered_error_x
            self.filtered_error_y = self.filter_alpha * error_y_pixels + (1 - self.filter_alpha) * self.filtered_error_y
            self.filtered_error_depth = self.filter_alpha * error_depth + (1 - self.filter_alpha) * self.filtered_error_depth

            # Convert pixel error to Cartesian displacement in camera frame
            # Camera frame: X=right, Y=down, Z=forward
            # We want to move end-effector in SAME direction as object to keep it in view

            # --- X-Axis (Camera Right/Left) ---
            # P-Term (Proportional)
            p_term_x = self.pixel_to_meter_x * self.filtered_error_x
            # D-Term (Derivative)
            error_delta_x = self.filtered_error_x - self.last_error_x_pixels
            self.last_error_x_pixels = self.filtered_error_x
            d_term_x = self.pixel_to_meter_x_d * error_delta_x
            # Total X command
            delta_x_camera = p_term_x + d_term_x

            # --- Y-Axis (Camera Up/Down) ---
            # P-Term (Proportional)
            p_term_y = self.pixel_to_meter_y * self.filtered_error_y
            # D-Term (Derivative)
            error_delta_y = self.filtered_error_y - self.last_error_y_pixels
            self.last_error_y_pixels = self.filtered_error_y
            d_term_y = self.pixel_to_meter_y_d * error_delta_y
            # Total Y command
            delta_y_camera = p_term_y + d_term_y

            p_term_z = self.depth_to_meter_z * self.filtered_error_depth
            error_delta_z = self.filtered_error_depth - self.last_error_depth
            self.last_error_depth = self.filtered_error_depth
            d_term_z = self.depth_to_meter_z_d * error_delta_z
            delta_z_camera = p_term_z + d_term_z

            # --- END NEW PD LOGIC ---
            # delta_x_camera = self.filtered_error_x * self.pixel_to_meter_x  # Move right if object on right
            # delta_y_camera = self.filtered_error_y * self.pixel_to_meter_y  # Move up if object below
            # delta_z_camera = self.filtered_error_depth * self.depth_to_meter_z  # Move back if object farther (depth)
            
            # Get current end-effector state
            ee_state = p.getLinkState(self.robot_id, self.ee_link)
            current_pos = ee_state[0]
            current_orn = ee_state[1]
            
            # Get rotation matrix to transform camera frame to world frame
            rot_matrix = p.getMatrixFromQuaternion(current_orn)
            rot_matrix = np.array(rot_matrix).reshape(3, 3)
            
            # Camera displacement in camera frame
            delta_camera_frame = np.array([delta_x_camera, delta_y_camera, delta_z_camera])
            
            # Transform to world frame
            delta_world_frame = rot_matrix.dot(delta_camera_frame)
            
            # Update target end-effector position
            self.target_ee_pos[0] = current_pos[0] + delta_world_frame[0]
            self.target_ee_pos[1] = current_pos[1] + delta_world_frame[1]
            self.target_ee_pos[2] = current_pos[2] + delta_world_frame[2]
            
            # Compute IK to reach target position (use INITIAL orientation to prevent drift)
            target_joints = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link,
                self.target_ee_pos,
                self.initial_ee_orn,  # Use initial orientation to maintain constant tilt
                maxNumIterations=20,
                residualThreshold=1e-4
            )
            
            # Send motor commands
            for i, joint_idx in enumerate(self.controllable_joints[:7]):
                if i < len(target_joints):
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target_joints[i],
                        force=500,
                        maxVelocity=1.0,
                        positionGain=0.3
                    )
            
            # Debug output every 60 frames (~2 Hz)
            if self.frame_counter % 60 == 0:
                print(f"[Tracking] Pixel error: ({error_x_pixels:.1f}, {error_y_pixels:.1f}) px, "
                      f"[Tracking] Depth error: {error_depth*1000:.1f} mm, "
                    #   f"Area: {area:.0f}px² (target: {self.target_area:.0f}), "
                      f"Delta: ({delta_world_frame[0]*1000:.1f}, {delta_world_frame[1]*1000:.1f}, "
                      f"{delta_world_frame[2]*1000:.1f}) mm")
            
            # Collect trajectory data every 4 frames for plotting
            if self.frame_counter % 3 == 0:
                ee_state = p.getLinkState(self.robot_id, self.ee_link)
                pos = ee_state[0]
                self.trajectory_data.append([pos[0], pos[1], pos[2]])
            
        
        self.frame_counter += 1


def generate_robotB_trajectory_plots(trajectory_data, output_filename="robotB_trajectory.png"):
    """Generate trajectory plots for Robot B (Franka Panda) end-effector path."""
    
    if len(trajectory_data) == 0:
        print("No trajectory data to plot for Robot B")
        return
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract data
    data = np.array(trajectory_data)
    x_vals = data[:, 0]
    y_vals = data[:, 1]
    z_vals = data[:, 2]
    time_vals = np.arange(len(data)) / 60.0  # Assuming 60 Hz data collection
    
    # Create figure with 2-and-1 layout (same as Robot A)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Y-Z plane (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(y_vals, z_vals, 'b-', linewidth=2, label='Tracking path')
    ax1.plot(y_vals[0], z_vals[0], 'go', markersize=10, label='Start')
    ax1.plot(y_vals[-1], z_vals[-1], 'ro', markersize=10, label='End')
    ax1.set_xlabel('Y Position (m)', fontsize=12)
    ax1.set_ylabel('Z Position (m)', fontsize=12)
    ax1.set_title('Robot B End-Effector: Y-Z Plane', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # Plot 2: 3D trajectory (top-right)
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.plot(x_vals, y_vals, z_vals, 'b-', linewidth=2, label='Tracking path')
    ax2.plot([x_vals[0]], [y_vals[0]], [z_vals[0]], 'go', markersize=10, label='Start')
    ax2.plot([x_vals[-1]], [y_vals[-1]], [z_vals[-1]], 'ro', markersize=10, label='End')
    ax2.set_xlabel('X Position (m)', fontsize=10)
    ax2.set_ylabel('Y Position (m)', fontsize=10)
    ax2.set_zlabel('Z Position (m)', fontsize=10)
    ax2.set_title('Robot B End-Effector: 3D Trajectory', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.locator_params(axis='x', nbins=4)
    ax2.locator_params(axis='y', nbins=4)
    ax2.locator_params(axis='z', nbins=4)
    
    # Plot 3: X position over time (bottom, spanning full width)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time_vals, x_vals, 'b-', linewidth=2)
    ax3.plot(time_vals[0], x_vals[0], 'go', markersize=10, label='Start')
    ax3.plot(time_vals[-1], x_vals[-1], 'ro', markersize=10, label='End')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('X Position (m)', fontsize=12)
    ax3.set_title('Robot B End-Effector: X Position vs Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"Total points: {len(data)}\n"
    stats_text += f"Duration: {time_vals[-1]:.1f}s\n"
    stats_text += f"X range: [{x_vals.min():.3f}, {x_vals.max():.3f}]m\n"
    stats_text += f"Y range: [{y_vals.min():.3f}, {y_vals.max():.3f}]m\n"
    stats_text += f"Z range: [{z_vals.min():.3f}, {z_vals.max():.3f}]m"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Robot B trajectory plot saved: {output_filename}")
    plt.close()


def generate_overlay_trajectory_plots(robotA_data, robotB_data, output_filename="dual_robot_overlay.png"):
    """Generate overlay plots showing both Robot A and Robot B trajectories together."""
    
    if len(robotA_data) == 0 or len(robotB_data) == 0:
        print("Insufficient data for overlay plot")
        return
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract Robot A data (format: [x, y, z, trajectory_name])
    # Need to handle mixed types (floats and strings)
    xA = np.array([d[0] for d in robotA_data], dtype=float)
    yA = np.array([d[1] for d in robotA_data], dtype=float)
    zA = np.array([d[2] for d in robotA_data], dtype=float)
    timeA = np.arange(len(robotA_data)) / 60.0

    # Offset X so both start at 0
    xA_offset = xA - xA[0]

    # Extract Robot B data (format: [x, y, z])
    dataB = np.array(robotB_data)
    xB = dataB[:, 0]
    yB = dataB[:, 1]
    zB = dataB[:, 2]
    timeB = np.arange(len(dataB)) / 60.0
    xB_offset = xB - xB[0]
    
    # Create figure with 2-and-1 layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Y-Z plane overlay (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(yA, zA, 'b-', linewidth=2, alpha=0.7, label='Robot A (UR5)')
    ax1.plot(yA[0], zA[0], 'bo', markersize=8)
    ax1.plot(yB, zB, 'r-', linewidth=2, alpha=0.7, label='Robot B (Franka)')
    ax1.plot(yB[0], zB[0], 'ro', markersize=8)
    ax1.set_xlabel('Y Position (m)', fontsize=12)
    ax1.set_ylabel('Z Position (m)', fontsize=12)
    ax1.set_title('Dual Robot Overlay: Y-Z Plane', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.axis('equal')
    
    # Plot 2: 3D trajectory overlay (top-right)
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.plot(xA, yA, zA, 'b-', linewidth=2, alpha=0.7, label='Robot A (UR5)')
    ax2.plot([xA[0]], [yA[0]], [zA[0]], 'bo', markersize=8)
    ax2.plot(xB, yB, zB, 'r-', linewidth=2, alpha=0.7, label='Robot B (Franka)')
    ax2.plot([xB[0]], [yB[0]], [zB[0]], 'ro', markersize=8)
    ax2.set_xlabel('X Position (m)', fontsize=10)
    ax2.set_ylabel('Y Position (m)', fontsize=10)
    ax2.set_zlabel('Z Position (m)', fontsize=10)
    ax2.set_title('Dual Robot Overlay: 3D Trajectories', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.locator_params(axis='x', nbins=4)
    ax2.locator_params(axis='y', nbins=4)
    ax2.locator_params(axis='z', nbins=4)
    
    # Plot 3: X position over time overlay (bottom, spanning full width)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(timeA, xA_offset, 'b-', linewidth=2, alpha=0.7, label='Robot A (UR5)')
    ax3.plot(timeB, xB_offset, 'r-', linewidth=2, alpha=0.7, label='Robot B (Franka)')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('X Position (m)', fontsize=12)
    ax3.set_title('Dual Robot Overlay: X Position vs Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    
    # Add statistics for both robots
    stats_text = f"Robot A (UR5):\n"
    stats_text += f"  Points: {len(robotA_data)}, Duration: {timeA[-1]:.1f}s\n"
    stats_text += f"  X: [{xA.min():.3f}, {xA.max():.3f}]m\n"
    stats_text += f"  Y: [{yA.min():.3f}, {yA.max():.3f}]m\n"
    stats_text += f"  Z: [{zA.min():.3f}, {zA.max():.3f}]m\n\n"
    stats_text += f"Robot B (Franka):\n"
    stats_text += f"  Points: {len(robotB_data)}, Duration: {timeB[-1]:.1f}s\n"
    stats_text += f"  X: [{xB.min():.3f}, {xB.max():.3f}]m\n"
    stats_text += f"  Y: [{yB.min():.3f}, {yB.max():.3f}]m\n"
    stats_text += f"  Z: [{zB.min():.3f}, {zB.max():.3f}]m"
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Dual robot overlay plot saved: {output_filename}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main entry point for the dual-robot simulation.
    Sets up both robots in a shared PyBullet environment.
    """
    
    print("\n" + "="*70)
    print("DUAL ROBOT SYSTEM - MAIN CONTROLLER")
    print("="*70)
    print("Initializing shared environment with both robots...\n")
    
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=1.0/120.0, numSolverIterations=12, numSubSteps=1)
    sim_hz = 240.0
    
    # Performance optimizations: disable shadows and heavy debug visuals
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    
    # Load ground plane
    plane = p.loadURDF("plane.urdf")
    print("✓ Ground plane loaded\n")
    
    # ========== Robot A (UR5) Setup ==========
    print("Setting up Robot A (UR5)...")
    # Import robotA here to use existing functions
    import robotA
    
    # Create UR5's table and robot
    ur5_table_height = 0.5
    ur5_table_collision = p.createCollisionShape(
        p.GEOM_BOX, 
        halfExtents=[0.5, 0.5, ur5_table_height/2]
    )
    ur5_table_visual = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=[0.5, 0.5, ur5_table_height/2], 
        rgbaColor=[1, 1, 1, 1]
    )
    ur5_table = p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=ur5_table_collision,
        baseVisualShapeIndex=ur5_table_visual,
        basePosition=[0.5, 0, ur5_table_height/2]
    )
    
    # Load UR5 robot (self-collision disabled for performance)
    ur5 = p.loadURDF(
        "../urdf/ur5.urdf", 
        basePosition=[0.5, 0, ur5_table_height + 0.02],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE
    )
    print(f"✓ UR5 loaded at [0.5, 0, {ur5_table_height + 0.02}]")
    
    # Initialize UR5 and create cube
    initial_ee_pos, initial_ee_orn, ur5_ee_link = robotA.initialize_robot(ur5)
    cube, cube_pos = robotA.create_cube(ur5_table_height)
    obstacles = [ur5_table, plane]
    print()
    
    # ========== Robot B (Franka Panda) Setup ==========
    print("Setting up Robot B (Franka Panda)...")
    # Position 1.5m in front (positive X), rotated 180° to face UR5
    franka_base_x = 0.5 + 1.5  # 1.5m forward from UR5
    franka_base_pos = [franka_base_x, 0, robotB.TABLE_HEIGHT + 0.02]
    franka_orientation = [0, 0, np.pi]  # 180° rotation around Z-axis
    
    franka_table, panda, ee_link = robotB.setup_franka_robot(
        physics_client,
        base_position=franka_base_pos,
        base_orientation=franka_orientation
    )
    
    # ========== Simulation Loop ==========
    print("="*70)
    print("DUAL ROBOT SYSTEM READY")
    print("="*70)
    print(f"Robot A (UR5):    Position [0.5, 0, {ur5_table_height + 0.02}]")
    print(f"Robot B (Franka): Position {franka_base_pos}, facing UR5")
    print(f"Distance between robots: 1.5m")
    print("\nExecuting UR5 pick-and-place and trajectories...")
    print("Press Ctrl+C to exit.\n")
    
    # Execute UR5 pick and return
    constraint_id = robotA.pick_and_return(
        robot_id=ur5, ee_link=ur5_ee_link, cube_id=cube,
        cube_pos=cube_pos, initial_ee_pos=initial_ee_pos,
        initial_ee_orn=initial_ee_orn, obstacle_ids=obstacles
    )
    
    if constraint_id is None:
        print("✗ UR5 pick-and-place failed.\n")
        p.disconnect()
        return
    
    time.sleep(1.0)
    
    # Pre-compute UR5 trajectories
    circle_waypoints = None
    lissajous_waypoints = None
    
    if robotA.USE_CIRCLE:
        circle_waypoints = robotA.precompute_circular_trajectory(
            robot_id=ur5, ee_link=ur5_ee_link,
            center_pos=initial_ee_pos, radius=robotA.CIRCLE_RADIUS,
            num_points=robotA.NUM_CIRCLE_POINTS
        )
    
    if robotA.USE_LISSAJOUS:
        lissajous_waypoints = robotA.precompute_lissajous_trajectory(
            robot_id=ur5, ee_link=ur5_ee_link,
            center_pos=initial_ee_pos,
            amplitude_y=robotA.LISSAJOUS_AMPLITUDE_Y,
            amplitude_z=robotA.LISSAJOUS_AMPLITUDE_Z,
            num_points=robotA.NUM_LISSAJOUS_POINTS
        )
    
    # Continuous trajectory execution loop
    cycle_count = 0
    max_cycles = 2 if robotA.PLOT_GRAPHS else float('inf')
    
    # Start vision processing worker (separate process for CV)
    print("\n" + "="*70)
    print("STARTING VISION PROCESSOR")
    print("="*70)
    vision_process, image_queue, result_queue = vision_processor.start_vision_process(
        camera_width=robotB.CAMERA_WIDTH,
        camera_height=robotB.CAMERA_HEIGHT,
        debug=False  # Set to True for vision debug output
    )
    print("✓ Vision processor started in separate process\n")
    
    # Initialize independent robot controllers
    controller_A = RobotAController(
        robot_id=ur5,
        ee_link=ur5_ee_link,
        waypoints_circle=circle_waypoints,
        waypoints_lissajous=lissajous_waypoints,
        duration_circle=robotA.CIRCLE_DURATION,
        duration_lissajous=robotA.LISSAJOUS_DURATION
    )
    
    controller_B = RobotBController(
        robot_id=panda,
        ee_link=ee_link,
        image_queue=image_queue,
        result_queue=result_queue
    )
    
    print("\n✓ Starting independent robot controllers in unified simulation loop...\n")
    print(f"[Cycle 1] Starting circle trajectory...")
    
    try:
        while controller_A.cycle_count < max_cycles:
            # Robot A: Execute trajectory control
            controller_A.control_step()
            
            # Robot B: Update camera tracking
            controller_B.control_step()
            
            # Single physics step
            p.stepSimulation()
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    # Cleanup vision processor
    print("Terminating vision processor...")
    vision_process.terminate()
    vision_process.join(timeout=1.0)
    
    # Generate plots if enabled
    if robotA.PLOT_GRAPHS and len(robotA.trajectory_data) > 0:
        robotA.generate_trajectory_plots()
    
    # Generate Robot B trajectory plot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"robotB_trajectory_{timestamp}.png"
    graphs_dir = os.path.join(os.path.dirname(__file__), "..", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    output_path = os.path.join(graphs_dir, filename)

    if len(controller_B.trajectory_data) > 0:
        generate_robotB_trajectory_plots(controller_B.trajectory_data, output_filename=output_path)
    
    # Generate dual robot overlay plot
    if robotA.PLOT_GRAPHS and len(robotA.trajectory_data) > 0 and len(controller_B.trajectory_data) > 0:
        overlay_filename = f"dual_robot_overlay_{timestamp}.png"
        overlay_path = os.path.join(graphs_dir, overlay_filename)
        generate_overlay_trajectory_plots(robotA.trajectory_data, controller_B.trajectory_data, output_filename=overlay_path)
    
    # Disconnect PyBullet
    p.disconnect()


if __name__ == "__main__":
    main()
