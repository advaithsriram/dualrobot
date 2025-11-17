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
                    current_pos[0], current_pos[1], current_pos[2], traj_name
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
        
        # Visual servoing gains (proportional control)
        self.kp_pan = 0.002   # Pan control gain (base joint rotation)
        self.kp_tilt = 0.001  # Tilt control gain (shoulder joint)
        
        # Get controllable joints
        self.controllable_joints = []
        num_joints = p.getNumJoints(robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                self.controllable_joints.append(i)
        
        # Current joint positions (for incremental control)
        self.current_joint_targets = [p.getJointState(robot_id, j)[0] 
                                      for j in self.controllable_joints[:7]]
        
    def control_step(self):
        """Execute one control step for Franka - update camera and track object"""
        
        # Update camera and send to vision processor every 8 frames (30 Hz)
        if self.frame_counter % 8 == 0:
            rgb_image = robotB.get_camera_image(self.robot_id, self.ee_link)
            
            # Send image to vision processor (non-blocking)
            if self.image_queue is not None:
                try:
                    self.image_queue.put_nowait(rgb_image)
                except:
                    pass  # Queue full, skip this frame
        
        # Get latest detection result (non-blocking)
        if self.result_queue is not None:
            try:
                while not self.result_queue.empty():
                    self.latest_detection = self.result_queue.get_nowait()
            except:
                pass
        
        # Visual servoing control
        if self.latest_detection is not None and self.latest_detection['detected']:
            pixel_x = self.latest_detection['pixel_x']
            pixel_y = self.latest_detection['pixel_y']
            
            # Compute error from image center
            error_x = pixel_x - self.camera_width / 2   # Positive = object right of center
            error_y = pixel_y - self.camera_height / 2  # Positive = object below center
            
            # Proportional control to align camera with object
            # Joint 0: Base rotation (pan left/right)
            # Joint 1: Shoulder (tilt up/down)
            delta_pan = -self.kp_pan * error_x   # Negative to turn towards object
            delta_tilt = self.kp_tilt * error_y  # Positive to tilt down when object below
            
            # Update target positions (incremental control)
            self.current_joint_targets[0] += delta_pan
            self.current_joint_targets[1] += delta_tilt
            
            # Apply joint limits (safety)
            self.current_joint_targets[0] = np.clip(self.current_joint_targets[0], -np.pi, np.pi)
            self.current_joint_targets[1] = np.clip(self.current_joint_targets[1], -np.pi/2, 0)
            
            # Send motor commands
            for i, joint_idx in enumerate(self.controllable_joints[:7]):
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=self.current_joint_targets[i],
                    force=500,
                    maxVelocity=1.0,
                    positionGain=0.3
                )
            
            # Debug output every 60 frames (~2 Hz)
            if self.frame_counter % 60 == 0:
                print(f"[Tracking] Error: ({error_x:.1f}, {error_y:.1f}) px, "
                      f"Control: pan={np.rad2deg(delta_pan):.2f}°, tilt={np.rad2deg(delta_tilt):.2f}°")
        
        self.frame_counter += 1

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
    ur5_table_height = 0.4
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
    
    # Disconnect PyBullet
    p.disconnect()


if __name__ == "__main__":
    main()
