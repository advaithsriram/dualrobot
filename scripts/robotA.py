"""
Main demonstration with SMOOTH trajectory execution.

Key improvements over main.py:
1. Pre-computes entire circular trajectory upfront (no drift)
2. Uses velocity control for smoother motion
3. Higher positionGain for better tracking
4. Proper timing between waypoints

Flow:
1. Setup environment (table, robot, cube)
2. Pick up cube from specified position
3. Return to initial position with cube
4. Execute smooth circular trajectory while holding cube
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from pick_place import create_graspable_cube, attach_object_to_robot, detach_object

# ============================================================================
# CONFIGURATION
# ============================================================================

# Trajectory selection
USE_CIRCLE = True  # Set to True for circular trajectory
USE_LISSAJOUS = True  # Set to True for Lissajous (figure-8/infinity) trajectory

# Trajectory parameters
CIRCLE_DIAMETER = 0.3  # Diameter of circular trajectory (in meters)
CIRCLE_RADIUS = CIRCLE_DIAMETER / 2
NUM_CIRCLE_POINTS = 200  # More points for smoother pre-computed trajectory
CIRCLE_DURATION = 3.0  # Time to complete one circle (seconds)

# Lissajous parameters
LISSAJOUS_AMPLITUDE_Y = 0.15  # Horizontal amplitude (2x vertical for proper figure-8)
LISSAJOUS_AMPLITUDE_Z = 0.075  # Vertical amplitude (half of horizontal)
NUM_LISSAJOUS_POINTS = 200  # Number of waypoints
LISSAJOUS_DURATION = 3.0  # Time to complete one figure-8 (seconds)

# Sinusoidal motion in X-axis (perpendicular to Y-Z plane)
USE_SINUSOIDAL_X = True  # Enable 3D motion with X-axis oscillation
SINUSOIDAL_X_AMPLITUDE = 0.08  # Amplitude of back-and-forth motion (meters)
SINUSOIDAL_X_FREQUENCY = 1.0  # Frequency factor (1.0 = one complete cycle per trajectory)

# Environment parameters
TABLE_HEIGHT = 0.4  # Height of table in meters
CUBE_SIZE = 0.04  # Size of cube to pick up
CUBE_POSITION = [0.8, -0.3, None]  # [x, y, z] - z will be calculated

# Robot initial joint configuration (degrees, will be converted to radians)
INITIAL_JOINT_ANGLES = [0, -90, 90, 180, -90, 0]

# Debug flags
DEBUG_COLLISION_PREVENTION = True
USE_STL_COLLISION_MESHES = True
PLOT_GRAPHS = True  # Generate and save trajectory plots after simulation
PLOT_GRAPHS = True  # Generate and save trajectory plots after simulation

# Global trajectory data storage
trajectory_data = []  # Stores [x, y, z, trajectory_type] for each waypoint

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Initialize PyBullet environment with table and robot."""
    
    print("\n" + "="*70)
    print("UR5 SMOOTH TRAJECTORY DEMO")
    print("="*70)
    print("Using pre-computed trajectories with velocity profiling for smooth motion\n")
    
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load ground plane
    plane = p.loadURDF("plane.urdf")
    
    # Create table (white box)
    table_collision = p.createCollisionShape(
        p.GEOM_BOX, 
        halfExtents=[0.5, 0.5, TABLE_HEIGHT/2]
    )
    table_visual = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=[0.5, 0.5, TABLE_HEIGHT/2], 
        rgbaColor=[1, 1, 1, 1]
    )
    table = p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=table_collision,
        baseVisualShapeIndex=table_visual,
        basePosition=[0.5, 0, TABLE_HEIGHT/2]
    )
    
    # Load UR5 robot with STL collision meshes
    ur5 = p.loadURDF(
        "../urdf/ur5.urdf", 
        basePosition=[0.5, 0, TABLE_HEIGHT + 0.02],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE | 
              p.URDF_USE_SELF_COLLISION |
              p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    )
    
    print(f"✓ Environment loaded")
    print(f"  - Table height: {TABLE_HEIGHT}m")
    print(f"  - Robot base: [0.5, 0, {TABLE_HEIGHT + 0.02}]")
    print(f"  - Using STL collision meshes\n")
    
    return plane, table, ur5


def initialize_robot(robot_id):
    """Set robot to initial joint configuration with strong motor control."""
    
    # Convert initial angles to radians
    initial_positions = [np.deg2rad(angle) for angle in INITIAL_JOINT_ANGLES]
    
    # Set joint states
    for i in range(6):
        p.resetJointState(robot_id, i, initial_positions[i])
    
    # Enable strong position control
    for i in range(6):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=initial_positions[i],
            force=5000,
            maxVelocity=2.0,
            positionGain=0.5
        )
    
    # Let physics settle
    for _ in range(100):
        p.stepSimulation()
    
    # Get initial end-effector state
    num_joints = p.getNumJoints(robot_id)
    ee_link = num_joints - 1
    ee_state = p.getLinkState(robot_id, ee_link)
    
    print(f"✓ Robot initialized")
    print(f"  - Initial joint config: {INITIAL_JOINT_ANGLES} (degrees)")
    print(f"  - End-effector position: [{ee_state[0][0]:.3f}, {ee_state[0][1]:.3f}, {ee_state[0][2]:.3f}]\n")
    
    return ee_state[0], ee_state[1], ee_link


def create_cube(table_height):
    """Create a graspable cube at specified position."""
    
    table_top_z = table_height + 0.02
    cube_z = table_top_z + CUBE_SIZE/2 + 0.001
    cube_position = [CUBE_POSITION[0], CUBE_POSITION[1], cube_z]
    
    cube = create_graspable_cube(
        position=cube_position,
        size=CUBE_SIZE,
        color=[1, 0, 0, 1],
        mass=0.05
    )
    
    # Let cube settle
    for _ in range(100):
        p.stepSimulation()
    
    settled_pos, _ = p.getBasePositionAndOrientation(cube)
    
    print(f"✓ Cube created at [{settled_pos[0]:.3f}, {settled_pos[1]:.3f}, {settled_pos[2]:.3f}]\n")
    
    return cube, settled_pos


# ============================================================================
# MOTION CONTROL - Pick and Place (same as before)
# ============================================================================

def check_collision_free(robot_id, joint_positions, obstacle_ids):
    """Check if current configuration is collision-free."""
    p.performCollisionDetection()
    
    self_contacts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    if self_contacts is not None and len(self_contacts) > 0:
        strong_contacts = [c for c in self_contacts if len(c) > 9 and c[9] > 0.1]
        if len(strong_contacts) > 0:
            return False
    
    for obstacle_id in obstacle_ids:
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)
        if contacts is not None and len(contacts) > 0:
            strong_contacts = [c for c in contacts if len(c) > 9 and c[9] > 0.1]
            if len(strong_contacts) > 0:
                return False
    
    return True


def move_to_position(robot_id, ee_link, target_pos, target_orn=None, obstacle_ids=[], speed_factor=1.0, silent=False):
    """Move robot to target position with collision prevention."""
    
    if target_orn is None:
        target_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
    
    if not silent:
        print(f"  Moving to [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]...")
    
    current_joints = [p.getJointState(robot_id, i)[0] for i in range(6)]
    
    target_joints = p.calculateInverseKinematics(
        robot_id, ee_link, target_pos, target_orn,
        maxNumIterations=100, residualThreshold=1e-5
    )
    
    if target_joints is None:
        if not silent:
            print("  ✗ IK failed!")
        return False
    
    num_waypoints = 50
    collision_count = 0
    
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)
        waypoint = [current_joints[j] + t * (target_joints[j] - current_joints[j]) for j in range(6)]
        
        for j in range(6):
            p.setJointMotorControl2(
                bodyIndex=robot_id, jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=waypoint[j],
                maxVelocity=2.0 * speed_factor,
                force=5000, positionGain=0.5
            )
        
        for _ in range(10):
            p.stepSimulation()
            
            if DEBUG_COLLISION_PREVENTION and not check_collision_free(robot_id, waypoint, obstacle_ids):
                collision_count += 1
                if collision_count > 5:
                    if not silent:
                        print(f"  ✗ COLLISION! Stopped at waypoint {i}/{num_waypoints}")
                    current_state = [p.getJointState(robot_id, j)[0] for j in range(6)]
                    for j in range(6):
                        p.setJointMotorControl2(
                            bodyIndex=robot_id, jointIndex=j,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=current_state[j],
                            force=5000, maxVelocity=0.1
                        )
                    return False
            else:
                collision_count = max(0, collision_count - 1)
    
    ee_state = p.getLinkState(robot_id, ee_link)
    distance = np.linalg.norm(np.array(target_pos) - np.array(ee_state[0]))
    
    if not silent:
        if distance < 0.02:
            print(f"  ✓ Reached target (error: {distance:.4f}m)")
        else:
            print(f"  ⚠ Close to target (error: {distance:.4f}m)")
    
    return distance < 0.05


def pick_and_return(robot_id, ee_link, cube_id, cube_pos, initial_ee_pos, initial_ee_orn, obstacle_ids):
    """Execute pick-and-place sequence."""
    
    print("="*70)
    print("PHASE 1: PICK AND RETURN TO INITIAL POSITION")
    print("="*70 + "\n")
    
    hover_height = 0.05
    pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2]]
    hover_pick_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + hover_height]
    
    print("Step 1: Moving to hover position")
    if not move_to_position(robot_id, ee_link, hover_pick_pos, obstacle_ids=obstacle_ids):
        return None
    time.sleep(0.5)
    
    print("\nStep 2: Moving down to pick")
    if not move_to_position(robot_id, ee_link, pick_pos, obstacle_ids=obstacle_ids, speed_factor=0.5):
        return None
    time.sleep(0.5)
    
    print("\nStep 3: Grasping cube")
    constraint_id = attach_object_to_robot(robot_id, cube_id, ee_link)
    time.sleep(0.5)
    
    print("\nStep 4: Lifting cube")
    if not move_to_position(robot_id, ee_link, hover_pick_pos, obstacle_ids=obstacle_ids, speed_factor=0.5):
        return None
    time.sleep(0.5)
    
    print("\nStep 5: Returning to initial position")
    if not move_to_position(robot_id, ee_link, initial_ee_pos, target_orn=initial_ee_orn, obstacle_ids=obstacle_ids):
        return None
    time.sleep(0.5)
    
    print("\n✓ Pick-and-place completed!\n")
    return constraint_id


# ============================================================================
# SMOOTH CIRCULAR TRAJECTORY EXECUTION
# ============================================================================

def precompute_circular_trajectory(robot_id, ee_link, center_pos, radius, num_points):
    """
    Pre-compute entire circular trajectory (all joint positions).
    This eliminates drift and allows for velocity profiling.
    
    Returns:
        List of joint configurations forming a smooth circle
    """
    
    print("="*70)
    print("PHASE 2: PRE-COMPUTING SMOOTH CIRCULAR TRAJECTORY")
    print("="*70)
    print(f"Parameters:")
    print(f"  - Diameter: {radius*2:.3f}m")
    print(f"  - Center: [{center_pos[0]:.3f}, {center_pos[1]:.3f}, {center_pos[2]:.3f}]")
    print(f"  - Waypoints: {num_points}")
    print(f"  - Duration: {CIRCLE_DURATION}s")
    if USE_SINUSOIDAL_X:
        print(f"  - X-axis sinusoidal: amplitude={SINUSOIDAL_X_AMPLITUDE:.3f}m, freq={SINUSOIDAL_X_FREQUENCY}x")
    print()
    
    # Get current orientation
    ee_state = p.getLinkState(robot_id, ee_link)
    trajectory_orn = ee_state[1]
    
    print("Computing IK for all waypoints...")
    waypoints = []
    
    for step in range(num_points):
        t = (step / num_points) * 2 * np.pi
        
        # Circular motion on Y-Z plane with optional X-axis sinusoidal motion
        x_offset = 0
        if USE_SINUSOIDAL_X:
            # Sinusoidal oscillation in X (perpendicular to circle plane)
            x_offset = SINUSOIDAL_X_AMPLITUDE * np.sin(SINUSOIDAL_X_FREQUENCY * t)
        
        target_pos = [
            center_pos[0] + x_offset,              # X: sinusoidal back-and-forth
            center_pos[1] + radius * np.cos(t),    # Y: circle horizontal
            center_pos[2] + radius * np.sin(t)     # Z: circle vertical
        ]
        
        # Calculate IK
        target_joints = p.calculateInverseKinematics(
            robot_id, ee_link, target_pos, trajectory_orn,
            maxNumIterations=100, residualThreshold=1e-5
        )
        
        if target_joints is None:
            print(f"  ⚠ IK failed at waypoint {step}/{num_points}")
            continue
        
        waypoints.append(list(target_joints[:6]))
    
    print(f"✓ Pre-computed {len(waypoints)} waypoints\n")
    return waypoints


def precompute_lissajous_trajectory(robot_id, ee_link, center_pos, amplitude_y, amplitude_z, num_points):
    """
    Pre-compute entire Lissajous curve (figure-8/infinity symbol) trajectory.
    
    Lissajous curve with 2:1 frequency ratio creates a figure-8 shape:
    - y(t) = A_y * sin(t)      → oscillates once per cycle
    - z(t) = A_z * sin(2t)     → oscillates twice per cycle (creates figure-8)
    
    Returns:
        List of joint configurations forming a smooth figure-8
    """
    
    print("="*70)
    print("PRE-COMPUTING SMOOTH LISSAJOUS TRAJECTORY")
    print("="*70)
    print(f"Parameters:")
    print(f"  - Amplitude Y: {amplitude_y:.3f}m (horizontal)")
    print(f"  - Amplitude Z: {amplitude_z:.3f}m (vertical)")
    print(f"  - Ratio: 2:1 (creates figure-8/infinity symbol)")
    print(f"  - Center: [{center_pos[0]:.3f}, {center_pos[1]:.3f}, {center_pos[2]:.3f}]")
    print(f"  - Waypoints: {num_points}")
    print(f"  - Duration: {LISSAJOUS_DURATION}s")
    if USE_SINUSOIDAL_X:
        print(f"  - X-axis sinusoidal: amplitude={SINUSOIDAL_X_AMPLITUDE:.3f}m, freq={SINUSOIDAL_X_FREQUENCY}x")
    print()
    
    # Get current orientation
    ee_state = p.getLinkState(robot_id, ee_link)
    trajectory_orn = ee_state[1]
    
    print("Computing IK for all waypoints...")
    waypoints = []
    
    for step in range(num_points):
        t = (step / num_points) * 2 * np.pi  # 0 to 2π for one complete cycle
        
        # Lissajous curve: sin(t) vs sin(2t) creates figure-8
        # Phase shift by +π/2 makes it start at right side (y=amplitude_y, z=0)
        # Negate z-component to reverse direction (match circle's counter-clockwise flow)
        # This makes it go down-right first instead of down-left
        
        # Add sinusoidal motion in X-axis (perpendicular to figure-8 plane)
        x_offset = 0
        if USE_SINUSOIDAL_X:
            x_offset = SINUSOIDAL_X_AMPLITUDE * np.sin(SINUSOIDAL_X_FREQUENCY * t)
        
        target_pos = [
            center_pos[0] + x_offset,                                   # X: sinusoidal back-and-forth
            center_pos[1] + amplitude_y * np.sin(t + np.pi/2),          # Y: figure-8 horizontal
            center_pos[2] - amplitude_z * np.sin(2 * (t + np.pi/2))     # Z: figure-8 vertical (negated)
        ]
        
        # Calculate IK
        target_joints = p.calculateInverseKinematics(
            robot_id, ee_link, target_pos, trajectory_orn,
            maxNumIterations=100, residualThreshold=1e-5
        )
        
        if target_joints is None:
            print(f"  ⚠ IK failed at waypoint {step}/{num_points}")
            continue
        
        waypoints.append(list(target_joints[:6]))
    
    print(f"✓ Pre-computed {len(waypoints)} waypoints\n")
    return waypoints


def execute_smooth_trajectory(robot_id, ee_link, waypoints, duration, trajectory_name="trajectory", color=[0, 0, 1]):
    """
    Execute pre-computed trajectory with smooth velocity profiling.
    
    This is MUCH smoother because:
    1. All IK computed upfront (no per-step computation)
    2. Proper timing ensures smooth motion
    3. Higher position gain for better tracking
    """
    
    global trajectory_data
    
    # print("="*70)
    # print(f"EXECUTING SMOOTH {trajectory_name.upper()}")
    # print("="*70)
    # print("Using pre-computed waypoints with velocity profiling\n")
    
    num_waypoints = len(waypoints)
    sim_hz = 240.0  # Simulation frequency
    time_per_waypoint = duration / num_waypoints
    steps_per_waypoint = int(time_per_waypoint * sim_hz)
    
    # print(f"Execution parameters:")
    # print(f"  - Time per waypoint: {time_per_waypoint*1000:.1f}ms")
    # print(f"  - Simulation steps per waypoint: {steps_per_waypoint}")
    # print(f"  - Total duration: {duration}s\n")
    
    prev_pos = None
    start_time = time.time()
    
    for step, waypoint in enumerate(waypoints):
        # Set joint targets with higher gain for smooth tracking
        for j in range(6):
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=waypoint[j],
                maxVelocity=10.0,  # Higher max velocity for continuous motion
                force=5000,
                positionGain=0.3  # Lower gain = smoother but less accurate
            )
        
        # Step simulation for this waypoint
        for _ in range(steps_per_waypoint):
            p.stepSimulation()
            # time.sleep(1./sim_hz)  # Control speed via duration parameter
        
        # Draw trajectory trace
        ee_state = p.getLinkState(robot_id, ee_link)
        current_pos = ee_state[0]
        
        # Store position data for plotting (only for trajectory execution, not pick-and-place)
        if PLOT_GRAPHS and ("circular" in trajectory_name or "lissajous" in trajectory_name):
            trajectory_data.append([
                current_pos[0],  # x
                current_pos[1],  # y
                current_pos[2],  # z
                trajectory_name  # trajectory type
            ])
        
        if prev_pos is not None:
            p.addUserDebugLine(prev_pos, current_pos, lineColorRGB=color, lineWidth=2, lifeTime=0)
        
        prev_pos = current_pos
        
        # Progress indicator
        # if step % (num_waypoints // 4) == 0 and step > 0:
        #     elapsed = time.time() - start_time
        #     progress = (step / num_waypoints) * 100
        #     print(f"  Progress: {progress:.0f}% ({elapsed:.1f}s elapsed)")
    
    # elapsed = time.time() - start_time
    # print(f"\n✓ Trajectory completed in {elapsed:.1f}s!\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Setup
    plane, table, ur5 = setup_environment()
    initial_ee_pos, initial_ee_orn, ee_link = initialize_robot(ur5)
    cube, cube_pos = create_cube(TABLE_HEIGHT)
    obstacles = [table, plane]
    
    # Phase 1: Pick and return
    constraint_id = pick_and_return(
        robot_id=ur5, ee_link=ee_link, cube_id=cube,
        cube_pos=cube_pos, initial_ee_pos=initial_ee_pos,
        initial_ee_orn=initial_ee_orn, obstacle_ids=obstacles
    )
    
    if constraint_id is None:
        print("✗ Pick-and-place failed.\n")
        p.disconnect()
        return
    
    time.sleep(1.0)
    
    # Phase 2: Pre-compute both trajectories once
    # print("\n" + "="*70)
    # print("PHASE 2: PRE-COMPUTING TRAJECTORIES")
    # print("="*70 + "\n")
    
    circle_waypoints = None
    lissajous_waypoints = None
    
    if USE_CIRCLE:
        circle_waypoints = precompute_circular_trajectory(
            robot_id=ur5, ee_link=ee_link,
            center_pos=initial_ee_pos, radius=CIRCLE_RADIUS,
            num_points=NUM_CIRCLE_POINTS
        )
    
    if USE_LISSAJOUS:
        lissajous_waypoints = precompute_lissajous_trajectory(
            robot_id=ur5, ee_link=ee_link,
            center_pos=initial_ee_pos,
            amplitude_y=LISSAJOUS_AMPLITUDE_Y,
            amplitude_z=LISSAJOUS_AMPLITUDE_Z,
            num_points=NUM_LISSAJOUS_POINTS
        )
    
    # Phase 3: Infinite loop alternating between trajectories
    # print("="*70)
    # print("PHASE 3: CONTINUOUS TRAJECTORY EXECUTION")
    # print("="*70)
    # print("Alternating between circle (blue) and figure-8 (red)...")
    # print("Close window to exit\n")
    
    cycle_count = 0
    max_cycles = 2 if PLOT_GRAPHS else float('inf')  # Run 2 cycles for plotting, infinite otherwise
    
    while cycle_count < max_cycles:
        if USE_CIRCLE and circle_waypoints:
            # print(f"\n[Cycle {cycle_count}] Executing circle...")
            execute_smooth_trajectory(
                robot_id=ur5, ee_link=ee_link,
                waypoints=circle_waypoints, duration=CIRCLE_DURATION,
                trajectory_name="circular trajectory",
                color=[0, 0, 1]  # Blue
            )
        
        if USE_LISSAJOUS and lissajous_waypoints:
            # print(f"[Cycle {cycle_count}] Executing figure-8...")
            execute_smooth_trajectory(
                robot_id=ur5, ee_link=ee_link,
                waypoints=lissajous_waypoints, duration=LISSAJOUS_DURATION,
                trajectory_name="lissajous trajectory (figure-8)",
                color=[1, 0, 0]  # Red
            )
        
        cycle_count += 1
        # print(f"✓ Cycle {cycle_count} complete\n")
    
    # Generate plots after simulation
    if PLOT_GRAPHS and len(trajectory_data) > 0:
        generate_trajectory_plots()
    
    # Disconnect PyBullet
    p.disconnect()


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def generate_trajectory_plots():
    """
    Generate and save trajectory plots showing end-effector motion.
    Creates two plots:
    1. Y-Z plane: Shows overall trajectory pattern (circle + Lissajous combined)
    2. X-Z plane: Shows sinusoidal wave motion perpendicular to main trajectories
    """
    
    print("\n" + "="*70)
    print("GENERATING TRAJECTORY PLOTS")
    print("="*70)
    
    # Create graphs directory if it doesn't exist
    graphs_dir = os.path.join(os.path.dirname(__file__), "..", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    print(f"Plots will be saved to: {graphs_dir}\n")
    
    # Convert trajectory data to numpy array for easier processing
    data = np.array(trajectory_data, dtype=object)
    x = np.array([d[0] for d in data], dtype=float)
    y = np.array([d[1] for d in data], dtype=float)
    z = np.array([d[2] for d in data], dtype=float)
    
    print(f"Total data points recorded: {len(x)}")
    print(f"X range: [{x.min():.3f}, {x.max():.3f}] m")
    print(f"Y range: [{y.min():.3f}, {y.max():.3f}] m")
    print(f"Z range: [{z.min():.3f}, {z.max():.3f}] m\n")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========== Plot 1: Y-Z Plane (Circle and Lissajous Combined) ==========
    ax1.set_title('End-Effector Trajectory: Y-Z Plane\n(Circle + Lissajous Combined)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Y Position (m)', fontsize=11)
    ax1.set_ylabel('Z Position (m)', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot complete trajectory in Y-Z plane
    ax1.plot(y, z, 'b-', linewidth=1.5, alpha=0.8)
    ax1.scatter(y[0], z[0], c='green', s=150, marker='o', 
               edgecolors='black', linewidths=2, zorder=5, label='Start')
    ax1.scatter(y[-1], z[-1], c='red', s=150, marker='X', 
               edgecolors='black', linewidths=2, zorder=5, label='End')
    ax1.legend(loc='best', fontsize=10)
    
    # ========== Plot 2: X-Z Plane (Sinusoidal Wave) ==========
    ax2.set_title('End-Effector Trajectory: X Position Over Time\n(Sinusoidal Motion)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('X Position (m)', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot X position over time (shows pure sine wave)
    time_steps = np.arange(len(x))
    ax2.plot(time_steps, x, 'r-', linewidth=1.5, alpha=0.8)
    ax2.scatter(time_steps[0], x[0], c='green', s=150, marker='o', 
               edgecolors='black', linewidths=2, zorder=5, label='Start')
    ax2.scatter(time_steps[-1], x[-1], c='red', s=150, marker='X', 
               edgecolors='black', linewidths=2, zorder=5, label='End')
    ax2.legend(loc='best', fontsize=10)
    
    # Add overall title
    fig.suptitle('UR5 Robot End-Effector 3D Trajectory Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_analysis_{timestamp}.png"
    filepath = os.path.join(graphs_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {filename}")
    print(f"  - Y-Z Plane: Shows combined circle and Lissajous (∞) patterns")
    print(f"  - X Position vs Time: Shows sinusoidal wave (back-and-forth motion)")
    print("="*70 + "\n")
    
    # Close the plot to free memory
    plt.close()


if __name__ == "__main__":
    main()
