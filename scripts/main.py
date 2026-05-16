"""
Main script for dual robot system.

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
import argparse

# Import robot modules
import robotB
import vision_processor
from controllers.franka_policies import PIDTrackingPolicy, PPOTrackingPolicy

CAMERA_FREQ = 30.0  # Camera update frequency in Hz
SIM_HZ = 120.0     # Simulation frequency in Hz
VIDEO_RECORDING = False  # Enable video recording

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
    """Controller for Franka Panda robot with swappable tracking policies."""

    def __init__(
        self,
        robot_id,
        ee_link,
        policy,
        image_queue=None,
        result_queue=None,
        target_body_id=None,
        control_mode="pid",
        trajectory_points=200,
        observation_mode="vision",
    ):
        self.robot_id = robot_id
        self.ee_link = ee_link
        self.policy = policy
        self.frame_counter = 0
        self.control_mode = control_mode
        self.observation_mode = observation_mode
        self.target_body_id = target_body_id
        self.trajectory_points = trajectory_points

        self.image_queue = image_queue
        self.result_queue = result_queue
        self.latest_detection = None

        self.controllable_joints = []
        num_joints = p.getNumJoints(robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.controllable_joints.append(i)

        ee_state = p.getLinkState(robot_id, ee_link)
        self.target_ee_pos = list(ee_state[0])
        self.initial_ee_orn = ee_state[1]
        self.previous_ee_pos = np.array(ee_state[0], dtype=np.float32)
        self.trajectory_data = []

    def _get_target_phase(self):
        return self.frame_counter * (2 * np.pi / max(1.0, self.trajectory_points))

    def _get_target_state(self):
        if self.target_body_id is None:
            return None, None
        target_pos, _ = p.getBasePositionAndOrientation(self.target_body_id)
        target_vel, _ = p.getBaseVelocity(self.target_body_id)
        return np.array(target_pos, dtype=np.float32), np.array(target_vel, dtype=np.float32)

    def control_step(self):
        """Execute one control step for Franka."""

        needs_camera_detection = (
            self.control_mode == "pid"
            or (self.control_mode == "rl" and self.observation_mode == "vision")
        )

        if needs_camera_detection and self.frame_counter % 4 == 0:
            rgb_image, depth_array = robotB.get_camera_image(self.robot_id, self.ee_link)
            if self.control_mode == "pid" and self.image_queue is not None:
                try:
                    self.image_queue.put_nowait((rgb_image, depth_array))
                except:
                    pass
            elif self.control_mode == "rl":
                pixel_x, pixel_y, detected, area = vision_processor.detect_red_object(
                    rgb_image,
                    robotB.CAMERA_WIDTH,
                    robotB.CAMERA_HEIGHT,
                    debug=False,
                )
                depth_value = None
                if detected:
                    px = int(np.clip(pixel_x, 0, robotB.CAMERA_WIDTH - 1))
                    py = int(np.clip(pixel_y, 0, robotB.CAMERA_HEIGHT - 1))
                    depth_value = float(depth_array[py, px])
                self.latest_detection = {
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "detected": detected,
                    "area": area,
                    "depth": depth_value,
                    "timestamp": time.time(),
                }

        if self.control_mode == "pid" and self.result_queue is not None:
            try:
                while not self.result_queue.empty():
                    self.latest_detection = self.result_queue.get_nowait()
            except:
                pass

        ee_state = p.getLinkState(self.robot_id, self.ee_link, computeLinkVelocity=1)
        current_pos = np.array(ee_state[0], dtype=np.float32)
        current_orn = ee_state[1]
        ee_vel = np.array(ee_state[6], dtype=np.float32) if len(ee_state) > 6 else current_pos - self.previous_ee_pos
        target_pos, target_vel = self._get_target_state()

        command = self.policy.act({
            "detection": self.latest_detection,
            "current_pos": current_pos,
            "current_orn": current_orn,
            "ee_vel": ee_vel,
            "target_pos": target_pos,
            "target_vel": target_vel,
            "phase": self._get_target_phase(),
        })

        if command is not None:
            delta_world_frame = command.delta_world
            self.target_ee_pos = (current_pos + delta_world_frame).tolist()
            target_joints = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link,
                self.target_ee_pos,
                self.initial_ee_orn,
                maxNumIterations=20,
                residualThreshold=1e-4
            )

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

            if self.frame_counter % 60 == 0:
                if self.control_mode == "pid":
                    pixel_error = command.debug.get("pixel_error", (0.0, 0.0))
                    depth_error = command.debug.get("depth_error", 0.0)
                    print(f"[Tracking:PID] Pixel error: ({pixel_error[0]:.1f}, {pixel_error[1]:.1f}) px, "
                          f"Depth error: {depth_error*1000:.1f} mm, "
                          f"Delta: ({delta_world_frame[0]*1000:.1f}, {delta_world_frame[1]*1000:.1f}, "
                          f"{delta_world_frame[2]*1000:.1f}) mm")
                elif target_pos is not None:
                    err = np.linalg.norm(target_pos - current_pos)
                    action = command.debug.get("action", np.zeros(3))
                    print(f"[Tracking:RL] 3D error: {err*1000:.1f} mm, "
                          f"Action: ({action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}), "
                          f"Delta: ({delta_world_frame[0]*1000:.1f}, {delta_world_frame[1]*1000:.1f}, "
                          f"{delta_world_frame[2]*1000:.1f}) mm")

        if self.frame_counter % 3 == 0:
            self.trajectory_data.append([current_pos[0], current_pos[1], current_pos[2]])

        self.previous_ee_pos = current_pos
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
    time_vals = np.arange(len(data)) / CAMERA_FREQ  # 60 Hz data collection
    
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
    
    # # Add statistics
    # stats_text = f"Total points: {len(data)}\n"
    # stats_text += f"Duration: {time_vals[-1]:.1f}s\n"
    # stats_text += f"X range: [{x_vals.min():.3f}, {x_vals.max():.3f}]m\n"
    # stats_text += f"Y range: [{y_vals.min():.3f}, {y_vals.max():.3f}]m\n"
    # stats_text += f"Z range: [{z_vals.min():.3f}, {z_vals.max():.3f}]m"
    
    # fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
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
    timeA = np.arange(len(robotA_data)) / CAMERA_FREQ

    # Offset X so both start at 0
    xA_offset = xA - xA[0]

    # Extract Robot B data (format: [x, y, z])
    dataB = np.array(robotB_data)
    xB = dataB[:, 0]
    yB = dataB[:, 1]
    zB = dataB[:, 2]
    timeB = np.arange(len(dataB)) / CAMERA_FREQ
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
    
    # # Add statistics for both robots
    # stats_text = f"Robot A (UR5):\n"
    # stats_text += f"  Points: {len(robotA_data)}, Duration: {timeA[-1]:.1f}s\n"
    # stats_text += f"  X: [{xA.min():.3f}, {xA.max():.3f}]m\n"
    # stats_text += f"  Y: [{yA.min():.3f}, {yA.max():.3f}]m\n"
    # stats_text += f"  Z: [{zA.min():.3f}, {zA.max():.3f}]m\n\n"
    # stats_text += f"Robot B (Franka):\n"
    # stats_text += f"  Points: {len(robotB_data)}, Duration: {timeB[-1]:.1f}s\n"
    # stats_text += f"  X: [{xB.min():.3f}, {xB.max():.3f}]m\n"
    # stats_text += f"  Y: [{yB.min():.3f}, {yB.max():.3f}]m\n"
    # stats_text += f"  Z: [{zB.min():.3f}, {zB.max():.3f}]m"
    
    # fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Dual robot overlay plot saved: {output_filename}")
    plt.close()

def plot_tracking_errors(robotA_data, robotB_data, output_dir="../graphs"):
    # Calculate error metrics
    def calc_metrics(error):
        mae = np.mean(np.abs(error))
        mse = np.mean(error**2)
        rmse = np.sqrt(mse)
        return mae, mse, rmse

    """Plot tracking errors (Robot B - Robot A) in x, y, z over time as a 2x1 subplot."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import time

    dataA = np.array(robotA_data)
    dataB = np.array(robotB_data)
    n = min(len(dataA), len(dataB))
    xA = dataA[:n, 0]
    yA = dataA[:n, 1]
    zA = dataA[:n, 2]
    xB = dataB[:n, 0]
    yB = dataB[:n, 1]
    zB = dataB[:n, 2]
    time_vals = np.arange(n) / CAMERA_FREQ  # 60 Hz data collection

    # Offset X so both start at 0
    xA_offset = xA - xA[0]
    xB_offset = xB - xB[0]

    # Calculate tracking errors (Robot B - Robot A)
    error_x = xB_offset - xA_offset
    error_y = yB - yA
    error_z = zB - zA


    mae_x, mse_x, rmse_x = calc_metrics(error_x)
    mae_y, mse_y, rmse_y = calc_metrics(error_y)
    mae_z, mse_z, rmse_z = calc_metrics(error_z)

    print("\nTracking Error Metrics:")
    print(f"X axis:   MAE={mae_x:.5f}  MSE={mse_x:.5f}  RMSE={rmse_x:.5f}")
    print(f"Y axis:   MAE={mae_y:.5f}  MSE={mse_y:.5f}  RMSE={rmse_y:.5f}")
    print(f"Z axis:   MAE={mae_z:.5f}  MSE={mse_z:.5f}  RMSE={rmse_z:.5f}")

    print("\nRecommended: RMSE is most useful for penalizing large errors, but MAE is easier to interpret. Use both for a complete picture.")


    # Compute absolute errors
    abs_error_x = np.abs(error_x)
    abs_error_y = np.abs(error_y)
    abs_error_z = np.abs(error_z)

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top subplot: Absolute Y and Z errors
    axs[0].plot(time_vals, abs_error_y, 'b-', label='|Error in Y| (B - A)')
    axs[0].plot(time_vals, abs_error_z, 'g-', label='|Error in Z| (B - A)')
    axs[0].set_xlabel('Time (s)', fontsize=12)
    axs[0].set_ylabel('Absolute Error Y/Z (m)', fontsize=12)
    axs[0].set_title('Absolute Tracking Error in Y and Z Over Time', fontsize=14, fontweight='bold')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=11)

    # Bottom subplot: Absolute X error
    axs[1].plot(time_vals, abs_error_x, 'r-', label='|Error in X| (B - A)')
    axs[1].set_ylabel('Absolute Error X (m)', fontsize=12)
    axs[1].set_title('Absolute Tracking Error in X Over Time', fontsize=14, fontweight='bold')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=11)

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"tracking_error_{timestamp}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Tracking error plot saved: {output_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Dual robot tracking simulation")
    parser.add_argument(
        "--control-mode",
        choices=["pid", "rl"],
        default="pid",
        help="Tracking policy for the Franka controller.",
    )
    parser.add_argument(
        "--rl-model-path",
        default=None,
        help="Path to a Stable-Baselines3 PPO model zip for --control-mode rl.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=["vision"],
        default="vision",
        help="Observation format expected by the PPO model. RL policies use 17-D vision features.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["xyz", "yz", "x"],
        default="xyz",
        help="Action axes enabled for the PPO model.",
    )
    return parser.parse_args()


def main():
    """
    Main entry point for the dual-robot simulation.
    Sets up both robots in a shared PyBullet environment.
    """
    
    print("\n" + "="*70)
    print("DUAL ROBOT SYSTEM - MAIN CONTROLLER")
    print("="*70)
    args = parse_args()
    if args.control_mode == "rl" and not args.rl_model_path:
        raise ValueError("--rl-model-path is required when --control-mode rl")

    print("Initializing shared environment with both robots...\n")
    print(f"Control mode: {args.control_mode.upper()}\n")
    if args.control_mode == "rl":
        print(f"Observation mode: {args.observation_mode}")
        print(f"Action mode: {args.action_mode}\n")
    
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    sim_hz = 240.0
    p.setPhysicsEngineParameter(fixedTimeStep=1.0/SIM_HZ, numSolverIterations=12, numSubSteps=1)
    
    # Performance optimizations: disable shadows and heavy debug visuals
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    
    # Load ground plane
    plane = p.loadURDF("plane.urdf")
    print("✓ Ground plane loaded\n")

    if VIDEO_RECORDING:
        # Start video recording
        video_filename = f"simulation_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_filename)
        print(f"✓ Video recording started: {video_filename}\n")
    
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
    franka_base_y = 0.2
    franka_base_pos = [franka_base_x, franka_base_y, robotB.TABLE_HEIGHT + 0.02]
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
    
    vision_process = None
    image_queue = None
    result_queue = None
    if args.control_mode == "pid":
        print("\n" + "="*70)
        print("STARTING VISION PROCESSOR")
        print("="*70)
        vision_process, image_queue, result_queue = vision_processor.start_vision_process(
            camera_width=robotB.CAMERA_WIDTH,
            camera_height=robotB.CAMERA_HEIGHT,
            debug=False
        )
        print("✓ Vision processor started in separate process\n")
        tracking_policy = PIDTrackingPolicy(robotB.CAMERA_WIDTH, robotB.CAMERA_HEIGHT)
    else:
        print("\n" + "="*70)
        print("LOADING PPO TRACKING POLICY")
        print("="*70)
        tracking_policy = PPOTrackingPolicy(
            args.rl_model_path,
            observation_mode=args.observation_mode,
            action_mode=args.action_mode,
            camera_width=robotB.CAMERA_WIDTH,
            camera_height=robotB.CAMERA_HEIGHT,
            camera_far=robotB.CAMERA_FAR,
        )
        print(f"✓ PPO policy loaded: {args.rl_model_path}\n")
    
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
        policy=tracking_policy,
        image_queue=image_queue,
        result_queue=result_queue,
        target_body_id=cube,
        control_mode=args.control_mode,
        trajectory_points=robotA.NUM_CIRCLE_POINTS,
        observation_mode=args.observation_mode
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
            # time.sleep(1.0 / (4*SIM_HZ))
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    # Cleanup vision processor
    if vision_process is not None:
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

        # Generate tracking error plot (Robot B - Robot A)
        plot_tracking_errors(robotA.trajectory_data, controller_B.trajectory_data, output_dir=graphs_dir)
    
    if VIDEO_RECORDING:
        # Stop video recording
        p.stopStateLogging(video_id)
        print(f"\n✓ Video recording stopped: {video_filename}\n")
    # Disconnect PyBullet
    p.disconnect()


if __name__ == "__main__":
    main()
