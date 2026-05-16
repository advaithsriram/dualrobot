"""Ground-truth PyBullet tracking environment for Stage 1 RL."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required for RL training. Install dependencies from requirements.txt."
    ) from exc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SCRIPTS_ROOT)
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

import robotA
import robotB
import vision_processor
from controllers.franka_policies import make_ground_truth_observation
from pick_place import attach_object_to_robot, create_graspable_cube


@contextmanager
def suppress_native_output(enabled: bool):
    """Suppress Python and native-library stdout/stderr during noisy resets."""
    if not enabled:
        yield
        return

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with open(os.devnull, "w") as devnull:
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


class FrankaGroundTruthTrackingEnv(gym.Env):
    """Train Franka to track the UR5-held target using true target pose."""

    metadata = {"render_modes": ["human", "direct"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str = "direct",
        episode_steps: int = 600,
        action_scale: float = 0.02,
        trajectory_mode: str = "mixed",
        action_mode: str = "xyz",
        observation_mode: str = "ground_truth",
        seed: Optional[int] = None,
        quiet: bool = True,
        position_x_reward_weight: float = 80.0,
        position_yz_reward_weight: float = 50.0,
        velocity_x_reward_weight: float = 1.0,
        velocity_yz_reward_weight: float = 0.5,
        vision_pixel_noise_std: float = 0.0,
        vision_depth_noise_std: float = 0.0,
        vision_dropout_prob: float = 0.0,
        vision_debug: bool = False,
        vision_debug_every: int = 100,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.episode_steps = episode_steps
        self.action_scale = action_scale
        self.trajectory_mode = trajectory_mode
        if action_mode not in {"xyz", "yz", "x"}:
            raise ValueError("action_mode must be one of: xyz, yz, x")
        self.action_mode = action_mode
        if observation_mode not in {"ground_truth", "vision"}:
            raise ValueError("observation_mode must be one of: ground_truth, vision")
        self.observation_mode = observation_mode
        self.rng = np.random.default_rng(seed)
        self.quiet = quiet
        self.position_x_reward_weight = position_x_reward_weight
        self.position_yz_reward_weight = position_yz_reward_weight
        self.velocity_x_reward_weight = velocity_x_reward_weight
        self.velocity_yz_reward_weight = velocity_yz_reward_weight
        self.vision_pixel_noise_std = vision_pixel_noise_std
        self.vision_depth_noise_std = vision_depth_noise_std
        self.vision_dropout_prob = vision_dropout_prob
        self.vision_debug = vision_debug
        self.vision_debug_every = vision_debug_every

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_dim = 20 if self.observation_mode == "ground_truth" else 17
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.client = None
        self.plane = None
        self.ur5 = None
        self.panda = None
        self.cube = None
        self.constraint_id = None
        self.franka_ee_link = None
        self.ur5_ee_link = None
        self.franka_joints = []
        self.ur5_waypoints = []
        self.step_count = 0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_ee_pos = np.zeros(3, dtype=np.float32)
        self.previous_vision_error = np.zeros(3, dtype=np.float32)
        self.target_depth = None
        self.last_vision_detected = 0.0
        self.last_vision_area_norm = 0.0
        self.initial_franka_orn = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        with suppress_native_output(self.quiet):
            self._build_world()
        self.step_count = 0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_vision_error = np.zeros(3, dtype=np.float32)
        self.target_depth = None
        self.last_vision_detected = 0.0
        self.last_vision_area_norm = 0.0

        ee_state = p.getLinkState(self.panda, self.franka_ee_link, computeLinkVelocity=1)
        self.previous_ee_pos = np.array(ee_state[0], dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        raw_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action = self._apply_action_mode(raw_action)
        self._drive_ur5_target()
        self._drive_franka(action)

        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)

        tracking_state = self._get_tracking_state()
        error = tracking_state["target_pos"] - tracking_state["ee_pos"]
        velocity_error = tracking_state["target_vel"] - tracking_state["ee_vel"]
        error_norm = float(np.linalg.norm(error))
        error_x = float(abs(error[0]))
        error_yz = float(np.linalg.norm(error[1:3]))
        velocity_error_norm = float(np.linalg.norm(velocity_error))
        velocity_error_x = float(abs(velocity_error[0]))
        velocity_error_yz = float(np.linalg.norm(velocity_error[1:3]))
        delta_action = action - self.previous_action

        position_x_weight = self.position_x_reward_weight if self.action_mode in {"xyz", "x"} else 0.0
        position_yz_weight = self.position_yz_reward_weight if self.action_mode in {"xyz", "yz"} else 0.0
        velocity_x_weight = self.velocity_x_reward_weight if self.action_mode in {"xyz", "x"} else 0.0
        velocity_yz_weight = self.velocity_yz_reward_weight if self.action_mode in {"xyz", "yz"} else 0.0

        active_error = error_norm
        if self.action_mode == "yz":
            active_error = error_yz
        elif self.action_mode == "x":
            active_error = error_x

        reward = (
            -position_x_weight * error_x**2
            -position_yz_weight * error_yz**2
            -velocity_x_weight * velocity_error_x**2
            -velocity_yz_weight * velocity_error_yz**2
            -0.05 * float(np.dot(action, action))
            -0.2 * float(np.dot(delta_action, delta_action))
        )
        if active_error < 0.02:
            reward += 1.0
        if active_error > 0.45:
            reward -= 5.0

        self.previous_action = action
        self.step_count += 1
        obs = self._get_obs(tracking_state)
        terminated = False
        truncated = self.step_count >= self.episode_steps
        info = {
            "tracking_error": error_norm,
            "tracking_error_x": error_x,
            "tracking_error_yz": error_yz,
            "velocity_error": velocity_error_norm,
            "velocity_error_x": velocity_error_x,
            "velocity_error_yz": velocity_error_yz,
            "action_mode": self.action_mode,
            "observation_mode": self.observation_mode,
            "vision_detected": self.last_vision_detected,
            "vision_area_norm": self.last_vision_area_norm,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.client is not None:
            p.disconnect(physicsClientId=self.client)
            self.client = None

    def _build_world(self):
        self.close()
        connection_mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self.client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 120.0,
            numSolverIterations=12,
            numSubSteps=1,
            physicsClientId=self.client,
        )

        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self._load_ur5_target_robot()
        self._load_franka_tracker()

        for _ in range(80):
            p.stepSimulation(physicsClientId=self.client)

    def _load_ur5_target_robot(self):
        ur5_table_height = 0.5
        table_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.5, 0.5, ur5_table_height / 2], physicsClientId=self.client
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.5, ur5_table_height / 2],
            rgbaColor=[1, 1, 1, 1],
            physicsClientId=self.client,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.5, 0, ur5_table_height / 2],
            physicsClientId=self.client,
        )

        ur5_path = os.path.join(REPO_ROOT, "urdf", "ur5.urdf")
        self.ur5 = p.loadURDF(
            ur5_path,
            basePosition=[0.5, 0, ur5_table_height + 0.02],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client,
        )
        initial_positions = [np.deg2rad(angle) for angle in robotA.INITIAL_JOINT_ANGLES]
        for i, joint_pos in enumerate(initial_positions):
            p.resetJointState(self.ur5, i, joint_pos, physicsClientId=self.client)

        self.ur5_ee_link = p.getNumJoints(self.ur5, physicsClientId=self.client) - 1
        ee_state = p.getLinkState(self.ur5, self.ur5_ee_link, physicsClientId=self.client)
        center_pos = ee_state[0]

        self.cube = create_graspable_cube(
            position=center_pos,
            size=robotA.CUBE_SIZE,
            color=[1, 0, 0, 1],
            mass=0.05,
        )
        self.constraint_id = attach_object_to_robot(self.ur5, self.cube, self.ur5_ee_link)

        circle = robotA.precompute_circular_trajectory(
            self.ur5,
            self.ur5_ee_link,
            center_pos,
            robotA.CIRCLE_RADIUS,
            robotA.NUM_CIRCLE_POINTS,
        )
        lissajous = robotA.precompute_lissajous_trajectory(
            self.ur5,
            self.ur5_ee_link,
            center_pos,
            robotA.LISSAJOUS_AMPLITUDE_Y,
            robotA.LISSAJOUS_AMPLITUDE_Z,
            robotA.NUM_LISSAJOUS_POINTS,
        )

        if self.trajectory_mode == "circle":
            self.ur5_waypoints = circle
        elif self.trajectory_mode == "lissajous":
            self.ur5_waypoints = lissajous
        else:
            self.ur5_waypoints = circle + lissajous

    def _load_franka_tracker(self):
        franka_base_pos = [2.0, 0.2, robotB.TABLE_HEIGHT + 0.02]
        franka_orientation = [0, 0, np.pi]
        table_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.5, 0.5, robotB.TABLE_HEIGHT / 2], physicsClientId=self.client
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.5, robotB.TABLE_HEIGHT / 2],
            rgbaColor=[1, 1, 1, 1],
            physicsClientId=self.client,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[franka_base_pos[0], franka_base_pos[1], robotB.TABLE_HEIGHT / 2],
            physicsClientId=self.client,
        )

        panda_path = os.path.join(REPO_ROOT, "urdf", "panda.urdf")
        self.panda = p.loadURDF(
            panda_path,
            basePosition=franka_base_pos,
            baseOrientation=p.getQuaternionFromEuler(franka_orientation),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client,
        )
        self.franka_joints = []
        for i in range(p.getNumJoints(self.panda, physicsClientId=self.client)):
            joint_info = p.getJointInfo(self.panda, i, physicsClientId=self.client)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.franka_joints.append(i)

        for i, joint_idx in enumerate(self.franka_joints[:7]):
            p.resetJointState(
                self.panda,
                joint_idx,
                robotB.INITIAL_JOINT_ANGLES[i],
                physicsClientId=self.client,
            )
        self.franka_ee_link = 7
        ee_state = p.getLinkState(self.panda, self.franka_ee_link, physicsClientId=self.client)
        self.initial_franka_orn = ee_state[1]

    def _drive_ur5_target(self):
        waypoint = self.ur5_waypoints[self.step_count % len(self.ur5_waypoints)]
        for j in range(6):
            p.setJointMotorControl2(
                bodyIndex=self.ur5,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=waypoint[j],
                maxVelocity=10.0,
                force=500,
                positionGain=0.3,
                physicsClientId=self.client,
            )

    def _drive_franka(self, action):
        ee_state = p.getLinkState(self.panda, self.franka_ee_link, physicsClientId=self.client)
        current_pos = np.array(ee_state[0], dtype=np.float32)
        target_pos = current_pos + action * self.action_scale
        target_joints = p.calculateInverseKinematics(
            self.panda,
            self.franka_ee_link,
            target_pos.tolist(),
            self.initial_franka_orn,
            maxNumIterations=20,
            residualThreshold=1e-4,
            physicsClientId=self.client,
        )
        for i, joint_idx in enumerate(self.franka_joints[:7]):
            if i < len(target_joints):
                p.setJointMotorControl2(
                    bodyIndex=self.panda,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_joints[i],
                    force=500,
                    maxVelocity=1.0,
                    positionGain=0.3,
                    physicsClientId=self.client,
                )

    def _apply_action_mode(self, action):
        masked_action = np.array(action, dtype=np.float32, copy=True)
        if self.action_mode == "yz":
            masked_action[0] = 0.0
        elif self.action_mode == "x":
            masked_action[1:] = 0.0
        return masked_action

    def _get_tracking_state(self):
        ee_state = p.getLinkState(
            self.panda,
            self.franka_ee_link,
            computeLinkVelocity=1,
            physicsClientId=self.client,
        )
        ee_pos = np.array(ee_state[0], dtype=np.float32)
        ee_vel = np.array(ee_state[6], dtype=np.float32) if len(ee_state) > 6 else ee_pos - self.previous_ee_pos
        target_pos, _ = p.getBasePositionAndOrientation(self.cube, physicsClientId=self.client)
        target_vel, _ = p.getBaseVelocity(self.cube, physicsClientId=self.client)
        phase = 2 * np.pi * (self.step_count % len(self.ur5_waypoints)) / len(self.ur5_waypoints)
        self.previous_ee_pos = ee_pos
        return {
            "ee_pos": ee_pos,
            "ee_vel": ee_vel,
            "target_pos": np.asarray(target_pos, dtype=np.float32),
            "target_vel": np.asarray(target_vel, dtype=np.float32),
            "phase": phase,
        }

    def _get_obs(self, tracking_state=None):
        if tracking_state is None:
            tracking_state = self._get_tracking_state()

        if self.observation_mode == "vision":
            return self._get_vision_observation(tracking_state)

        return make_ground_truth_observation(
            ee_pos=tracking_state["ee_pos"],
            ee_vel=tracking_state["ee_vel"],
            target_pos=tracking_state["target_pos"],
            target_vel=tracking_state["target_vel"],
            previous_action=self.previous_action,
            phase=tracking_state["phase"],
        )

    def _get_vision_observation(self, tracking_state):
        rgb_image, depth_array = robotB.get_camera_image(self.panda, self.franka_ee_link)
        pixel_x, pixel_y, detected, area = vision_processor.detect_red_object(
            rgb_image,
            robotB.CAMERA_WIDTH,
            robotB.CAMERA_HEIGHT,
            debug=False,
        )
        if detected and self.vision_dropout_prob > 0.0:
            detected = bool(self.rng.random() >= self.vision_dropout_prob)

        if detected and self.vision_pixel_noise_std > 0.0:
            pixel_x += float(self.rng.normal(0.0, self.vision_pixel_noise_std))
            pixel_y += float(self.rng.normal(0.0, self.vision_pixel_noise_std))
            pixel_x = float(np.clip(pixel_x, 0, robotB.CAMERA_WIDTH - 1))
            pixel_y = float(np.clip(pixel_y, 0, robotB.CAMERA_HEIGHT - 1))

        depth_value = 0.0
        if detected:
            px = int(np.clip(pixel_x, 0, robotB.CAMERA_WIDTH - 1))
            py = int(np.clip(pixel_y, 0, robotB.CAMERA_HEIGHT - 1))
            depth_value = float(depth_array[py, px])
            if self.vision_depth_noise_std > 0.0:
                depth_value += float(self.rng.normal(0.0, self.vision_depth_noise_std))
            if self.target_depth is None:
                self.target_depth = depth_value

        if not detected:
            pixel_error_x = 0.0
            pixel_error_y = 0.0
            depth_error = 0.0
        else:
            pixel_error_x = (pixel_x - robotB.CAMERA_WIDTH / 2) / (robotB.CAMERA_WIDTH / 2)
            pixel_error_y = (pixel_y - robotB.CAMERA_HEIGHT / 2) / (robotB.CAMERA_HEIGHT / 2)
            depth_error = 0.0 if self.target_depth is None else depth_value - self.target_depth

        vision_error = np.array([pixel_error_x, pixel_error_y, depth_error], dtype=np.float32)
        vision_error_delta = vision_error - self.previous_vision_error
        self.previous_vision_error = vision_error

        area_norm = float(area) / float(robotB.CAMERA_WIDTH * robotB.CAMERA_HEIGHT)
        depth_norm = depth_value / robotB.CAMERA_FAR if detected else 0.0
        detected_flag = 1.0 if detected else 0.0
        self.last_vision_detected = detected_flag
        self.last_vision_area_norm = area_norm

        if self.vision_debug and self.step_count % max(1, self.vision_debug_every) == 0:
            print(
                "[vision] "
                f"step={self.step_count:,} "
                f"detected={bool(detected)} "
                f"pixel=({pixel_x:.1f},{pixel_y:.1f}) "
                f"pixel_error_norm=({pixel_error_x:.3f},{pixel_error_y:.3f}) "
                f"depth={depth_value:.4f}m "
                f"target_depth={0.0 if self.target_depth is None else self.target_depth:.4f}m "
                f"depth_error={depth_error:.4f}m "
                f"area_norm={area_norm:.5f}"
            )

        return np.concatenate(
            [
                vision_error,
                vision_error_delta,
                np.array([detected_flag, area_norm, depth_norm], dtype=np.float32),
                tracking_state["ee_vel"],
                self.previous_action,
                np.array(
                    [
                        np.sin(tracking_state["phase"]),
                        np.cos(tracking_state["phase"]),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
