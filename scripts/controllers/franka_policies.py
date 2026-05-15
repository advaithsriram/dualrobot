"""Swappable tracking policies for the Franka end-effector.

Policies return a Cartesian displacement in the world frame. The caller is
responsible for applying IK and sending joint commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pybullet as p


@dataclass
class TrackingCommand:
    delta_world: np.ndarray
    debug: dict


class BaseTrackingPolicy:
    def reset(self) -> None:
        pass

    def act(self, context: dict) -> Optional[TrackingCommand]:
        raise NotImplementedError


class PIDTrackingPolicy(BaseTrackingPolicy):
    """Frozen visual-servoing PID baseline behavior from the original controller."""

    def __init__(
        self,
        camera_width: int,
        camera_height: int,
        pixel_to_meter_x: float = 0.001,
        pixel_to_meter_y: float = 0.001,
        pixel_to_meter_x_d: float = 0.0005,
        pixel_to_meter_y_d: float = 0.0005,
        depth_to_meter_z: float = 0.25,
        depth_to_meter_z_d: float = 0.04,
        filter_alpha: float = 0.7,
    ):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.pixel_to_meter_x = pixel_to_meter_x
        self.pixel_to_meter_y = pixel_to_meter_y
        self.pixel_to_meter_x_d = pixel_to_meter_x_d
        self.pixel_to_meter_y_d = pixel_to_meter_y_d
        self.depth_to_meter_z = depth_to_meter_z
        self.depth_to_meter_z_d = depth_to_meter_z_d
        self.filter_alpha = filter_alpha
        self.reset()

    def reset(self) -> None:
        self.target_depth = None
        self.last_error_x_pixels = 0.0
        self.last_error_y_pixels = 0.0
        self.last_error_depth = 0.0
        self.filtered_error_x = 0.0
        self.filtered_error_y = 0.0
        self.filtered_error_depth = 0.005

    def act(self, context: dict) -> Optional[TrackingCommand]:
        detection = context.get("detection")
        if detection is None or not detection.get("detected", False):
            return None

        pixel_x = detection["pixel_x"]
        pixel_y = detection["pixel_y"]
        depth = detection.get("depth")

        if self.target_depth is None and depth is not None:
            self.target_depth = depth
            print(f"[Tracking] Target depth set to {depth:.4f} m (initial distance locked)")

        error_x_pixels = pixel_x - self.camera_width / 2
        error_y_pixels = pixel_y - self.camera_height / 2

        error_depth = 0.0
        if depth is not None and self.target_depth is not None:
            error_depth = depth - self.target_depth

        self.filtered_error_x = (
            self.filter_alpha * error_x_pixels
            + (1 - self.filter_alpha) * self.filtered_error_x
        )
        self.filtered_error_y = (
            self.filter_alpha * error_y_pixels
            + (1 - self.filter_alpha) * self.filtered_error_y
        )
        self.filtered_error_depth = (
            self.filter_alpha * error_depth
            + (1 - self.filter_alpha) * self.filtered_error_depth
        )

        error_delta_x = self.filtered_error_x - self.last_error_x_pixels
        error_delta_y = self.filtered_error_y - self.last_error_y_pixels
        error_delta_z = self.filtered_error_depth - self.last_error_depth

        self.last_error_x_pixels = self.filtered_error_x
        self.last_error_y_pixels = self.filtered_error_y
        self.last_error_depth = self.filtered_error_depth

        delta_x_camera = (
            self.pixel_to_meter_x * self.filtered_error_x
            + self.pixel_to_meter_x_d * error_delta_x
        )
        delta_y_camera = (
            self.pixel_to_meter_y * self.filtered_error_y
            + self.pixel_to_meter_y_d * error_delta_y
        )
        delta_z_camera = (
            self.depth_to_meter_z * self.filtered_error_depth
            + self.depth_to_meter_z_d * error_delta_z
        )

        current_orn = context["current_orn"]
        rot_matrix = np.array(p.getMatrixFromQuaternion(current_orn)).reshape(3, 3)
        delta_camera = np.array([delta_x_camera, delta_y_camera, delta_z_camera])
        delta_world = rot_matrix.dot(delta_camera)

        return TrackingCommand(
            delta_world=delta_world,
            debug={
                "pixel_error": (error_x_pixels, error_y_pixels),
                "depth_error": error_depth,
            },
        )


class PPOTrackingPolicy(BaseTrackingPolicy):
    """Ground-truth pose tracking policy loaded from Stable-Baselines3 PPO."""

    def __init__(self, model_path: str, action_scale: float = 0.02):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required for --control-mode rl. "
                "Install dependencies from requirements.txt."
            ) from exc

        self.model = PPO.load(model_path)
        self.action_scale = action_scale
        self.previous_action = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        self.previous_action = np.zeros(3, dtype=np.float32)

    def act(self, context: dict) -> Optional[TrackingCommand]:
        observation = make_ground_truth_observation(
            ee_pos=np.asarray(context["current_pos"], dtype=np.float32),
            ee_vel=np.asarray(context.get("ee_vel", np.zeros(3)), dtype=np.float32),
            target_pos=np.asarray(context["target_pos"], dtype=np.float32),
            target_vel=np.asarray(context.get("target_vel", np.zeros(3)), dtype=np.float32),
            previous_action=self.previous_action,
            phase=float(context.get("phase", 0.0)),
        )
        action, _ = self.model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        self.previous_action = action
        delta_world = np.clip(action, -1.0, 1.0) * self.action_scale
        return TrackingCommand(
            delta_world=delta_world,
            debug={"action": action, "tracking_error": observation[12:15]},
        )


def make_ground_truth_observation(
    ee_pos: np.ndarray,
    ee_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    previous_action: np.ndarray,
    phase: float,
) -> np.ndarray:
    error = target_pos - ee_pos
    return np.concatenate(
        [
            ee_pos,
            ee_vel,
            target_pos,
            target_vel,
            error,
            previous_action,
            np.array([np.sin(phase), np.cos(phase)], dtype=np.float32),
        ]
    ).astype(np.float32)
