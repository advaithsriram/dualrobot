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
    """PPO tracking policy loaded from Stable-Baselines3."""

    def __init__(
        self,
        model_path: str,
        action_scale: float = 0.02,
        observation_mode: str = "vision",
        action_mode: str = "xyz",
        camera_width: int = 320,
        camera_height: int = 240,
        camera_far: float = 5.0,
    ):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required for --control-mode rl. "
                "Install dependencies from requirements.txt."
            ) from exc

        self.model = PPO.load(model_path)
        self.action_scale = action_scale
        if observation_mode != "vision":
            raise ValueError("Only observation_mode='vision' is supported for PPO policies.")
        if action_mode not in {"xyz", "yz", "x"}:
            raise ValueError("action_mode must be one of: xyz, yz, x")
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_far = camera_far
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_vision_error = np.zeros(3, dtype=np.float32)
        self.target_depth = None

    def reset(self) -> None:
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_vision_error = np.zeros(3, dtype=np.float32)
        self.target_depth = None

    def act(self, context: dict) -> Optional[TrackingCommand]:
        observation = self._make_vision_observation(context)
        action, _ = self.model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        action = self._apply_action_mode(np.clip(action, -1.0, 1.0))
        self.previous_action = action
        delta_world = action * self.action_scale
        return TrackingCommand(
            delta_world=delta_world,
            debug={"action": action, "observation_mode": self.observation_mode},
        )

    def _make_vision_observation(self, context: dict) -> np.ndarray:
        detection = context.get("detection")
        detected = bool(detection and detection.get("detected", False))

        pixel_error_x = 0.0
        pixel_error_y = 0.0
        depth_error = 0.0
        area_norm = 0.0
        depth_norm = 0.0
        depth_value = 0.0

        if detected:
            pixel_x = float(detection["pixel_x"])
            pixel_y = float(detection["pixel_y"])
            depth = detection.get("depth")
            depth_value = 0.0 if depth is None else float(depth)

            pixel_error_x = (pixel_x - self.camera_width / 2) / (self.camera_width / 2)
            pixel_error_y = (pixel_y - self.camera_height / 2) / (self.camera_height / 2)
            if self.target_depth is None and depth is not None:
                self.target_depth = depth_value
            if self.target_depth is not None and depth is not None:
                depth_error = depth_value - self.target_depth

            area_norm = float(detection.get("area", 0.0)) / float(self.camera_width * self.camera_height)
            depth_norm = depth_value / self.camera_far

        vision_error = np.array([pixel_error_x, pixel_error_y, depth_error], dtype=np.float32)
        vision_error_delta = vision_error - self.previous_vision_error
        self.previous_vision_error = vision_error

        return np.concatenate(
            [
                vision_error,
                vision_error_delta,
                np.array([1.0 if detected else 0.0, area_norm, depth_norm], dtype=np.float32),
                np.asarray(context.get("ee_vel", np.zeros(3)), dtype=np.float32),
                self.previous_action,
                np.array(
                    [
                        np.sin(float(context.get("phase", 0.0))),
                        np.cos(float(context.get("phase", 0.0))),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

    def _apply_action_mode(self, action: np.ndarray) -> np.ndarray:
        masked_action = np.array(action, dtype=np.float32, copy=True)
        if self.action_mode == "yz":
            masked_action[0] = 0.0
        elif self.action_mode == "x":
            masked_action[1:] = 0.0
        return masked_action
