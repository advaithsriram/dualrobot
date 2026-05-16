"""Train a PPO policy for ground-truth 3D end-effector tracking."""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.tracking_env import FrankaGroundTruthTrackingEnv


class PeriodicTrainingPrinter(BaseCallback):
    def __init__(self, print_every: int):
        super().__init__()
        self.print_every = print_every
        self.next_print = print_every

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        tracking_errors = [
            info["tracking_error"]
            for info in infos
            if isinstance(info, dict) and "tracking_error" in info
        ]
        tracking_errors_x = [
            info["tracking_error_x"]
            for info in infos
            if isinstance(info, dict) and "tracking_error_x" in info
        ]
        tracking_errors_x_world = [
            info["tracking_error_x_world"]
            for info in infos
            if isinstance(info, dict) and "tracking_error_x_world" in info
        ]
        tracking_errors_yz = [
            info["tracking_error_yz"]
            for info in infos
            if isinstance(info, dict) and "tracking_error_yz" in info
        ]
        velocity_errors = [
            info["velocity_error"]
            for info in infos
            if isinstance(info, dict) and "velocity_error" in info
        ]
        velocity_errors_x = [
            info["velocity_error_x"]
            for info in infos
            if isinstance(info, dict) and "velocity_error_x" in info
        ]
        velocity_errors_yz = [
            info["velocity_error_yz"]
            for info in infos
            if isinstance(info, dict) and "velocity_error_yz" in info
        ]
        vision_detected = [
            info["vision_detected"]
            for info in infos
            if isinstance(info, dict) and "vision_detected" in info
        ]
        vision_area_norm = [
            info["vision_area_norm"]
            for info in infos
            if isinstance(info, dict) and "vision_area_norm" in info
        ]

        if len(rewards) > 0:
            mean_reward = sum(float(reward) for reward in rewards) / len(rewards)
            self.logger.record("tracking/step_reward", mean_reward)
        else:
            mean_reward = None

        if tracking_errors:
            mean_error = sum(tracking_errors) / len(tracking_errors)
            self.logger.record("tracking/error_m", mean_error)

        if tracking_errors_x:
            mean_error_x = sum(tracking_errors_x) / len(tracking_errors_x)
            self.logger.record("tracking/error_x_relative_m", mean_error_x)

        if tracking_errors_x_world:
            mean_error_x_world = sum(tracking_errors_x_world) / len(tracking_errors_x_world)
            self.logger.record("tracking/error_x_world_m", mean_error_x_world)

        if tracking_errors_yz:
            mean_error_yz = sum(tracking_errors_yz) / len(tracking_errors_yz)
            self.logger.record("tracking/error_yz_m", mean_error_yz)

        if velocity_errors:
            mean_velocity_error = sum(velocity_errors) / len(velocity_errors)
            self.logger.record("tracking/velocity_error_mps", mean_velocity_error)

        if velocity_errors_x:
            mean_velocity_error_x = sum(velocity_errors_x) / len(velocity_errors_x)
            self.logger.record("tracking/velocity_error_x_mps", mean_velocity_error_x)

        if velocity_errors_yz:
            mean_velocity_error_yz = sum(velocity_errors_yz) / len(velocity_errors_yz)
            self.logger.record("tracking/velocity_error_yz_mps", mean_velocity_error_yz)

        if vision_detected:
            mean_detected = sum(vision_detected) / len(vision_detected)
            self.logger.record("vision/detected", mean_detected)

        if vision_area_norm:
            mean_area_norm = sum(vision_area_norm) / len(vision_area_norm)
            self.logger.record("vision/area_norm", mean_area_norm)

        if self.print_every <= 0 or self.num_timesteps < self.next_print:
            return True

        reward_text = f" reward={mean_reward:.4f}" if mean_reward is not None else ""
        if tracking_errors:
            mean_error = sum(tracking_errors) / len(tracking_errors)
            if velocity_errors:
                mean_velocity_error = sum(velocity_errors) / len(velocity_errors)
                detail_text = ""
                if tracking_errors_x and tracking_errors_yz:
                    mean_error_x = sum(tracking_errors_x) / len(tracking_errors_x)
                    mean_error_yz = sum(tracking_errors_yz) / len(tracking_errors_yz)
                    detail_text = f" x_rel_error={mean_error_x:.5f} m yz_error={mean_error_yz:.5f} m"
                velocity_detail_text = ""
                if velocity_errors_x and velocity_errors_yz:
                    mean_velocity_error_x = sum(velocity_errors_x) / len(velocity_errors_x)
                    mean_velocity_error_yz = sum(velocity_errors_yz) / len(velocity_errors_yz)
                    velocity_detail_text = (
                        f" x_vel_error={mean_velocity_error_x:.5f} m/s"
                        f" yz_vel_error={mean_velocity_error_yz:.5f} m/s"
                    )
                print(
                    f"[train] step={self.num_timesteps:,} "
                    f"{reward_text} "
                    f"tracking_error={mean_error:.5f} m "
                    f"{detail_text} "
                    f"velocity_error={mean_velocity_error:.5f} m/s"
                    f"{velocity_detail_text}"
                )
            else:
                print(f"[train] step={self.num_timesteps:,}{reward_text} tracking_error={mean_error:.5f} m")
        else:
            print(f"[train] step={self.num_timesteps:,}{reward_text}")
        self.next_print += self.print_every
        return True


class PeriodicCheckpointSaver(BaseCallback):
    def __init__(self, save_every: int, checkpoint_dir: str, checkpoint_name: str):
        super().__init__()
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.next_save = save_every

    def _on_training_start(self) -> None:
        if self.save_every > 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.save_every <= 0 or self.num_timesteps < self.next_save:
            return True

        save_path = os.path.join(
            self.checkpoint_dir,
            f"{self.checkpoint_name}_{self.num_timesteps}_steps",
        )
        self.model.save(save_path)
        print(f"[checkpoint] saved {save_path}.zip")
        while self.next_save <= self.num_timesteps:
            self.next_save += self.save_every
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO tracker with ground-truth target pose.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--load-path", default=None)
    parser.add_argument("--save-path", default="../models/ppo_franka_tracker")
    parser.add_argument("--checkpoint-dir", default="../models/checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=200_000)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--log-dir", default="../runs/ppo_franka_tracker")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--trajectory-mode", choices=["circle", "lissajous", "mixed"], default="mixed")
    parser.add_argument("--action-mode", choices=["xyz", "yz", "x"], default="xyz")
    parser.add_argument("--observation-mode", choices=["ground_truth", "vision"], default="ground_truth")
    parser.add_argument("--print-every", type=int, default=10_000)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--sb3-verbose", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--env-verbose", action="store_true")
    parser.add_argument("--position-x-reward-weight", type=float, default=80.0)
    parser.add_argument("--position-yz-reward-weight", type=float, default=50.0)
    parser.add_argument("--velocity-x-reward-weight", type=float, default=1.0)
    parser.add_argument("--velocity-yz-reward-weight", type=float, default=0.5)
    parser.add_argument("--vision-pixel-noise-std", type=float, default=0.0)
    parser.add_argument("--vision-depth-noise-std", type=float, default=0.0)
    parser.add_argument("--vision-dropout-prob", type=float, default=0.0)
    parser.add_argument("--vision-debug", action="store_true")
    parser.add_argument("--vision-debug-every", type=int, default=100)
    return parser.parse_args()


def make_env(
    render_mode,
    trajectory_mode,
    action_mode,
    observation_mode,
    quiet,
    position_x_reward_weight,
    position_yz_reward_weight,
    velocity_x_reward_weight,
    velocity_yz_reward_weight,
    vision_pixel_noise_std,
    vision_depth_noise_std,
    vision_dropout_prob,
    vision_debug,
    vision_debug_every,
):
    def _factory():
        env = FrankaGroundTruthTrackingEnv(
            render_mode=render_mode,
            trajectory_mode=trajectory_mode,
            action_mode=action_mode,
            observation_mode=observation_mode,
            quiet=quiet,
            position_x_reward_weight=position_x_reward_weight,
            position_yz_reward_weight=position_yz_reward_weight,
            velocity_x_reward_weight=velocity_x_reward_weight,
            velocity_yz_reward_weight=velocity_yz_reward_weight,
            vision_pixel_noise_std=vision_pixel_noise_std,
            vision_depth_noise_std=vision_depth_noise_std,
            vision_dropout_prob=vision_dropout_prob,
            vision_debug=vision_debug,
            vision_debug_every=vision_debug_every,
        )
        return Monitor(env)

    return _factory


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    render_mode = "human" if args.render else "direct"
    env = DummyVecEnv([
        make_env(
            render_mode,
            args.trajectory_mode,
            args.action_mode,
            args.observation_mode,
            quiet=not args.env_verbose,
            position_x_reward_weight=args.position_x_reward_weight,
            position_yz_reward_weight=args.position_yz_reward_weight,
            velocity_x_reward_weight=args.velocity_x_reward_weight,
            velocity_yz_reward_weight=args.velocity_yz_reward_weight,
            vision_pixel_noise_std=args.vision_pixel_noise_std,
            vision_depth_noise_std=args.vision_depth_noise_std,
            vision_dropout_prob=args.vision_dropout_prob,
            vision_debug=args.vision_debug,
            vision_debug_every=args.vision_debug_every,
        )
    ])

    if args.load_path:
        model = PPO.load(
            args.load_path,
            env=env,
            verbose=args.sb3_verbose,
            tensorboard_log=args.log_dir,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=args.sb3_verbose,
            tensorboard_log=args.log_dir,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            clip_range=0.2,
        )

    checkpoint_name = args.checkpoint_name
    if checkpoint_name is None:
        checkpoint_name = os.path.basename(args.save_path.rstrip(os.sep))

    callback = CallbackList([
        PeriodicTrainingPrinter(args.print_every),
        PeriodicCheckpointSaver(
            save_every=args.checkpoint_every,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=checkpoint_name,
        ),
    ])

    try:
        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=args.progress_bar,
            callback=callback,
            reset_num_timesteps=args.load_path is None,
        )
    except KeyboardInterrupt:
        interrupted_path = f"{args.save_path}_interrupted"
        model.save(interrupted_path)
        print(f"\nInterrupted. Saved latest model to {interrupted_path}.zip")
    else:
        model.save(args.save_path)
        print(f"Saved PPO tracker to {args.save_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
