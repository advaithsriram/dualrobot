"""Evaluate a trained PPO policy in the Stage 1 ground-truth environment."""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from rl.tracking_env import FrankaGroundTruthTrackingEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO tracker with ground-truth target pose.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--trajectory-mode", choices=["circle", "lissajous", "mixed"], default="mixed")
    parser.add_argument("--action-mode", choices=["xyz", "yz", "x"], default="xyz")
    parser.add_argument("--position-x-reward-weight", type=float, default=80.0)
    parser.add_argument("--position-yz-reward-weight", type=float, default=50.0)
    parser.add_argument("--velocity-x-reward-weight", type=float, default=1.0)
    parser.add_argument("--velocity-yz-reward-weight", type=float, default=0.5)
    parser.add_argument("--env-verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    model = PPO.load(args.model_path)
    env = FrankaGroundTruthTrackingEnv(
        render_mode="human" if args.render else "direct",
        trajectory_mode=args.trajectory_mode,
        action_mode=args.action_mode,
        position_x_reward_weight=args.position_x_reward_weight,
        position_yz_reward_weight=args.position_yz_reward_weight,
        velocity_x_reward_weight=args.velocity_x_reward_weight,
        velocity_yz_reward_weight=args.velocity_yz_reward_weight,
        quiet=not args.env_verbose,
    )

    episode_rmse = []
    episode_mae = []
    episode_x_rmse = []
    episode_x_mae = []
    episode_yz_rmse = []
    episode_yz_mae = []
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        errors = []
        errors_x = []
        errors_yz = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            errors.append(info["tracking_error"])
            errors_x.append(info["tracking_error_x"])
            errors_yz.append(info["tracking_error_yz"])

        errors = np.asarray(errors)
        errors_x = np.asarray(errors_x)
        errors_yz = np.asarray(errors_yz)
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        x_rmse = float(np.sqrt(np.mean(errors_x**2)))
        x_mae = float(np.mean(np.abs(errors_x)))
        yz_rmse = float(np.sqrt(np.mean(errors_yz**2)))
        yz_mae = float(np.mean(np.abs(errors_yz)))
        episode_rmse.append(rmse)
        episode_mae.append(mae)
        episode_x_rmse.append(x_rmse)
        episode_x_mae.append(x_mae)
        episode_yz_rmse.append(yz_rmse)
        episode_yz_mae.append(yz_mae)
        print(
            f"Episode {episode + 1}: "
            f"3D RMSE={rmse:.5f} m, MAE={mae:.5f} m | "
            f"X RMSE={x_rmse:.5f} m, MAE={x_mae:.5f} m | "
            f"YZ RMSE={yz_rmse:.5f} m, MAE={yz_mae:.5f} m"
        )

    print("\nSummary")
    print(f"3D RMSE: mean={np.mean(episode_rmse):.5f} m, std={np.std(episode_rmse):.5f} m")
    print(f"3D MAE:  mean={np.mean(episode_mae):.5f} m, std={np.std(episode_mae):.5f} m")
    print(f"X RMSE:  mean={np.mean(episode_x_rmse):.5f} m, std={np.std(episode_x_rmse):.5f} m")
    print(f"X MAE:   mean={np.mean(episode_x_mae):.5f} m, std={np.std(episode_x_mae):.5f} m")
    print(f"YZ RMSE: mean={np.mean(episode_yz_rmse):.5f} m, std={np.std(episode_yz_rmse):.5f} m")
    print(f"YZ MAE:  mean={np.mean(episode_yz_mae):.5f} m, std={np.std(episode_yz_mae):.5f} m")
    env.close()


if __name__ == "__main__":
    main()
