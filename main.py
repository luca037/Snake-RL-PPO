import os
import torch
import csv
import numpy as np
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv

from Model import SnakeCNN
from Game import snake_make_env


###############################################################################
### Global vars.
###############################################################################

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("INFO: CUDA is available. Running on GPU.")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("INFO: MPS DEVICE found. Running on GPU.")
else:
    print("INFO: CUDA and MPS not available. Running on CPU.")

# Unified TensorBoard log path (fixes the path discrepancy bug).
TENSORBOARD_LOG_DIR = "./output/plots/tensorboard/"


# Custom callback to log average score during training.
class ScoreLoggingCallback(BaseCallback):
    """Logs the average episode score every N episodes to TensorBoard."""

    def __init__(self, log_interval=100, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_scores = []

    def _on_step(self) -> bool:
        # Check info dicts for completed episodes.
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if done and i < len(infos):
                score = infos[i].get("score", 0)
                self.episode_scores.append(score)

        # Log aggregated stats at the specified interval.
        if len(self.episode_scores) >= self.log_interval:
            avg_score = np.mean(self.episode_scores)
            max_score = np.max(self.episode_scores)
            win_rate = np.mean([1 if s >= 97 else 0 for s in self.episode_scores])
            self.logger.record("snake/avg_score", avg_score)
            self.logger.record("snake/max_score", max_score)
            self.logger.record("snake/win_rate", win_rate)
            if self.verbose:
                print(
                    f"[ScoreLog] Episodes: {len(self.episode_scores)} | "
                    f"Avg: {avg_score:.1f} | Max: {max_score} | WinRate: {win_rate:.3f}"
                )
            self.episode_scores = []
        return True


###############################################################################
### Functions.
###############################################################################

def evaluate_model(model_path="", num_games=1000, num_envs=64, csv_path="eval_scores.csv"):
    """
    Loads a trained SB3 model and evaluates it across multiple vectorized
    environments simultaneously for massive speed improvements.
    Saves the raw scores to a CSV file.
    """
    print(f"INFO: Loading model from '{model_path}'...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"ERROR: Could not load model. Details: {e}")
        return

    vec_env = make_vec_env(snake_make_env, n_envs=num_envs)

    scores = []
    record = 0

    print(f"\nSimulating {num_games} games across {num_envs} environments...")
    print("-" * 50)

    # Get the initial batch of observations.
    obs = vec_env.reset()

    while len(scores) < num_games:
        # Predict actions for all environments at once.
        actions, _ = model.predict(obs, deterministic=True)

        # Step all environments simultaneously.
        obs, _, dones, infos = vec_env.step(actions)

        # Check which environments finished in this step.
        for i, done in enumerate(dones):
            if done:
                score = infos[i].get("score", 0)
                scores.append(score)
                if score > record:
                    record = score
                # Print progress every 100 games.
                if not len(scores) % 100 or len(scores) == num_games:
                    current_avg = np.mean(scores)
                    print(f"Games Completed: {len(scores)}/{num_games} | Avg: {current_avg:.1f} | Record: {record}")

        if len(scores) >= num_games:
            break

    # Trim to exact num_games in case we overshot.
    scores = scores[:num_games]
    avg_score = np.mean(scores)
    win_count = sum(1 for s in scores if s >= 97)
    win_rate = win_count / len(scores)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Games Played: {len(scores)}")
    print(f"Average Score:      {avg_score:.2f}")
    print(f"Record Score:       {record}")
    print(f"Win Rate:           {win_rate:.4f} ({win_count}/{len(scores)})")
    print("=" * 50)

    # Save the raw scores to CSV.
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    print(f"\nINFO: Saving raw scores to '{csv_path}'...")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Game", "score"])
        for idx, score in enumerate(scores):
            writer.writerow([idx + 1, score])

    # Clean up.
    vec_env.close()


def watch_model(model_path="", num_games=5):
    """Loads the model and renders it on screen."""
    print(f"INFO: Loading model from '{model_path}'...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"ERROR: Could not load model. Details: {e}")
        return

    env = snake_make_env(gui=True)

    # Simulate games with rendering.
    for i in range(num_games):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        score = info.get("score", 0)
        print(f"Game {i + 1}/{num_games} finished! Score: {score}")


def make_model(env, pre_trained=""):
    """Create a new PPO model or load a pre-trained one."""
    if os.path.exists(pre_trained):
        print(f"INFO: Found existing model at '{pre_trained}'.")
        # Load and override the tensorboard log path to fix the path discrepancy bug.
        model = PPO.load(
            pre_trained,
            env=env,
            tensorboard_log=TENSORBOARD_LOG_DIR,
        )
        return model

    policy_kwargs = dict(
        features_extractor_class=SnakeCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256], vf=[256]),
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,

        # PPO hyperparameters (Attempt 3: balanced tuning).
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,

        # Logging and device.
        tensorboard_log=TENSORBOARD_LOG_DIR,
        verbose=1,
        device=DEVICE,
    )
    return model


###############################################################################

def main():
    # Define command line args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="If you want to train the agent.")
    parser.add_argument("--eval", action="store_true", help="If you want to evaluate the agent.")
    parser.add_argument("--watch", action="store_true", help="If you want to watch the agent play.")

    args = parser.parse_args()

    model_path = "./output/models/sb3_snake_ppo_best_TR1.zip"

    if args.train:
        num_envs = 16
        # Use SubprocVecEnv for true parallel rollouts with PPO.
        vec_env = make_vec_env(snake_make_env, n_envs=num_envs, vec_env_cls=SubprocVecEnv)

        total_steps = 50_000_000

        # Checkpoint callback to save periodic snapshots.
        checkpoint_callback = CheckpointCallback(
            save_freq=125_000,
            save_path='./output/models/sb3_checkpoints/',
            name_prefix='sb3_snake_ppo'
        )

        # Score logging callback for TensorBoard.
        score_callback = ScoreLoggingCallback(log_interval=1000, verbose=1)

        model = make_model(vec_env, model_path)

        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback, score_callback],
            reset_num_timesteps=False,
            progress_bar=True
        )

        print("INFO: Training finished. Saving model...")
        model.save("./output/models/sb3_snake_ppo_best_TR2")
    elif args.eval:
        evaluate_model(model_path=model_path, csv_path="./output/csv/eval.csv", num_games=10_000)
    elif args.watch:
        watch_model(model_path=model_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
