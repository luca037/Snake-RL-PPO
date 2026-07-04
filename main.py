import os
import torch
import csv
import numpy as np
import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from Model import CNNQNet
from Game import snake_make_env


###############################################################################
### Global vars
###############################################################################

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("INFO: CUDA is available. Running on GPU.")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print ("INFO: MPS DEVICE found. Running on GPU")
else:
    print("INFO: CUDA and MPS not available. Running on CPU.")


###############################################################################
### Functions
###############################################################################

def evaluate_model(model_path="", num_games=1000, num_envs=64, csv_path="eval_scores.csv"):
    """
    Loads a trained SB3 model and evaluates it across multiple vectorized 
    environments simultaneously for massive speed improvements.
    Saves the raw scores to a CSV file.
    """
    print(f"INFO: Loading model from '{model_path}'...")
    try:
        model = DQN.load(model_path)
        model.policy.compile()
    except Exception as e:
        print(f"ERROR: Could not load model. Details: {e}")
        return

    vec_env = make_vec_env(snake_make_env, n_envs=num_envs)
    
    scores = []
    record = 0
    
    print(f"\nSimulating {num_games} games across {num_envs} environments...")
    print("-" * 50)
    
    # Get the initial batch of observations (Shape: [num_envs, 4, 12, 12])
    obs = vec_env.reset()
    
    while len(scores) < num_games:
        # Pass the entire batch of observations to the model at once
        # (deterministic=True disables random exploration)
        actions, _ = model.predict(obs, deterministic=True)
        
        # Step all environments simultaneously.
        obs, _, dones, infos = vec_env.step(actions)
        
        # Check which environments finished in this step.
        for i, done in enumerate(dones):
            if done:
                # Infl list contains the score.
                score = infos[i].get("score", 0)
                scores.append(score)
                if score > record:
                    record = score
                # Print progress every 100 games
                if not len(scores) % 100 or len(scores) == num_games:
                    current_avg = np.mean(scores)
                    print(f"Games Completed: {len(scores)}/{num_games} | Avg: {current_avg:.1f} | Record: {record}")
                    
        # Since multiple environments can finish on the exact same step, 
        # we might slightly overshoot the num_games. 
        # We break here if we hit the target.
        if len(scores) >= num_games:
            break

    # Trim to exact num_games in case we overshot
    scores = scores[:num_games]
    avg_score = np.mean(scores)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Games Played: {len(scores)}")
    print(f"Average Score:      {avg_score:.2f}")
    print(f"Record Score:       {record}")
    print("="*50)
    
    # Save the raw scores to CSV
    print(f"\nINFO: Saving raw scores to '{csv_path}'...")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Game", "score"])
        for idx, score in enumerate(scores):
            writer.writerow([idx + 1, score])
    
    # Clean up
    vec_env.close()


def watch_model(model_path="", num_games=5):
    """
    Loads the model and renders it on screen.
    :param speed: Seconds to pause between frames. Increase to slow down the snake.
    """
    print(f"INFO: Loading model from '{model_path}'...")
    try:
        model = DQN.load(model_path)
        model.policy.compile()
    except Exception as e:
        print(f"ERROR: Could not load model. Details: {e}")
        return

    env = snake_make_env(gui=True)
    
    # Simualte.
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
    if os.path.exists(pre_trained):
        print(f"INFO: Found existing model at '{pre_trained}'.")
        # Load the model and pass the environment so it can continue interacting
        model = DQN.load(pre_trained, env=env)
        return model

    policy_kwargs = dict(
        features_extractor_class=CNNQNet,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[] # Handle the dense layers entirely inside our CNNQNet class
    )

    model = DQN(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        
        # --- Hyperparameters ---
        learning_rate=0.00025,
        buffer_size=2_000_000,        # SB3 stores this in RAM. 1M is safe, 2M might OOM.
        learning_starts=10_000,       # Populate buffer before training
        batch_size=512,               # Your batch size
        tau=1.0,                      # 1.0 means Hard Target Sync
        gamma=0.99,
        train_freq=4,                 # Train once every 4 steps
        gradient_steps=1,             # 1 gradient step per train_freq
        max_grad_norm=10.0,           # Max gradient norm
        target_update_interval=5_000,
        
        # --- Exploration (Epsilon) ---
        exploration_initial_eps=1.0,
        exploration_final_eps=0.0001,
        exploration_fraction=0.3,    # Reaches min after x%  of total_timesteps
        
        
        # --- Logging + Device ---
        tensorboard_log="./output/plots/tensorboard/",
        verbose=1,
        device=DEVICE
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
    
    #model_path = "./output/models/sb3_checkpoints/sb3_snake_model_35000000_steps.zip"
    model_path = "./output/models/sb3_snake_current_best.zip"

    if args.train:
        num_envs = 8
        vec_env = make_vec_env(snake_make_env, n_envs=num_envs)
        
        total_steps = 30_000_000

        checkpoint_callback = CheckpointCallback(
                save_freq=125_000, # Saves every save_freq * num_envs steps.
                save_path='./output/models/sb3_checkpoints/',
                name_prefix='sb3_snake_model'
        )

        model = make_model(vec_env, model_path)
        model.policy.compile()

        # CRITICAL SETTING: reset_num_timesteps
        # If False: Epsilon stays at its minimum (0.01). (Pure exploitation)
        # If True (default): Epsilon resets to 1.0 and decays again. (Warm restart)
        model.learn(
                total_timesteps=total_steps, 
                callback=checkpoint_callback,
                reset_num_timesteps=True,
                progress_bar=True
        )

        print("INFO: Training finished. Saving model...")
        model.save("./output/models/new_model")
    elif args.eval:
        evaluate_model(model_path=model_path, csv_path="./output/csv/eval.csv", num_games=10_000)
    elif args.watch:
        watch_model(model_path=model_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 
