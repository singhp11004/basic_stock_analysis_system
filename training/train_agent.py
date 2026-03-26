"""
PHASE 4: REINFORCEMENT LEARNING TRAINING LOOP
============================================

This file orchestrates the RL training process.
CRITICAL: Uses TRAIN data only to prevent overfitting.
"""

import os
import sys

import os
import sys
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train():
    config_path = "config/config.yaml"

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Use NORMALIZED training data for better generalization
    train_data_path = config["data"]["train_data_path"].replace(".csv", "_normalized.csv")

    # Initialize environment with TRAINING DATA ONLY
    # SB3 requires vectorized environments, DummyVecEnv is easiest wrapper
    env = DummyVecEnv([lambda: TradingEnv(config_path, data_path=train_data_path)])

    sb3_cfg = config["sb3"]

    print("=" * 60)
    print("PHASE 4: RL TRAINING STARTED (PPO SB3)")
    print(f"Training data: {train_data_path}")
    print(f"Total timesteps: {sb3_cfg['total_timesteps']}")
    print("=" * 60)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=sb3_cfg["learning_rate"],
        n_steps=sb3_cfg["n_steps"],
        batch_size=sb3_cfg["batch_size"],
        n_epochs=sb3_cfg["n_epochs"],
        gamma=sb3_cfg["gamma"],
        ent_coef=sb3_cfg["ent_coef"],
        verbose=1
    )

    model.learn(total_timesteps=sb3_cfg["total_timesteps"])

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_agent")
    print("Model saved to models/ppo_agent.zip")


if __name__ == "__main__":
    train()
