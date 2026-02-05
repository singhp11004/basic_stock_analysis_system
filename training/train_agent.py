"""
PHASE 4: REINFORCEMENT LEARNING TRAINING LOOP
============================================

This file orchestrates the RL training process.
CRITICAL: Uses TRAIN data only to prevent overfitting.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from env.trading_env import TradingEnv
from models.rl_agent import RLAgent


def train():
    config_path = "config/config.yaml"

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    episodes = config["rl"]["episodes"]
    
    # Use NORMALIZED training data for better generalization
    train_data_path = config["data"]["train_data_path"].replace(".csv", "_normalized.csv")

    # Initialize environment with TRAINING DATA ONLY
    env = TradingEnv(config_path, data_path=train_data_path)
    agent = RLAgent(config_path, n_actions=3)

    print("=" * 60)
    print("PHASE 4: RL TRAINING STARTED (TRAIN DATA ONLY)")
    print(f"Training data: {train_data_path}")
    print(f"Total episodes: {episodes}")
    print(f"Training samples: {len(env.data)}")
    print("=" * 60)

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Steps: {steps} | "
            f"Total Reward: {total_reward:.2f} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    agent.save("models/q_table.pkl")
    print("Q-table saved to models/q_table.pkl")


if __name__ == "__main__":
    train()
