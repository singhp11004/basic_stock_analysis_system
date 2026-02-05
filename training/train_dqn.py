"""
train_dqn.py

DQN training loop with normalized features.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from env.trading_env import TradingEnv
from models.dqn_agent import DQNAgent


def train_dqn():
    config_path = "config/config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    episodes = config["rl"]["episodes"]
    
    # Use NORMALIZED training data
    train_data_path = config["data"]["train_data_path"].replace(".csv", "_normalized.csv")

    # Initialize environment
    env = TradingEnv(config_path, data_path=train_data_path)
    
    # Get state dimension from first state
    initial_state = env.reset()
    state_dim = len(initial_state)
    
    # Initialize DQN agent
    agent = DQNAgent(config_path, state_dim=state_dim, n_actions=3)

    print("=" * 60)
    print("DQN TRAINING STARTED")
    print(f"Training data: {train_data_path}")
    print(f"State dimension: {state_dim}")
    print(f"Total episodes: {episodes}")
    print(f"Device: {agent.device}")
    print("=" * 60)

    best_reward = float("-inf")
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        total_loss = 0.0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            loss = agent.update(state, action, reward, next_state, done)
            total_loss += loss

            state = next_state
            total_reward += reward
            steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0
        
        # Log progress
        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Buffer: {len(agent.replay_buffer)}"
            )
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save("models/dqn_best.pth")

    print("=" * 60)
    print("DQN TRAINING COMPLETE")
    print(f"Best reward: {best_reward:.2f}")
    print("=" * 60)
    
    # Save final model
    agent.save("models/dqn_final.pth")


if __name__ == "__main__":
    train_dqn()
