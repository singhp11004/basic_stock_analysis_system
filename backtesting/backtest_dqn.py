"""
backtest_dqn.py

DQN backtesting on out-of-sample test data with normalized features.
"""

import os
import sys
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from env.trading_env import TradingEnv
from models.dqn_agent import DQNAgent
from evaluation.metrics import (
    evaluate_strategy,
    buy_and_hold_baseline,
    print_comparison
)
import pandas as pd


def backtest_dqn():
    config_path = "config/config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    initial_cash = config["trading"]["initial_cash"]
    transaction_cost = config["trading"]["transaction_cost"]
    
    # Use NORMALIZED test data
    test_data_path = config["data"]["test_data_path"].replace(".csv", "_normalized.csv")
    
    # Also load original test data for buy-and-hold comparison (uses raw prices)
    original_test_path = config["data"]["test_data_path"]
    original_test_df = pd.read_csv(original_test_path)

    # Create environment with normalized data
    env = TradingEnv(config_path, data_path=test_data_path)
    state = env.reset()
    state_dim = len(state)

    # Create and load trained DQN agent
    agent = DQNAgent(config_path, state_dim=state_dim, n_actions=3)
    agent.load("models/dqn_best.pth")
    agent.epsilon = 0.0  # Pure exploitation

    done = False
    portfolio_history = [env.portfolio_value]
    action_counts = {0: 0, 1: 0, 2: 0}

    print("=" * 60)
    print("DQN BACKTESTING (OUT-OF-SAMPLE TEST DATA)")
    print(f"Test data: {test_data_path}")
    print(f"Test samples: {len(env.data)}")
    print("=" * 60)

    while not done:
        action = agent.select_action(state)
        action_counts[action] += 1
        
        next_state, reward, done = env.step(action)
        portfolio_history.append(env.portfolio_value)

        # Safety checks
        assert env.cash >= 0
        assert env.shares >= 0
        assert not (env.cash > 0 and env.shares > 0)

        state = next_state

    # RL metrics
    rl_metrics = evaluate_strategy(portfolio_history, initial_cash)

    # Buy-and-hold baseline (use original prices, not normalized)
    baseline_metrics = buy_and_hold_baseline(
        original_test_df["Adj Close"], 
        initial_cash, 
        transaction_cost
    )

    # Print results
    print("\nBacktest Complete!")
    print(f"\nAction Distribution:")
    print(f"  HOLD: {action_counts[0]} | BUY: {action_counts[1]} | SELL: {action_counts[2]}")
    
    print_comparison(rl_metrics, baseline_metrics)

    return rl_metrics, baseline_metrics


if __name__ == "__main__":
    backtest_dqn()
