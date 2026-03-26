"""
backtester.py

PHASE 5A: Clean, cold-start RL backtesting with full evaluation.
CRITICAL: Uses TEST data only (out-of-sample evaluation).
"""

import os
import sys
import yaml
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from env.trading_env import TradingEnv
from stable_baselines3 import PPO
from evaluation.metrics import (
    evaluate_strategy,
    buy_and_hold_baseline,
    print_comparison
)


def backtest():
    config_path = "config/config.yaml"

    # -------- LOAD CONFIG --------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    initial_cash = config["trading"]["initial_cash"]
    transaction_cost = config["trading"]["transaction_cost"]
    
    # Use NORMALIZED test data (must match training)
    test_data_path = config["data"]["test_data_path"].replace(".csv", "_normalized.csv")
    
    # Also load original test data for buy-and-hold baseline (uses real prices)
    original_test_path = config["data"]["test_data_path"]
    import pandas as pd
    original_test_df = pd.read_csv(original_test_path)

    # -------- CREATE FRESH ENV WITH TEST DATA --------
    env = TradingEnv(config_path, data_path=test_data_path)
    state, _ = env.reset()

    print("DEBUG: Portfolio after reset:", env.portfolio_value)
    assert env.portfolio_value == initial_cash

    # -------- CREATE FRESH AGENT --------
    model = PPO.load("models/ppo_agent")

    done = False
    portfolio_history = [env.portfolio_value]
    action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL

    print("=" * 60)
    print("PHASE 5A: RL BACKTESTING (OUT-OF-SAMPLE TEST DATA)")
    print(f"Test data: {test_data_path}")
    print(f"Test samples: {len(env.data)}")
    print("=" * 60)

    while not done:
        action, _states = model.predict(state, deterministic=True)
        action_item = int(action)
        action_counts[action_item] += 1
        
        next_state, reward, done, truncated, info = env.step(action_item)

        # Track portfolio value
        portfolio_history.append(env.portfolio_value)

        # HARD SAFETY CHECK
        FLOAT_TOLERANCE = 1e-8
        assert not (env.cash > FLOAT_TOLERANCE and env.shares > FLOAT_TOLERANCE), \
            f"Inconsistent state: cash={env.cash:.4f}, shares={env.shares:.8f}"

        state = next_state

    # -------- RL STRATEGY METRICS --------
    rl_metrics = evaluate_strategy(portfolio_history, initial_cash)

    # -------- BUY-AND-HOLD BASELINE --------
    # Use original (non-normalized) prices for realistic baseline
    baseline_metrics = buy_and_hold_baseline(original_test_df["Adj Close"], initial_cash, transaction_cost)

    # -------- PRINT RESULTS --------
    print("\nBacktest Complete!")
    print(f"\nAction Distribution:")
    print(f"  HOLD: {action_counts[0]} | BUY: {action_counts[1]} | SELL: {action_counts[2]}")
    
    print_comparison(rl_metrics, baseline_metrics)

    return rl_metrics, baseline_metrics


if __name__ == "__main__":
    backtest()
