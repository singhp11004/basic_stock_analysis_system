"""
reward_function.py

Numerically stable, scale-invariant reward function for trading RL.

Reward components:
- Normalized profit (return)
- Drawdown penalty
- Volatility penalty

This reward is SAFE for RL training.
"""

import yaml
import numpy as np
from collections import deque


class RewardFunction:
    """
    Multi-objective reward calculator for trading environments.
    """

    def __init__(self, config_path: str, volatility_window: int = 20):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        reward_cfg = config["reward"]

        self.profit_weight = reward_cfg["profit_weight"]
        self.drawdown_penalty = reward_cfg["drawdown_penalty"]
        self.volatility_penalty = reward_cfg["volatility_penalty"]

        self.peak_portfolio_value = None
        self.returns_window = deque(maxlen=volatility_window)

    def reset(self, initial_portfolio_value: float):
        self.peak_portfolio_value = initial_portfolio_value
        self.returns_window.clear()

    def compute_reward(
        self,
        previous_portfolio_value: float,
        current_portfolio_value: float
    ) -> float:
        # -------- NORMALIZED PROFIT (RETURN) --------
        if previous_portfolio_value <= 0:
            step_return = 0.0
        else:
            step_return = (
                current_portfolio_value - previous_portfolio_value
            ) / previous_portfolio_value

        # -------- DRAWDOWN --------
        self.peak_portfolio_value = max(
            self.peak_portfolio_value, current_portfolio_value
        )

        drawdown = (
            (self.peak_portfolio_value - current_portfolio_value)
            / self.peak_portfolio_value
        )

        # -------- VOLATILITY --------
        self.returns_window.append(step_return)

        volatility = (
            np.std(self.returns_window)
            if len(self.returns_window) > 1
            else 0.0
        )

        # -------- FINAL REWARD --------
        reward = (
            self.profit_weight * step_return
            - self.drawdown_penalty * drawdown
            - self.volatility_penalty * volatility
        )

        return reward
