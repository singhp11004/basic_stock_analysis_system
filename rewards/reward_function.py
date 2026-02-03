"""
reward_function.py

Implements a multi-objective reward function for stock trading using:
- Profit
- Drawdown penalty
- Volatility penalty

Reward = profit_weight * profit
         - drawdown_penalty * drawdown
         - volatility_penalty * volatility
"""

import yaml
import numpy as np
from collections import deque


class RewardFunction:
    """
    Multi-objective reward calculator for trading environments.
    """

    def __init__(self, config_path: str, volatility_window: int = 20):
        """
        Initialize reward function using configuration file.

        Parameters
        ----------
        config_path : str
            Path to config.yaml
        volatility_window : int
            Window size for volatility calculation
        """

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        reward_cfg = config["reward"]

        self.profit_weight = reward_cfg["profit_weight"]
        self.drawdown_penalty = reward_cfg["drawdown_penalty"]
        self.volatility_penalty = reward_cfg["volatility_penalty"]

        # Track portfolio peak for drawdown calculation
        self.peak_portfolio_value = None

        # Store recent returns for volatility calculation
        self.returns_window = deque(maxlen=volatility_window)

    def reset(self, initial_portfolio_value: float):
        """
        Reset reward state at the beginning of an episode.

        Parameters
        ----------
        initial_portfolio_value : float
            Starting portfolio value
        """
        self.peak_portfolio_value = initial_portfolio_value
        self.returns_window.clear()

    def compute_reward(
        self,
        previous_portfolio_value: float,
        current_portfolio_value: float
    ) -> float:
        """
        Compute reward for one time step.

        Parameters
        ----------
        previous_portfolio_value : float
            Portfolio value at time t-1
        current_portfolio_value : float
            Portfolio value at time t

        Returns
        -------
        float
            Scalar reward
        """

        # -------- PROFIT TERM --------
        profit = current_portfolio_value - previous_portfolio_value

        # -------- DRAWDOWN TERM --------
        if self.peak_portfolio_value is None:
            self.peak_portfolio_value = current_portfolio_value

        self.peak_portfolio_value = max(
            self.peak_portfolio_value, current_portfolio_value
        )

        drawdown = (
            (self.peak_portfolio_value - current_portfolio_value)
            / self.peak_portfolio_value
        )

        # -------- VOLATILITY TERM --------
        if previous_portfolio_value > 0:
            step_return = profit / previous_portfolio_value
            self.returns_window.append(step_return)

        volatility = (
            np.std(self.returns_window)
            if len(self.returns_window) > 1
            else 0.0
        )

        # -------- FINAL REWARD --------
        reward = (
            self.profit_weight * profit
            - self.drawdown_penalty * drawdown
            - self.volatility_penalty * volatility
        )

        return reward
