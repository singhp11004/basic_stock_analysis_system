"""
trading_env.py

Final trading environment for reinforcement learning.
Implements correct single-stock portfolio accounting
with no leverage, no short selling, and transaction costs.

CRITICAL DESIGN:
- Uses NORMALIZED features for state representation (better generalization)
- Uses REAL prices for portfolio valuation (financially meaningful)
"""

import pandas as pd
import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

from rewards.reward_function import RewardFunction


class TradingEnv(gym.Env):
    """
    Single-stock trading environment with:
    - Discrete actions (HOLD, BUY, SELL)
    - Binary position (0 = no stock, 1 = fully invested)
    - No leverage, no short selling
    
    State: Normalized features (for generalization)
    Pricing: Real Adj Close (for correct portfolio valuation)
    """

    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(self, config_path: str, data_path: str = None, prices_path: str = None):
        """
        Args:
            config_path: Path to config.yaml
            data_path: Path to normalized features (for state)
            prices_path: Path to raw features with real prices (for trading)
                         If None, uses data_path (assumes prices in features)
        """
        # -------- LOAD CONFIG --------
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        data_cfg = self.config["data"]
        trading_cfg = self.config["trading"]

        # Load normalized feature data (for state representation)
        features_path = data_path if data_path else data_cfg["features_data_path"]
        self.data = pd.read_csv(features_path)
        self.data.reset_index(drop=True, inplace=True)
        
        # Load real price data (for portfolio valuation)
        if prices_path:
            self.prices_data = pd.read_csv(prices_path)
            self.prices_data.reset_index(drop=True, inplace=True)
        else:
            # If using normalized data, infer prices path from features path
            if "_normalized" in features_path:
                inferred_prices_path = features_path.replace("_normalized", "")
                self.prices_data = pd.read_csv(inferred_prices_path)
                self.prices_data.reset_index(drop=True, inplace=True)
            else:
                # Same file contains real prices
                self.prices_data = self.data

        # Trading parameters
        self.initial_cash = trading_cfg["initial_cash"]
        self.transaction_cost = trading_cfg["transaction_cost"]

        # Reward function
        self.reward_fn = RewardFunction(config_path)

        # Environment state
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.position = 0
        self.portfolio_value = self.initial_cash
        self.prev_portfolio_value = self.initial_cash
        
        # SB3 Gymnasium Spaces Action and Observation space
        self.action_space = spaces.Discrete(3)
        obs = self._get_state()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset environment to the beginning of the episode.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.position = 0

        self.portfolio_value = self.initial_cash
        self.prev_portfolio_value = self.portfolio_value

        self.reward_fn.reset(self.portfolio_value)

        return self._get_state(), {}

    def step(self, action: int):
        """
        Execute one environment step.

        Actions:
        0 = HOLD
        1 = BUY
        2 = SELL
        """
        done = False
        
        # Use REAL price for trading (from prices_data, not normalized data)
        price = self.prices_data.loc[self.current_step, "Adj Close"]

        # -------- EXECUTE ACTION --------
        if action == self.BUY and self.position == 0:
            # Buy with all cash (apply transaction cost)
            effective_cash = self.cash * (1 - self.transaction_cost)
            self.shares = effective_cash / price
            self.cash = 0.0
            self.position = 1

        elif action == self.SELL and self.position == 1:
            # Sell all shares (apply transaction cost)
            proceeds = self.shares * price
            self.cash = proceeds * (1 - self.transaction_cost)
            self.shares = 0.0
            self.position = 0

        # -------- UPDATE PORTFOLIO VALUE --------
        self.prev_portfolio_value = self.portfolio_value

        if self.position == 1:
            self.portfolio_value = self.shares * price
        else:
            self.portfolio_value = self.cash

        # -------- COMPUTE REWARD --------
        reward = self.reward_fn.compute_reward(
            self.prev_portfolio_value,
            self.portfolio_value
        )

        # -------- ADVANCE TIME --------
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        next_state = self._get_state()
        return next_state, float(reward), done, False, {}

    def _get_state(self):
        """
        Construct state vector from NORMALIZED features:
        [market features..., position_flag]
        """
        row = self.data.loc[self.current_step]

        # Market features (exclude Date and raw prices to prevent leakage)
        drop_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        cols_to_drop = [c for c in drop_cols if c in row.index]
        features = row.drop(labels=cols_to_drop).values.astype(np.float32)

        position_flag = np.array([float(self.position)], dtype=np.float32)

        return np.concatenate([features, position_flag])
