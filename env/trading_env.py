"""
trading_env.py

Final trading environment for reinforcement learning.
Implements correct portfolio accounting and integrates
the multi-objective reward function.
"""

import pandas as pd
import numpy as np
import yaml

from rewards.reward_function import RewardFunction


class TradingEnv:
    """
    Single-stock trading environment with:
    - Discrete actions
    - Binary position
    - Transaction costs
    """

    # Action definitions
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(self, config_path: str):
        # -------- LOAD CONFIG --------
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        data_cfg = self.config["data"]
        trading_cfg = self.config["trading"]

        # Load feature data
        self.data = pd.read_csv(data_cfg["features_data_path"])
        self.data.reset_index(drop=True, inplace=True)

        # Trading parameters
        self.initial_cash = trading_cfg["initial_cash"]
        self.transaction_cost = trading_cfg["transaction_cost"]

        # Reward function
        self.reward_fn = RewardFunction(config_path)

        # Environment state
        self.current_step = 0
        self.cash = None
        self.position = None          # number of shares held
        self.portfolio_value = None
        self.prev_portfolio_value = None

    def reset(self):
        """
        Reset environment to the beginning of the episode.
        """
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0.0

        self.portfolio_value = self.initial_cash
        self.prev_portfolio_value = self.portfolio_value

        # Reset reward tracking
        self.reward_fn.reset(self.portfolio_value)

        return self._get_state()

    def step(self, action: int):
        """
        Execute one environment step.

        Actions:
        0 = HOLD
        1 = BUY
        2 = SELL
        """
        done = False
        price = self.data.loc[self.current_step, "Close"]

        # -------- EXECUTE ACTION --------
        if action == self.BUY and self.position == 0:
            # Invest all cash
            shares = self.cash / price
            cost = self.cash * self.transaction_cost

            self.position = shares
            self.cash = -cost  # cash fully invested + transaction cost

        elif action == self.SELL and self.position > 0:
            # Sell all shares
            proceeds = self.position * price
            cost = proceeds * self.transaction_cost

            self.cash = proceeds - cost
            self.position = 0.0

        # -------- UPDATE PORTFOLIO VALUE --------
        self.prev_portfolio_value = self.portfolio_value

        self.portfolio_value = self.cash + self.position * price

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

        return next_state, reward, done

    def _get_state(self):
        """
        Construct state vector:
        [market features..., position_flag]
        """
        row = self.data.loc[self.current_step]

        # Market features (exclude Date)
        features = row.drop("Date").values.astype(np.float32)

        # Position flag (0 or 1)
        position_flag = np.array(
            [1.0 if self.position > 0 else 0.0],
            dtype=np.float32
        )

        state = np.concatenate([features, position_flag])
        return state
