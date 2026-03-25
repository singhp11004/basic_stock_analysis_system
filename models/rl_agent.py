"""
PHASE 4: Q-LEARNING AGENT
========================

WHAT THIS FILE DOES
-------------------
This file defines the reinforcement learning agent itself.

It answers the question:
    "HOW does the agent learn?"

Specifically, it implements:
- A tabular Q-learning algorithm
- An epsilon-greedy exploration strategy
- Q-value updates using the Bellman equation


WHY THIS FILE EXISTS
--------------------
We separate responsibilities clearly:

- env/trading_env.py
    Simulates the market and portfolio dynamics

- rewards/reward_function.py
    Defines what is considered good or bad behavior

- models/rl_agent.py (THIS FILE)
    Defines how the agent updates its knowledge

- training/train_agent.py
    Controls when and how often learning happens


HOW IT FITS INTO THE PROJECT
----------------------------
Phase 1: Data ingestion & feature engineering
Phase 2: Config & reward shaping
Phase 3: Trading environment
Phase 4: RL agent + training loop
Phase 5: Backtesting & evaluation


WHAT THIS FILE DOES NOT DO
--------------------------
- It does NOT interact with the environment directly
- It does NOT loop over episodes
- It does NOT load data or compute rewards
"""

import numpy as np
import yaml
from collections import defaultdict
import pickle


class RLAgent:
    """
    Tabular Q-learning agent for discrete action spaces.
    """

    def __init__(self, config_path: str, n_actions: int):
        # -------- LOAD CONFIG --------
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        rl_cfg = config["rl"]

        self.alpha = rl_cfg["learning_rate"]      # learning rate
        self.gamma = rl_cfg["discount_factor"]    # discount factor

        self.epsilon = rl_cfg["epsilon_start"]    # exploration probability
        self.epsilon_min = rl_cfg["epsilon_min"]
        self.epsilon_decay = rl_cfg["epsilon_decay"]

        self.n_actions = n_actions

        # Q-table: maps state -> action values
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

    def _discretize_state(self, state, n_bins=10):
        """
        Convert continuous state into a discrete representation
        usable as a dictionary key.

        Uses coarse binning to improve generalization:
        - Clips values to [-3, 3] (assumes normalized features)
        - Divides into n_bins equal buckets
        
        This ensures similar states map to the same Q-table entry.
        """
        # Clip to reasonable range (for normalized data, most values are in [-3, 3])
        clipped = np.clip(state, -3, 3)
        
        # Bin into n_bins buckets: value in [-3, 3] -> bin in [0, n_bins-1]
        binned = ((clipped + 3) / 6 * n_bins).astype(int)
        binned = np.clip(binned, 0, n_bins - 1)  # safety clamp
        
        return tuple(binned)

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        """
        state_key = self._discretize_state(state)

        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Exploitation
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-values using the Q-learning update rule.
        """
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)

        best_next_q = 0.0 if done else np.max(self.q_table[next_state_key])

        # Bellman update
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error

        # Decay exploration rate at the end of each episode
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def save(self, filepath: str):
        """Save Q-table to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath: str):
        """Load Q-table from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.q_table.update(data)
