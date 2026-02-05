"""
dqn_agent.py

Deep Q-Network agent for trading.
Uses neural network function approximation instead of tabular Q-learning.

Key improvements over tabular:
- Generalizes to unseen states
- Handles continuous state spaces
- Learns feature representations
"""

import numpy as np
import yaml
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.
    
    Architecture: state_dim -> 128 -> 128 -> n_actions
    """
    
    def __init__(self, state_dim: int, n_actions: int):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    
    Stores transitions and samples random minibatches.
    Breaks correlation between consecutive samples.
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    """
    
    def __init__(self, config_path: str, state_dim: int, n_actions: int):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        rl_cfg = config["rl"]
        dqn_cfg = config.get("dqn", {})
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        # Hyperparameters
        self.gamma = rl_cfg["discount_factor"]
        self.epsilon = rl_cfg["epsilon_start"]
        self.epsilon_min = rl_cfg["epsilon_min"]
        self.epsilon_decay = rl_cfg["epsilon_decay"]
        
        # DQN-specific params
        self.lr = dqn_cfg.get("learning_rate", 0.001)
        self.batch_size = dqn_cfg.get("batch_size", 64)
        self.buffer_size = dqn_cfg.get("buffer_size", 10000)
        self.target_update_freq = dqn_cfg.get("target_update_freq", 100)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is never trained directly
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # Training step counter
        self.steps = 0
    
    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, state, action, reward, next_state, done):
        """
        Store transition and perform learning step.
        """
        self.store_transition(state, action, reward, next_state, done)
        
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample minibatch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values (from target network)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss and optimization
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon at end of episode
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save model and optimizer state."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps
        }, filepath)
        print(f"DQN model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        print(f"DQN model loaded from {filepath}")
