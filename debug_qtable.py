"""
Debug script to investigate why the Q-learning agent always chooses HOLD.
"""

import os
import sys
import pickle
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from env.trading_env import TradingEnv
from models.rl_agent import RLAgent


def investigate():
    config_path = "config/config.yaml"
    
    # Load agent and Q-table
    agent = RLAgent(config_path, n_actions=3)
    agent.load("models/q_table.pkl")
    agent.epsilon = 0.0  # exploitation only
    
    print("=" * 60)
    print("Q-TABLE ANALYSIS")
    print("=" * 60)
    print(f"Total states in Q-table: {len(agent.q_table)}")
    
    # Load test environment
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    test_data_path = config["data"]["test_data_path"]
    env = TradingEnv(config_path, data_path=test_data_path)
    
    # Get first few test states
    state = env.reset()
    
    print("\n" + "=" * 60)
    print("SAMPLE TEST STATES & Q-VALUES")
    print("=" * 60)
    
    for i in range(5):
        state_key = agent._discretize_state(state)
        q_values = agent.q_table[state_key]
        action = np.argmax(q_values)
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        
        print(f"\nStep {i}:")
        print(f"  State (first 5 features): {state[:5]}")
        print(f"  Discretized key (first 5): {state_key[:5]}")
        print(f"  Q-values: HOLD={q_values[0]:.4f}, BUY={q_values[1]:.4f}, SELL={q_values[2]:.4f}")
        print(f"  Chosen action: {action_names[action]}")
        print(f"  State in Q-table? {state_key in agent.q_table}")
        
        # Take step
        next_state, _, done = env.step(action)
        state = next_state
        if done:
            break
    
    # Check if test states exist in Q-table
    env.reset()
    state = env.reset()
    
    states_in_qtable = 0
    states_not_in_qtable = 0
    
    done = False
    while not done:
        state_key = agent._discretize_state(state)
        if state_key in agent.q_table:
            states_in_qtable += 1
        else:
            states_not_in_qtable += 1
        action = agent.select_action(state)
        state, _, done = env.step(action)
    
    print("\n" + "=" * 60)
    print("STATE COVERAGE ANALYSIS")
    print("=" * 60)
    print(f"Test states found in Q-table: {states_in_qtable}")
    print(f"Test states NOT in Q-table: {states_not_in_qtable}")
    print(f"Coverage: {100 * states_in_qtable / (states_in_qtable + states_not_in_qtable):.2f}%")
    
    # Check what Q-values look like for unknown states (defaultdict returns zeros)
    print("\n" + "=" * 60)
    print("ROOT CAUSE")
    print("=" * 60)
    if states_not_in_qtable > states_in_qtable:
        print("⚠️  Most test states are NOT in the Q-table!")
        print("   Reason: The discretized states in test data differ from train data.")
        print("   Effect: defaultdict returns [0, 0, 0] for Q-values.")
        print("   Result: argmax([0, 0, 0]) = 0 = HOLD")
        print("\n   FIX: The state discretization (rounding to 2 decimals) creates")
        print("   states that don't generalize between train and test periods.")


if __name__ == "__main__":
    investigate()
