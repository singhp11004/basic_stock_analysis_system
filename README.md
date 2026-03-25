# Basic Stock Analysis System

## Problem Statement

Traditional stock trading relies heavily on human intuition and manual analysis, which are subject to emotional biases, fatigue, and limited processing capacity. While automated algorithmic trading systems exist, many rely on rigid, hard-coded rules that fail to adapt to evolving market regimes and unseen volatility. 

The problem is to design an **adaptive, data-driven trading system** that learns optimal trading strategies directly from historical market data without relying on predefined heuristics. We aim to build a Reinforcement Learning (RL) agent that interacts with a simulated single-stock market environment, learning to maximize long-term portfolio returns while managing risk (drawdowns/volatility) and transaction costs. The agent must make optimal discrete decisions (Buy, Hold, Sell) based on a rich state representation encompassing technical indicators, momentum metrics, and machine learning-generated predictive signals.

## Project Overview

This project implements an end-to-end Reinforcement Learning (RL) pipeline for algorithmic stock trading. It uses a **Tabular Q-Learning Agent** to navigate a custom OpenAI Gym-like single-stock trading environment. 

The system features a rigorous data pipeline that prevents data leakage by explicitly separating training and testing data temporally. It normalizes features using statistics strictly from the training set and uses real prices for accurate portfolio valuation during backtesting.

## Key Features

- **RL Trading Agent**: A Tabular Q-learning agent with an epsilon-greedy exploration strategy.
- **Custom Trading Environment**: Simulates realistic trading with transaction costs, binary positions (fully invested or uninvested), and no short-selling/leverage.
- **Robust Feature Engineering**: 
  - *Technical Indicators*: Moving averages, RSI, MACD, Bollinger Bands, momentum.
  - *Machine Learning Signals*: Random Forest classifier predictions to forecast next-day positive returns.
- **Strict Data Segregation**: Strict temporal train/test split. Feature normalization and ML model training occur *only* on the training split to ensure valid out-of-sample backtesting.

## System Architecture

The project is structured into a multi-phase pipeline:

1. **Phase 1: Data Ingestion** (`data_ingestion/fetch_data.py`) - Downloads raw stock data using `yfinance`.
2. **Phase 2: Preprocessing** (`preprocessing/data_cleaner.py`) - Cleans missing values and sorts by date.
3. **Phase 3: Feature Engineering** 
   - `features/technical_indicators.py` - Computes financial indicators based on Adjusted Close prices.
   - `features/ml_signals.py` - Trains an ML model on historical data to generate predictive signals for the RL agent.
4. **Phase 4: Train/Test Split & Normalization** (`preprocessing/data_splitter.py`, `preprocessing/feature_normalizer.py`) - Splits data chronologically and normalizes features to improve agent generalization.
5. **Phase 5: RL Training** (`training/train_agent.py`) - Trains the Q-Learning agent on the training data.
6. **Phase 6: Backtesting** (`backtesting/backtester.py`) - Evaluates the trained agent on unseen test data, comparing its performance against a standard Buy-and-Hold baseline.

## Installation

Ensure you have Python 3.9+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Note: If you encounter issues with `packaging` on certain OS environments, you may need to ensure `packaging==23.1` is used).*

## Usage

To run the entire system pipeline sequentially, execute the following commands from the project root directory:

```bash
# 1. Fetch data
PYTHONPATH=. python data_ingestion/fetch_data.py

# 2. Clean data
PYTHONPATH=. python preprocessing/data_cleaner.py

# 3. Generate technical indicators
PYTHONPATH=. python features/technical_indicators.py

# 4. Generate ML signals
PYTHONPATH=. python features/ml_signals.py

# 5. Split data into train/test sets
PYTHONPATH=. python preprocessing/data_splitter.py

# 6. Normalize features
PYTHONPATH=. python preprocessing/feature_normalizer.py

# 7. Train the RL Agent
PYTHONPATH=. python training/train_agent.py

# 8. Evaluate through Backtesting
PYTHONPATH=. python backtesting/backtester.py
```

### Configuration

System hyperparameters, file paths, and RL settings (learning rate, discount factor, epsilon decay, rewards weights) can be configured in `config/config.yaml`.
