# Current System Status — Basic Stock Analysis System
**As of: 2026-03-26**

---

## 1. Overall System Health

| Area | Status | Notes |
|---|---|---|
| Data Ingestion | ✅ Functional | Config-driven, `auto_adjust=False` |
| Preprocessing / Cleaning | ✅ Functional | Sorts by date, drops NaN |
| Feature Engineering | ✅ Functional | 10 features, Adj Close-based, Close removed |
| ML Signals | ✅ Functional | Random Forest, separate output file |
| Train/Test Split | ✅ Functional | Strict temporal split |
| Feature Normalization | ✅ Functional | Z-score, fitted on train only |
| Trading Environment | ✅ Functional | Gym-compatible, DummyVecEnv wrapped |
| RL Agent (PPO) | ✅ Functional | Deep RL via Stable-Baselines3 |
| Training Loop | ✅ Functional | PPO training on normalized data |
| Backtesting | ✅ Functional | Out-of-sample, realistic evaluation |
| Evaluation Metrics | ✅ Functional | Return, Sharpe, Max Drawdown |
| Visualization | ❌ Empty stub | `plots.py` has no implementation |
| Strategy Module | ❌ Empty stub | `trading_strategy.py` has no implementation |
| CI / Tests | ❌ Missing | No test suite exists |
| Orchestration | ❌ Manual | Sequential shell commands |

---

## 2. Module-by-Module Assessment

### 2.1 `data_ingestion/fetch_data.py`
- **Status:** ✅ Functional (Upgraded)
- Now fully integrated with `config/config.yaml` for symbol, start date, and end date.
- Uses `yfinance` with `auto_adjust=False` to preserve `Adj Close`.
- **Gap:** No automatic retry logic for network failures.

### 2.2 `preprocessing/data_cleaner.py`
- **Status:** Working (minimal)
- Parses dates, sorts, drops NaN.
- **Gap:** No advanced outlier detection or smoothing.

### 2.3 `preprocessing/data_splitter.py`
- **Status:** Working
- Clean temporal split based on config `split_date`.

### 2.4 `preprocessing/feature_normalizer.py`
- **Status:** Working
- Fits Z-score normalizer on training data only.
- Saves artifact to `models/feature_normalizer.pkl`.

### 2.5 `features/technical_indicators.py`
- **Status:** Working
- 10 features: Return, SMA_10, SMA_30, RSI, Volatility, MACD_Signal, BB_Position, Momentum, Trend, Vol_Regime.
- `Close` column explicitly dropped to prevent leakage.

### 2.6 `features/ml_signals.py`
- **Status:** ✅ Functional (Fixed)
- Now saves to a separate `ml_features.csv` as per config, avoiding file corruption.
- Random Forest trained on train split only.
- **Gap:** No test accuracy reported in logs.

### 2.7 `env/trading_env.py`
- **Status:** Working (Gym-compatible)
- Correctly separates normalized features for state and real prices for valuation.
- Compatible with Stable-Baselines3 vectorized wrappers.

### 2.8 `models/rl_agent.py` → `PPO Agent`
- **Status:** ✅ Functional (Upgraded)
- Replaced Tabular Q-Learning with **Proximal Policy Optimization (PPO)**.
- Neural network function approximation solves the state-space explosion.
- Model: `models/ppo_agent.zip`.

### 2.9 `training/train_agent.py`
- **Status:** ✅ Functional (Upgraded)
- Uses Stable-Baselines3 `PPO` for training.
- Fully integrated with `config/config.yaml` for hyperparameters.

### 2.10 `backtesting/backtester.py`
- **Status:** ✅ Functional (Fixed)
- Fixed floating-point assertion crash using `FLOAT_TOLERANCE`.
- Uses frozen PPO model for out-of-sample evaluation.
- **Gap:** Results not yet saved to CSV/JSON files.

### 2.11 `rewards/reward_function.py`
- **Status:** Working (well-structured)
- Multi-objective: profit + drawdown penalty + volatility penalty
- `profit_weight = 5.0`, `drawdown_penalty = 0.01`, `volatility_penalty = 0.01`
- **Gap:** Drawdown and volatility penalties are so minimal (0.01) they are effectively ignored — the agent optimizes almost purely for raw return, making it likely to overfit on momentum; no Sortino ratio component; reward is not clipped, so large single-step gains can dominate learning

### 2.12 `evaluation/metrics.py`
- **Status:** Working
- Computes: total return, max drawdown, Sharpe ratio, annualized volatility
- Side-by-side comparison vs buy-and-hold with verdict
- **Gap:** Risk-free rate hardcoded to 0.0; Calmar ratio missing; no trade-level statistics (win rate, avg win/loss, trade count); no rolling Sharpe (only full-period); output is purely console text — no file output

### 2.13 `strategy/trading_strategy.py` + `visualization/plots.py`
- **Status:** ❌ Empty files (0 bytes)
- These are placeholders with no implementation

### 2.14 `config/config.yaml`
- **Status:** Functional
- Contains a `dqn:` section (batch_size, buffer_size, target_update_freq) that is **completely unused** — there is no DQN implementation, only a reference ghost config
- `end_date: "auto"` is set in YAML but `fetch_data.py` does not read from config — it hardcodes dates in `__main__`

---

## 3. Saved Artifacts (models/)

| Artifact | Size | Description |
|---|---|---|
| `models/q_table.pkl` | 914 KB | Trained Q-table (very large — possibly overfit signature) |
| `models/ml_model.pkl` | 262 KB | Trained Random Forest |
| `models/ml_scaler.pkl` | 786 bytes | StandardScaler for ML features |
| `models/feature_normalizer.pkl` | 1.4 KB | Z-score normalizer for RL features |

> **Note:** A 914 KB Q-table for a 10-bin, 14-dim state implies ~914,000 unique discretized states have been visited during training. This is simultaneously evidence of reasonable exploration and a warning sign of effective state-space explosion.

---

## 4. Pipeline Execution Status

Steps required to run the full system (currently manual):
```
Step 1: PYTHONPATH=. python data_ingestion/fetch_data.py
Step 2: PYTHONPATH=. python preprocessing/data_cleaner.py
Step 3: PYTHONPATH=. python features/technical_indicators.py
Step 4: PYTHONPATH=. python features/ml_signals.py
Step 5: PYTHONPATH=. python preprocessing/data_splitter.py
Step 6: PYTHONPATH=. python preprocessing/feature_normalizer.py
Step 7: PYTHONPATH=. python training/train_agent.py
Step 8: PYTHONPATH=. python backtesting/backtester.py
```

There is **no orchestrator** — a failure at step 4 is not detected at step 5. This is a significant operational fragility for a research pipeline.

---

## 5. Data Flow Validation

```
raw/stock_prices.csv
  └─► [clean] processed/cleaned_data.csv
        └─► [technical_indicators] features/features.csv
              └─► [ml_signals: overwrites features.csv, adds ML_Signal]
                    └─► [data_splitter] train_features.csv / test_features.csv
                          └─► [normalizer] train_features_normalized.csv / test_features_normalized.csv
                                └─► [train_agent] models/q_table.pkl
                                └─► [backtester] Console output (no file artifact)
```

**Known fragility:** `ml_signals.py` overwrites the original `features.csv`. If re-run independently it produces a different file (the `Target` column is dropped, and indexing changes by 1 row due to `df.iloc[:-1]`). This makes the pipeline non-idempotent.
