# Issues, Bugs & Technical Defects
**Basic Stock Analysis System — Critical Review**

---

## Severity Legend
- 🔴 **CRITICAL** — Will cause incorrect results, silent data leakage, or runtime crashes
- 🟠 **HIGH** — Significant logical flaw degrading results meaningfully
- 🟡 **MEDIUM** — Operational fragility or important missing validation
- 🟢 **LOW** — Code quality, best practice violations, minor gaps

---

## STATUS UPDATE (2026-03-26)
Several critical and high-severity issues identified on 2026-03-25 have been resolved following the migration to a research-grade Deep RL architecture (PPO).

---

## Resolved Issues (2026-03-26)

### ✅ BUG-001 RESOLVED — Floating-Point Assertion Failure in Backtester
**Fix:** Introduced `FLOAT_TOLERANCE = 1e-8` to handle residual fractional shares.
**Status:** Verified. Backtester now runs to completion without assertion errors.

### ✅ BUG-002 RESOLVED — `fetch_data.py` Now Reads Config
**Fix:** Connected `fetch_data.py` `__main__` block to `config/config.yaml`.
**Status:** Verified. Symobl and date ranges are now configuration-driven.

### ✅ BUG-003 RESOLVED — Pipeline Idempotency Fixed
**Fix:** `ml_signals.py` now writes to a separate `ml_features.csv` file instead of overwriting the base features.
**Status:** Verified. Multiple runs no longer corrupt the dataset.

### ✅ ISSUE-001 RESOLVED — Q-Table State Explosion Solved
**Fix:** Replaced Tabular Q-Learning with **Proximal Policy Optimization (PPO)** using neural network function approximation.
**Status:** Verified. The agent now generalizes across the continuous state space.

---

## Active Issues
**File:** `backtesting/backtester.py:77`
```python
assert not (env.cash > 0 and env.shares > 0)
```
**Problem:** After a SELL action, residual floating-point errors can leave `env.shares` as a tiny non-zero value (e.g., `1e-15`). Under `position == 0`, `portfolio_value = self.cash` so this is financially harmless, but the assertion will crash the backtest.

**Evidence:** `env.shares = effective_cash / price` — fractional share arithmetic is exact only in theory.

**Impact:** Crashes backtest run silently. No output produced.

---

### 🔴 BUG-002 — `fetch_data.py` Ignores Config for Date Range
**File:** `data_ingestion/fetch_data.py:46-55` vs `config/config.yaml:7-9`
```yaml
# config.yaml
data:
  start_date: "2018-01-01"
  end_date: "auto"
```
```python
# fetch_data.py __main__ — completely ignores config
SYMBOL = "AAPL"
START_DATE = "2018-01-01"
```
**Problem:** The config has `start_date` and `end_date` fields that `fetch_data.py` does not read. If a user changes the config date, the data will NOT change.

**Impact:** Silent mismatch between configured and actually fetched date range. Particularly dangerous if `split_date` is changed.

---

### 🔴 BUG-003 — Non-Idempotent Pipeline (ml_signals.py Overwrites Features)
**File:** `features/ml_signals.py:100`
```python
df_for_pred.to_csv(features_path, index=False)  # overwrites features.csv
```
**Problem:** `ml_signals.py` reads `features.csv`, drops the last row (`df.iloc[:-1]`), and writes back to the same path, **shortening the file by 1 row permanently**. Running the script twice in sequence produces a features file that is 2 rows shorter than the original.

**Impact:** Silent data loss. Makes the pipeline non-idempotent. Downstream split and normalization sizes will differ depending on how many times `ml_signals.py` was run.

---

### 🔴 BUG-004 — State Contains Raw Price Columns (Potential Future-Price Leakage)
**File:** `env/trading_env.py:159`
```python
features = row.drop("Date").values.astype(np.float32)
```
**Problem:** The normalized feature CSV (produced by `feature_normalizer.py`) normalizes **all non-Date columns** — including `Adj Close`, `Open`, `High`, `Low`, `Volume`. These raw price values, even when Z-scored, still encode absolute price level information. Because normalization is fit on the training set, test-period prices are encoded relative to training means, which is fine — but `Adj Close` at time `t` should not appear as a raw feature since Adj Close is also used as the **reward signal** at time `t`. This creates a direct cycle: the agent is simultaneously observing and being rewarded on the same value.

**Impact:** The agent can learn to predict its own reward from the state, rather than learning a generalizable market strategy. Mild form of look-ahead bias.

---

### 🔴 BUG-005 — ML Signal Uses Different Scaler than RL Features
**Files:** `features/ml_signals.py:66-67`, `preprocessing/feature_normalizer.py`
```python
# ml_signals.py uses:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```
**Problem:** The ML prediction model has its own internal `StandardScaler` (saved to `models/ml_scaler.pkl`), while the RL environment uses a separate `FeatureNormalizer` (saved to `models/feature_normalizer.pkl`). These are fitted on the same data but at different times and on different feature subsets (ML scaler excludes `Return` from features, RL normalizer includes it).

After `ml_signals.py` adds `ML_Signal` to `features.csv`, the RL normalizer is re-fit by `feature_normalizer.py` — but now the normalizer sees an `ML_Signal` column that was computed using a different normalization scheme. The normalized `ML_Signal` in the RL state is thus double-normalized (once by its own scaler, once by the RL normalizer).

**Impact:** The `ML_Signal` feature entering the RL state is numerically mangled and may convey noise rather than signal.

---

## HIGH Issues

### 🟠 ISSUE-001 — Q-Table State Explosion (10^14 States)
**File:** `models/rl_agent.py:81-99`

The state vector has approximately 13-14 dimensions (10 technical features + ML_Signal + Adj Close columns + position flag). With 10 bins per dimension, the theoretical Q-table space is 10^14 states.

The Q-table file is **914 KB**, implying ~914,000 unique states were visited. This is 0.00000001% of the possible space — the agent has explored virtually nothing. The Q-table thus functions as a heavily sparse lookup table with essentially random behaviour for unseen (majority) states.

**Consequence:** Backtest actions for most time steps are governed by the default Q-value of zero (all actions equally preferred → effectively random action on ties via `np.argmax` selecting index 0 = HOLD).

---

### 🟠 ISSUE-002 — Reward Weights Effectively Nullify Risk Management
**File:** `config/config.yaml:42-44`
```yaml
reward:
  profit_weight: 5.0
  drawdown_penalty: 0.01
  volatility_penalty: 0.01
```
The drawdown and volatility penalties are 500× smaller than the profit weight. This means the agent is overwhelmingly incentivized to maximize raw return with no meaningful risk control. The system will produce an agent that:
- Does not preserve capital during draw-downs
- Has volatility equal to or exceeding buy-and-hold
- Fails the stated design goal of "managing risk (drawdowns/volatility)"

---

### 🟠 ISSUE-003 — No Convergence Validation During Training
**File:** `training/train_agent.py`

Training runs for exactly 1000 episodes with no convergence criterion. The agent may:
- Converge in 200 episodes (remaining 800 overfitting to training order)
- Never converge (reward oscillating, no plateau)

With no logged reward curve and no early stopping, it is impossible to determine which scenario occurred.

---

### 🟠 ISSUE-004 — Fragile Price Path Inference via String Replacement
**File:** `env/trading_env.py:61-63`
```python
if "_normalized" in features_path:
    inferred_prices_path = features_path.replace("_normalized", "")
```
If the directory or file name contains `"_normalized"` anywhere in the path (e.g., a folder named `data_normalized/`), this replacement produces a wrong path silently. No `os.path.exists()` check is performed afterward — so the environment will fail at step time rather than at init.

---

### 🟠 ISSUE-005 — Ghost DQN Config Section
**File:** `config/config.yaml:48-53`
```yaml
dqn:
  learning_rate: 0.001
  batch_size: 64
  buffer_size: 10000
  target_update_freq: 100
```
There is no DQN implementation anywhere in the codebase. This config block is dead configuration that misleads anyone reading the project about implemented capabilities. It also suggests a partially planned upgrade that was abandoned.

---

## MEDIUM Issues

### 🟡 ISSUE-006 — RSI Uses Simple Rolling Mean (Non-Standard)
**File:** `features/technical_indicators.py:45-46`
```python
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
```
The industry-standard RSI (Wilder, 1978) uses exponential smoothing (EMA with alpha = 1/14), not a simple rolling mean. The simple mean version produces different values — especially at the beginning of the series — and will not match any reference RSI data or library implementation.

---

### 🟡 ISSUE-007 — Bollinger Band Division by Zero (Silent NaN)
**File:** `features/technical_indicators.py:68`
```python
df["BB_Position"] = (price - bb_lower) / (bb_upper - bb_lower)
```
When `bb_upper == bb_lower` (price has been constant for 20 days — rare but possible during holidays or halted trading), this produces `NaN`. The subsequent `df.dropna()` silently removes these rows without warning.

---

### 🟡 ISSUE-008 — No ML Signal Test Accuracy Reported
**File:** `features/ml_signals.py:79-80`
Only training accuracy is printed. Without test accuracy, it's impossible to know if the Random Forest has overfit, which directly affects the quality of the `ML_Signal` feature provided to the RL agent.

---

### 🟡 ISSUE-009 — `data_cleaner.py` Is Insufficiently Defensive
**File:** `preprocessing/data_cleaner.py`
- Uses `dropna()` without specifying which columns must be non-null
- No check that `Adj Close` is numeric and positive
- No validation that the date range after cleaning meets minimum length requirements
- No logging of how many rows were dropped

---

### 🟡 ISSUE-010 — Backtester Prints Only to Console, No Artifacts
**File:** `backtesting/backtester.py`
Portfolio history, action counts, and performance metrics are printed to stdout but never saved. This means:
- Results cannot be compared across experiments
- No equity curve CSV for visualization
- No persistent audit trail

---

## LOW Issues

### 🟢 ISSUE-011 — `f` is Missing from `fetch_stock_data` Config Integration
`fetch_stock_data()` exists as a standalone function but `__main__` doesn't read config — hardcoding `SYMBOL`, `START_DATE`, `END_DATE`, `OUTPUT_FILE`. The function signature accepts them as arguments but is not connected to `config.yaml`.

### 🟢 ISSUE-012 — No `__init__.py` Files in Packages
None of the module directories (`features/`, `models/`, `rewards/`, etc.) have `__init__.py` files. The project relies entirely on `PYTHONPATH=.` to resolve imports, which is fragile and non-standard for Python packaging.

### 🟢 ISSUE-013 — No Requirements Version Pinning Strategy
`requirements.txt` pins **all** packages to exact versions (including low-level networking libraries like `certifi==2026.1.4`). This is too rigid — future installs on different platforms may fail due to binary incompatibilities. Should use semantic version ranges for core libs and let the resolver handle transitive deps.

### 🟢 ISSUE-014 — `packaging` Dependency Is a Local Build Path
**File:** `requirements.txt:66`
```
packaging @ file:///home/task_176104877067765/conda-bld/packaging_1761049113113/work
```
This is an absolute path to a local Conda build directory on a specific machine. It will fail for any other developer or CI system. This is a blocker for reproducibility.

### 🟢 ISSUE-015 — `strategy/trading_strategy.py` and `visualization/plots.py` Are 0-Byte Files
These files are committed as empty placeholders, contributing nothing to the system but implying functionality that doesn't exist. They will cause import errors if any other module tries to import from them.
