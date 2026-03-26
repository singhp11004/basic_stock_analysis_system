# RL & ML Integration — Deep Technical Reference
**Basic Stock Analysis System**
**Document Date: 2026-03-26**

This document provides a complete, element-by-element walkthrough of every Machine Learning (ML) and Reinforcement Learning (RL) component in the system — what it does, where it lives in the pipeline, how it connects to its neighbors, and why it was designed that way.

---

## System Pipeline at a Glance

```
Raw Prices (yfinance)
    │
    ▼
[PHASE 1 — Data Ingestion]         → data/raw/stock_prices.csv
    │
    ▼
[PHASE 2 — Preprocessing]          → data/processed/cleaned_data.csv
    │
    ▼
[PHASE 3 — ML: Technical Indicators] → data/features/features.csv
    │
    ▼
[PHASE 4 — ML: ML Signal (RF)]     → data/features/features_with_ml.csv
    │
    ▼
[PHASE 5 — Train/Test Split]       → train_features.csv / test_features.csv
    │
    ▼
[PHASE 6 — ML: Feature Normalization] → *_normalized.csv
    │
    ▼
[PHASE 7 — RL: Training (PPO)]     → models/ppo_agent.zip
    │
    ▼
[PHASE 8 — RL: Backtesting]        → Console Metrics
```

---

## Part 1 — Machine Learning Layer

The ML layer is responsible for constructing the **state representation** that the RL agent observes. It has two distinct components: hand-crafted technical indicators and a data-driven ML signal.

---

### 1.1 Technical Indicators
**File:** `features/technical_indicators.py`
**Role:** Transform raw OHLCV prices into 10 normalized, engineered features.

#### Design Rule
```
ALL features use Adj Close, never Close.
```
This is critical. Using `Close` instead of `Adj Close` causes artificial price jumps on stock split dates (AAPL has split multiple times), which the agent would incorrectly interpret as market signals.

---

#### Feature 1 — `Return` (Daily Log Return)
```python
df["Return"] = price.pct_change()
```
**What it is:** The percentage change in Adj Close from the previous day.
**Formula:** `(price[t] - price[t-1]) / price[t-1]`
**Why:** The most direct measure of recent price momentum. This is the primary signal the agent uses to understand whether the market went up or down in the last step.
**Range:** Typically -0.05 to +0.05 (±5%) for large-cap stocks.

---

#### Feature 2 — `SMA_10` (10-Day Simple Moving Average)
```python
df["SMA_10"] = price.rolling(10).mean()
```
**What it is:** The average of the last 10 daily closing prices.
**Why:** Captures short-term trend direction. When price > SMA_10, momentum is upward.
**Note:** The raw SMA level is meaningless without context (AAPL at $150 vs $200). After normalization it encodes *relative* deviation from the training mean, which is informative.

---

#### Feature 3 — `SMA_30` (30-Day Simple Moving Average)
```python
df["SMA_30"] = price.rolling(30).mean()
```
**What it is:** The average of the last 30 daily prices.
**Why:** Captures medium-term trend. Used in combination with SMA_10 to compute `Trend`.

---

#### Feature 4 — `RSI` (Relative Strength Index)
```python
delta = price.diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))
```
**What it is:** An oscillator bounded [0, 100] measuring the speed and magnitude of recent price changes.
**Formula:** `RSI = 100 - (100 / (1 + RS))` where `RS = avg_gain / avg_loss` over 14 days.
**Interpretation:**
- RSI < 30 → Oversold (potential BUY signal)
- RSI > 70 → Overbought (potential SELL signal)
- RSI = 50 → Neutral

> [!NOTE]
> The current implementation uses a simple 14-day rolling mean. The industry standard (Wilder 1978) uses exponential smoothing (EMA with α = 1/14). This is a known gap — values near the start of the series differ from reference implementations.

---

#### Feature 5 — `Volatility` (20-Day Rolling Std of Returns)
```python
df["Volatility"] = df["Return"].rolling(20).std()
```
**What it is:** The standard deviation of daily returns over the last 20 trading days (~1 month).
**Why:** Encodes the current risk regime. Low volatility = calm market; high volatility = uncertain, potentially risky. The reward function penalizes volatility, so the agent needs to observe it to act upon it.

---

#### Feature 6 — `MACD_Signal` (Trend Confluence Signal)
```python
ema_12 = price.ewm(span=12, adjust=False).mean()
ema_26 = price.ewm(span=26, adjust=False).mean()
macd = ema_12 - ema_26
signal_line = macd.ewm(span=9, adjust=False).mean()
df["MACD_Signal"] = np.where(macd > signal_line, 1, -1)
```
**What it is:** A binary signal (`+1` or `-1`) representing whether the MACD line is above or below its signal line.
**MACD itself:** Difference between 12-day EMA and 26-day EMA. Positive when short-term trend outpaces long-term (bullish momentum).
**Signal line:** 9-day EMA of the MACD. When MACD > signal line → bullish crossover.
**Why binary:** Reduces noise. The agent doesn't need the exact MACD magnitude, only whether momentum is bullish or bearish at this moment.

---

#### Feature 7 — `BB_Position` (Bollinger Band Position)
```python
bb_middle = price.rolling(20).mean()
bb_std = price.rolling(20).std()
bb_upper = bb_middle + 2 * bb_std
bb_lower = bb_middle - 2 * bb_std
df["BB_Position"] = (price - bb_lower) / (bb_upper - bb_lower)
df["BB_Position"] = df["BB_Position"].clip(0, 1)
```
**What it is:** Where the current price sits within its 20-day Bollinger Bands, mapped to [0, 1].
- `0.0` → Price at the lower band (statistically low, potential oversold)
- `0.5` → Price at the 20-day average (neutral)
- `1.0` → Price at the upper band (statistically high, potential overbought)

**Why:** Encodes mean-reversion context. A BBPosition of 0.05 combined with RSI=28 is a strong oversold double-signal that the agent can learn to exploit.

---

#### Feature 8 — `Momentum` (10-Day Rate of Change)
```python
df["Momentum"] = price.pct_change(10)
```
**What it is:** The total percentage price change over 10 trading days (2 weeks).
**Formula:** `(price[t] - price[t-10]) / price[t-10]`
**Why:** Distinct from `Return` (which is 1-day). Momentum captures sustained directional movement that is too slow to appear in daily returns but too fast for SMA_30 to capture clearly.

---

#### Feature 9 — `Trend` (SMA Ratio)
```python
df["Trend"] = df["SMA_10"] / df["SMA_30"]
```
**What it is:** The ratio of the 10-day SMA to the 30-day SMA.
- `> 1.0` → Short-term average above long-term average → **uptrend**
- `< 1.0` → Short-term average below long-term average → **downtrend**
- `= 1.0` → Convergence / transition point

**Why:** A clean, dimensionless signal for the general market trend. Unlike the raw SMA values (which depend on price level), the ratio is scale-invariant.

---

#### Feature 10 — `Vol_Regime` (Volatility Regime Flag)
```python
avg_volatility = df["Volatility"].rolling(60).mean()
df["Vol_Regime"] = np.where(df["Volatility"] > avg_volatility, 1, 0)
```
**What it is:** A binary flag indicating whether current volatility is above its own 60-day rolling average.
- `1` → High-volatility regime (market is fearful / uncertain)
- `0` → Low-volatility regime (market is calm / confident)

**Why:** Acts as a proxy for the VIX (fear index). When `Vol_Regime=1`, the agent should be more conservative. When `Vol_Regime=0`, momentum strategies tend to work better.

---

### 1.2 ML Signal — Random Forest Classifier
**File:** `features/ml_signals.py`
**Role:** Generate a predictive signal (`ML_Signal`) representing the probability the next day's return will be positive.

#### Architecture
```
Input:  10 technical features (all except Return, Date, Target)
Model:  RandomForestClassifier(n_estimators=100, max_depth=5, 
                               min_samples_leaf=10, random_state=42)
Output: Probability of next-day positive return ∈ [0.0, 1.0]
```

#### Target Construction
```python
# Target: Will tomorrow's return be positive?
df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
```
**Importance:** `shift(-1)` shifts the return column backwards by 1, so `Target[t]` = whether `Return[t+1] > 0`. This is a **next-day direction classification** problem.

#### Strict Data Segregation
```python
train_df = df[df["Date"] < split_date].copy()   # Model fits on this
test_df  = df[df["Date"] >= split_date].copy()   # Model NEVER sees these labels during training

# Only train rows (minus last row with no known target):
train_df = train_df.iloc[:-1]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # ← Fit on TRAIN ONLY
model.fit(X_train_scaled, y_train)

# Predictions are generated for ALL data (including test), 
# but model was never exposed to test labels:
X_all_scaled = scaler.transform(X_all)
ml_probs = model.predict_proba(X_all_scaled)[:, 1]
df_for_pred["ML_Signal"] = ml_probs
```
**Why this matters:** If the model was trained on test-data labels, the ML_Signal would encode future information into the RL state — a form of data leakage that would produce unrealistically good backtest results.

#### ML_Signal interpretation
| Value Range | Meaning |
|---|---|
| 0.40 – 0.60 | No strong directional edge |
| > 0.65 | Strong bullish signal |
| < 0.35 | Strong bearish signal |

---

### 1.3 Feature Normalization (Z-Score)
**File:** `preprocessing/feature_normalizer.py`
**Role:** Scale all features to zero mean and unit variance using training statistics only.

```python
class FeatureNormalizer:
    def fit(self, df):
        # Compute statistics from TRAINING data only
        self.means = df[self.feature_cols].mean()
        self.stds  = df[self.feature_cols].std()
        self.stds  = self.stds.replace(0, 1)     # avoid divide-by-zero
    
    def transform(self, df):
        # Apply: z = (x - μ) / σ
        for col in self.feature_cols:
            df_norm[col] = (df_norm[col] - self.means[col]) / self.stds[col]
```

**Why normalize?**
- Raw features have wildly different scales: `RSI ∈ [0, 100]`, `Return ∈ [-0.1, 0.1]`, `SMA_30 ∈ [$40, $250]`.
- The PPO neural network's weight gradients are sensitive to feature scale. Without normalization, large-magnitude features (SMA values) dominate the gradient updates.
- Z-scoring ensures every feature contributes proportionally to the agent's decision.

**Key constraint — Fit on Train Only:**
```
train stats → used to normalize train data ✅
train stats → used to normalize test data  ✅  (same stats, no leakage)
test stats  → NEVER used for normalization ❌
```
If test statistics were used, the normalizer would implicitly encode future information (the statistical distribution of the test period) into the training process.

---

## Part 2 — Reinforcement Learning Layer

### 2.1 The Trading Environment
**File:** `env/trading_env.py`
**Role:** Define the Markov Decision Process (MDP) the agent operates in.

#### MDP Definition
| Element | Value |
|---|---|
| **State S** | 11 normalized market features + 1 position flag = 12-dim continuous vector |
| **Action A** | `{0: HOLD, 1: BUY, 2: SELL}` — discrete, 3 actions |
| **Reward R** | Multi-objective: profit + drawdown penalty + volatility penalty |
| **Transition** | Deterministic: price data is fixed; agent actions change portfolio, not market |
| **Termination** | End of time series (episode = one pass over the price history) |

#### State Construction
```python
def _get_state(self):
    row = self.data.loc[self.current_step]

    # Exclude raw price columns — they are used for trading, not state
    drop_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols_to_drop = [c for c in drop_cols if c in row.index]
    features = row.drop(labels=cols_to_drop).values.astype(np.float32)

    # Append position flag (0 = holding cash, 1 = holding stock)
    position_flag = np.array([float(self.position)], dtype=np.float32)

    return np.concatenate([features, position_flag])
```

State vector at any timestep `t`:
```
[Return, SMA_10, SMA_30, RSI, Volatility, MACD_Signal, BB_Position,
 Momentum, Trend, Vol_Regime, ML_Signal, position_flag]
```
→ 12 dimensions total (all normalized except `position_flag` which is already binary).

#### Action Execution
```python
def step(self, action: int):
    price = self.prices_data.loc[self.current_step, "Adj Close"]  # REAL price

    if action == BUY and self.position == 0:
        effective_cash = self.cash * (1 - self.transaction_cost)  # pay 0.1%
        self.shares = effective_cash / price
        self.cash = 0.0
        self.position = 1

    elif action == SELL and self.position == 1:
        proceeds = self.shares * price
        self.cash = proceeds * (1 - self.transaction_cost)        # pay 0.1%
        self.shares = 0.0
        self.position = 0
```

**Key design choice:** Normalized features are used for the **state** (generalization), while real `Adj Close` prices are used for **portfolio valuation** (financial accuracy). These come from two separate files (the normalized CSV and the original), which the environment loads simultaneously.

#### Transaction Cost
```
transaction_cost: 0.001  # = 0.1% per trade side
```
Applied on both BUY and SELL. A round-trip trade costs ~0.2%, incentivising the agent to hold positions rather than trade randomly.

---

### 2.2 Reward Function
**File:** `rewards/reward_function.py`
**Role:** Translate the portfolio's step-by-step performance into a scalar signal the agent optimizes.

#### Multi-Objective Reward Formula
```python
reward = (
    profit_weight    * step_return    # +5.0 × return: maximize profit
    - drawdown_penalty * drawdown     # -0.01 × drawdown: penalize peak-to-trough losses
    - volatility_penalty * volatility # -0.01 × rolling_std: penalize erratic returns
)
```

#### Component 1 — Normalized Step Return
```python
step_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
```
**What it is:** The per-step portfolio return (not in dollar terms, but as a fraction).
**Why normalize?** A +$500 gain means nothing without context — it's huge on $10,000 but tiny on $1,000,000. The fractional return is scale-invariant and comparable across portfolio sizes.
**Weight:** `profit_weight = 5.0` — the dominant signal.

#### Component 2 — Drawdown Penalty
```python
self.peak_portfolio_value = max(self.peak_portfolio_value, current_portfolio_value)
drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
```
**What it is:** The current percentage decline from the highest portfolio value ever reached in this episode.
**Example:** If peak was $110,000 and current is $95,000 → drawdown = (110,000 - 95,000) / 110,000 = 13.6%
**Weight:** `drawdown_penalty = 0.01` — currently minimal vs profit weight.

#### Component 3 — Volatility Penalty
```python
self.returns_window.append(step_return)  # rolling window of last 20 step returns
volatility = np.std(self.returns_window)
```
**What it is:** The standard deviation of recent step returns (a rolling 20-step volatility).
**Weight:** `volatility_penalty = 0.01` — currently minimal.

> [!CAUTION]
> The current drawdown and volatility penalties (0.01 each) are **500× smaller** than the profit weight (5.0). This means the agent is effectively trained to maximize raw return with near-zero risk consideration. Future work should rebalance these weights toward Sortino-style penalization of only downside volatility.

---

### 2.3 RL Algorithm — Proximal Policy Optimization (PPO)
**File:** `training/train_agent.py`
**Library:** Stable-Baselines3
**Role:** The core decision-making algorithm that learns a policy from environment interactions.

#### What PPO Is
PPO is an **Actor-Critic, on-policy** policy gradient algorithm. It learns two things simultaneously:
- **Actor (Policy π):** Given a state, which action probabilities should I output?
- **Critic (Value V):** Given a state, how good is this state (expected future reward)?

```
State Input (12-dim)
       │
  Shared MLP Backbone:
    Linear(12 → 64)  + Tanh
    Linear(64 → 64)  + Tanh
       │
  ┌────┴────┐
  │         │
Actor      Critic
  │         │
Linear(64→3) Linear(64→1)
Softmax      (state value)
(action probs)
```

#### The PPO Objective (Clipped Surrogate)
```
L_CLIP(θ) = E[ min(r_t(θ) × A_t,  clip(r_t(θ), 1-ε, 1+ε) × A_t) ]

where:
  r_t(θ)  = π_θ(a|s) / π_θ_old(a|s)   ← policy ratio (new/old)
  A_t     = advantage estimate           ← how much better was this action?
  ε       = 0.2                          ← clipping range
```
**Why clip?** Without clipping, a single large gradient update could move the policy too far in one direction (e.g., "always BUY"), destroying previously learned behaviour. Clipping ensures each update is conservative, making training far more stable than classic REINFORCE or DQN.

#### Advantage Estimation (GAE)
```
A_t = δ_t + γλδ_{t+1} + (γλ)²δ_{t+2} + ...
where δ_t = r_t + γV(s_{t+1}) - V(s_t)   ← TD residual
      γ = 0.99  (discount factor)
      λ = 0.95  (GAE lambda from SB3 default)
```
GAE (Generalized Advantage Estimation) is a bias-variance trade-off:
- `λ=0` → Pure 1-step TD (low variance, high bias)
- `λ=1` → Full Monte Carlo (low bias, high variance)
- `λ=0.95` → The sweet spot used here

#### Training Configuration
```yaml
# config/config.yaml — sb3 section
sb3:
  learning_rate: 0.0003     # Adam optimizer step size
  n_steps: 2048             # steps collected per policy update rollout
  batch_size: 64            # mini-batch size for gradient updates
  n_epochs: 10              # how many times to reuse each rollout
  gamma: 0.99               # discount factor (future reward weighting)
  ent_coef: 0.01            # entropy bonus (encourages exploration)
  total_timesteps: 500000   # total environment steps to train for
```

**n_steps = 2048:** The agent interacts with the environment for 2048 steps before performing a policy update. Given ~3 years of daily training data (~750 steps per episode), this spans roughly 2–3 full simulated trading years per batch.

**ent_coef = 0.01:** Adds `0.01 × H(π)` to the loss where H is the policy entropy. This prevents premature convergence to a deterministic (and potentially wrong) action like "always BUY".

#### Training Code
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Vectorize environment (required by SB3)
env = DummyVecEnv([lambda: TradingEnv(config_path, data_path=train_data_path)])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate = 0.0003,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    ent_coef      = 0.01,
    verbose       = 1
)

model.learn(total_timesteps=500_000)
model.save("models/ppo_agent")
```

---

### 2.4 Backtesting — How the Agent Is Evaluated
**File:** `backtesting/backtester.py`
**Role:** Run the frozen trained agent on **unseen test data** and measure performance.

```python
model = PPO.load("models/ppo_agent")   # Load frozen weights

done = False
while not done:
    # Deterministic inference — no exploration noise
    action, _states = model.predict(state, deterministic=True)

    next_state, reward, done, truncated, info = env.step(int(action))
    portfolio_history.append(env.portfolio_value)

    # Float-tolerance safety check
    FLOAT_TOLERANCE = 1e-8
    assert not (env.cash > FLOAT_TOLERANCE and env.shares > FLOAT_TOLERANCE)
    
    state = next_state
```

**`deterministic=True`:** During evaluation, no exploration. The agent always picks the highest-probability action, ensuring reproducible results.

**Cold start:** The environment is re-initialized to `$100,000` cash before the backtest. No knowledge from training carries over to portfolio state — only the learned weights.

---

### 2.5 Evaluation Metrics
**File:** `evaluation/metrics.py`
**Role:** Quantify performance against a buy-and-hold baseline.

#### Total Return
```python
total_return = ((final_value - initial_value) / initial_value) * 100
```
**What it measures:** Did the agent grow the portfolio? Compare against buy-and-hold.

#### Maximum Drawdown
```python
values = np.array(portfolio_values)
peak = np.maximum.accumulate(values)      # running maximum
drawdown = (peak - values) / peak         # fraction below peak at each point
max_drawdown = np.max(drawdown) * 100
```
**What it measures:** The worst peak-to-trough decline during the test period (%). Key risk metric — a 50% drawdown requires a 100% gain to recover.

#### Sharpe Ratio
```python
# Annualized Sharpe (assumes daily returns, 252 trading days/year)
excess_returns = returns - risk_free_rate / 252
sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
```
**What it measures:** Return per unit of risk. Sharpe > 1.0 is typically considered good for a trading strategy.
**Limitation:** `risk_free_rate = 0.0` is hardcoded. In 2023–2025 (Fed Funds at 5%+), this significantly overstates the attractiveness of returns.

#### Annualized Volatility
```python
volatility = np.std(returns) * np.sqrt(252) * 100
```
**What it measures:** How erratic the portfolio returns are, expressed as an annualized percentage. Lower is better, all else equal.

---

## Part 3 — Information Flow: ML → RL

The key integration point is how ML features feed into the RL observation space:

```
Raw price data (Adj Close, OHLC, Volume)
        │
        ▼
[technical_indicators.py]
  Compute: Return, SMA_10, SMA_30, RSI, Volatility,
           MACD_Signal, BB_Position, Momentum, Trend, Vol_Regime
        │
        ▼
[ml_signals.py]
  Train RF on: Return, SMA_10, SMA_30, RSI, Volatility,
               MACD_Signal, BB_Position, Momentum, Trend, Vol_Regime
  Predict:     P(next-day return > 0) → ML_Signal ∈ [0, 1]
  Append:      ML_Signal to feature table
        │
        ▼
[feature_normalizer.py]
  Z-score all 11 features using TRAIN STATISTICS ONLY
        │
        ▼
[TradingEnv._get_state()]
  Concatenate: [11 normalized features] + [position_flag]
  → 12-dim observation vector delivered to PPO
        │
        ▼
[PPO Actor Network]
  Input: 12-dim state → Output: P(HOLD), P(BUY), P(SELL)
        │
        ▼
[TradingEnv.step()]
  Execute action using REAL Adj Close prices
  Compute reward from portfolio value change
        │
        ▼
[PPO updates weights to maximize expected discounted reward]
```

This architecture strictly isolates:
1. **Signal domain** (ML features) — computed from historical price data
2. **Decision domain** (RL agent) — acts on normalized signals, never sees raw prices
3. **Evaluation domain** (real prices) — used only for portfolio accounting and baseline comparison

---

## Part 4 — Config Reference

All hyperparameters in one place (`config/config.yaml`):

```yaml
trading:
  initial_cash: 100000     # Starting portfolio value
  transaction_cost: 0.001  # 0.1% per trade — incentivizes holding

reward:
  profit_weight: 5.0       # Weight on step return (dominant signal)
  drawdown_penalty: 0.01   # Weight on drawdown (currently minimal)
  volatility_penalty: 0.01 # Weight on volatility (currently minimal)

sb3:                        # PPO Hyperparameters
  learning_rate: 0.0003    # Adam optimizer LR
  n_steps: 2048            # Steps per rollout
  batch_size: 64           # SGD mini-batch
  n_epochs: 10             # Policy update iterations per rollout
  gamma: 0.99              # Discount factor (long-term focus)
  ent_coef: 0.01           # Exploration entropy bonus
  total_timesteps: 500000  # Total training budget

data:
  split_date: "2023-01-01"  # Train < this date; Test >= this date
```
