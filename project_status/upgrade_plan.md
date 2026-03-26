# Advanced Deep Reinforcement Learning Upgrade Plan
## Basic Stock Analysis System — Research-Grade Redesign

**Analysis Date: 2026-03-25**
**Scope:** Deep RL architecture, algorithmic upgrades, feature modernization, evaluation rigour, real-world deployment

---

> [!IMPORTANT]
> **Update: 2026-03-26**
> **PHASE A and PHASE B have been successfully implemented.** The system now utilizes a Deep RL architecture with PPO (Stable-Baselines3). This document remains as a roadmap for future enhancements (Phase C and beyond).

---

## Why the Current Q-Learning Agent [REPLACED] Cannot Work

Before proposing upgrades, it is critical to understand *why* tabular Q-learning is fundamentally unsuitable here — not just suboptimal, but structurally incorrect for this problem.

| Property | Tabular Q-Learning | What Trading Requires |
|---|---|---|
| State space | Discrete, enumerable | Continuous, high-dimensional |
| Generalization | Zero (lookup only) | Must generalize unseen states |
| Sample efficiency | Very low | Must learn from limited history |
| Temporal memory | None (Markov only) | Needs multi-step market context |
| Regime adaptation | None | Essential (bull/bear/sideways) |
| Position sizing | Binary only | Continuous allocation |

**The fundamental mismatch:** Financial markets produce continuous observations where two states that are numerically "similar" (adjacent bins) often carry completely different trading signals (e.g., RSI=29.9 vs RSI=30.1 — the oversold boundary). Discretization destroys this critical boundary information.

---

## Architecture Roadmap Summary

```
CURRENT                 PHASE A            PHASE B              PHASE C
───────────────         ──────────────     ─────────────────    ──────────────────
Tabular Q-Learning  →   DQN + DDQN    →   PPO / SAC / TD3  →   Transformer + 
(Q-table, 914KB)        (Neural Net)       (Actor-Critic)        Multi-Head Attention
                                                                  + FinRL Ecosystem
```

---

## PHASE A — Deep Q-Network Family (Immediate Upgrade, 1–2 Weeks)

This phase replaces the tabular agent with a neural function approximator. It is the minimum viable deep RL upgrade and solves the state-space explosion entirely.

### A.1 Vanilla DQN → Double DQN → Dueling DQN

Implement all three progressively. Each resolves a specific failure mode of plain Q-learning.

#### A.1.1 Deep Q-Network (DQN) — Foundation

**Core Architecture:**
```
Input State (12-14 dims, continuous + normalized)
    │
BatchNorm1d(input_dim)          ← stabilizes training across varied feature scales
    │
Linear(input_dim → 256) + ReLU
    │
Linear(256 → 128) + ReLU
    │
Dropout(0.2)                    ← prevents overfitting to training regime
    │
Linear(128 → 64) + ReLU
    │
Linear(64 → 3)                  ← Q(s, HOLD), Q(s, BUY), Q(s, SELL)
```

**Key components required:**

| Component | Purpose | Implementation |
|---|---|---|
| Experience Replay Buffer | Break temporal correlations; reuse transitions | `collections.deque(maxlen=100_000)` |
| Target Network | Stabilize Q-target; prevent moving target problem | Soft update: `θ_target ← τθ + (1-τ)θ_target` (τ=0.005) |
| ε-greedy with linear decay | Exploration → exploitation schedule | Decay over 50,000 steps, not episodes |
| Huber Loss | Robust to large TD errors (reward outliers common in finance) | `F.smooth_l1_loss()` in PyTorch |
| Gradient Clipping | Prevent exploding gradients | `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` |
| Batch Normalization | Normalize activations across batch | `nn.BatchNorm1d()` after input |
| Prioritized Experience Replay | Sample important transitions more frequently | SumTree data structure; α=0.6, β=0.4→1.0 |

**Training hyperparameters (recommended starting point):**
```yaml
dqn:
  learning_rate: 0.0003          # Adam optimizer
  batch_size: 128
  replay_buffer_size: 100000
  gamma: 0.99                    # discount factor
  tau: 0.005                     # soft target update rate
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay_steps: 50000     # step-based, not episode-based
  min_replay_size: 10000         # don't train until buffer has enough samples
  target_update_freq: 1          # use soft updates (tau) rather than hard updates
  train_freq: 4                  # train every 4 environment steps
```

#### A.1.2 Double DQN (DDQN) — Fixes Overestimation

**The Problem with Vanilla DQN:**
```
Q-target = r + γ * max_a Q_target(s', a)
```
The `max` operator selects the highest Q-value action AND evaluates it with the same network. When Q-values have noise (always early in training), this systematically **overestimates** returns.

**DDQN Fix (one line change):**
```python
# VANILLA DQN (overestimates):
next_q = target_net(next_states).max(1)[0]

# DOUBLE DQN (decoupled selection + evaluation):
next_actions = online_net(next_states).argmax(1)          # SELECT with online net
next_q = target_net(next_states).gather(1, next_actions)  # EVALUATE with target net
```

**Why this matters for trading:** In financial markets, overestimated Q-values cause the agent to hold positions too long during drawdowns (it thinks future value is higher than it is). DDQN produces more realistic Q-values and earlier sell signals.

#### A.1.3 Dueling DQN — Learns "When to Hold" Better

**The Problem:** Standard DQN must learn a separate Q-value for every (state, action) pair. In trading, **HOLD** dominates most time steps — but the agent wastes capacity learning Q(s, HOLD) for every state independently.

**Dueling Architecture:**
```
Shared Feature Layers (same as DQN)
         │
    ┌────┴────┐
    │         │
  V(s)      A(s,a)            ← TWO separate "heads"
State      Advantage
Value      Function
(scalar)   (3-dim vector)
    │         │
    └────┬────┘
         │
Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```

- **V(s):** "How good is this state overall?" — learned once regardless of action
- **A(s,a):** "How much better is action a versus the average?" — learned as a differential

**Trading benefit:** V(s) encodes the market's general quality (trending vs choppy). A(s,a) encodes the marginal value of buying vs selling vs holding. The agent can learn "this is a volatile regime (V is low) so holding is best" without re-learning it for every state.

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.feature = nn.Sequential(
            nn.BatchNorm1d(state_dim),
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        features = self.feature(x)
        V = self.value_stream(features)
        A = self.advantage_stream(features)
        return V + A - A.mean(dim=1, keepdim=True)  # Q = V + (A - mean(A))
```

### A.2 Prioritized Experience Replay (PER)

Standard replay samples uniformly — a catastrophic miss in trading where rare, high-impact events (market crashes, breakouts) are the most important transitions to learn from.

**PER prioritizes transitions with high TD error:**
```python
priority = |TD error| + ε     # ε prevents zero-priority transitions
P(i) = priority_i^α / Σ priority_j^α    # α=0.6 controls how much prioritization
```

Importance sampling weights correct for the resulting bias:
```python
w_i = (1 / (N * P(i)))^β     # β annealed from 0.4 → 1.0 during training
```

**Implementation:** Use a **SumTree** data structure for O(log n) priority sampling. Available in `stable-baselines3` or implement with ~100 lines of code.

**Why critical for trading:** A day with a 10% market crash has 100× more learning value than a typical flat day. Without PER, the crash transition is sampled equally with boring days and effectively ignored. PER will replay it repeatedly until the agent learns the correct response.

---

## PHASE B — Policy Gradient Methods (2–4 Weeks After Phase A)

Policy gradient methods directly optimize the policy rather than estimating Q-values. They are better suited for:
- Continuous action spaces (later: position sizing)
- Highly non-stationary environments (financial markets are strongly non-stationary)
- Multi-objective rewards (natural for trading: return + risk + drawdown)

### B.1 Proximal Policy Optimization (PPO)

PPO is the default choice for trading applications. It's the algorithm in the reference notebook (`reference_ppo-reinforcement-learning-trading-agent.ipynb`) and is validated across thousands of real-world RL deployments.

**Core Idea:** Prevent catastrophic policy updates by clipping the policy ratio:
```
L_CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

where r_t(θ) = π_θ(a|s) / π_θ_old(a|s)   ← policy ratio
      A_t = advantage estimate (how much better this action was vs average)
      ε = 0.2                               ← clipping range
```

This clip prevents the new policy from moving too far from the old policy in any single update — crucial for financial markets where a single bad update can destroy weeks of learning.

**Actor-Critic Architecture for PPO:**
```
State Input (continuous)
        │
Shared Backbone:
  Linear(state_dim → 256) + Tanh     ← Tanh preferred over ReLU for Actor-Critic
  Linear(256 → 128) + Tanh
        │
   ┌────┴────┐
   │         │
Actor      Critic
(Policy)   (Value)
   │         │
Linear(128 → 64) + Tanh    Linear(128 → 64) + Tanh
   │         │
Linear(64 → 3)             Linear(64 → 1)
Softmax                    (state value estimate)
(action probabilities)
```

**Training components:**

| Component | Configuration | Justification |
|---|---|---|
| Generalized Advantage Estimation (GAE) | λ=0.95 | Variance reduction; standard for PPO |
| Value function coefficient | c1=0.5 | Balance policy vs value loss |
| Entropy coefficient | c2=0.01 | Encourage exploration; prevent premature convergence |
| PPO epochs per rollout | 10 | Multiple gradient updates per collected batch |
| Mini-batch size | 64 | |
| Rollout length (horizon) | 2048 steps | Full buffer before each update |
| Learning rate schedule | 3e-4 → 1e-5 (linear decay) | |
| Max gradient norm | 0.5 | |

**PPO vs DQN for trading:**

| Aspect | DQN | PPO |
|---|---|---|
| Action space | Discrete only | Discrete and continuous |
| Sample efficiency | Higher (off-policy) | Lower (on-policy, samples discarded after update) |
| Stability | Moderate | High |
| Hyperparameter sensitivity | High | Low |
| Position sizing later | No | Yes (Gaussian policy for % allocation) |

**Recommendation:** Start with DQN (Phase A), validate the pipeline, then switch to PPO for its superior stability in non-stationary environments. This matches the reference notebook trajectory.

### B.2 Soft Actor-Critic (SAC)

SAC is the single best off-policy algorithm for contexts with:
- Complex, non-convex reward landscapes (multi-objective trading rewards)
- Strong non-stationarity (regime changes in markets)
- Need for automatic exploration without manual ε-tuning

**Core Idea — Maximum Entropy RL:**
Standard RL maximizes expected return:
```
J(π) = E[Σ γ^t r_t]
```

SAC maximizes return **plus entropy** of the policy:
```
J(π) = E[Σ γ^t (r_t + α * H(π(·|s_t)))]

where H(π(·|s)) = -E[log π(a|s)]   ← entropy: how "spread out" the policy is
      α = temperature parameter (controls exploration vs exploitation balance)
```

**Why this is powerful for trading:**
Maximizing entropy means the agent learns to stay uncertain (spread probability across actions) when the market signal is unclear — exactly the correct behaviour. A deterministic agent exploits noise; a maximum-entropy agent hedges when unsure.

**Auto-tuning of temperature (α):**
```python
# α is automatically tuned to maintain target entropy
target_entropy = -np.log(1.0 / n_actions) * 0.98  # ≈ desired minimum entropy
log_alpha = nn.Parameter(torch.zeros(1))
alpha = log_alpha.exp()
```
This eliminates the need to manually tune the exploration-exploitation trade-off — one of the most difficult hyperparameters in DQN.

**SAC Architecture (Twin Critics — Critical for Stability):**
```
Actor: μ(s), σ(s) → Gaussian(a)     ← stochastic policy
Critic 1: Q1(s, a)
Critic 2: Q2(s, a)                   ← two critics; use minimum for targets
Target Critic 1 + Target Critic 2    ← soft-updated copies

Target: r + γ * min(Q1, Q2)(s', a') - α * log π(a'|s')
                ↑
         Take the MINIMUM of twin critics → prevents overestimation
```

**Research evidence for trading:** In comparative studies (arXiv 2024), SAC shows the best sample efficiency for trading environments with multi-objective rewards, producing agents that maximize return while naturally controlling volatility (due to entropy maximization = distributing risk across actions).

### B.3 Twin Delayed Deep Deterministic Policy Gradient (TD3)

TD3 is the deterministic counterpart to SAC. Better when:
- You can tolerate less exploration
- You want a deterministic policy at inference (cleaner execution signals)
- You need to model position sizing as a continuous fraction

**Three innovations over DDPG:**

1. **Twin Critics** (same as SAC) — prevents Q-value overestimation
2. **Delayed Actor Updates** — actor updates every 2 critic updates (prevents oscillation)
3. **Target Policy Smoothing** — add noise to target actions:
   ```python
   target_action = (actor_target(next_state) + noise).clamp(-1, 1)
   # noise ~ N(0, 0.2) clipped to [-0.5, 0.5]
   ```
   This prevents the critic from over-fitting to a specific deterministic action.

**TD3 for position sizing (continuous action):**
With TD3, the action can be a **continuous value in [-1, +1]**:
- `a = -1.0`: Sell everything
- `a = 0.0`: Hold current position
- `a = +1.0`: Buy with all cash
- `a = 0.5`: Buy with 50% of available cash

This is far more realistic than binary buy/sell and allows the agent to express confidence through allocation size — a key capability for real-world risk management.

---

## PHASE C — Temporal Models and Transformer Architectures (4–8 Weeks)

Phases A and B use **feedforward** networks that treat each time step independently. Financial markets are **deeply temporal** — the meaning of today's RSI=65 depends on whether it was 40 last week (rising momentum) or 85 last week (cooling momentum). The current system has zero temporal memory.

### C.1 LSTM / GRU State Encoder

Replace the feedforward feature backbone with a recurrent encoder:

```
State history: [s_{t-29}, s_{t-28}, ..., s_{t-1}, s_t]  ← last 30 days
        │
LSTM(input_size=feature_dim, hidden_size=128, num_layers=2, dropout=0.2)
        │
  LSTM output at t → encoded_state (128-dim)
        │
Actor head / Critic head / DQN head (same as before)
```

**Key design decisions:**

| Decision | Recommendation | Reason |
|---|---|---|
| Sequence length | 30 trading days (~6 weeks) | Captures medium-term momentum; avoids vanishing gradients |
| LSTM vs GRU | GRU | Fewer parameters, comparable performance, faster training |
| Bidirectional | No | Trading is causal; no future data allowed |
| Layer norm | Yes (`nn.LayerNorm`) | More stable than batch norm for sequences |

```python
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, seq_len=30):
        super().__init__()
        self.seq_len = seq_len
        self.norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, 
                          batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 64)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.norm(x)
        out, _ = self.gru(x)
        return F.relu(self.fc(out[:, -1, :]))  # use last timestep output
```

**Training modification:** The replay buffer must now store **sequences** of transitions rather than individual (s, a, r, s') tuples. Each sample becomes (s_seq, a, r, s'_seq) where s_seq is the last 30 states.

### C.2 Multi-Head Attention / Transformer Encoder

State-of-the-art approach as of 2025. The Transformer's self-attention mechanism learns which past time steps are most relevant for the current decision — a property uniquely valuable for financial time series.

**Why Attention Outperforms LSTM for Financial Data:**
- LSTM forgets distant context (gradient vanishing over long sequences)
- Attention directly computes relevance weight between any two time steps
- Can learn "earnings season patterns" (specific-period correlations every 90 days)
- Parallelizable across time steps (much faster training)

**Temporal Transformer Architecture:**
```
State sequence: [s_{t-L}, ..., s_t]  shape: (seq_len, feature_dim)
        │
Linear projection → d_model=128
        │
Positional Encoding (sinusoidal or learned)
        │
Transformer Encoder Block × N_layers:
  ┌──────────────────────────────────┐
  │ Multi-Head Self-Attention        │  n_heads=4
  │   (causal mask: past only)       │  d_k = d_model / n_heads = 32
  │ + Residual + LayerNorm           │
  │                                  │
  │ Feed-Forward Network             │  d_ff = 512
  │ + Residual + LayerNorm           │
  └──────────────────────────────────┘ × 3 layers
        │
 Output at position t (current step)  (128-dim encoded state)
        │
 Actor / Critic / DQN Head
```

**Causal masking is mandatory** — the attention at time t must only attend to positions ≤ t:
```python
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
attn_output = F.multi_head_attention_forward(..., attn_mask=causal_mask)
```

**Research validation:** The "TRONformer" paper (NeurIPS 2024) shows Transformer-PPO outperforming LSTM-PPO and MLP-PPO on S&P 500 constituent trading by 15-25% Sharpe ratio improvement. The attention mechanism learns to focus on earnings weeks and FOMC meetings automatically from historical patterns.

### C.3 Market Regime Awareness — Contextual RL

Financial markets cycle through distinct regimes: bull trending, bear trending, high-volatility sideways, low-volatility sideways. An agent trained without regime awareness learns an averaged-out policy that is mediocre in all regimes.

**Approach: Regime-Conditioned Policy**

1. **Detect regime:** Use an auxiliary classifier (HMM or K-means on realized volatility + trend) to assign a discrete regime label r ∈ {0,1,2,3}

2. **Condition the policy:** Concatenate regime embedding to the state:
```python
# Regime one-hot (or learned embedding)
regime_emb = nn.Embedding(n_regimes=4, embedding_dim=8)
conditioned_state = torch.cat([encoded_state, regime_emb(regime)], dim=-1)
```

3. **Train regime-wise:** The agent learns different behaviour for each regime. In high-volatility (VIX-like) regimes, entropy bonus is increased to encourage holding cash.

**Implementation using Gaussian HMM:**
```python
from hmmlearn import hmm
hmm_model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
features_for_hmm = df[["Return", "Volatility", "Trend"]].values
hmm_model.fit(features_for_hmm[train_idx])
regimes = hmm_model.predict(features_for_hmm)
```

---

## PHASE D — Advanced Reward Engineering

The reward function is as important as the algorithm. The current function is technically correct but financially naive. These upgrades make the reward signal more aligned with what sophisticated investors actually optimize.

### D.1 Sortino Ratio as Reward (Replace Sharpe Component)

The current volatility penalty penalizes both upside and downside variance equally. In finance, **only downside variance is harmful**. The Sortino ratio penalizes only negative returns:

```python
def sortino_reward(returns_window, target_return=0.0):
    downside = [min(r - target_return, 0) for r in returns_window]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-6
    return np.mean(returns_window) / (downside_std + 1e-8)
```

**New reward function:**
```python
reward = (
    profit_weight * step_return                         # maximize returns
    - drawdown_penalty * current_drawdown               # penalize drawdowns
    - sortino_penalty * neg_return_volatility           # penalize only losses, not gains
    - transaction_penalty * |action_changed|            # penalize excessive trading
    + regime_bonus * regime_aligned_action              # reward regime-appropriate actions
)
```

### D.2 Risk-Adjusted Return as Primary Signal

Replace raw return with immediate Sharpe/Sortino as the per-step reward. This teaches the agent to think in risk-adjusted terms from the first training step:

```python
step_sharpe = step_return / (rolling_volatility + 1e-8) * np.sqrt(252)
```

### D.3 Reward Clipping and Normalization

Financial returns vary dramatically across market conditions. A 3% gain on a volatile day is less impressive than 3% on a flat day. Normalize rewards:

```python
# Reward normalization using running statistics
class RunningMeanStd:
    def __init__(self, clip=10.0):
        self.mean, self.var, self.count = 0, 1, 0
        self.clip = clip
    
    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += delta * (x - self.mean)
    
    def normalize(self, x):
        std = np.sqrt(self.var / max(self.count, 1))
        return np.clip((x - self.mean) / (std + 1e-8), -self.clip, self.clip)
```

This prevents rare high-reward days from dominating the gradient update — a critical stability fix for trading RL.

### D.4 Multi-Objective Reward with Pareto Optimal Weighting

Instead of fixed weights, use **dynamically adapted weights** based on current portfolio state:

```python
def adaptive_reward(step_return, drawdown, in_drawdown_phase):
    if in_drawdown_phase:
        # During drawdown: prioritize capital preservation
        profit_w, drawdown_w = 0.3, 3.0
    else:
        # During growth: prioritize returns
        profit_w, drawdown_w = 2.0, 0.5
    return profit_w * step_return - drawdown_w * drawdown
```

---

## PHASE E — Feature Engineering Upgrades

The quality of the input features determines the ceiling of any RL algorithm. These additions create a richer, more predictive state representation.

### E.1 Multi-Timeframe Features (MTF)

The current system uses a single timeframe (daily). Markets operate across multiple timeframes simultaneously — a daily RSI of 30 (oversold) during a weekly uptrend is bullish; the same RSI in a weekly downtrend is bearish.

```python
# Add weekly (5-day) and monthly (21-day) aggregates alongside daily
features_weekly = compute_features(df, timeframe=5)   # 5-day bars
features_monthly = compute_features(df, timeframe=21)  # 21-day bars

# Concatenate into multi-timeframe state
state = pd.concat([features_daily, features_weekly, features_monthly], axis=1)
```

**Additional features to add:**

| Feature | Formula/Source | Why |
|---|---|---|
| ATR (Average True Range) | `max(H-L, |H-Cp|, |L-Cp|)` rolling | Better volatility measure than std |
| OBV (On-Balance Volume) | Cumulative ± volume on up/down days | Volume-confirmed price moves |
| VWAP Deviation | `(Price - VWAP) / VWAP` | Institutional buy/sell zone proximity |
| 52-week High/Low Position | `(P - 52wk_low) / (52wk_high - 52wk_low)` | Momentum context |
| Realized Volatility (5-min bars) | Parkinson estimator from H/L | More accurate than close-to-close |
| Autocorrelation (lag 1, 5, 21) | `df["Return"].autocorr(lag)` | Mean-reversion vs trend signal |

### E.2 Macro Context Features

Add external signals that the stock doesn't encode internally:

```python
import yfinance as yf

def fetch_macro_features(start, end):
    VIX = yf.download("^VIX", start, end)["Adj Close"].rename("VIX")
    SPY_return = yf.download("SPY", start, end)["Adj Close"].pct_change().rename("Market_Return")
    TNX = yf.download("^TNX", start, end)["Adj Close"].rename("TNX_10Y")  # 10yr yield
    DXY = yf.download("DX-Y.NYB", start, end)["Adj Close"].rename("DXY")  # USD index
    return pd.concat([VIX, SPY_return, TNX, DXY], axis=1)
```

**Why macro features matter:**
- **VIX > 30:** Market fear regime — AAPL historically falls with high VIX
- **10Y yield rising:** Tech stocks (AAPL) compress on rate sensitivity
- **SPY underperforming:** Systematic risk — no stock-specific alpha from buying

### E.3 LSTM-Based Learned Features (AutoEncoder Pre-Training)

Before RL training, pre-train a temporal autoencoder on the full feature sequence. The bottleneck representation becomes a compressed "market state embedding" that feeds the RL agent:

```
Input sequence (30 days × 15 features)
        │
LSTM Encoder → latent z (32-dim compact representation)
        │
LSTM Decoder → reconstruct 30 days × 15 features
        │
Loss: MSE reconstruction loss (pre-training only)
```

Then freeze the encoder and use `z` as the RL state. The encoder has learned to compress multi-day market dynamics into a dense representation — a much richer state than manually crafted features.

---

## PHASE F — Evaluation and Robustness Upgrades

### F.1 Walk-Forward with Anchored Window

Replace single split with rolling anchored-window evaluation:

```
Anchor: 2018-01-01
────────────────────────────────────────────────────────
│◄──── Train 2018-2020 ────►│◄─ Test 2021 ─►│
│◄─────────── Train 2018-2021 ─────────────►│◄─ Test 2022 ─►│
│◄───────────────── Train 2018-2022 ───────────────────►│◄─ Test 2023 ─►│
│◄────────────────────── Train 2018-2023 ──────────────────────────►│◄─ Test 2024 ─►│
```

Report **distribution** of metrics across all test windows:
```
Sharpe Ratio: mean=0.82, std=0.31, min=0.41, max=1.35
Total Return: mean=14.2%, std=5.8%, min=4.1%, max=22.7%
Max Drawdown: mean=11.4%, std=3.2%, worst=18.9%
```

A system with consistent performance across regime changes is deployable. A system with Sharpe=1.8 in one window and -0.3 in another is overfit.

### F.2 Monte Carlo Bootstrap for Statistical Significance

```python
def bootstrap_sharpe(returns, n_bootstrap=5000):
    sharpes = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        sharpes.append(calculate_sharpe(sample))
    return {
        "mean": np.mean(sharpes),
        "std": np.std(sharpes),
        "ci_95": (np.percentile(sharpes, 2.5), np.percentile(sharpes, 97.5)),
        "p_positive": (np.array(sharpes) > 0).mean()
    }
```

### F.3 Permutation Test Against Null Hypothesis

Test whether the agent's performance is above chance:
```python
def permutation_test(strategy_returns, n_permutations=1000):
    """Test if Sharpe > 0 is statistically significant."""
    observed_sharpe = calculate_sharpe(strategy_returns)
    null_sharpes = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(strategy_returns)
        null_sharpes.append(calculate_sharpe(shuffled))
    p_value = (np.array(null_sharpes) >= observed_sharpe).mean()
    return observed_sharpe, p_value  # reject null if p_value < 0.05
```

### F.4 Comprehensive Metrics Suite

Add missing professional-grade metrics:

| Metric | Formula | Threshold for "Good" |
|---|---|---|
| Calmar Ratio | Annual Return / Max Drawdown | > 1.0 |
| Information Ratio | (Return - Benchmark) / Tracking Error | > 0.5 |
| Win Rate | % of days with positive P&L | > 52% |
| Profit Factor | Gross Win / Gross Loss | > 1.5 |
| Average Win/Loss Ratio | Mean Win / Mean Loss | > 1.2 |
| Maximum Consecutive Losses | Longest losing streak | < 15 days |
| Recovery Factor | Total Return / Max Drawdown | > 3.0 |
| Omega Ratio | P(return > threshold) / P(return < threshold) | > 1.0 |

---

## PHASE G — Advanced System Architecture

### G.1 FinRL Ecosystem Integration

[FinRL](https://github.com/AI4Finance-Foundation/FinRL) is the leading open-source DRL trading framework from Columbia University / JP Morgan. Integrating it provides:
- Pre-built environments for stocks, crypto, futures, forex
- Tested implementations of DQN, PPO, SAC, TD3, A2C
- Realistic market simulation with liquidity shocks and partial fills
- Multi-asset portfolio environment out of the box

```bash
pip install finrl
```

Migrating to FinRL-compatible environments future-proofs the system and allows direct comparison against published research benchmarks.

### G.2 Stable-Baselines3 Integration

[Stable-Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/) provides production-quality, tested implementations of all Phase B algorithms with one-line training:

```python
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# PPO in one block:
model = PPO(
    "MlpPolicy",
    env=trading_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./logs/ppo/",
)
callbacks = [
    EvalCallback(eval_env, best_model_save_path="./models/", eval_freq=5000),
    CheckpointCallback(save_freq=10000, save_path="./checkpoints/"),
]
model.learn(total_timesteps=500_000, callback=callbacks)
```

Advantages: built-in TensorBoard logging, checkpoint saving, hyperparameter validation, documented reproduction across seeds.

### G.3 Experiment Tracking with MLflow

```python
import mlflow

with mlflow.start_run(run_name="SAC_Transformer_v2"):
    mlflow.log_params({
        "algorithm": "SAC",
        "state_encoder": "Transformer",
        "seq_len": 30,
        "train_period": "2018-2022",
        "test_period": "2023-2025",
    })
    mlflow.log_metrics({
        "sharpe_ratio": 1.24,
        "calmar_ratio": 2.1,
        "total_return_pct": 22.4,
        "max_drawdown_pct": 10.6,
    })
    mlflow.log_artifact("models/sac_agent.zip")
```

---

## Recommended Implementation Sequence

### Immediate (1 week)
1. Apply all Phase 1 bug fixes from `upgrade_plan.md`
2. Install PyTorch, Stable-Baselines3: `pip install torch stable-baselines3`
3. Implement Dueling Double DQN with PER using SB3 or from scratch

### Short-term (2–4 weeks)
4. Implement PPO via SB3 — compare directly to DQN on same test data
5. Add SAC — enable continuous position sizing
6. Add Walk-Forward validation (5 windows)
7. Implement all Phase D reward engineering upgrades

### Medium-term (1–3 months)
8. Add GRU/LSTM temporal encoder
9. Add macro features (VIX, SPY, TNX)
10. Implement full metrics suite with bootstrap CI
11. Integrate MLflow for experiment tracking

### Long-term (3–6 months)
12. Implement Transformer attention encoder
13. Integrate FinRL for multi-asset environment
14. Add regime detection (HMM)
15. Live paper trading via Alpaca API

---

## Algorithm Selection Guide

> Use this table to decide which algorithm to implement for your goals:

| Goal | Recommended Algorithm | Library |
|---|---|---|
| Quick wins, binary buy/sell | Double DQN + PER | PyTorch (from scratch) |
| Stability + simplicity | PPO | Stable-Baselines3 |
| Continuous position sizing | SAC or TD3 | Stable-Baselines3 |
| Best exploration | SAC (automatic entropy) | Stable-Baselines3 |
| Multi-asset portfolio | PPO + FinRL | FinRL + SB3 |
| Temporal pattern learning | PPO + GRU/LSTM backbone | Custom SB3 policy |
| Best possible performance | Transformer + SAC | Custom implementation |
| Research comparison | A2C, PPO, SAC, TD3 all | SB3 (multi-agent comparison) |

---

## Dependencies to Add

```bash
# Core deep learning
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# RL libraries
pip install stable-baselines3[extra]   # PPO, SAC, TD3, A2C + extras
pip install gymnasium                   # modern OpenAI Gym replacement

# Financial RL
pip install finrl                       # FinRL ecosystem

# Experiment tracking
pip install mlflow

# Hidden Markov Model (regime detection)
pip install hmmlearn

# Statistics
pip install scipy statsmodels arch      # financial econometrics

# Visualization
pip install plotly seaborn tensorboard
```

**Total additional dependencies:** ~12 packages
**Estimated disk space:** ~2GB (primarily PyTorch)
