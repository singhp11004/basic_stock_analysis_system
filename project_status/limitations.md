> [!NOTE]
> **Update: 2026-03-26**
> The limitations regarding **Tabular Q-Learning (Section 1)** have been fundamentally addressed by the implementation of a Deep RL agent using **PPO**. The following analysis remains for historical context and to document the rationale for the upgrade.

## 1. Algorithm Limitations — [REPLACED BY PPO]

### 1.1 The Curse of Dimensionality (Primary Bottleneck)
The single most limiting factor in this system is the use of **Tabular Q-Learning** with a 13-14 dimensional continuous state space.

| Parameter | Value | Implication |
|---|---|---|
| State dimensions | ~13 (10 features + ML_Signal + Adj Close + position) | |
| Bins per dimension | 10 | |
| Theoretical state space | 10^13 — 10^14 states | ~100 trillion combinations |
| States actually visited (inferred from q_table.pkl) | ~914,000 | 0.000001% coverage |
| Generalization for unseen states | 0% (returns default zeros) | Effectively random |

**Bottom line:** Tabular Q-learning cannot generalize. It memorizes training state-action pairs. On test data with any distributional shift (different market regime, post-split prices, etc.), the agent behaves randomly for the vast majority of steps.

### 1.2 Discrete Binning Destroys Information
The 10-bin discretization of each continuous feature compresses fine-grained market signals into 10 coarse buckets. For example, RSI = 29.9 and RSI = 30.1 map to the same bin despite meaning "just below oversold" vs "just into oversold" — a meaningful distinction traders act on. This information loss is unavoidable in tabular Q-learning and represents a fundamental capability gap.

### 1.3 No Generalization Across Market Regimes
The Q-table is trained on data from 2018–2022 (primarily a bull market with the COVID crash). The test period (2023–2025) includes:
- Post-rate-hike environment
- High-interest rate period
- Tech sector volatility (2022–2023)

A tabular agent cannot generalize patterns learned in one regime to another. This is a structural limitation, not a fixable bug.

---

## 2. Market Simulation Limitations

### 2.1 Single-Stock Universe Only
The environment simulates exactly one stock (AAPL, hardcoded). There is no:
- Portfolio diversification
- Correlation between assets
- Sector-level analysis
- Cross-sectional momentum (comparing this stock vs market)

This means the agent learns a single-instrument trading strategy with no concept of relative value or risk-adjusted asset allocation — a significant departure from real trading systems.

### 2.2 Binary Position (0 or 100% Invested)
The agent can only be either fully out of the market (100% cash) or fully invested (100% in stock). There is no:
- Partial position sizing
- Kelly criterion
- Position scaling by confidence
- Portfolio construction

This makes the strategy extremely concentrated and high-risk — binary all-in/all-out does not represent how professional trading systems operate.

### 2.3 No Short Selling or Hedging
By design, the agent cannot profit in falling markets. During the 2022 bear market, the optimal strategy (in the training data) is simply "hold cash" — a trivial outcome that doesn't require RL. A more realistic system would allow shorting or options.

### 2.4 Simplified Transaction Costs
The system uses a flat 0.1% transaction cost (`transaction_cost: 0.001`) applied uniformly. Real-world trading involves:
- Bid-ask spread (not modeled)
- Market impact (not modeled — critical for large orders)
- Slippage (not modeled)
- Brokerage tiers (not modeled)
- Short-term capital gains tax (not modeled)

The 0.1% cost underestimates actual trading friction by a significant margin for retail investors.

### 2.5 Fill at Close Price (Lookahead Bias in Execution)
The environment executes trades at the **same day's** `Adj Close` from which the state features were derived. In reality, a decision made at market close can only be executed at the **next day's open**. This is a common but impactful form of lookahead bias — the agent "knows" the closing price before deciding whether to buy at it.

---

## 3. Data Limitations

### 3.1 Single Asset, Single Data Source
All data comes from Yahoo Finance (yfinance), which is a free public API. This means:
- No Level-2 order book data
- No institutional flows/dark pool data
- No options market signals (implied volatility, put/call ratio)
- No fundamental data (P/E ratio, earnings, dividends)
- No macro data (interest rates, CPI, market breadth)
- Yahoo Finance data quality varies — corporate actions may not always be correctly adjusted

### 3.2 Survivorship Bias Risk
AAPL has been one of the world's best-performing stocks over 2018–2025. Testing exclusively on AAPL introduces strong survivorship bias — any strategy that held AAPL long for this period would show exceptional returns. The RL agent's learned behaviour may be trivially "buy-and-hold Apple" rather than a generalizable market strategy.

### 3.3 Limited Test Window
With a split at 2023-01-01 and data through ~2025-03-25, the test set covers approximately 27 months. This is:
- Too short to validate performance across a full market cycle
- Confounded by the strong bull market of 2023–2024
- Insufficient for statistical significance testing of the Sharpe ratio

### 3.4 No Walk-Forward Validation
The system uses a single fixed train/test split. A more robust evaluation would use **walk-forward optimization** (rolling windows with expanding or fixed-size training sets), which:
- Reduces split-date sensitivity
- Tests adaptability to regime changes
- Provides distribution of performance outcomes rather than single point

---

## 4. ML Signal Limitations

### 4.1 Random Forest Predicts Direction, Not Magnitude
The `ML_Signal` is a probability of positive next-day return (0 to 1). It does not encode:
- Magnitude of expected return
- Risk/uncertainty around the prediction
- Confidence calibration quality

A Random Forest with 100 trees on financial data will typically show poor out-of-sample calibration — probabilities cluster around 0.5 and the signal has low information content in trending markets.

### 4.2 No Feature Selection or Regularization Analysis
The Random Forest uses all 9 technical features as inputs without analyzing which are redundant or harmful. Collinear features (e.g., SMA_10, SMA_30, Trend are derived from the same prices) may add noise rather than signal.

### 4.3 No Temporal Information in ML Model
The Random Forest is a point-in-time classifier — it does not capture temporal structure (momentum, autocorrelation, volatility clustering). An LSTM or temporal convolutional model would be significantly more appropriate for sequential financial data.

---

## 5. Evaluation Limitations

### 5.1 No Statistical Significance Testing
The system reports a single Sharpe ratio and return comparison. There is no:
- Bootstrap confidence interval on Sharpe ratio
- t-test on outperformance vs baseline
- Permutation testing
- Monte Carlo simulation of different market paths

Without statistical testing, outperformance might be luck.

### 5.2 No Transaction Cost Sensitivity Analysis
There is no analysis of how performance degrades as transaction costs increase — important because the agent may make excessive buy-sell cycles that only appear profitable under the optimistic 0.1% assumption.

### 5.3 Risk-Free Rate Hardcoded to Zero
In the current high-rate environment (2023–2025: Fed Funds Rate 5%+), comparing to a 0% risk-free rate dramatically overstates the attractiveness of equity returns. The Sharpe ratio computed here is not a valid comparison to a real investor who could earn 5% in T-bills.

### 5.4 No Benchmark Beyond Simple Buy-and-Hold
The only comparison baseline is single-stock buy-and-hold. There is no comparison to:
- S&P 500 passive index
- 60/40 portfolio
- Momentum/mean-reversion simple rule-based strategies
- A random agent (lower bound baseline)

---

## 6. Operational Limitations

### 6.1 No Live Trading Capability
The system is purely offline/historical. There is no:
- Real-time data feed integration
- Order management system (OMS) connection
- Paper trading simulation
- Brokerage API integration (Alpaca, Interactive Brokers)

### 6.2 No Experiment Tracking
There is no integration with MLflow, Weights & Biases, or any experiment tracker. This means:
- Hyperparameter changes cannot be compared reliably
- Training runs produce no persistent records
- Reproducibility depends solely on saved model files

### 6.3 No Automated Pipeline
The 8-step manual shell command sequence has no automation. Any step failure requires manual diagnosis and restart from scratch (or manual intermediate restart, which itself can cause the non-idempotent issues described in bugs).

### 6.4 Not Scalable to Multiple Stocks or Strategies
The entire codebase is architected around a single stock and a single RL algorithm. Extending to multiple assets or multiple strategies would require significant refactoring — the environment, reward function, and agent are not parametrically designed for flexibility.
