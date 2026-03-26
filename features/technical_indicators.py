"""
technical_indicators.py

Computes technical indicators for the RL trading agent.

CRITICAL DESIGN RULE:
- ALL price-based features use ADJ CLOSE
- This guarantees consistency with portfolio valuation and rewards

FEATURES:
- Basic: Return, SMA, RSI, Volatility
- Market Timing: MACD_Signal, BB_Position, Momentum, Trend
- Sentiment Proxy: Vol_Regime (high/low volatility state)
"""

import pandas as pd
import numpy as np
import os


def compute_features(input_path, output_path):
    print("Computing technical indicators...")

    df = pd.read_csv(input_path)

    # -------------------------------
    # USE ADJ CLOSE EVERYWHERE
    # -------------------------------
    price = df["Adj Close"]

    # ============ BASIC FEATURES ============
    
    # Returns
    df["Return"] = price.pct_change()

    # Moving averages
    df["SMA_10"] = price.rolling(10).mean()
    df["SMA_30"] = price.rolling(30).mean()

    # RSI
    delta = price.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Volatility (of adjusted returns)
    df["Volatility"] = df["Return"].rolling(20).std()

    # ============ MARKET TIMING FEATURES ============
    
    # MACD Signal: +1 when MACD > Signal, -1 when MACD < Signal
    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Signal"] = np.where(macd > signal_line, 1, -1)
    
    # Bollinger Band Position: 0 = at lower band, 1 = at upper band
    bb_middle = price.rolling(20).mean()
    bb_std = price.rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    df["BB_Position"] = (price - bb_lower) / (bb_upper - bb_lower)
    df["BB_Position"] = df["BB_Position"].clip(0, 1)  # Clamp to 0-1
    
    # Momentum: Rate of change over 10 days
    df["Momentum"] = price.pct_change(10)
    
    # Trend: SMA ratio (>1 = uptrend, <1 = downtrend)
    df["Trend"] = df["SMA_10"] / df["SMA_30"]
    
    # ============ SENTIMENT PROXY (VOLATILITY REGIME) ============
    
    # Vol_Regime: 1 = high volatility (fear), 0 = low volatility (greed)
    avg_volatility = df["Volatility"].rolling(60).mean()
    df["Vol_Regime"] = np.where(df["Volatility"] > avg_volatility, 1, 0)

    # ============ CLEANUP ============
    
    # Drop rows with NaNs from rolling windows
    df.dropna(inplace=True)

    # We KEEP raw price columns here because TradingEnv needs them for portfolio valuation.
    # They will be explicitly dropped from the state representation in TradingEnv._get_state()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Features saved to {output_path}")
    print(f"Final rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    INPUT_FILE = "data/processed/cleaned_data.csv"
    OUTPUT_FILE = "data/features/features.csv"

    compute_features(INPUT_FILE, OUTPUT_FILE)
