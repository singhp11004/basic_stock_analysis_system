"""
technical_indicators.py

Computes technical indicators for the RL trading agent.

CRITICAL DESIGN RULE:
- ALL price-based features use ADJ CLOSE
- This guarantees consistency with portfolio valuation and rewards
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

    # Drop rows with NaNs from rolling windows
    df.dropna(inplace=True)

    # CRITICAL: Remove Close column to prevent state leakage
    # Agent must only see Adj Close-based features
    if "Close" in df.columns:
        df.drop(columns=["Close"], inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Features saved to {output_path}")
    print(f"Final rows: {len(df)}")


if __name__ == "__main__":
    INPUT_FILE = "data/processed/cleaned_data.csv"
    OUTPUT_FILE = "data/features/features.csv"

    compute_features(INPUT_FILE, OUTPUT_FILE)
