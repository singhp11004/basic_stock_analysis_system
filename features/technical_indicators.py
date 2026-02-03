import pandas as pd
import numpy as np
import os


def compute_features(input_path, output_path):
    print("Computing technical indicators...")

    df = pd.read_csv(input_path)

    df["Return"] = df["Close"].pct_change()

    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Volatility"] = df["Return"].rolling(20).std()

    df.dropna(inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Features saved to {output_path}")
    print(f"Final rows: {len(df)}")


if __name__ == "__main__":
    INPUT_FILE = "data/processed/cleaned_data.csv"
    OUTPUT_FILE = "data/features/features.csv"

    compute_features(INPUT_FILE, OUTPUT_FILE)
