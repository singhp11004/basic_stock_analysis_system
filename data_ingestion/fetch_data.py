import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_stock_data(symbol, start_date, end_date, output_path):
    print(f"Fetching {symbol} data...")

    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=False
        )
    except Exception as e:
        raise RuntimeError(f"yfinance download failed: {e}")

    # Handle Yahoo failure (VERY IMPORTANT)
    if df is None or df.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    df.reset_index(inplace=True)

    # Flatten MultiIndex columns safely
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Hard validation (NON-NEGOTIABLE)
    required_cols = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved data to {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    SYMBOL = "AAPL"
    START_DATE = "2018-01-01"

    # 🔒 Avoid 'today' edge case (use yesterday)
    END_DATE = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    OUTPUT_FILE = "data/raw/stock_prices.csv"

    fetch_stock_data(SYMBOL, START_DATE, END_DATE, OUTPUT_FILE)
