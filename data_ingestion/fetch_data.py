import os
import yfinance as yf
import pandas as pd
from datetime import datetime


def fetch_stock_data(symbol, start_date, end_date, output_path):
    print(f"Fetching {symbol} data...")

    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        progress=False
    )

    if df.empty:
        raise ValueError("No data returned. Check symbol or dates.")

    df.reset_index(inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved data to {output_path}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    SYMBOL = "AAPL"
    START_DATE = "2018-01-01"
    END_DATE = datetime.today().strftime("%Y-%m-%d")

    OUTPUT_FILE = "data/raw/stock_prices.csv"

    fetch_stock_data(SYMBOL, START_DATE, END_DATE, OUTPUT_FILE)
