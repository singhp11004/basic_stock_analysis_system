import pandas as pd
import os


def clean_stock_data(input_path, output_path):
    print("Loading raw data...")
    df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    df.dropna(inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Cleaned data saved to {output_path}")
    print(f"Rows after cleaning: {len(df)}")


if __name__ == "__main__":
    INPUT_FILE = "data/raw/stock_prices.csv"
    OUTPUT_FILE = "data/processed/cleaned_data.csv"

    clean_stock_data(INPUT_FILE, OUTPUT_FILE)
