"""
data_splitter.py

Splits feature data into train and test sets based on date.
CRITICAL: Prevents overfitting by ensuring the agent never sees test data during training.
"""

import pandas as pd
import yaml
import os


def split_data(config_path: str = "config/config.yaml"):
    """
    Split features data into training and testing sets.
    
    Train: Date < split_date
    Test: Date >= split_date
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_cfg = config["data"]
    
    features_path = data_cfg.get("ml_features_data_path", data_cfg["features_data_path"])
    train_path = data_cfg["train_data_path"]
    test_path = data_cfg["test_data_path"]
    split_date = data_cfg["split_date"]
    
    print(f"Loading features from: {features_path}")
    df = pd.read_csv(features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    split_dt = pd.to_datetime(split_date)
    
    # Split data
    train_df = df[df["Date"] < split_dt].copy()
    test_df = df[df["Date"] >= split_dt].copy()
    
    # Reset indices for clean environment stepping
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    # Save splits
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("=" * 60)
    print("DATA SPLIT COMPLETE")
    print("=" * 60)
    print(f"Split date: {split_date}")
    print(f"Training set: {len(train_df)} rows ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"Testing set:  {len(test_df)} rows ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    print(f"Train saved to: {train_path}")
    print(f"Test saved to:  {test_path}")
    
    return train_df, test_df


if __name__ == "__main__":
    split_data()
