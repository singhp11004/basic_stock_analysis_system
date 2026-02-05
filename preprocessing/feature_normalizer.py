"""
feature_normalizer.py

Normalizes features for improved RL generalization.
CRITICAL: Fit ONLY on training data to prevent data leakage.
"""

import pandas as pd
import numpy as np
import yaml
import os
import pickle


class FeatureNormalizer:
    """
    Z-score normalizer that fits on training data only.
    
    Why normalize?
    - Raw prices (e.g., $40 vs $230) confuse the agent
    - Normalized features let agent learn relative patterns
    - Improves generalization to unseen data
    """
    
    def __init__(self):
        self.means = None
        self.stds = None
        self.feature_cols = None
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, exclude_cols: list = None):
        """
        Compute mean and std from training data.
        
        Args:
            df: Training dataframe
            exclude_cols: Columns to exclude from normalization (e.g., Date)
        """
        exclude_cols = exclude_cols or ["Date"]
        
        # Get numeric feature columns
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Compute statistics
        self.means = df[self.feature_cols].mean()
        self.stds = df[self.feature_cols].std()
        
        # Replace zero std with 1 to avoid division by zero
        self.stds = self.stds.replace(0, 1)
        
        self.is_fitted = True
        print(f"Normalizer fitted on {len(self.feature_cols)} features")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization using stored statistics.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        df_norm = df.copy()
        
        for col in self.feature_cols:
            if col in df_norm.columns:
                df_norm[col] = (df_norm[col] - self.means[col]) / self.stds[col]
        
        return df_norm
    
    def fit_transform(self, df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
        """Fit on data and transform it."""
        self.fit(df, exclude_cols)
        return self.transform(df)
    
    def save(self, filepath: str):
        """Save normalizer parameters."""
        with open(filepath, "wb") as f:
            pickle.dump({
                "means": self.means,
                "stds": self.stds,
                "feature_cols": self.feature_cols
            }, f)
        print(f"Normalizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load normalizer parameters."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.means = data["means"]
            self.stds = data["stds"]
            self.feature_cols = data["feature_cols"]
            self.is_fitted = True
        print(f"Normalizer loaded from {filepath}")


def normalize_train_test(config_path: str = "config/config.yaml"):
    """
    Normalize train and test data, fitting only on train.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_cfg = config["data"]
    
    train_path = data_cfg["train_data_path"]
    test_path = data_cfg["test_data_path"]
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Create and fit normalizer on TRAIN ONLY
    normalizer = FeatureNormalizer()
    train_norm = normalizer.fit_transform(train_df, exclude_cols=["Date"])
    
    # Transform test using train statistics
    test_norm = normalizer.transform(test_df)
    
    # Save normalized data
    train_norm_path = train_path.replace(".csv", "_normalized.csv")
    test_norm_path = test_path.replace(".csv", "_normalized.csv")
    
    train_norm.to_csv(train_norm_path, index=False)
    test_norm.to_csv(test_norm_path, index=False)
    
    # Save normalizer for inference
    normalizer.save("models/feature_normalizer.pkl")
    
    print("=" * 60)
    print("NORMALIZATION COMPLETE")
    print("=" * 60)
    print(f"Train normalized: {train_norm_path}")
    print(f"Test normalized:  {test_norm_path}")
    
    # Show sample statistics
    print("\nSample normalized train stats:")
    print(train_norm[normalizer.feature_cols].describe().loc[["mean", "std"]])
    
    return normalizer


if __name__ == "__main__":
    normalize_train_test()
