"""
ml_signals.py

Generates ML-based prediction signals for the RL trading agent.

CRITICAL DESIGN:
- Model is trained ONLY on training data (before split_date)
- Predictions are made for ALL data, but model never sees test labels during training
- This prevents data leakage and ensures valid out-of-sample testing
"""

import pandas as pd
import numpy as np
import yaml
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def generate_ml_signals(config_path="config/config.yaml"):
    """
    Generate ML prediction signals and add them to the feature files.
    """
    print("=" * 60)
    print("GENERATING ML SIGNALS")
    print("=" * 60)
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    split_date = config["data"]["split_date"]
    features_path = config["data"]["features_data_path"]
    ml_features_path = config["data"]["ml_features_data_path"]
    
    # Load full feature data
    df = pd.read_csv(features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    print(f"Total samples: {len(df)}")
    print(f"Split date: {split_date}")
    
    # ============ CREATE TARGET VARIABLE ============
    # Target: Will next day's return be positive? (1 = yes, 0 = no)
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
    
    # ============ SPLIT DATA ============
    train_df = df[df["Date"] < split_date].copy()
    test_df = df[df["Date"] >= split_date].copy()
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Features for ML model (exclude Date, Target, and Return to avoid leakage)
    feature_cols = [col for col in df.columns 
                    if col not in ["Date", "Target", "Return"]]
    
    # Drop last row of train (no target for it)
    train_df = train_df.iloc[:-1]
    
    # ============ TRAIN ML MODEL ============
    X_train = train_df[feature_cols].values
    y_train = train_df["Target"].values
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    train_accuracy = model.score(X_train_scaled, y_train)
    print(f"Train accuracy: {train_accuracy:.4f}")
    
    # ============ GENERATE PREDICTIONS FOR ALL DATA ============
    # Drop the last row (no target) for consistent indexing
    df_for_pred = df.iloc[:-1].copy()
    
    X_all = df_for_pred[feature_cols].values
    X_all_scaled = scaler.transform(X_all)
    
    # Get probability of positive return
    ml_probs = model.predict_proba(X_all_scaled)[:, 1]
    
    # Add ML_Signal to dataframe
    df_for_pred["ML_Signal"] = ml_probs
    
    # ============ SAVE UPDATED FEATURES ============
    # Remove target column (it was only for training)
    df_for_pred.drop(columns=["Target"], inplace=True)
    
    # Save to ML features file
    df_for_pred.to_csv(ml_features_path, index=False)
    print(f"Updated features saved to {ml_features_path}")
    
    # ============ SAVE MODEL AND SCALER ============
    os.makedirs("models", exist_ok=True)
    
    with open("models/ml_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open("models/ml_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print("ML model saved to models/ml_model.pkl")
    print("ML scaler saved to models/ml_scaler.pkl")
    
    # ============ FEATURE IMPORTANCE ============
    print("\nTop 5 Feature Importances:")
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    for i, row in importances.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    generate_ml_signals()
