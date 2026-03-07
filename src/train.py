# src/train.py

import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.config import (
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH,
    MODEL_PATH,
    PROCESSED_DIR
)
from src.feature_engineering import engineer_features


# ── 1. Load Processed Data ────────────────────────────────────────────────────
def load_processed_data():
    """Pickle se train/test data load karta hai"""
    with open(os.path.join(PROCESSED_DIR, "X_train.pkl"), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "X_test.pkl"), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "y_train.pkl"), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "y_test.pkl"), 'rb') as f:
        y_test = pickle.load(f)

    print(f"Data loaded - X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── 2. Train Model ────────────────────────────────────────────────────────────
def train_model(X_train, y_train):
    """Random Forest train karta hai"""
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    print("Model training done!")
    return model


# ── 3. Save Model ─────────────────────────────────────────────────────────────
def save_model(model):
    """Model ko pickle mein save karta hai"""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")


# ── 4. Main Function ──────────────────────────────────────────────────────────
def train():
    """Pura training pipeline"""

    # Data load karo
    X_train, X_test, y_train, y_test = load_processed_data()

    # Feature engineering
    X_train, X_test, top_features = engineer_features(X_train, X_test, y_train)

    # Train karo
    model = train_model(X_train, y_train)

    # Save karo
    save_model(model)

    return model, X_test, y_test, top_features

