# src/feature_engineering.py

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_STATE, TOP_N_FEATURES, PROCESSED_DIR


# ── 1. Encoding ───────────────────────────────────────────────────────────────
def encode_columns(X):
    """
    High cardinality → Frequency Encoding (DeviceInfo, id_31, P_emaildomain)
    Low cardinality  → Label Encoding (baaki sab object cols)
    """
    high_card_cols = ['DeviceInfo', 'id_31', 'P_emaildomain']

    # Frequency Encoding
    for col in high_card_cols:
        if col in X.columns:
            freq = X[col].value_counts()
            X[col] = X[col].map(freq)

    # Label Encoding
    le = LabelEncoder()
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    print("Encoding done")
    return X


# ── 2. Log Transform ──────────────────────────────────────────────────────────
def log_transform(X):
   
    if 'TransactionAmt' in X.columns:
        X['TransactionAmt_Log']     = np.log1p(X['TransactionAmt'])
        X['TransactionAmt_decimal'] = X['TransactionAmt'] - X['TransactionAmt'].astype(int)
        print("Log transform done")
    return X


# ── 3. Time Features ──────────────────────────────────────────────────────────
def extract_time_features(X):
  
    if 'TransactionDT' in X.columns:
        X['Transaction_hour'] = X['TransactionDT'] % 86400 // 3600
        X['Transaction_day']  = X['TransactionDT'] // 86400
        print("Time features extracted")
    return X


# ── 4. Top 50 Feature Selection ───────────────────────────────────────────────
def select_top_features(X_train, y_train, X_test):
   
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    importances  = pd.Series(rf.feature_importances_, index=X_train.columns)
    top_features = importances.nlargest(TOP_N_FEATURES).index.tolist()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, "top_features.pkl"), 'wb') as f:
        pickle.dump(top_features, f)
    print(f"Top {TOP_N_FEATURES} features saved")

    return X_train[top_features], X_test[top_features], top_features


# ── 5. Main Function ──────────────────────────────────────────────────────────
def engineer_features(X_train, X_test, y_train):
    

    X_train = encode_columns(X_train)
    X_test  = encode_columns(X_test)

    X_train = log_transform(X_train)
    X_test  = log_transform(X_test)

    X_train = extract_time_features(X_train)
    X_test  = extract_time_features(X_test)

    X_train, X_test, top_features = select_top_features(X_train, y_train, X_test)

    return X_train, X_test, top_features


if __name__ == "__main__":
    engineer_features()