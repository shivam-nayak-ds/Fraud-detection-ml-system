# src/data_preprocessing.py

import pandas as pd
import numpy as np
import pickle
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import (
    DATA_PATH, PROCESSED_DIR, RANDOM_STATE,
    MISSING_THRESHOLD, HIGH_CARDINALITY_THREESHOLD, TEST_SIZE
)


# ── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data():
   
    transaction = pd.read_csv("data/raw/train_transaction.csv")
    identity    = pd.read_csv("data/raw/train_identity.csv")
    df = transaction.merge(identity, on='TransactionID', how='left')
    print(f"Data loaded: {df.shape}")
    return df


# ── 2. Clean Data ─────────────────────────────────────────────────────────────
def clean_data(df):
    
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {before - len(df)}")

    if 'TransactionAmt' in df.columns:
        df = df[df['TransactionAmt'] > 0]
        upper_limit = df['TransactionAmt'].quantile(0.99)
        df['TransactionAmt'] = df['TransactionAmt'].clip(upper=upper_limit)
        print(f"TransactionAmt capped at {upper_limit:.2f}")

    return df


# ── 3. Drop Missing Columns ───────────────────────────────────────────────────
def drop_missing_columns(df):
    """50% se zyada missing columns drop karta hai"""
    threshold = MISSING_THRESHOLD * len(df)
    df = df.dropna(thresh=threshold, axis=1)
    print(f"After dropping missing cols: {df.shape}")
    return df


# ── 4. Fill Missing Values ────────────────────────────────────────────────────
def fill_missing_values(df):
    """Numerical → median, Categorical → mode"""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    print("Missing values filled")
    return df


# ── 5. Encode Columns ─────────────────────────────────────────────────────────
def encode_columns(df):
    """
    Low cardinality  → Label Encoding
    High cardinality → Frequency Encoding
    """
    cat_cols = df.select_dtypes(include='object').columns

    for col in cat_cols:
        if df[col].nunique() <= HIGH_CARDINALITY_THREESHOLD:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            freq = df[col].value_counts() / len(df)
            df[col] = df[col].map(freq)

    print("Encoding done")
    return df


# ── 6. Scale Features ─────────────────────────────────────────────────────────
def scale_features(df, target_col='isFraud'):
    
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    scaler_path = os.path.join(PROCESSED_DIR, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    return df, scaler


# ── 7. Train/Test Split ───────────────────────────────────────────────────────
def split_data(df, target_col='isFraud'):
    """80/20 split"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── 8. Apply SMOTE ────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - X: {X_resampled.shape} | Fraud cases: {y_resampled.sum()}")
    return X_resampled, y_resampled


# ── 9. Save Processed Data ────────────────────────────────────────────────────
def save_processed(X_train, X_test, y_train, y_test):
   
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open(os.path.join(PROCESSED_DIR, "X_train.pkl"), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(PROCESSED_DIR, "X_test.pkl"), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(PROCESSED_DIR, "y_train.pkl"), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(PROCESSED_DIR, "y_test.pkl"), 'wb') as f:
        pickle.dump(y_test, f)

    print("All processed files saved!")


# ── 10. Main Function ─────────────────────────────────────────────────────────
def preprocess():
    
    df = load_data()
    df = clean_data(df)
    df = drop_missing_columns(df)
    df = fill_missing_values(df)
    df = encode_columns(df)
    df, scaler = scale_features(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, y_train = apply_smote(X_train, y_train)

    save_processed(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    preprocess()
