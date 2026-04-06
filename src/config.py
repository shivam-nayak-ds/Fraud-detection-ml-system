import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Updated to nested folder: ieee-fraud-detection
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "ieee-fraud-detection") 

# Database configuration
DATABASE_URL = "sqlite:///data/fraud_detection.db"
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Data files
FILES_TO_INGEST = [
    "train_transaction.csv", 
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv"
]
