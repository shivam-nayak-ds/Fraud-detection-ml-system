# src/db_ingestion.py
import pandas as pd
from sqlalchemy import create_engine
import os
import sys

# ── Configuration (DATABASE URL) ──────────────────────────────────────────────
# Change this according to your database:
# PostgreSQL: 'postgresql://user:password@localhost:5432/db_name'
# MySQL:      'mysql+mysqlconnector://user:password@localhost:3306/db_name'
# SQLite:     'sqlite:///data/fraud_detection.db'
DATABASE_URL = 'sqlite:///data/fraud_detection.db'

# Files paths
RAW_DATA_PATH = "data/raw"
TRANSACTION_CSV = os.path.join(RAW_DATA_PATH, "train_transaction.csv")
IDENTITY_CSV = os.path.join(RAW_DATA_PATH, "train_identity.csv")

def ingest_data(csv_path, table_name, chunksize=10000):
    """
    Reads a large CSV file in chunks and inserts it into a SQL table.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    print(f"Starting ingestion of {csv_path} into table '{table_name}'...")
    
    # Create database engine
    engine = create_engine(DATABASE_URL)
    
    try:
        # Read and write in chunks to handle large files
        count = 0
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
            # On first chunk, replace table; on subsequent, append
            mode = 'replace' if i == 0 else 'append'
            chunk.to_sql(table_name, engine, if_exists=mode, index=False)
            
            count += len(chunk)
            print(f"Ingested {count} records so far...")
            
        print(f"SUCCESS: Total {count} records loaded into '{table_name}'.")
        
    except Exception as e:
        print(f"FAILED: An error occurred during ingestion: {e}")

if __name__ == "__main__":
    # Ensure data/ folder exists for SQLite
    if not os.path.exists('data'):
        os.makedirs('data')

    print("--- Database Ingestion Process Started ---")
    
    # Load Transaction Data
    ingest_data(TRANSACTION_CSV, "transactions")
    
    # Load Identity Data
    ingest_data(IDENTITY_CSV, "identities")
    
    print("--- Process Completed ---")
