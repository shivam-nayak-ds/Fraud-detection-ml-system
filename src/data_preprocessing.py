import pandas as pd
from sqlalchemy import create_engine
import os

# Database Path
DB_URL = 'sqlite:///data/fraud_detection.db'
RAW_DATA = 'data/raw/'

def run_full_ingestion():
    # Database create karein (folder setup)
    if not os.path.exists('data'): os.makedirs('data')
    engine = create_engine(DB_URL)
    
    # Ye 4 files Kaggle se download karke 'data/raw/' mein rakh lo:
    files_to_load = [
        "train_transaction.csv", 
        "train_identity.csv",
        "test_transaction.csv",
        "test_identity.csv"
    ]

    for filename in files_to_load:
        filepath = os.path.join(RAW_DATA, filename)
        
        if not os.path.exists(filepath):
            print(f"FAILED: {filename} nahi mila! Isay download karke data/raw/ mein rakhein.")
            continue
            
        # Table name define karein (transaction_train, identity_train, etc.)
        table_name = filename.replace(".csv", "")
        
        print(f"Starting: {filename} -> Table: {table_name}")
        
        # CHUNK size: 10k rows (Best for slow systems)
        for i, chunk in enumerate(pd.read_csv(filepath, chunksize=10000)):
            mode = 'replace' if i == 0 else 'append'
            chunk.to_sql(table_name, engine, if_exists=mode, index=False)
            print(f"  Processed chunk {i+1} ({ (i+1)*10000 } rows)...")

    print("\n---  SUCCESS: Pura data database mein chala gaya! ---")

if __name__ == "__main__":
    run_full_ingestion()
