import os
import polars as pl
from src.config import DATABASE_URL, PROCESSED_DATA_DIR
from src.logger import logger
from src.utils import save_data

def fetch_data_phased(part=1):
    """
    Phased Data Fetching for Retraining Simulation:
    Part 1: First 50% of data.
    Part 2: Remaining 50% of data.
    """
    try:
        total_records = 590540  # Approximately
        limit = total_records // 2
        offset = 0 if part == 1 else limit
        
        logger.info(f"Initializing PART-{part} Data Fetch with Polars (Offset: {offset}, Limit: {limit})...")
        
        # SQL Query with LIMIT and OFFSET
        query = f"SELECT * FROM train_transaction LIMIT {limit} OFFSET {offset}"
        
        conn_uri = DATABASE_URL.replace("sqlite:///", "sqlite://")
        df = pl.read_database_uri(query, conn_uri)
        
        logger.info(f"SUCCESS: PART-{part} Dataset ({len(df)} records) loaded!")
        return df
    except Exception as e:
        logger.error(f"FAILURE: Polars phased fetch error: {str(e)}")
        raise e

if __name__ == "__main__":
    # 1. Fetch data from SQL database (Extracted)
    df = fetch_data_phased(part=1)
    
    # 2. Save data to Disk (Stored)
    save_path = os.path.join(PROCESSED_DATA_DIR, "train_extracted_part1.parquet")
    save_data(df, save_path)
    
    print(f"\n--- FAST Data Extraction & Storage Success (PART-1) ---")
    print(f"Data saved to: {save_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns (First 10): {df.columns[:10]} ...")
