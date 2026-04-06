import pandas as pd
from sqlalchemy import create_engine
import os
import sys

# Import settings from our config and logger
from src.config import DATABASE_URL, RAW_DATA_DIR, FILES_TO_INGEST
from src.logger import logger

def run_db_ingestion():
    """
    Industry standard data ingestion script for IEEE-CIS dataset.
    Reads large CSV files in chunks and loads them into a SQL database.
    """
    logger.info("Starting professional database ingestion process...")
    
    # Engine setup
    engine = create_engine(DATABASE_URL)
    
    for filename in FILES_TO_INGEST:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            logger.warning(f"SKIP ERROR: {filename} not found at {RAW_DATA_DIR}. Please download from Kaggle.")
            continue
            
        # Table name (e.g., train_transaction)
        table_name = filename.replace(".csv", "")
        logger.info(f"Processing '{filename}' mapping to table '{table_name}'...")
        
        try:
            # Batch Processing to prevent OOM (Out of Memory)
            chunk_size = 10000
            for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
                mode = 'replace' if i == 0 else 'append'
                chunk.to_sql(table_name, engine, if_exists=mode, index=False)
                
                if (i + 1) % 10 == 0:  # Log every 100k records
                    logger.info(f"--- Ingested { (i+1) * chunk_size } records so far for {filename}...")
            
            logger.info(f"SUCCESS: {filename} successfully loaded into database.")
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR loading {filename}: {e}")
            sys.exit(1)

    logger.info("--- Data Ingestion Process Fully Completed! ---")

if __name__ == "__main__":
    run_db_ingestion()
