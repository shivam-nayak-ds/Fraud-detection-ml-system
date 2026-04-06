# src/utils.py
# Yahan helper functions likhne hai — save/load pickle, create directories

import os
import polars as pl
from src.logger import logger

def save_data(df, file_path):
    """
    Saves a Polars DataFrame to a parquet file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.write_parquet(file_path)
        logger.info(f"SUCCESS: Data saved to {file_path}")
    except Exception as e:
        logger.error(f"FAILURE: Could not save data to {file_path}: {e}")
        raise e

def load_data(file_path):
    """
    Loads a Polars DataFrame from a parquet file.
    """
    try:
        df = pl.read_parquet(file_path)
        logger.info(f"SUCCESS: Data loaded from {file_path}")
        return df
    except Exception as e:
        logger.error(f"FAILURE: Could not load data from {file_path}: {e}")
        raise e