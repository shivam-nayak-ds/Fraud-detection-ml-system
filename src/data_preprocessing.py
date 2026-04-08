import polars as pl
import os
import numpy as np
from src.config import PROCESSED_DATA_DIR
from src.logger import logger
from src.utils import save_data, load_data

def reduce_memory_usage(df: pl.DataFrame) -> pl.DataFrame:
    """
    Optimizes memory usage by downcasting numeric types.
    Supports both Integer and Float types.
    """
    initial_mem = df.estimated_size() / (1024**2)
    
    for col in df.columns:
        dtype = df[col].dtype
        
        # Downcast Integers
        if dtype in [pl.Int64, pl.Int32]:
            c_min, c_max = df[col].min(), df[col].max()
            if c_min is not None and c_max is not None:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df = df.with_columns(pl.col(col).cast(pl.Int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df = df.with_columns(pl.col(col).cast(pl.Int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df = df.with_columns(pl.col(col).cast(pl.Int32))
        
        # Downcast Floats
        elif dtype == pl.Float64:
            c_min, c_max = df[col].min(), df[col].max()
            if c_min is not None and c_max is not None:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_columns(pl.col(col).cast(pl.Float32))
                    
    final_mem = df.estimated_size() / (1024**2)
    logger.info(f"Memory: {initial_mem:.1f}MB -> {final_mem:.1f}MB (Reduction: {100*(initial_mem-final_mem)/initial_mem:.1f}%)")
    return df

def handle_missing_data(df: pl.DataFrame, threshold: float = 0.7) -> pl.DataFrame:
    """Drops columns exceeding the missing values threshold."""
    total = len(df)
    null_counts = df.null_count().to_dicts()[0]
    cols_to_drop = [col for col, count in null_counts.items() if (count / total) > threshold]
    logger.info(f"Dropping {len(cols_to_drop)} high-null columns.")
    return df.drop(cols_to_drop)

def clean_categorical_features(df: pl.DataFrame) -> pl.DataFrame:
    """Standardizes string features (lowercase, stripping, null filling)."""
    cat_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
    for col in cat_cols:
        df = df.with_columns(
            pl.col(col).str.strip_chars().str.to_lowercase().fill_null("unknown")
        )
    return df

def preprocess_pipeline(df_trans: pl.DataFrame, df_id: pl.DataFrame = None) -> pl.DataFrame:
    """Main Orchestrator for the preprocessing pipeline."""
    try:
        logger.info("Initializing Data Preprocessing Pipeline...")
        
        # Merge if identity data exists
        if df_id is not None:
            df = df_trans.join(df_id, on="TransactionID", how="left")
            logger.info("Joined Transaction and Identity datasets.")
        else:
            df = df_trans

        # Sequence of operations
        df = reduce_memory_usage(df)
        df = handle_missing_data(df)
        df = clean_categorical_features(df)
        
        logger.info(f"Pipeline complete. Output shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Preprocessing Pipeline Failed: {str(e)}")
        raise

if __name__ == "__main__":
    raw_input = os.path.join(PROCESSED_DATA_DIR, "train_extracted_part1.parquet")
    processed_output = os.path.join(PROCESSED_DATA_DIR, "train_preprocessed.parquet")
    
    if os.path.exists(raw_input):
        df = load_data(raw_input)
        df_final = preprocess_pipeline(df)
        save_data(df_final, processed_output)
    else:
        logger.error(f"Input file not found at {raw_input}. Run data_fetch first.")
