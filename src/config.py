# ══════════════════════════════════════════════════════════════════════════════
# Configuration Management
# Reads settings from .env file using Pydantic v2
# ══════════════════════════════════════════════════════════════════════════════

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Load all configuration from .env file
    Pydantic v2 with settings support
    """
    
    # ── Database ────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///fraud_detection.db"
    DATABASE_ECHO: bool = False
    
    # ── API ─────────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = True
    
    # ── Security ────────────────────────────────────────────────────────────
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    API_KEY_HEADER: str = "X-API-Key"
    
    # ── MLflow ──────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "fraud_detection"
    MLFLOW_BACKEND_STORE_URI: str = "file:///mlruns"
    
    # ── Logging ─────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/fraud_detection.log"
    LOG_FORMAT: str = "json"
    
    # ── Model ───────────────────────────────────────────────────────────────
    MODEL_VERSION: str = "1"
    MODEL_PATH: str = "models/"
    SCALER_PATH: str = "models/scaler.pkl"
    FEATURE_COLUMNS_PATH: str = "models/feature_columns.pkl"
    
    # ── Data ────────────────────────────────────────────────────────────────
    RAW_DATA_DIR: str = "data/raw/ieee-fraud-detection/"
    PROCESSED_DATA_DIR: str = "data/processed/"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # ── AWS ─────────────────────────────────────────────────────────────────
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET_NAME: str = "fraud-detection-models"
    
    # ── Monitoring ──────────────────────────────────────────────────────────
    MONITORING_ENABLED: bool = True
    DRIFT_THRESHOLD: float = 0.05
    RETRAINING_FREQUENCY_DAYS: int = 7
    
    # ── Environment ─────────────────────────────────────────────────────────
    ENVIRONMENT: str = "development"  # development, staging, production
    
    # Modern Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # Ignore extra env variables
    )


# ── Global settings object ─────────────────────────────────────────────────
settings = Settings()


# ── Helper functions ───────────────────────────────────────────────────────
def get_settings() -> Settings:
    """
    Get settings object
    Usage: from src.config import get_settings
    """
    return settings


def is_production() -> bool:
    """Check if running in production environment"""
    return settings.ENVIRONMENT == "production"


def is_development() -> bool:
    """Check if running in development environment"""
    return settings.ENVIRONMENT == "development"


def is_staging() -> bool:
    """Check if running in staging environment"""
    return settings.ENVIRONMENT == "staging"