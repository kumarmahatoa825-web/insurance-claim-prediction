"""
InsureAI — Core Settings & Centralized Configuration
======================================================
Requirements 411-415, 416-418:
  - Centralized risk thresholds
  - Model and dataset paths
  - Security configurations (CORS, file size limits)
  - Business cost assumptions
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "InsureAI"
    APP_VERSION: str = "4.1.1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    REAL_TRAIN_DATA_PATH: str = "data/real/train.csv"
    REAL_TEST_DATA_PATH: str = "data/real/test.csv"
    SYNTHETIC_DATA_PATH: str = "data/synthetic/insurance_claims.csv"
    PROCESSED_DATA_DIR: str = "data/processed"
    MODELS_DIR: str = "models"
    CURRENT_POINTER_PATH: str = "models/current.json"
    REGISTRY_PATH: str = "models/registry.json"
    REPORTS_DIR: str = "reports"
    DATABASE_URL: str = "sqlite+aiosqlite:///insureai.db"

    # Security & Networking
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
    MAX_UPLOAD_SIZE_MB: int = 15
    MAX_BATCH_ROWS: int = 50000

    # Risk Scoring (Project Scale 0-100)
    LOW_RISK_MAX: int = 25
    MODERATE_RISK_MAX: int = 50
    HIGH_RISK_MAX: int = 75
    # Critical risk is > 75

    # Business Decision Costs (in INR ₹)
    FP_COST: float = 500.0    # Cost of manual review on non-claim
    FN_COST: float = 5000.0   # Cost of missed claim fraud/risk

    # ML & Monitoring Defaults
    RANDOM_SEED: int = 42
    MONITORING_WINDOW_SIZE: int = 500
    SMOTE_K_NEIGHBORS: int = 3

    # Security — Registry API Key (override in .env for production)
    API_KEY: str = "insureai-dev-key-change-in-production"

    # Champion metadata path shortcut (for routes.py convenience)
    CHAMPION_METADATA_PATH: str = ""

    model_config = {
        "env_file": ".env",
        "extra": "allow"
    }


settings = Settings()

