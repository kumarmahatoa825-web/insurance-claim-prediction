from pydantic_settings import BaseSettings

class MLConfig(BaseSettings):
    NUMBER_OF_ROWS: int = 5000
    RANDOM_SEED: int = 42
    CLAIM_RATE: float = 0.30
    NOISE_LEVEL: float = 0.15
    MODEL_SELECTION_METRIC: str = 'f1'
    DEFAULT_THRESHOLD: float = 0.5
    LOW_RISK_MAX: int = 25
    MODERATE_RISK_MAX: int = 50
    HIGH_RISK_MAX: int = 75
    FP_COST: float = 500.0
    FN_COST: float = 5000.0

config = MLConfig()
