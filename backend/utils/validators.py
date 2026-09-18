import pandas as pd

def validate_csv_schema(df: pd.DataFrame, required_columns: list) -> list:
    errors = []
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    return errors

def validate_categorical_value(value: str, allowed: list) -> bool:
    return value in allowed

def validate_numeric_range(value: float, min_val: float, max_val: float) -> bool:
    return min_val <= value <= max_val

def detect_csv_issues(df: pd.DataFrame) -> list:
    warnings = []
    if df.isnull().values.any():
        warnings.append("Dataset contains missing values.")
    return warnings

def sanitize_input(data: dict) -> dict:
    return {k: v for k, v in data.items() if v is not None}
