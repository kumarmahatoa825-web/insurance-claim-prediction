"""
InsureAI — Data Validation Engine
===================================
Implements Sections A, P, X (requirements 9-24, 176-179, 247-258):
  - Schema consistency validation between train and test/inference
  - Leakage checks (ensures target or derived target features are never passed during inference)
  - Strict range checks for vehicle insurance variables
  - Categorical allowed-value verification
  - Missingness and infinity detection
  - Batch error reporting (preserves valid rows, separates failing rows)
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

# Expected features for the real vehicle insurance dataset
REAL_FEATURE_COLUMNS = [
    "policy_tenure", "age_of_car", "age_of_policyholder", "area_cluster",
    "population_density", "make", "segment", "model", "fuel_type",
    "max_torque", "max_power", "engine_type", "airbags", "is_esc",
    "is_adjustable_steering", "is_tpms", "is_parking_sensors",
    "is_parking_camera", "rear_brakes_type", "displacement", "cylinder",
    "transmission_type", "gear_box", "steering_type", "turning_radius",
    "length", "width", "height", "gross_weight", "is_front_fog_lights",
    "is_rear_window_wiper", "is_rear_window_washer", "is_rear_window_defogger",
    "is_brake_assist", "is_power_door_locks", "is_central_locking",
    "is_power_steering", "is_driver_seat_height_adjustable",
    "is_day_night_rear_view_mirror", "is_ecw", "is_speed_alert", "ncap_rating"
]

SYNTHETIC_FEATURE_COLUMNS = [
    "age", "gender", "income", "occupation", "location", "policy_type",
    "policy_tenure", "premium", "coverage_amount", "vehicle_age",
    "num_dependents", "previous_claims", "payment_history", "renewal_count", "health_score"
]

# Critical physical bounds: Violations of these are strictly rejected as errors (physical impossibilities)
CRITICAL_PHYSICAL_BOUNDS = {
    # Real dataset features (strictly non-negative)
    "policy_tenure": (0.0, None),
    "age_of_car": (0.0, None),
    "age_of_policyholder": (0.0, None),
    "population_density": (0.0, None),
    "airbags": (0, None),
    "displacement": (0, None),
    "cylinder": (1, None),
    "gear_box": (1, None),
    "turning_radius": (0.0, None),
    "length": (0, None),
    "width": (0, None),
    "height": (0, None),
    "gross_weight": (0, None),
    "ncap_rating": (0, 5),
    # Synthetic dataset features
    "age": (18, 120),
    "income": (0, None),
    "premium": (0, None),
    "coverage_amount": (0, None),
    "vehicle_age": (0, None),
    "num_dependents": (0, None),
    "previous_claims": (0, None),
    "payment_history": (0.0, 1.0),
    "renewal_count": (0, None),
    "health_score": (0.0, 1.0),
}

# Realistic bounds for real-world vehicle insurance features
NUMERIC_RANGES = {
    "policy_tenure": (0.0, 3.0),         # normalized tenure in dataset
    "age_of_car": (0.0, 1.0),            # normalized car age
    "age_of_policyholder": (0.0, 1.0),   # normalized policyholder age
    "population_density": (1, 100000),   # people / sq km
    "airbags": (0, 12),
    "displacement": (500, 5000),         # cc
    "cylinder": (2, 8),
    "gear_box": (3, 8),
    "turning_radius": (3.5, 8.0),        # meters
    "length": (2500, 6000),              # mm
    "width": (1200, 2500),               # mm
    "height": (1200, 2500),              # mm
    "gross_weight": (500, 4000),         # kg
    "ncap_rating": (0, 5),
    # Synthetic dataset ranges
    "age": (18, 100),
    "income": (0, 10000000),
    "premium": (0, 1000000),
    "coverage_amount": (0, 50000000),
    "vehicle_age": (0, 40),
    "num_dependents": (0, 15),
    "previous_claims": (0, 30),
    "payment_history": (0.0, 1.0),
    "renewal_count": (0, 50),
    "health_score": (0.0, 1.0),
}

ALLOWED_CATEGORICALS = {
    "fuel_type": {"Petrol", "Diesel", "CNG"},
    "transmission_type": {"Manual", "Automatic"},
    "steering_type": {"Power", "Manual", "Electric"},
    "rear_brakes_type": {"Drum", "Disc"},
    "segment": {"A", "B1", "B2", "C1", "C2", "Utility"},
    "area_cluster": {f"C{i}" for i in range(1, 23)},
    "make": {1, 2, 3, 4, 5, "1", "2", "3", "4", "5"},
    "model": {f"M{i}" for i in range(1, 12)},
    "engine_type": {
        "1.0 SCe", "1.2 L K Series Engine", "1.2 L K12N Dualjet", "1.5 L U2 CRDi",
        "1.5 Turbocharged Revotorq", "1.5 Turbocharged Revotron", "F8D Petrol Engine",
        "G12B", "K Series Dual jet", "K10C", "i-DTEC"
    },
}

TARGET_LEAKAGE_COLUMNS = ["is_claim", "claim_filed", "claim_amount", "claim_date", "claim_status"]


class DataValidator:
    """
    Validates input schemas, bounds, types, and flags potential data leakage.
    """

    @staticmethod
    def validate_prediction_input(
        input_dict: dict,
        data_mode: str = "real",
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validates a single prediction payload before inference.
        Returns: (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # 1. Leakage Check: Ensure target column is NOT in input
        for target in TARGET_LEAKAGE_COLUMNS:
            if target in input_dict and input_dict[target] is not None:
                errors.append(f"Data leakage risk: Target column '{target}' cannot be passed during prediction.")

        # 2. Check for NaN or Inf in numeric fields
        for k, v in input_dict.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                errors.append(f"Invalid value in field '{k}': NaN or Inf detected.")

        # 3. Critical physical constraints check (Errors)
        critical_violated = set()
        for col, (c_min, c_max) in CRITICAL_PHYSICAL_BOUNDS.items():
            if col in input_dict and input_dict[col] is not None:
                try:
                    val = float(input_dict[col])
                    if (c_min is not None and val < c_min) or (c_max is not None and val > c_max):
                        errors.append(
                            f"Critical physical constraint violation: Field '{col}' value {val} is outside allowed limits [{c_min}, {c_max}]."
                        )
                        critical_violated.add(col)
                except (ValueError, TypeError):
                    errors.append(f"Field '{col}' must be a numeric value, got '{input_dict[col]}'.")
                    critical_violated.add(col)

        # 4. Soft numeric range checks (Warnings for unusual but physically possible values)
        for col, (min_val, max_val) in NUMERIC_RANGES.items():
            if col in input_dict and input_dict[col] is not None and col not in critical_violated:
                try:
                    val = float(input_dict[col])
                    if val < min_val or val > max_val:
                        warnings.append(
                            f"Field '{col}' value {val} is outside expected range [{min_val}, {max_val}]."
                        )
                except (ValueError, TypeError):
                    pass

        # 5. Categorical checks
        for col, allowed in ALLOWED_CATEGORICALS.items():
            if col in input_dict and input_dict[col] is not None:
                val = str(input_dict[col])
                if val not in allowed:
                    warnings.append(
                        f"Field '{col}' value '{val}' is outside standard categories: {sorted(list(allowed))}."
                    )

        # 6. Core required fields check
        required = ["policy_tenure", "age_of_policyholder", "make"] if data_mode == "real" else ["age", "policy_type"]
        for req in required:
            if req not in input_dict or input_dict[req] is None:
                errors.append(f"Missing required field: '{req}'.")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    @staticmethod
    def validate_batch_dataframe(
        df: pd.DataFrame,
        data_mode: str = "real",
        max_rows: int = 50000,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Validates an uploaded batch DataFrame.
        Splits into valid_df (ready for prediction) and error_df (with failure reasons).
        Requirement 265: Never crash entire batch because of one bad row.
        """
        summary: Dict[str, Any] = {
            "total_rows": len(df),
            "valid_rows": 0,
            "failed_rows": 0,
            "duplicate_policy_ids": 0,
            "warnings": [],
            "errors": [],
        }

        if len(df) > max_rows:
            summary["errors"].append(f"Batch size {len(df)} exceeds maximum allowed of {max_rows} rows.")
            return pd.DataFrame(), df, summary

        # Duplicate ID check (Fatal in production batch)
        id_col = next((c for c in ["policy_id", "id"] if c in df.columns), None)
        if id_col:
            dups = int(df[id_col].duplicated().sum())
            if dups > 0:
                summary["duplicate_policy_ids"] = dups
                summary["errors"].append(
                    f"Duplicate '{id_col}' values detected ({dups} duplicates). Duplicate policy IDs are not permitted."
                )
                return pd.DataFrame(), df, summary

        # Leakage check: Reject immediately if target column is provided
        clean_df = df.copy()
        for target in TARGET_LEAKAGE_COLUMNS:
            if target in clean_df.columns:
                summary["errors"].append(
                    f"Data leakage risk: target column '{target}' must not be supplied in prediction data."
                )
                return pd.DataFrame(), df, summary

        # Row-level validation
        valid_indices = []
        error_records = []

        required_cols = ["policy_tenure", "age_of_policyholder", "make"] if data_mode == "real" else ["age", "policy_type"]
        missing_req = [c for c in required_cols if c not in clean_df.columns]
        if missing_req:
            summary["errors"].append(f"Missing essential columns for {data_mode} mode: {missing_req}")
            return pd.DataFrame(), df, summary

        for idx, row in clean_df.iterrows():
            row_errors = []
            # Check required fields non-null
            for rc in required_cols:
                if pd.isna(row[rc]):
                    row_errors.append(f"Missing {rc}")

            # Check critical physical bounds
            for col, (c_min, c_max) in CRITICAL_PHYSICAL_BOUNDS.items():
                if col in row and pd.notna(row[col]):
                    try:
                        v = float(row[col])
                        if (c_min is not None and v < c_min) or (c_max is not None and v > c_max):
                            row_errors.append(f"Physical constraint violation in {col} ({v})")
                    except (ValueError, TypeError):
                        row_errors.append(f"Invalid numeric value in {col}")

            if row_errors:
                err_dict = row.to_dict()
                err_dict["_validation_error"] = "; ".join(row_errors)
                error_records.append(err_dict)
            else:
                valid_indices.append(idx)

        valid_df = clean_df.loc[valid_indices].reset_index(drop=True)
        failed_df = pd.DataFrame(error_records)

        summary["valid_rows"] = len(valid_df)
        summary["failed_rows"] = len(failed_df)

        return valid_df, failed_df, summary
