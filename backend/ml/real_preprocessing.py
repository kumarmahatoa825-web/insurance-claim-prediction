"""
Real-data preprocessing pipeline for the Vehicle Insurance Claims dataset.

FOLD-SAFE DESIGN: This module provides two classes:
  1. VehicleFeatureEngineer — stateless sklearn transformer (fit is a no-op).
     Parses max_torque/max_power strings, maps Yes/No binary cols, engineers features.
  2. RealInsurancePreprocessor — sklearn BaseEstimator+TransformerMixin that wraps
     VehicleFeatureEngineer + ColumnTransformer.
     - fit(X) → fits ColumnTransformer on training data ONLY
     - transform(X) → applies fitted ColumnTransformer
     - fit_transform(X) → fit + transform in one step
     Compatible with imblearn.Pipeline for fold-safe cross-validation.
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


# ── Column groups ─────────────────────────────────────────────────────────────

# Columns to drop (ID, or post-hoc leakage risk)
DROP_COLS = ["policy_id", "is_claim"]

# Free-text numeric columns that embed units (e.g., "60Nm@3500rpm")
STRING_NUMERIC_COLS = ["max_torque", "max_power"]

# Binary Yes/No categorical columns → mapped to 0/1 (stateless)
BINARY_COLS = [
    "is_esc", "is_adjustable_steering", "is_tpms", "is_parking_sensors",
    "is_parking_camera", "is_front_fog_lights", "is_rear_window_wiper",
    "is_rear_window_washer", "is_rear_window_defogger", "is_brake_assist",
    "is_power_door_locks", "is_central_locking", "is_power_steering",
    "is_driver_seat_height_adjustable", "is_day_night_rear_view_mirror",
    "is_ecw", "is_speed_alert",
]

# Low-cardinality categorical columns (one-hot encoded)
LOW_CARD_CAT_COLS = [
    "area_cluster", "segment", "fuel_type", "rear_brakes_type",
    "transmission_type", "steering_type", "engine_type",
    "make", "model",
]

# High-cardinality categorical columns (none - model is M1..M11, one-hot encoded)
HIGH_CARD_CAT_COLS = []

# Continuous numeric base columns (after string-numeric parsing)
BASE_NUMERIC_COLS = [
    "policy_tenure", "age_of_car", "age_of_policyholder",
    "population_density", "airbags", "displacement",
    "cylinder", "gear_box", "turning_radius", "length", "width",
    "height", "gross_weight", "ncap_rating",
]

# Engineered feature names (added by VehicleFeatureEngineer)
ENGINEERED_NUMERIC_COLS = [
    "torque_value", "power_value", "torque_rpm", "power_rpm",
    "safety_score", "safety_missing_count", "age_tenure_ratio", "car_size",
    "power_to_weight", "torque_to_weight",
]

TARGET_COL = "is_claim"


# ── String parser & safe math helpers ─────────────────────────────────────────

def _parse_value_at_rpm(series: pd.Series):
    """Extract numeric value and RPM from strings like '60Nm@3500rpm' or '80bhp@5500rpm'."""
    value = series.str.extract(r"([0-9]+(?:\.[0-9]+)?)(?:Nm|bhp)", expand=False).astype(float)
    rpm = series.str.extract(r"@([0-9]+(?:\.[0-9]+)?)rpm", expand=False).astype(float)
    return value, rpm


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """
    Mathematically safe division returning NaN when denominator <= 0 or missing.
    Prevents ratio explosion when denominator approaches zero.
    """
    num_vals = num.astype(float).values
    den_vals = den.astype(float).values
    out = np.full_like(num_vals, np.nan, dtype=float)
    valid_mask = (~np.isnan(num_vals)) & (~np.isnan(den_vals)) & (den_vals > 0)
    np.divide(num_vals, den_vals, out=out, where=valid_mask)
    return pd.Series(out, index=num.index)


# ── Feature Engineering Transformer ──────────────────────────────────────────

class VehicleFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stateless sklearn-compatible transformer.
    All operations are deterministic on the raw data — no learned state.
    Inherits BaseEstimator + TransformerMixin so it can be used in sklearn Pipelines.

    Operations:
      - Drop policy_id (identifier, not a feature)
      - Parse max_torque / max_power string → numeric values + RPM
      - Map binary Yes/No → 0/1
      - Engineer: safety_score, age_tenure_ratio, car_size, power_to_weight, torque_to_weight
    """

    def fit(self, X, y=None):
        """No-op: this transformer is stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all stateless feature engineering."""
        X = X.copy()

        # Drop ID columns if present
        X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])

        # ── Parse max_torque and max_power ──────────────────────────────────
        for col in STRING_NUMERIC_COLS:
            if col in X.columns:
                if col == "max_torque":
                    X["torque_value"], X["torque_rpm"] = _parse_value_at_rpm(X[col].astype(str))
                elif col == "max_power":
                    X["power_value"], X["power_rpm"] = _parse_value_at_rpm(X[col].astype(str))
                X.drop(columns=[col], inplace=True)

        # ── Binary Yes/No → 1.0/0.0 (Preserve NaNs for SimpleImputer) ──────
        for col in BINARY_COLS:
            if col in X.columns:
                X[col] = X[col].map({"Yes": 1.0, "No": 0.0})

        # Ensure nominal categoricals are treated as strings
        if "make" in X.columns:
            X["make"] = X["make"].astype(str)
        if "model" in X.columns:
            X["model"] = X["model"].astype(str)

        # ── Feature engineering ─────────────────────────────────────────────
        safety_cols = [
            "is_esc", "is_tpms", "is_parking_sensors", "is_parking_camera",
            "is_brake_assist", "is_ecw", "is_speed_alert",
        ]
        present_safety = [c for c in safety_cols if c in X.columns]
        if present_safety:
            # Semantic fix: preserve NaNs, sum with min_count=1, and track missing count explicitly
            X["safety_score"] = X[present_safety].sum(axis=1, min_count=1)
            X["safety_missing_count"] = X[present_safety].isna().sum(axis=1).astype(float)

        if "age_of_policyholder" in X.columns and "policy_tenure" in X.columns:
            X["age_tenure_ratio"] = safe_div(X["age_of_policyholder"], X["policy_tenure"])

        if all(c in X.columns for c in ["length", "width", "height"]):
            X["car_size"] = (X["length"].astype(float) * X["width"].astype(float) * X["height"].astype(float)) / 1e8

        if "power_value" in X.columns and "gross_weight" in X.columns:
            X["power_to_weight"] = safe_div(X["power_value"], X["gross_weight"])

        if "torque_value" in X.columns and "gross_weight" in X.columns:
            X["torque_to_weight"] = safe_div(X["torque_value"], X["gross_weight"])

        return X


# ── Full Preprocessor (sklearn-compatible) ────────────────────────────────────

class RealInsurancePreprocessor(BaseEstimator, TransformerMixin):
    """
    Full sklearn-compatible preprocessing pipeline for the real vehicle insurance dataset.

    Inherits BaseEstimator + TransformerMixin:
      - fit(X, y=None)       → fits the ColumnTransformer on training data ONLY
      - transform(X)         → applies fitted ColumnTransformer to any data
      - fit_transform(X, y)  → sklearn default: fit then transform

    Usage in imblearn.Pipeline (FOLD-SAFE):
        from imblearn.pipeline import Pipeline as ImbPipeline
        pipe = ImbPipeline([
            ('preprocessor', RealInsurancePreprocessor()),
            ('smote', SMOTE()),
            ('classifier', RandomForestClassifier()),
        ])
        pipe.fit(X_train_raw, y_train)   # fits preprocessor on ONLY this fold's training data
        pipe.predict_proba(X_val_raw)    # transforms val with training-fold preprocessor (no leakage)
    """

    def __init__(self):
        self.engineer_ = VehicleFeatureEngineer()
        self._column_transformer = None
        self._fitted = False
        self._feature_names: list = []
        self._num_cols: list = []
        self._binary_present: list = []
        self._low_cat_cols: list = []
        self._high_cat_cols: list = []

    def _build_column_transformer(self, X_engineered: pd.DataFrame) -> ColumnTransformer:
        """
        Dynamically build ColumnTransformer based on what columns exist after engineering.
        Called inside fit() to adapt to the transformed column set.
        """
        all_cols = list(X_engineered.columns)

        # Numeric columns: everything not binary, not categorical
        num_cols = [
            c for c in all_cols
            if c not in BINARY_COLS
            and c not in LOW_CARD_CAT_COLS
            and c not in HIGH_CARD_CAT_COLS
            and c not in DROP_COLS
            and c != TARGET_COL
            and X_engineered[c].dtype != object
        ]

        low_cat_present = [c for c in LOW_CARD_CAT_COLS if c in all_cols]
        high_cat_present = [c for c in HIGH_CARD_CAT_COLS if c in all_cols]
        binary_present = [c for c in BINARY_COLS if c in all_cols]

        # Store for feature name computation
        self._num_cols = num_cols
        self._low_cat_cols = low_cat_present
        self._high_cat_cols = high_cat_present
        self._binary_present = binary_present

        # Build sub-pipelines
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        low_cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ])
        high_cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        binary_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ])

        transformers = []
        if num_cols:
            transformers.append(("num", num_pipe, num_cols))
        if binary_present:
            transformers.append(("bin", binary_pipe, binary_present))
        if low_cat_present:
            transformers.append(("cat_low", low_cat_pipe, low_cat_present))
        if high_cat_present:
            transformers.append(("cat_high", high_cat_pipe, high_cat_present))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, X: pd.DataFrame, y=None) -> "RealInsurancePreprocessor":
        """
        Fit the preprocessor on training data only.
        The VehicleFeatureEngineer is stateless; only the ColumnTransformer learns state.

        Args:
            X: Raw training DataFrame (must include all vehicle feature columns).
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            self (fitted transformer)
        """
        X_eng = self.engineer_.transform(X)
        ct = self._build_column_transformer(X_eng)
        ct.fit(X_eng)
        self._column_transformer = ct
        self._fitted = True
        self._feature_names = self._compute_feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted ColumnTransformer.
        Must call fit() first.

        Args:
            X: Raw DataFrame (any split — val/test/inference).

        Returns:
            np.ndarray of shape (n_samples, n_features)
        """
        if not self._fitted:
            raise RuntimeError(
                "RealInsurancePreprocessor is not fitted. "
                "Call fit() or fit_transform() on training data first."
            )
        X_eng = self.engineer_.transform(X)
        return self._column_transformer.transform(X_eng)

    def _compute_feature_names(self) -> list:
        """Compute output feature names from the fitted ColumnTransformer."""
        if hasattr(self._column_transformer, "get_feature_names_out"):
            try:
                raw_names = self._column_transformer.get_feature_names_out()
                clean_names = []
                for n in raw_names:
                    clean = n.split("__", 1)[-1] if "__" in n else n
                    clean_names.append(clean)
                return clean_names
            except Exception:
                pass

        names = list(self._num_cols) + list(self._binary_present)

        if self._low_cat_cols:
            try:
                ohe = self._column_transformer.named_transformers_["cat_low"].named_steps["ohe"]
                names += ohe.get_feature_names_out(self._low_cat_cols).tolist()
            except (KeyError, AttributeError):
                names += self._low_cat_cols

        names += list(self._high_cat_cols)
        return names

    def get_feature_names(self) -> list:
        """Return list of output feature names (after OHE expansion)."""
        return self._feature_names

    # Legacy interface compatibility
    def fit_transform_legacy(self, X_train: pd.DataFrame) -> tuple:
        """
        Legacy interface returning (X_processed, feature_names).
        Kept for backward compatibility with scripts that call fit_transform directly.
        """
        X_proc = self.fit_transform(X_train)
        return X_proc, self._feature_names
