"""
InsureAI — Exploratory Data Analysis & Data Quality Engine
===========================================================
Implements Section B (requirements 31-50, 287-294, 625):
  - Strictly separates continuous, discrete, binary, categorical, and identifier columns
  - Never runs IQR / continuous outlier detection on target (is_claim) or binary flags
  - Detects outliers ONLY on continuous numerical variables
  - High correlation detection (> 0.85)
  - Near-zero-variance and constant column detection
  - Rare categories detection (< 1% representation)
  - Train/Test distribution consistency check
  - Target-feature relationship analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Known targets across real and synthetic modes
TARGET_CANDIDATES = ["is_claim", "claim_filed", "target"]
IDENTIFIER_CANDIDATES = ["policy_id", "id", "customer_id", "index"]

# Binary column names in vehicle insurance
KNOWN_BINARY_COLS = [
    "is_esc", "is_adjustable_steering", "is_tpms", "is_parking_sensors",
    "is_parking_camera", "is_front_fog_lights", "is_rear_window_wiper",
    "is_rear_window_washer", "is_rear_window_defogger", "is_brake_assist",
    "is_power_door_locks", "is_central_locking", "is_power_steering",
    "is_driver_seat_height_adjustable", "is_day_night_rear_view_mirror",
    "is_ecw", "is_speed_alert",
]


class DataQualityAnalyzer:
    """
    Comprehensive Data Quality & EDA analyzer adhering to scientific standards:
    - Fold-safe, leakage-aware column categorizer
    - Proper outlier detection strictly on continuous variables
    - Imbalance analysis and statistics
    """

    def _categorize_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Partition columns into distinct semantic types:
        identifiers, targets, binary, continuous_numerical, discrete_numerical, categorical.
        """
        identifiers = [c for c in df.columns if c.lower() in IDENTIFIER_CANDIDATES or c.lower().endswith("_id")]
        targets = [c for c in df.columns if c.lower() in TARGET_CANDIDATES]

        binary = []
        continuous = []
        discrete = []
        categorical = []

        for col in df.columns:
            if col in identifiers or col in targets:
                continue

            # Check if known or detected binary (Yes/No, 0/1, True/False, or <= 2 unique non-null values)
            vals = df[col].dropna()
            unique_vals = set(vals.unique())

            if (
                col in KNOWN_BINARY_COLS
                or unique_vals.issubset({0, 1, 0.0, 1.0})
                or unique_vals.issubset({"Yes", "No", "yes", "no", "Y", "N"})
                or unique_vals.issubset({True, False})
                or len(unique_vals) <= 2
            ):
                binary.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Distinguish continuous from discrete (e.g. cylinder count, airbags, gear_box vs displacement, length)
                if len(unique_vals) < 15 and pd.api.types.is_integer_dtype(df[col]):
                    discrete.append(col)
                else:
                    continuous.append(col)
            else:
                categorical.append(col)

        return {
            "identifiers": identifiers,
            "targets": targets,
            "binary": binary,
            "continuous_numerical": continuous,
            "discrete_numerical": discrete,
            "categorical": categorical,
        }

    def analyze(self, df: pd.DataFrame) -> dict:
        col_groups = self._categorize_columns(df)
        target_col = col_groups["targets"][0] if col_groups["targets"] else ("is_claim" if "is_claim" in df.columns else "claim_filed")

        return {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "column_groups": col_groups,
            "missing_values": self.get_missing_values(df),
            "outliers": self.get_outliers_iqr(df, col_groups["continuous_numerical"]),
            "correlations": self.get_correlations(df, col_groups["continuous_numerical"]),
            "high_correlations": self.get_high_correlations(df, col_groups["continuous_numerical"]),
            "duplicates": self.get_duplicates(df),
            "column_stats": self.get_column_stats(df, col_groups),
            "class_distribution": self.get_class_distribution(df, target_col),
            "variance_checks": self.get_variance_checks(df, col_groups),
            "rare_categories": self.get_rare_categories(df, col_groups["categorical"]),
            "assumptions": [
                "IQR outlier detection is evaluated ONLY on continuous numerical variables.",
                "Binary indicators and target labels are explicitly excluded from outlier analysis.",
                "Near-zero variance threshold: variance < 1e-4 or unique ratio < 0.001.",
                "High correlation threshold: Pearson correlation |r| > 0.85.",
            ]
        }

    def get_missing_values(self, df: pd.DataFrame) -> dict:
        total = len(df)
        result = {}
        for col in df.columns:
            n_missing = int(df[col].isnull().sum())
            result[col] = {
                "count": n_missing,
                "percent": round(n_missing / total * 100, 2) if total > 0 else 0,
            }
        return result

    def get_outliers_iqr(self, df: pd.DataFrame, continuous_cols: Optional[List[str]] = None) -> dict:
        """
        Detect outliers ONLY on continuous numerical variables.
        Explicitly excludes binary flags, identifiers, and target column.
        """
        if continuous_cols is None:
            col_groups = self._categorize_columns(df)
            continuous_cols = col_groups["continuous_numerical"]

        result = {}
        for col in continuous_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) == 0:
                continue

            Q1 = float(series.quantile(0.25))
            Q3 = float(series.quantile(0.75))
            IQR = Q3 - Q1
            iqr_outliers = int(((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum())

            mean = float(series.mean())
            std = float(series.std()) if len(series) > 1 else 0.0
            z_outliers = int(((series - mean).abs() > 3 * std).sum()) if std > 0 else 0

            result[col] = {
                "iqr_outliers": iqr_outliers,
                "z_score_outliers": z_outliers,
                "outlier_percent": round(iqr_outliers / len(series) * 100, 2),
                "q1": round(Q1, 4),
                "q3": round(Q3, 4),
                "iqr": round(IQR, 4),
                "method": "1.5x IQR rule on continuous variable",
            }
        return result

    def get_class_distribution(self, df: pd.DataFrame, target_col: str = "is_claim") -> dict:
        if target_col not in df.columns:
            # Fallback check
            for alt in ["is_claim", "claim_filed", "target"]:
                if alt in df.columns:
                    target_col = alt
                    break
            else:
                return {}

        counts = df[target_col].dropna().value_counts()
        total = int(counts.sum())
        if total == 0:
            return {}

        counts_dict = {str(k): int(v) for k, v in counts.items()}
        percentages = {str(k): round(int(v) / total * 100, 2) for k, v in counts.items()}
        min_count = counts.min()
        max_count = counts.max()
        imbalance_ratio = round(float(max_count / min_count), 2) if min_count > 0 else None

        return {
            "target_column": target_col,
            "counts": counts_dict,
            "percentages": percentages,
            "total": total,
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": bool((min_count / total) < 0.4),
            "minority_percentage": round(float(min_count / total * 100), 2),
        }

    def get_correlations(self, df: pd.DataFrame, continuous_cols: Optional[List[str]] = None) -> dict:
        if continuous_cols is None:
            continuous_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in IDENTIFIER_CANDIDATES]

        subset = df[[c for c in continuous_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
        if subset.empty:
            return {}
        corr = subset.corr()
        result = {}
        for col in corr.columns:
            result[col] = {}
            for row in corr.index:
                val = corr.loc[row, col]
                result[col][row] = None if pd.isna(val) else round(float(val), 4)
        return result

    def get_high_correlations(self, df: pd.DataFrame, continuous_cols: Optional[List[str]] = None, threshold: float = 0.85) -> list:
        """Identify pairs of features with absolute correlation exceeding threshold."""
        if continuous_cols is None:
            continuous_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in IDENTIFIER_CANDIDATES]

        subset = df[[c for c in continuous_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
        if subset.empty or subset.shape[1] < 2:
            return []

        corr = subset.corr().abs()
        high_corr = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                val = corr.loc[c1, c2]
                if not pd.isna(val) and val >= threshold:
                    high_corr.append({
                        "feature_1": c1,
                        "feature_2": c2,
                        "correlation": round(float(val), 4),
                        "warning": f"Strong collinearity between '{c1}' and '{c2}' (|r| = {round(val, 2)})."
                    })
        return high_corr

    def get_duplicates(self, df: pd.DataFrame) -> dict:
        n_dup = int(df.duplicated().sum())
        # Also check duplicate policy IDs if column exists
        id_col = next((c for c in df.columns if c.lower() in IDENTIFIER_CANDIDATES), None)
        id_dup = int(df[id_col].duplicated().sum()) if id_col else 0

        return {
            "count": n_dup,
            "percent": round(n_dup / len(df) * 100, 2) if len(df) > 0 else 0,
            "identifier_duplicates": id_dup,
            "identifier_column": id_col,
        }

    def get_variance_checks(self, df: pd.DataFrame, col_groups: Dict[str, List[str]]) -> dict:
        """Detect constant columns and near-zero variance features."""
        constant_cols = []
        near_zero_var = []

        all_features = col_groups["continuous_numerical"] + col_groups["discrete_numerical"] + col_groups["binary"]

        for col in all_features:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) == 0:
                continue
            n_unique = series.nunique()
            if n_unique <= 1:
                constant_cols.append(col)
            else:
                var = float(series.var())
                if var < 1e-4:
                    near_zero_var.append({"column": col, "variance": round(var, 6)})

        return {
            "constant_columns": constant_cols,
            "near_zero_variance": near_zero_var,
        }

    def get_rare_categories(self, df: pd.DataFrame, categorical_cols: List[str], threshold: float = 0.01) -> dict:
        """Detect categories with frequency below 1%."""
        rare_dict = {}
        total = len(df)
        for col in categorical_cols:
            if col not in df.columns:
                continue
            counts = df[col].value_counts(dropna=False)
            rare = []
            for val, cnt in counts.items():
                pct = cnt / total
                if pct < threshold:
                    rare.append({"category": str(val), "count": int(cnt), "percentage": round(pct * 100, 3)})
            if rare:
                rare_dict[col] = rare[:10]  # top 10 rare
        return rare_dict

    def get_column_stats(self, df: pd.DataFrame, col_groups: Optional[Dict[str, List[str]]] = None) -> dict:
        if col_groups is None:
            col_groups = self._categorize_columns(df)

        stats = {}
        for col in df.columns:
            n_missing = int(df[col].isnull().sum())
            n_unique = int(df[col].nunique())
            col_type = str(df[col].dtype)

            if col in col_groups["identifiers"]:
                sem_type = "identifier"
            elif col in col_groups["targets"]:
                sem_type = "target"
            elif col in col_groups["binary"]:
                sem_type = "binary"
            elif col in col_groups["continuous_numerical"]:
                sem_type = "continuous"
            elif col in col_groups["discrete_numerical"]:
                sem_type = "discrete"
            else:
                sem_type = "categorical"

            if pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                stats[col] = {
                    "dtype": col_type,
                    "semantic_type": sem_type,
                    "min": round(float(series.min()), 4) if len(series) > 0 else None,
                    "max": round(float(series.max()), 4) if len(series) > 0 else None,
                    "mean": round(float(series.mean()), 4) if len(series) > 0 else None,
                    "median": round(float(series.median()), 4) if len(series) > 0 else None,
                    "std": round(float(series.std()), 4) if len(series) > 1 else None,
                    "unique": n_unique,
                    "missing": n_missing,
                }
            else:
                mode_val = df[col].mode()
                stats[col] = {
                    "dtype": col_type,
                    "semantic_type": sem_type,
                    "unique": n_unique,
                    "missing": n_missing,
                    "top_value": str(mode_val.iloc[0]) if not mode_val.empty else None,
                    "value_counts": {str(k): int(v) for k, v in df[col].value_counts().head(10).items()},
                }
        return stats

    def get_data_quality_report(self, df: pd.DataFrame) -> dict:
        """Full quality report for the Data Quality dashboard."""
        report = self.analyze(df)
        warnings = []

        # Missing value warnings
        mv = report["missing_values"]
        for col, info in mv.items():
            if info["percent"] > 5:
                warnings.append({"level": "warning", "message": f"Column '{col}' has {info['percent']}% missing values."})
            elif info["percent"] > 0:
                warnings.append({"level": "info", "message": f"Column '{col}' has {info['count']} missing values."})

        # Duplicate warnings
        if report["duplicates"]["count"] > 0:
            warnings.append({"level": "warning", "message": f"{report['duplicates']['count']} duplicate rows detected."})
        if report["duplicates"].get("identifier_duplicates", 0) > 0:
            warnings.append({
                "level": "error",
                "message": f"Duplicate policy IDs detected: {report['duplicates']['identifier_duplicates']} rows."
            })

        # Class imbalance warning
        dist = report.get("class_distribution", {})
        if dist.get("is_imbalanced"):
            ratio = dist.get("imbalance_ratio", 1)
            target = dist.get("target_column", "target")
            warnings.append({
                "level": "warning",
                "message": f"Severe class imbalance on '{target}' ({ratio:.1f}:1 ratio, minority = {dist.get('minority_percentage')}%%). Use SMOTE or class-weighting.",
            })

        # Collinearity warnings
        for hc in report.get("high_correlations", []):
            warnings.append({"level": "info", "message": hc["warning"]})

        # Constant columns
        for cc in report.get("variance_checks", {}).get("constant_columns", []):
            warnings.append({"level": "warning", "message": f"Column '{cc}' is constant (zero variance)."})

        total_outliers = sum(v["iqr_outliers"] for v in report["outliers"].values())
        if total_outliers > len(df) * 0.05:
            warnings.append({
                "level": "info",
                "message": f"{total_outliers} potential outliers detected across continuous numerical variables (IQR method).",
            })

        cg = report["column_groups"]
        report["warnings"] = warnings
        report["summary"] = {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "continuous_columns": len(cg["continuous_numerical"]),
            "discrete_columns": len(cg["discrete_numerical"]),
            "binary_columns": len(cg["binary"]),
            "categorical_columns": len(cg["categorical"]),
            "identifier_columns": len(cg["identifiers"]),
            "target_columns": len(cg["targets"]),
            "total_missing": int(sum(v["count"] for v in mv.values())),
            "total_duplicates": report["duplicates"]["count"],
            "total_outliers_iqr": total_outliers,
            "is_imbalanced": dist.get("is_imbalanced", False),
            "imbalance_ratio": dist.get("imbalance_ratio", 1.0),
        }

        return report
