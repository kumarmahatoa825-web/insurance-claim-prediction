"""
InsureAI — Model Fairness & Bias Analysis Engine
==================================================
Implements Section S (requirements 295-307):
  - Demographic parity / disparate impact across sensitive vehicle & policy attributes
  - Group precision, recall, false positive rate (FPR), false negative rate (FNR)
  - Reference group documentation
  - Small group warnings (< 30 observations)
  - Clear responsible AI disclaimer: analytical purposes only
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, confusion_matrix

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Historical insurance data may reflect systemic socio-economic and demographic differences. "
    "These fairness evaluations provide algorithmic auditing transparency and must not be used "
    "for autonomous underwriting discrimination."
)


class FairnessAnalyzer:
    """Evaluates disparate impact, error rate parity, and demographic metrics across groups."""

    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        df_eval: pd.DataFrame,
        sensitive_cols: Optional[List[str]] = None,
    ) -> dict:
        df_copy = df_eval.copy()

        # Adapt sensitive columns to real vehicle dataset vs synthetic dataset
        if sensitive_cols is None:
            if "fuel_type" in df_copy.columns:
                sensitive_cols = ["fuel_type", "segment", "transmission_type"]
                if "area_cluster" in df_copy.columns:
                    sensitive_cols.append("area_cluster")
                if "age_of_policyholder" in df_copy.columns:
                    df_copy["age_group"] = pd.cut(
                        df_copy["age_of_policyholder"],
                        bins=[0, 0.35, 0.55, 2.0],
                        labels=["Younger (<0.35)", "Middle (0.35-0.55)", "Senior (>0.55)"],
                    ).astype(str)
                    sensitive_cols.append("age_group")
            else:
                sensitive_cols = ["gender", "occupation"]
                if "age" in df_copy.columns:
                    df_copy["age_group"] = self.create_age_groups(df_copy["age"])
                    sensitive_cols.append("age_group")

        group_metrics: Dict[str, Any] = {}
        sample_warnings: List[str] = []

        for col in sensitive_cols:
            if col in df_copy.columns:
                metrics, warns = self.compute_group_metrics(y_true, y_pred, y_prob, df_copy[col])
                group_metrics[col] = metrics
                sample_warnings.extend(warns)

        disparate_impact = self.compute_disparate_impact(group_metrics)

        return {
            "group_metrics": group_metrics,
            "disparate_impact": disparate_impact,
            "sensitive_attributes_evaluated": list(group_metrics.keys()),
            "small_sample_warnings": sample_warnings,
            "reference_group_rule": "The group with the highest positive prediction rate is used as the reference (denominator).",
            "disclaimer": DISCLAIMER,
        }

    def compute_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        groups: pd.Series,
        min_sample_warning: int = 30,
    ) -> Tuple[Dict[str, Any], List[str]]:
        group_metrics = {}
        warnings = []
        unique_groups = groups.dropna().unique()

        y_t_arr = np.array(y_true)
        y_p_arr = np.array(y_pred)
        y_pr_arr = np.array(y_prob)

        for g in unique_groups:
            mask = (groups == g).values
            count = int(mask.sum())
            if count == 0:
                continue

            if count < min_sample_warning:
                warnings.append(
                    f"Group '{groups.name}={g}' has only {count} observations (< {min_sample_warning}). "
                    "Metrics may have high variance."
                )

            y_t = y_t_arr[mask]
            y_p = y_p_arr[mask]
            y_pr = y_pr_arr[mask]

            cm = confusion_matrix(y_t, y_p, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            ppr = float((tp + fp) / count) if count > 0 else 0.0
            avg_prob = float(np.mean(y_pr)) if count > 0 else 0.0

            group_metrics[str(g)] = {
                "count": count,
                "precision": round(float(precision_score(y_t, y_p, zero_division=0)), 4),
                "recall": round(float(recall_score(y_t, y_p, zero_division=0)), 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
                "positive_prediction_rate": round(ppr, 4),
                "avg_predicted_prob": round(avg_prob, 4),
                "actual_claim_rate": round(float(np.mean(y_t)), 4),
                "has_sufficient_sample": count >= min_sample_warning,
            }

        return group_metrics, warnings

    def create_age_groups(self, ages: pd.Series) -> pd.Series:
        bins = [0, 30, 45, 60, 150]
        labels = ["18-30", "31-45", "46-60", "60+"]
        return pd.cut(ages, bins=bins, labels=labels, right=True).astype(str)

    def compute_disparate_impact(self, group_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute Disparate Impact Ratio = (Selection Rate for Group) / (Selection Rate for Reference Group).
        Reference group is the group with the highest selection rate.
        Ratio < 0.80 conventionally flags potential disparate impact under the 4/5ths rule.
        """
        di: Dict[str, Any] = {}
        for feature, metrics in group_metrics.items():
            rates = {k: v["positive_prediction_rate"] for k, v in metrics.items()}
            if not rates:
                continue

            ref_group = max(rates, key=rates.get)
            max_rate = rates[ref_group]

            ratios = {}
            for group, rate in rates.items():
                ratio = round(rate / max_rate, 3) if max_rate > 0 else 1.0
                ratios[group] = {
                    "disparate_impact_ratio": ratio,
                    "is_under_four_fifths_rule": ratio < 0.80,
                }

            di[feature] = {
                "reference_group": ref_group,
                "reference_selection_rate": max_rate,
                "ratios": ratios,
            }

        return di
