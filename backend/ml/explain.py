"""
InsureAI — Explainability Engine (SHAP + Global Feature Importance)
=====================================================================
Implements Section J (requirements 156-173, 623):
  - TreeExplainer for tree models (Random Forest, Decision Tree, Gradient Boosting, XGBoost)
  - KernelExplainer / LinearExplainer fallback for non-tree models
  - Precomputes and caches global feature importance (No empty global SHAP!)
  - Local waterfall explanations: top risk factors (positive SHAP) vs protective factors (negative SHAP)
  - Safe fallback mechanisms with graceful degradation
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ExplainabilityEngine:
    """
    Production-grade Explainability Engine integrating SHAP and feature attribution.
    Provides local prediction explanations (waterfall) and global feature importance.
    """

    def __init__(self):
        self.explainer = None
        self.feature_names: List[str] = []
        self.is_tree: bool = False
        self.background_data = None
        self._global_importance_cache: List[Dict[str, Any]] = []
        self._base_model = None

    def setup(self, pipeline: Any, X_train_raw: pd.DataFrame, feature_names: List[str]) -> None:
        """
        Initialize the explainer and precompute global importance.
        Supports both dict pipeline format and imblearn/sklearn Pipeline objects.
        """
        self.feature_names = list(feature_names) if feature_names else []

        # Extract preprocessor and classifier
        if isinstance(pipeline, dict):
            preprocessor = pipeline.get("preprocessor")
            model = pipeline.get("base_estimator") or pipeline.get("model")
        elif hasattr(pipeline, "named_steps"):
            preprocessor = pipeline.named_steps.get("preprocessor")
            model = pipeline.named_steps.get("classifier")
        else:
            preprocessor = pipeline[:-1] if hasattr(pipeline, "__getitem__") else None
            model = pipeline[-1] if hasattr(pipeline, "__getitem__") else pipeline

        # Unpack CalibratedClassifierCV or Pipeline if needed
        base_model = model
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            cal_est = model.calibrated_classifiers_[0].estimator
            if hasattr(cal_est, "named_steps") and "classifier" in cal_est.named_steps:
                base_model = cal_est.named_steps["classifier"]
                if preprocessor is None:
                    preprocessor = cal_est.named_steps.get("preprocessor")
            else:
                base_model = cal_est
        elif hasattr(model, "named_steps") and "classifier" in model.named_steps:
            base_model = model.named_steps["classifier"]
            if preprocessor is None:
                preprocessor = model.named_steps.get("preprocessor")
        elif hasattr(model, "estimator") and not hasattr(model, "estimators_"):
            base_model = model.estimator

        self._base_model = base_model
        self.preprocessor = preprocessor

        tree_types = ("RandomForest", "DecisionTree", "GradientBoosting", "XGB", "ExtraTrees")
        model_class_str = type(base_model).__name__
        self.is_tree = any(t in model_class_str for t in tree_types)

        if not HAS_SHAP:
            logger.warning("SHAP library not installed. Using model feature importance fallback.")
            self._global_importance_cache = self.get_global_model_importance(base_model, n_features=50)
            return

        try:
            sample_df = X_train_raw.head(100) if isinstance(X_train_raw, pd.DataFrame) else None
            if preprocessor is not None and sample_df is not None:
                X_sample_proc = preprocessor.transform(sample_df)
            else:
                X_sample_proc = None

            if self.is_tree:
                self.explainer = shap.TreeExplainer(base_model)
                logger.info("Initialized shap.TreeExplainer for %s", model_class_str)
            elif X_sample_proc is not None:
                self.background_data = X_sample_proc[:50]
                if hasattr(base_model, "predict_proba"):
                    self.explainer = shap.KernelExplainer(base_model.predict_proba, self.background_data)
                else:
                    self.explainer = shap.KernelExplainer(base_model.predict, self.background_data)
                logger.info("Initialized shap.KernelExplainer with background sample")

            # Precompute global importance using a background sample
            if X_sample_proc is not None and self.explainer is not None:
                self._compute_shap_global_importance(X_sample_proc[:80])
            else:
                self._global_importance_cache = self.get_global_model_importance(n_features=50)

        except Exception as e:
            logger.warning("SHAP initialization failed (%s). Falling back to feature importance.", e)
            self._global_importance_cache = self.get_global_model_importance(n_features=50)

    def _compute_shap_global_importance(self, X_proc: np.ndarray) -> None:
        """Compute mean absolute SHAP values across sample records."""
        try:
            sv = self.explainer.shap_values(X_proc)
            if isinstance(sv, list):
                # Binary classification: use positive class (index 1)
                sv_pos = sv[1] if len(sv) > 1 else sv[0]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv_pos = sv[:, :, 1]
            else:
                sv_pos = sv

            mean_abs = np.abs(sv_pos).mean(axis=0)
            mean_sign = sv_pos.mean(axis=0)

            pairs = []
            for i, feat in enumerate(self.feature_names):
                if i < len(mean_abs):
                    pairs.append({
                        "feature": feat,
                        "importance": float(round(mean_abs[i], 5)),
                        "direction": "increase" if mean_sign[i] >= 0 else "decrease",
                    })

            pairs.sort(key=lambda x: x["importance"], reverse=True)
            self._global_importance_cache = pairs
            logger.info("Computed global SHAP importance for %d features", len(pairs))
        except Exception as e:
            logger.warning("Global SHAP computation failed (%s), using model weights.", e)
            self._global_importance_cache = self.get_global_model_importance(n_features=50)

    def get_global_model_importance(
        self,
        base_model: Any = None,
        feature_names: Optional[List[str]] = None,
        n_features: int = 15,
    ) -> List[Dict[str, Any]]:
        """Extract Tree Mean Gini / Mean Decrease in Impurity (MDI) or linear weights."""
        model = base_model if base_model is not None else self._base_model
        names = feature_names if feature_names is not None else self.feature_names
        if model is None:
            return []
        pairs = []
        if hasattr(model, "feature_importances_") and names:
            importances = model.feature_importances_
            for feat, imp in zip(names, importances):
                pairs.append({
                    "feature": feat,
                    "importance": float(round(imp, 5)),
                    "direction": "increase",
                    "type": "gini_mdi",
                })
        elif hasattr(model, "coef_") and names:
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            for feat, c in zip(names, coefs):
                pairs.append({
                    "feature": feat,
                    "importance": float(round(abs(c), 5)),
                    "direction": "increase" if c >= 0 else "decrease",
                    "type": "coefficient",
                })

        pairs.sort(key=lambda x: x["importance"], reverse=True)
        return pairs[:n_features]

    def get_global_mdi(self, n_features: int = 15) -> List[Dict[str, Any]]:
        """Extract Tree Gini / Mean Decrease in Impurity (MDI)."""
        return self.get_global_model_importance(n_features=n_features)

    def get_global_shap_importance(self, n_features: int = 15) -> List[Dict[str, Any]]:
        """
        Return global mean absolute SHAP values across representative samples.
        Checks active champion release directory first, then in-memory cache,
        with fallback to model weights.
        """
        try:
            import os
            import json
            from .registry import ModelRegistry
            reg = ModelRegistry()
            champ_ver = reg.get_current_champion_version()
            if champ_ver:
                shap_path = os.path.join(reg.releases_dir, champ_ver, "global_shap_importance.json")
                if os.path.exists(shap_path):
                    with open(shap_path, "r", encoding="utf-8") as f:
                        items = json.load(f)
                        if items:
                            return items[:n_features]
        except Exception as e:
            logger.debug("Could not read release global_shap_importance.json: %s", e)

        if self._global_importance_cache:
            return self._global_importance_cache[:n_features]
        return self.get_global_model_importance(n_features=n_features)

    def get_global_importance(self, n_features: int = 15) -> List[Dict[str, Any]]:
        """
        Global importance method. Returns SHAP importance if available,
        otherwise falls back to model Gini/MDI weights.
        """
        return self.get_global_shap_importance(n_features=n_features)

    def explain_local_shap(self, input_dict: dict, preprocessor: Any = None) -> dict:
        """Decoupled local waterfall SHAP explanation for a single policyholder."""
        return self.explain_single(input_dict, preprocessor)

    def explain_single(self, input_dict: dict, preprocessor: Any = None) -> dict:
        """
        Explain a single policyholder prediction.
        Returns waterfall chart data: shap_values, base_value, top_risk_factors, protective_factors.
        Returns { 'available': False, 'reason': ... } if SHAP is unavailable.
        """
        if self.explainer is None:
            return {
                "available": False,
                "reason": "SHAP explainer is not initialized or unavailable.",
                "shap_values": {},
                "base_value": 0.0,
                "top_risk_factors": [],
                "protective_factors": [],
            }

        if preprocessor is None:
            preprocessor = self.preprocessor

        if preprocessor is None:
            if isinstance(input_dict, dict) and all(k in input_dict for k in self.feature_names):
                X_proc = np.array([[input_dict[k] for k in self.feature_names]])
            else:
                return {
                    "available": False,
                    "reason": "Preprocessor is required for local SHAP attribution.",
                    "shap_values": {},
                    "base_value": 0.0,
                    "top_risk_factors": [],
                    "protective_factors": [],
                }
        else:
            df = pd.DataFrame([input_dict])
            try:
                X_proc = preprocessor.transform(df)
            except Exception as e:
                return {
                    "available": False,
                    "reason": f"Input preprocessing failed: {e}",
                    "shap_values": {},
                    "base_value": 0.0,
                    "top_risk_factors": [],
                    "protective_factors": [],
                }

        try:
            if self.is_tree:
                shap_vals = self.explainer.shap_values(X_proc)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    shap_vals = shap_vals[0, :, 1]
                else:
                    shap_vals = shap_vals[0]

                expected = self.explainer.expected_value
                if isinstance(expected, (list, np.ndarray)):
                    expected = expected[1] if len(expected) > 1 else expected[0]
            else:
                shap_vals = self.explainer.shap_values(X_proc)
                if isinstance(shap_vals, list) and len(shap_vals) > 1:
                    shap_vals = shap_vals[1][0]
                else:
                    shap_vals = shap_vals[0]
                expected = getattr(self.explainer, "expected_value", None)
                if isinstance(expected, (list, np.ndarray)):
                    expected = expected[1] if len(expected) > 1 else expected[0]
                if expected is None:
                    expected = 0.064

            shap_dict = {}
            factors = []
            for i, feat in enumerate(self.feature_names):
                if i < len(shap_vals):
                    val = float(shap_vals[i])
                    shap_dict[feat] = round(val, 5)
                    factors.append({
                        "feature": feat,
                        "contribution": float(round(abs(val), 5)),
                        "value": float(round(val, 5)),
                        "direction": "increase" if val > 0 else "decrease",
                    })

            factors.sort(key=lambda x: x["contribution"], reverse=True)

            return {
                "available": True,
                "shap_values": shap_dict,
                "base_value": float(round(float(expected), 4)),
                "top_risk_factors": [f for f in factors if f["direction"] == "increase"][:6],
                "protective_factors": [f for f in factors if f["direction"] == "decrease"][:6],
                "total_factors_evaluated": len(factors),
                "attribution_note": "Local SHAP attribution for the underlying base estimator. The final displayed claim probability is produced by the cross-fitted calibrated model.",
            }

        except Exception as e:
            logger.warning("Error computing SHAP values: %s", e)
            return {
                "available": False,
                "reason": f"SHAP calculation error: {e}",
            }

    def get_shap_values(self, X_processed: np.ndarray) -> np.ndarray:
        """Compute raw SHAP values for a batch of processed features."""
        if self.explainer is not None and X_processed is not None and len(X_processed) > 0:
            try:
                sv = self.explainer.shap_values(X_processed)
                if isinstance(sv, list) and len(sv) > 1:
                    return np.array(sv[1])
                return np.array(sv)
            except Exception as e:
                logger.error("get_shap_values failed: %s", e)
        return np.array([])
