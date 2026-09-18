"""
PredictionEngine — schema-agnostic prediction for both real and synthetic data modes.

The loaded joblib pipeline dict contains:
  {
    'preprocessor': InsurancePreprocessor | RealInsurancePreprocessor,
    'model':        CalibratedClassifierCV,
    'data_mode':    'real' | 'synthetic'   (optional, defaults to 'synthetic')
  }
"""

import os
import json
from typing import Optional, List, Dict, Any
import joblib
import pandas as pd
import numpy as np
from .config import config


# ── Risk profiles & recommendations ──────────────────────────────────────────

RISK_PROFILES = {
    "Low": (
        "This policyholder has a low predicted claim risk. Current indicators "
        "suggest stable policy behavior with no major risk flags."
    ),
    "Moderate": (
        "This policyholder has a moderate predicted claim risk. Some factors "
        "warrant monitoring, but overall risk remains within normal bounds."
    ),
    "High": (
        "This policyholder has a moderately high predicted claim risk. Key "
        "attributes are contributing elevated risk signals."
    ),
    "Critical": (
        "This policyholder has a high predicted claim risk. Multiple significant "
        "risk factors are present. Additional review may be warranted."
    ),
}

RECOMMENDATIONS = {
    "Low": [
        "Vehicle risk profile is within standard underwriting guidelines.",
        "Eligible for standard annual policy renewal terms and no-claim discount retention.",
        "Encourage continued safe vehicle maintenance and active safety feature upkeep.",
    ],
    "Moderate": [
        "Review coverage deductibles and roadside assistance options based on regional cluster density.",
        "Verify vehicle safety equipment status (ESC, TPMS, braking assist).",
        "Monitor policy tenure progression at subsequent annual renewals.",
        "Consider standard vehicle safety feature verification at renewal.",
    ],
    "High": [
        "Perform detailed underwriting inspection on vehicle age and engine displacement profile.",
        "Assess geographic exposure in high-density urban clusters.",
        "Evaluate premium adjustment or enhanced deductible structure given elevated risk indicators.",
        "Recommend driver safety briefing or anti-theft device installation.",
    ],
    "Critical": [
        "Escalate to senior automotive underwriter for manual risk assessment.",
        "Require physical vehicle condition inspection and verification of safety specifications.",
        "Review historical claim activity in the designated area cluster before binding terms.",
        "Document all underwriting exceptions and risk mitigation constraints.",
        "Consider structured deductible or conditional coverage limits.",
    ],
}


class PredictionEngine:
    """
    Loads a saved joblib pipeline dict and provides single + batch prediction.
    Works with both real-data and synthetic-data preprocessors.
    """

    def __init__(self):
        self.pipeline = None       # {'preprocessor': ..., 'model': ..., 'data_mode': ...}
        self.metadata = None
        self._data_mode = "synthetic"

    # ── Loading ────────────────────────────────────────────────────────────────

    def get_active_threshold(self) -> float:
        """
        Retrieve authoritative decision threshold from active model metadata.
        Decision threshold converts probability into Likely/Unlikely binary decision.
        Risk bands convert probability into Low/Moderate/High/Critical project tiers.
        """
        if not self.metadata:
            raise RuntimeError("Model metadata not loaded.")

        thresh = self.metadata.get("default_threshold")
        if thresh is None and isinstance(self.metadata.get("threshold"), dict):
            thresh = self.metadata["threshold"].get("selected")

        if thresh is None:
            raise RuntimeError("Active model threshold is missing from metadata.")

        return float(thresh)

    def load_model(self, path: str = None) -> None:
        from .registry import ModelRegistry
        reg = ModelRegistry()
        if path:
            target_path = path
            active_meta = os.path.join(os.path.dirname(path), "metadata.json")
        else:
            target_path, active_meta = reg.get_active_champion_paths()

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Model file not found: {target_path}")

        self.pipeline = joblib.load(target_path)
        self._data_mode = self.pipeline.get("data_mode", "real") if isinstance(self.pipeline, dict) else "real"

        if not self.metadata:
            if active_meta and os.path.exists(active_meta):
                self.load_metadata(active_meta)

    def load_metadata(self, path: str = "models/model_metadata.json") -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_preprocessor_and_model(self):
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if isinstance(self.pipeline, dict):
            return self.pipeline.get("preprocessor"), self.pipeline["model"]
        if hasattr(self.pipeline, "__getitem__"):
            return self.pipeline[:-1], self.pipeline[-1]
        return None, self.pipeline

    def _predict_prob_array(self, df: pd.DataFrame) -> np.ndarray:
        """Compute probabilities handling both end-to-end pipelines and separate preprocessors."""
        preprocessor, model = self._get_preprocessor_and_model()
        try:
            # Model is end-to-end pipeline or CalibratedClassifierCV wrapping base pipeline
            probs = model.predict_proba(df)[:, 1]
            return probs
        except Exception:
            if preprocessor is not None:
                X_proc = preprocessor.transform(df)
                return model.predict_proba(X_proc)[:, 1]
            raise

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_single(self, input_dict: dict) -> dict:
        """Predict a single record given as a dict of feature values using strictly the authoritative active threshold."""
        threshold = self.get_active_threshold()

        df = pd.DataFrame([input_dict])
        try:
            probs = self._predict_prob_array(df)
            prob = float(probs[0])
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")

        risk_score = self.compute_risk_score(prob)
        risk_category = self.get_risk_category(risk_score)
        label = self.get_prediction_label(prob, threshold)

        model_name = "Unknown"
        model_version = "v1.0"
        if self.metadata:
            model_name = self.metadata.get("model_name", "Unknown")
            model_version = self.metadata.get("version", "v1.0")

        return {
            "prediction": label,
            "claim_probability": round(prob, 4),
            "non_claim_probability": round(1.0 - prob, 4),
            "risk_score": risk_score,
            "risk_category": risk_category,
            "threshold_used": threshold,
            "model_name": model_name,
            "model_version": model_version,
            "data_mode": self._data_mode,
            "top_risk_factors": [],     # enriched by ExplainabilityEngine in routes
            "protective_factors": [],
            "risk_profile": self.generate_risk_profile([], risk_category),
            "recommendations": self.get_recommendations(risk_category, []),
        }

    def predict_batch(self, df: pd.DataFrame) -> list:
        """Predict all rows of a DataFrame using strictly the authoritative active threshold."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded.")
        threshold = self.get_active_threshold()

        try:
            probs = self._predict_prob_array(df)
        except Exception as e:
            raise ValueError(f"Batch prediction failed: {e}")

        model_version = self.metadata.get("version", "v1.0") if self.metadata else "v1.0"

        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            prob = float(probs[i])
            risk_score = self.compute_risk_score(prob)
            risk_category = self.get_risk_category(risk_score)
            label = self.get_prediction_label(prob, threshold)

            policy_id = str(row.get("policy_id", f"row_{i+1}"))

            results.append({
                "policy_id": policy_id,
                "prediction": label,
                "claim_probability": round(prob, 4),
                "non_claim_probability": round(1.0 - prob, 4),
                "risk_score": risk_score,
                "risk_category": risk_category,
                "model_version": model_version,
            })
        return results

    # ── Risk scoring ──────────────────────────────────────────────────────────

    @staticmethod
    def compute_risk_score(probability: float) -> int:
        """Map probability [0,1] to AI Risk Score [0,100]. Labelled as Project Scale."""
        return int(round(probability * 100))

    @staticmethod
    def get_risk_category(risk_score: int) -> str:
        if risk_score <= config.LOW_RISK_MAX:
            return "Low"
        elif risk_score <= config.MODERATE_RISK_MAX:
            return "Moderate"
        elif risk_score <= config.HIGH_RISK_MAX:
            return "High"
        else:
            return "Critical"

    @staticmethod
    def get_prediction_label(probability: float, threshold: float) -> str:
        return "Likely to Claim" if probability >= threshold else "Unlikely to Claim"

    @staticmethod
    def generate_risk_profile(factors: list, risk_category: str) -> str:
        base = RISK_PROFILES.get(risk_category, "Risk profile unavailable.")
        if factors:
            top = factors[:2]
            names = [f["feature"].replace("_", " ") for f in top if isinstance(f, dict)]
            if names:
                base += f" Key contributing factors: {', '.join(names)}."
        return base

    @staticmethod
    def get_recommendations(risk_category: str, top_factors: list) -> list:
        recs = RECOMMENDATIONS.get(risk_category, ["Standard review applies."])
        return recs + [
            "NOTE: These are analytical suggestions, not automatic insurance decisions."
        ]
