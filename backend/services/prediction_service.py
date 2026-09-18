from datetime import datetime, timezone
import json
from fastapi import HTTPException
from backend.database.db import AsyncSessionLocal, save_prediction


class PredictionService:
    def __init__(self, app_state):
        self.app_state = app_state
        self.disclaimer = (
            "This system is an educational AI/ML project. Predictions are probabilistic estimates "
            "and should not be used as the sole basis for real insurance decisions."
        )

    async def process_prediction(self, req_data: dict) -> dict:
        predict_engine = getattr(self.app_state, "predict_engine", None)
        if predict_engine is None:
            raise HTTPException(
                status_code=503,
                detail="Model prediction engine is not initialized or unavailable. Please run model training.",
            )

        predict_result = predict_engine.predict_single(req_data)

        explain_engine = getattr(self.app_state, "explain_engine", None)
        if explain_engine is not None:
            try:
                explain_result = explain_engine.explain_single(req_data)
                predict_result["top_risk_factors"] = explain_result.get("top_risk_factors", [])
                predict_result["protective_factors"] = explain_result.get("protective_factors", [])
            except Exception:
                predict_result["top_risk_factors"] = []
                predict_result["protective_factors"] = []

        meta = getattr(self.app_state, "metadata", {}) or {}

        response_data = {
            "prediction": predict_result.get("prediction", "No Claim"),
            "claim_probability": predict_result.get("claim_probability", 0.0),
            "non_claim_probability": predict_result.get("non_claim_probability", 1.0),
            "risk_score": predict_result.get("risk_score", 0),
            "risk_category": predict_result.get("risk_category", "Low"),
            "threshold_used": predict_result.get("threshold_used") or meta.get("default_threshold"),
            "model_name": meta.get("model_name", "unknown"),
            "model_version": meta.get("version", "unknown"),
            "top_risk_factors": predict_result.get("top_risk_factors", []),
            "protective_factors": predict_result.get("protective_factors", []),
            "risk_profile": predict_result.get("risk_profile", "Normal"),
            "recommendations": predict_result.get("recommendations", []),
            "prediction_id": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "disclaimer": self.disclaimer,
        }

        if req_data.get("store_prediction"):
            try:
                data_to_store = dict(response_data)
                data_to_store["input_data"] = req_data
                data_to_store["policy_id"] = req_data.get("policy_id", "manual")
                async with AsyncSessionLocal() as db:
                    saved = await save_prediction(db, data_to_store)
                    response_data["prediction_id"] = saved.id
            except Exception:
                # Prediction still succeeds even if DB storage fails
                pass

        return response_data

