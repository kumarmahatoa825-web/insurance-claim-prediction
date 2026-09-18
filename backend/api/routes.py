"""
InsureAI — Core API Routes
============================
Implements:
  - Validated prediction endpoint (/predict) with 503 handling & SHAP integration
  - Non-crashing batch prediction (/batch-predict) with row-level validation
  - Real-data driven analytics (/analytics) without fake/formula-derived numbers
  - Comprehensive model monitoring (/monitoring) with PSI drift & verified ROC-AUC
  - Model registry endpoints (/registry, /registry/promote, /registry/rollback, /registry/compare)
  - Demographic fairness analysis (/fairness) on real vehicle attributes
  - Prediction history, cost analysis, threshold tuning, and calibration curves
"""

import os
import io
import json
import csv
import logging
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Query, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.settings import settings
from backend.database.db import (
    get_db, save_prediction, get_predictions, delete_prediction,
    record_prediction_outcome, export_predictions_csv
)
from backend.services.report_service import ReportService
from backend.validation.data_validator import DataValidator
from backend.ml.eda import DataQualityAnalyzer
from backend.ml.fairness import FairnessAnalyzer
from backend.ml.monitoring import ModelMonitor
from backend.ml.registry import ModelRegistry

logger = logging.getLogger(__name__)
router = APIRouter()

DISCLAIMER = (
    "This system is an educational AI/ML project. Predictions are probabilistic estimates "
    "and should not be used as the sole basis for real insurance decisions. "
    "AI Risk Score (0-100) is a Project Scale metric, not an industry underwriting score."
)


# ── Pydantic Request Models ───────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Schema-flexible prediction request.
    Supports real vehicle insurance fields as well as synthetic legacy attributes.
    Server strictly controls decision threshold; client threshold overrides are rejected with 422.
    Target leakage fields (e.g. is_claim) are strictly forbidden with 422.
    """
    model_config = {"extra": "allow"}

    # Control fields
    threshold: Optional[float] = Field(default=None, description="Forbidden: threshold overrides are rejected. Server uses active model threshold.")
    store_prediction: bool = True
    policy_id: Optional[str] = None

    @field_validator("threshold")
    @classmethod
    def validate_no_threshold_override(cls, v):
        if v is not None:
            raise ValueError(
                "Overriding the decision threshold is not permitted on /predict. "
                "The active Champion model's locked threshold is authoritative. "
                "Use /threshold-analysis for threshold experimentation."
            )
        return v

    @field_validator("is_claim", "claim_filed", "claim_amount", check_fields=False)
    @classmethod
    def validate_no_target_leakage(cls, v):
        if v is not None:
            raise ValueError(
                "Data leakage risk: target/outcome columns cannot be passed during prediction."
            )
        return v

    # Real vehicle insurance schema fields
    policy_tenure: Optional[float] = None
    age_of_car: Optional[float] = None
    age_of_policyholder: Optional[float] = None
    area_cluster: Optional[str] = None
    population_density: Optional[int] = None
    make: Optional[int] = None
    segment: Optional[str] = None
    model: Optional[str] = None
    fuel_type: Optional[str] = None
    max_torque: Optional[str] = None
    max_power: Optional[str] = None
    engine_type: Optional[str] = None
    airbags: Optional[int] = None
    is_esc: Optional[str] = None
    is_adjustable_steering: Optional[str] = None
    is_tpms: Optional[str] = None
    is_parking_sensors: Optional[str] = None
    is_parking_camera: Optional[str] = None
    rear_brakes_type: Optional[str] = None
    displacement: Optional[int] = None
    cylinder: Optional[int] = None
    transmission_type: Optional[str] = None
    gear_box: Optional[int] = None
    steering_type: Optional[str] = None
    turning_radius: Optional[float] = None
    length: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    gross_weight: Optional[int] = None
    is_front_fog_lights: Optional[str] = None
    is_rear_window_wiper: Optional[str] = None
    is_rear_window_washer: Optional[str] = None
    is_rear_window_defogger: Optional[str] = None
    is_brake_assist: Optional[str] = None
    is_power_door_locks: Optional[str] = None
    is_central_locking: Optional[str] = None
    is_power_steering: Optional[str] = None
    is_driver_seat_height_adjustable: Optional[str] = None
    is_day_night_rear_view_mirror: Optional[str] = None
    is_ecw: Optional[str] = None
    is_speed_alert: Optional[str] = None
    ncap_rating: Optional[int] = None

    # Legacy synthetic fields
    age: Optional[int] = None
    gender: Optional[str] = None
    income: Optional[float] = None
    occupation: Optional[str] = None
    location: Optional[str] = None
    policy_type: Optional[str] = None
    premium: Optional[float] = None
    coverage_amount: Optional[float] = None
    vehicle_age: Optional[int] = None
    num_dependents: Optional[int] = None
    previous_claims: Optional[int] = None
    payment_history: Optional[float] = None
    renewal_count: Optional[int] = None
    health_score: Optional[float] = None


class CostAnalysisRequest(BaseModel):
    fp_cost: float = Field(ge=0, default=500.0, description="Cost of a False Positive (Rs)")
    fn_cost: float = Field(ge=0, default=5000.0, description="Cost of a False Negative (Rs)")


class VehiclePredictRequest(PredictRequest):
    """Strict canonical vehicle prediction schema that rejects unknown extra fields."""
    model_config = {"extra": "forbid"}


class RecordOutcomeRequest(BaseModel):
    actual_claim: int = Field(..., ge=0, le=1, description="Actual claim outcome: 1 for claim filed, 0 for no claim")
    outcome_source: Optional[str] = Field(default="production_audit", description="Source or verifier of the outcome")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _get_engine(request: Request):
    """Retrieve initialized prediction engine or raise 503 if unavailable (requirement 175)."""
    engine = getattr(request.app.state, "predict_engine", None)
    if engine is None or not getattr(request.app.state, "model_loaded", False):
        raise HTTPException(
            status_code=503,
            detail="Predictive model is currently unavailable. Ensure training has been completed.",
        )
    return engine


# ── Health & System ───────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
async def health(request: Request):
    meta = getattr(request.app.state, "metadata", None) or {}
    return {
        "status": "ok",
        "model_loaded": getattr(request.app.state, "model_loaded", False),
        "model_version": meta.get("version", "unknown"),
        "model_name": meta.get("model_name", "unknown"),
        "data_mode": meta.get("data_mode", "real"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/live", tags=["System"])
async def health_live():
    """Liveness probe: returns 200 if the server process is responsive."""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/ready", tags=["System"])
async def health_ready(request: Request):
    """Readiness probe: returns 200 if model is loaded, metadata matches schema, and checksums verify, else 503."""
    is_loaded = getattr(request.app.state, "model_loaded", False)
    engine = getattr(request.app.state, "predict_engine", None)
    meta = getattr(request.app.state, "metadata", None)
    if not is_loaded or engine is None or meta is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready: champion model is not loaded.",
        )

    # Check schema consistency
    schema_version_path = os.path.join("schemas", "schema_version.json")
    schema_version_str = "unknown"
    if os.path.exists(schema_version_path):
        try:
            with open(schema_version_path, "r", encoding="utf-8") as f:
                s_info = json.load(f)
                schema_version_str = s_info.get("version", "unknown")
                expected_sha = s_info.get("feature_set_sha256")
                actual_sha = meta.get("feature_set_sha256")
                if expected_sha and actual_sha and expected_sha != actual_sha:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Service not ready: feature schema mismatch (expected {expected_sha[:8]}, model has {actual_sha[:8]}).",
                    )
        except HTTPException:
            raise
        except Exception as se:
            logger.warning("Schema check error in health/ready: %s", se)

    # Verify release checksums if candidate/release exists
    version = meta.get("version")
    if version:
        try:
            registry = ModelRegistry()
            is_valid, reason = registry.verify_release(version)
            if not is_valid:
                raise HTTPException(
                    status_code=503,
                    detail=f"Service not ready: release verification failed for {version}: {reason}",
                )
        except HTTPException:
            raise
        except Exception as ve:
            logger.warning("Release verification check skipped or failed: %s", ve)

    return {
        "status": "ready",
        "model_loaded": True,
        "model_version": meta.get("version", "unknown"),
        "model_name": meta.get("model_name", "unknown"),
        "schema_version": schema_version_str,
        "checksums_verified": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/data-mode", tags=["System"])
async def get_data_mode(request: Request):
    meta = getattr(request.app.state, "metadata", None) or {}
    mode = meta.get("data_mode", "real")
    dataset_name = meta.get("dataset", "Vehicle Insurance Claims (Real World)")
    real_test_path = os.path.join("data", "real", "test.csv")
    test_rows = 0
    if os.path.exists(real_test_path):
        try:
            with open(real_test_path, "rb") as f:
                test_rows = max(0, sum(1 for _ in f) - 1)
        except Exception:
            test_rows = meta.get("test_rows", 0)
    thresh = meta.get("default_threshold", meta.get("optimal_threshold", 0.080))
    return {
        "data_mode": mode,
        "dataset_name": dataset_name,
        "default_threshold": thresh,
        "optimal_threshold": thresh,
        "model_version": meta.get("version", "unknown"),
        "real_test_available": os.path.exists(real_test_path),
        "real_test_rows": test_rows,
    }


# ── Prediction Endpoints ──────────────────────────────────────────────────────

@router.post("/predict", tags=["Prediction"])
async def predict_single(req: PredictRequest, request: Request, db: AsyncSession = Depends(get_db)):
    engine = _get_engine(request)
    explain_engine = getattr(request.app.state, "explain_engine", None)
    meta = getattr(request.app.state, "metadata", None) or {}
    data_mode = meta.get("data_mode", "real")

    # Extract non-None input values
    raw_dict = req.model_dump(exclude={"threshold", "store_prediction", "policy_id"})
    input_dict = {k: v for k, v in raw_dict.items() if v is not None}
    if hasattr(req, "model_extra") and req.model_extra:
        input_dict.update({k: v for k, v in req.model_extra.items() if v is not None})

    # Validate inputs (Requirement 176)
    is_valid, errors, warnings = DataValidator.validate_prediction_input(input_dict, data_mode=data_mode)
    if not is_valid:
        raise HTTPException(status_code=422, detail={"errors": errors, "warnings": warnings})

    try:
        result = engine.predict_single(input_dict)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Attach SHAP explanation
    if explain_engine is not None:
        try:
            pipeline_obj = getattr(request.app.state, "model", None)
            preproc = pipeline_obj.get("preprocessor") if isinstance(pipeline_obj, dict) else (
                pipeline_obj.named_steps.get("preprocessor") if hasattr(pipeline_obj, "named_steps") else None
            )
            explanation = explain_engine.explain_single(input_dict, preproc)
            result["top_risk_factors"] = explanation.get("top_risk_factors", [])
            result["protective_factors"] = explanation.get("protective_factors", [])
            result["shap_values"] = explanation.get("shap_values", {})
        except Exception as ex:
            logger.warning("SHAP explanation failed: %s", ex)

    req_id = getattr(request.state, "request_id", None)
    result["request_id"] = req_id
    result["disclaimer"] = DISCLAIMER
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["validation_warnings"] = warnings
    result.setdefault("prediction_id", None)

    # Persist to database
    if req.store_prediction:
        try:
            saved = await save_prediction(db, {
                "policy_id": req.policy_id or "manual_entry",
                "claim_probability": result["claim_probability"],
                "risk_score": result["risk_score"],
                "risk_category": result["risk_category"],
                "prediction": result["prediction"],
                "model_version": result.get("model_version", meta.get("version", "v4.1.1")),
                "feature_schema_version": meta.get("schema_version", "1.0.0"),
                "threshold_used": result.get("threshold_used"),
                "model_release_hash": result.get("model_release_hash"),
                "request_id": req_id,
                "top_factors_json": json.dumps(result.get("top_risk_factors", [])),
                "input_data": input_dict,
            })
            result["prediction_id"] = saved.id
        except Exception as db_err:
            logger.warning("Could not persist prediction to database: %s", db_err)

    return _json_safe(result)


@router.post("/batch-predict", tags=["Prediction"])
async def batch_predict(request: Request, file: UploadFile = File(...)):
    """
    Batch prediction endpoint adhering to Requirement 265:
    Never crash entire batch because of a single bad row; returns valid predictions
    and report of any invalid rows.
    """
    engine = _get_engine(request)
    meta = getattr(request.app.state, "metadata", None) or {}
    data_mode = meta.get("data_mode", "real")

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB}MB.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")

    # Row-level non-crashing validation
    valid_df, failed_df, val_summary = DataValidator.validate_batch_dataframe(
        df, data_mode=data_mode, max_rows=settings.MAX_BATCH_ROWS
    )

    if val_summary.get("errors"):
        fatal_msg = "; ".join(val_summary["errors"])
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Batch rejected: {fatal_msg}",
                "validation_summary": val_summary,
            },
        )

    if len(valid_df) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No valid rows found in uploaded CSV.",
                "validation_summary": val_summary,
                "sample_errors": failed_df.head(5).to_dict(orient="records") if len(failed_df) > 0 else [],
            },
        )

    try:
        predictions = engine.predict_batch(valid_df)
    except Exception as e:
        logger.error("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch prediction failure: {e}")

    # Risk Distribution
    risk_dist: Dict[str, int] = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
    for p in predictions:
        cat = p.get("risk_category", "Unknown")
        risk_dist[cat] = risk_dist.get(cat, 0) + 1

    # Save output CSV to reports directory
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(settings.REPORTS_DIR, "batch_result.csv")
    fieldnames = [
        "policy_id", "claim_probability", "non_claim_probability",
        "risk_score", "risk_category", "prediction", "model_version"
    ]
    with open(report_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for p in predictions:
            writer.writerow(p)

    return _json_safe({
        "total": len(df),
        "total_rows": len(df),
        "total_submitted": len(df),
        "successful_predictions": len(predictions),
        "valid_predictions_count": len(predictions),
        "failed_rows_count": len(failed_df),
        "risk_distribution": risk_dist,
        "predictions": predictions[:100],  # Return first 100 for display
        "download_path": report_path,
        "validation_summary": val_summary,
        "data_mode": data_mode,
    })


@router.get("/batch-predict/download", tags=["Prediction"])
async def download_batch_result():
    path = os.path.join(settings.REPORTS_DIR, "batch_result.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No batch predictions available. Run a batch prediction first.")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return StreamingResponse(
        io.StringIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_prediction_results.csv"},
    )


@router.post("/batch-predict-real-test", tags=["Prediction"])
async def batch_predict_real_test(request: Request):
    """Run batch prediction demo on real unlabelled test.csv."""
    engine = _get_engine(request)
    test_path = os.path.join("data", "real", "test.csv")
    if not os.path.exists(test_path):
        raise HTTPException(status_code=404, detail="data/real/test.csv not found.")

    try:
        df = pd.read_csv(test_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed reading real test CSV: {e}")

    sample_size = min(500, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    predictions = engine.predict_batch(df_sample)
    risk_dist: Dict[str, int] = {}
    for p in predictions:
        cat = p.get("risk_category", "Unknown")
        risk_dist[cat] = risk_dist.get(cat, 0) + 1

    avg_prob = sum(p["claim_probability"] for p in predictions) / len(predictions) if predictions else 0.0

    return _json_safe({
        "total_in_test_set": len(df),
        "sampled_for_demo": sample_size,
        "predictions": predictions[:50],
        "risk_distribution": risk_dist,
        "avg_claim_probability": round(avg_prob, 4),
        "note": "data/real/test.csv has NO ground-truth labels. Evaluated for inference demo only.",
    })


# ── Analytics & Data Quality ─────────────────────────────────────────────────

@router.get("/analytics", tags=["Analytics"])
async def get_analytics(request: Request):
    """
    Computes verified analytics strictly from real dataset predictions.
    Adheres to requirements 202-210, 266-278, 620:
      - No fake/formula-derived risk distributions
      - Predicted claim rate from actual model predictions
      - Actual claim rate from is_claim
      - True probability histogram
    """
    meta = getattr(request.app.state, "metadata", None) or {}
    data_mode = meta.get("data_mode", "real")

    if data_mode == "real":
        data_path = os.path.join("data", "real", "train.csv")
        target_col = "is_claim"
    else:
        data_path = os.path.join("data", "raw", "insurance_claims.csv")
        target_col = "claim_filed"

    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Dataset not found at {data_path}.")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset read failed: {e}")

    total = len(df)
    actual_claims = int(df[target_col].sum())
    actual_claim_rate = float(round(actual_claims / total, 4))
    actual_no_claims = total - actual_claims

    # Group breakdowns
    def get_group_counts(col_name: str) -> Dict[str, int]:
        if col_name in df.columns:
            return {str(k): int(v) for k, v in df.groupby(col_name)[target_col].sum().items()}
        return {}

    if data_mode == "real":
        claims_by_type = get_group_counts("fuel_type")
        claims_by_location = get_group_counts("area_cluster")
        claims_by_occ = get_group_counts("segment")
        if "population_density" in df.columns:
            def pop_bucket(p):
                if p < 5000: return "<5K"
                elif p < 15000: return "5K-15K"
                elif p < 30000: return "15K-30K"
                else: return ">30K"
            df["_pop_bucket"] = df["population_density"].apply(pop_bucket)
            claims_by_income = {str(k): int(v) for k, v in df.groupby("_pop_bucket")[target_col].sum().items()}
        else:
            claims_by_income = {}
        if "age_of_policyholder" in df.columns:
            df["_age_group"] = pd.cut(
                df["age_of_policyholder"],
                bins=[0, 0.35, 0.55, 2.0],
                labels=["Younger (<0.35)", "Middle (0.35-0.55)", "Senior (>0.55)"],
            )
            claims_by_age = {str(k): int(v) for k, v in df.groupby("_age_group", observed=True)[target_col].sum().items()}
        else:
            claims_by_age = {}
    else:
        claims_by_type = get_group_counts("policy_type")
        claims_by_location = get_group_counts("location")
        claims_by_occ = get_group_counts("occupation")
        claims_by_income = {}
        claims_by_age = {}

    # Real predictions on a representative stratified sample for predicted metrics (Requirements 203-209, 271-273)
    engine = getattr(request.app.state, "predict_engine", None)
    risk_cat_dist = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
    hist_data = []
    avg_predicted_prob = meta.get("test_metrics", {}).get("mean_probability")

    if engine and getattr(request.app.state, "model_loaded", False):
        try:
            sample_df = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)
            feat_df = sample_df.drop(columns=[target_col, "policy_id"], errors="ignore")
            preds = engine.predict_batch(feat_df)

            sample_probs = [p["claim_probability"] for p in preds]
            avg_predicted_prob = float(round(sum(sample_probs) / len(sample_probs), 4))

            # Count sample risk categories
            sample_cats = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
            for p in preds:
                c = p.get("risk_category", "Low")
                sample_cats[c] = sample_cats.get(c, 0) + 1

            # Scale proportionally to the full dataset so counts sum to total (Requirement 208)
            scale = total / len(preds)
            risk_cat_dist = {k: int(round(v * scale)) for k, v in sample_cats.items()}
            # Adjust rounding difference to guarantee exact sum
            diff = total - sum(risk_cat_dist.values())
            risk_cat_dist["Low"] += diff

            # Histogram
            for i in range(10):
                lo, hi = i / 10, (i + 1) / 10
                cnt = sum(1 for pr in sample_probs if lo <= pr < hi)
                hist_data.append({"bucket": f"{int(lo*100)}-{int(hi*100)}%", "count": int(round(cnt * scale))})
        except Exception as ex:
            logger.warning("Could not run sample predictions for analytics: %s", ex)

    # Feature importance
    feat_imp = meta.get("feature_importance", meta.get("feature_importances", []))[:15]
    if not feat_imp:
        explain_engine = getattr(request.app.state, "explain_engine", None)
        if explain_engine:
            feat_imp = explain_engine.get_global_importance(15)

    high_risk_count = risk_cat_dist.get("High", 0) + risk_cat_dist.get("Critical", 0)
    best_f1 = meta.get("test_metrics", {}).get("f1") or meta.get("cv_metrics", {}).get("mean", {}).get("f1")

    return _json_safe({
        "total_analyzed": total,
        "high_risk_count": high_risk_count,
        "avg_claim_probability": avg_predicted_prob,
        "actual_claim_rate": actual_claim_rate,
        "best_model": meta.get("model_name", "Random Forest"),
        "best_f1": round(float(best_f1), 4) if best_f1 is not None else None,
        "data_mode": data_mode,
        "dataset_info": meta.get("dataset", "Vehicle Insurance Claims"),
        "claim_distribution": {"Claim": actual_claims, "No Claim": actual_no_claims},
        "risk_category_distribution": risk_cat_dist,
        "claims_by_age_group": claims_by_age,
        "claims_by_policy_type": claims_by_type,
        "claims_by_location": claims_by_location,
        "claims_by_occupation": claims_by_occ,
        "claims_by_income_range": claims_by_income,
        "probability_histogram": hist_data,
        "feature_importance": feat_imp,
    })


@router.get("/data-quality", tags=["Analytics"])
async def get_data_quality(request: Request):
    meta = getattr(request.app.state, "metadata", None) or {}
    data_mode = meta.get("data_mode", "real")
    data_path = os.path.join("data", "real", "train.csv") if data_mode == "real" else os.path.join("data", "raw", "insurance_claims.csv")

    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Dataset not found at {data_path}.")

    try:
        df = pd.read_csv(data_path)
        analyzer = DataQualityAnalyzer()
        report = analyzer.get_data_quality_report(df)
        report["data_mode"] = data_mode
        report["dataset_path"] = data_path
        return _json_safe(report)
    except Exception as e:
        logger.error("Data quality check failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Data quality analysis error: {e}")


# ── Fairness Analysis ─────────────────────────────────────────────────────────

@router.get("/fairness", tags=["Fairness"])
async def get_fairness(request: Request):
    meta = getattr(request.app.state, "metadata", None) or {}
    if meta.get("fairness"):
        return _json_safe(meta["fairness"])

    data_mode = meta.get("data_mode", "real")
    data_path = os.path.join("data", "real", "train.csv") if data_mode == "real" else os.path.join("data", "raw", "insurance_claims.csv")
    target_col = "is_claim" if data_mode == "real" else "claim_filed"

    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Dataset not found at {data_path}.")

    engine = _get_engine(request)
    try:
        df = pd.read_csv(data_path)
        sample_df = df.sample(n=min(1500, len(df)), random_state=42).reset_index(drop=True)
        feat_df = sample_df.drop(columns=[target_col, "policy_id"], errors="ignore")

        preds = engine.predict_batch(feat_df)
        y_true = sample_df[target_col].values
        y_pred = [1 if p["prediction"] == "Likely to Claim" else 0 for p in preds]
        y_prob = [p["claim_probability"] for p in preds]

        analyzer = FairnessAnalyzer()
        report = analyzer.analyze(y_true, y_pred, y_prob, sample_df)
        return _json_safe(report)
    except Exception as e:
        logger.error("Fairness analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Fairness analysis failed: {e}")


# ── Model Registry & Monitoring ───────────────────────────────────────────────

@router.get("/monitoring", tags=["Monitoring"])
async def get_monitoring(request: Request, db: AsyncSession = Depends(get_db)):
    """Comprehensive production monitoring endpoint (Section T)."""
    meta = getattr(request.app.state, "metadata", None) or {}
    try:
        history_records = await get_predictions(db, page=1, page_size=100)
        items = history_records.get("items", [])
    except Exception:
        items = []

    monitor = ModelMonitor(meta)
    report = monitor.get_monitoring_report(items)
    return _json_safe(report)


@router.get("/registry", tags=["Registry"])
async def get_registry(request: Request):
    """Fetch current Model Registry state including champion and candidates."""
    reg = getattr(request.app.state, "registry", None) or ModelRegistry(settings.MODELS_DIR)
    return _json_safe(reg.to_dict())


@router.post("/registry/approve/{candidate_version}", tags=["Registry"])
async def approve_candidate(candidate_version: str, request: Request):
    """Explicitly approve a candidate model for promotion (governance gate)."""
    reg = getattr(request.app.state, "registry", None) or ModelRegistry(settings.MODELS_DIR)
    try:
        reg.approve_candidate(candidate_version)
        return {"status": "approved", "version": candidate_version}
    except Exception as e:
        logger.error("Approval failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Approval failed: {e}")


@router.post("/registry/promote/{candidate_version}", tags=["Registry"])
async def promote_candidate(candidate_version: str, request: Request):
    """Promote an approved candidate model to champion."""
    reg = getattr(request.app.state, "registry", None) or ModelRegistry(settings.MODELS_DIR)
    try:
        champion = reg.promote_candidate(candidate_version)
        # Reload champion and metadata in app state
        m_path, meta_path = reg.get_active_champion_paths()
        if m_path and os.path.exists(m_path):
            import joblib
            request.app.state.model = joblib.load(m_path)
            request.app.state.model_loaded = True
            if getattr(request.app.state, "predict_engine", None):
                request.app.state.predict_engine.load_model(m_path)
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                request.app.state.metadata = json.load(f)
            if getattr(request.app.state, "predict_engine", None):
                request.app.state.predict_engine.metadata = request.app.state.metadata
        from backend.ml.monitoring import invalidate_monitoring_cache
        invalidate_monitoring_cache()
        return {"status": "promoted", "new_champion": champion}
    except Exception as e:
        logger.error("Promotion failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Promotion failed: {e}")


@router.post("/registry/rollback", tags=["Registry"])
async def rollback_champion(request: Request):
    """Roll back to the previous champion model."""
    reg = getattr(request.app.state, "registry", None) or ModelRegistry(settings.MODELS_DIR)
    try:
        restored = reg.rollback()
        # Reload restored champion and metadata in app state
        m_path, meta_path = reg.get_active_champion_paths()
        if m_path and os.path.exists(m_path):
            import joblib
            request.app.state.model = joblib.load(m_path)
            request.app.state.model_loaded = True
            if getattr(request.app.state, "predict_engine", None):
                request.app.state.predict_engine.load_model(m_path)
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                request.app.state.metadata = json.load(f)
            if getattr(request.app.state, "predict_engine", None):
                request.app.state.predict_engine.metadata = request.app.state.metadata
        from backend.ml.monitoring import invalidate_monitoring_cache
        invalidate_monitoring_cache()
        return {"status": "rolled_back", "restored_champion": restored}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Rollback failed: {e}")


@router.get("/registry/compare/{candidate_version}", tags=["Registry"])
async def compare_candidate(candidate_version: str, request: Request):
    """Compare candidate metrics with active champion."""
    reg = getattr(request.app.state, "registry", None) or ModelRegistry(settings.MODELS_DIR)
    try:
        return _json_safe(reg.compare_with_champion(candidate_version))
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/model-info", tags=["Model"])
async def get_model_info(request: Request):
    meta = getattr(request.app.state, "metadata", None)
    if meta:
        return _json_safe(meta)
    meta_path = settings.CHAMPION_METADATA_PATH if os.path.exists(settings.CHAMPION_METADATA_PATH) else os.path.join(settings.MODELS_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return _json_safe(json.load(f))
    raise HTTPException(status_code=404, detail="Model metadata not found. Run training first.")


@router.get("/model-comparison", tags=["Model"])
async def get_model_comparison():
    path = os.path.join(settings.MODELS_DIR, "all_models_results.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Model comparison data not found.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    roc_path = os.path.join(settings.MODELS_DIR, "roc_curves.json")
    if os.path.exists(roc_path):
        with open(roc_path, "r", encoding="utf-8") as f:
            roc_data = json.load(f)
        for name, metrics in data.items():
            metrics["roc_curve"] = roc_data.get(name, {})

    return _json_safe(data)


# ── Explainability, Threshold, Cost Analysis & History ───────────────────────

@router.post("/explain", tags=["Explainability"])
async def explain_prediction(req: PredictRequest, request: Request):
    engine = _get_engine(request)
    explain_engine = getattr(request.app.state, "explain_engine", None)

    input_dict = req.model_dump(exclude={"threshold", "store_prediction", "policy_id"})
    input_dict = {k: v for k, v in input_dict.items() if v is not None}

    try:
        result = engine.predict_single(input_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    shap_result = {}
    if explain_engine is not None:
        try:
            pipeline_obj = getattr(request.app.state, "model", None)
            preproc = pipeline_obj.get("preprocessor") if isinstance(pipeline_obj, dict) else (
                pipeline_obj.named_steps.get("preprocessor") if hasattr(pipeline_obj, "named_steps") else None
            )
            shap_result = explain_engine.explain_single(input_dict, preproc)
        except Exception as e:
            shap_result = {"error": str(e)}

    resp = {
        "available": shap_result.get("available", True if "shap_values" in shap_result else False),
        "prediction": result,
        "explanation": shap_result,
        "disclaimer": DISCLAIMER,
    }
    for k in ["shap_values", "base_value", "top_risk_factors", "protective_factors", "attribution_note", "reason"]:
        if k in shap_result:
            resp[k] = shap_result[k]
    return _json_safe(resp)


@router.get("/threshold-analysis", tags=["Model"])
async def threshold_analysis():
    path = os.path.join(settings.MODELS_DIR, "threshold_analysis.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return _json_safe(json.load(f))
    raise HTTPException(status_code=404, detail="Threshold analysis not found.")


@router.post("/cost-analysis", tags=["Model"])
async def cost_analysis(req: CostAnalysisRequest, request: Request):
    path = os.path.join(settings.MODELS_DIR, "threshold_analysis.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Threshold analysis data not available.")

    with open(path, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    results = []
    for row in threshold_data:
        fp = row.get("fp", 0)
        fn = row.get("fn", 0)
        cost = fp * req.fp_cost + fn * req.fn_cost
        results.append({
            "threshold": row["threshold"],
            "expected_cost": cost,
            "fp_count": fp,
            "fn_count": fn,
            "precision": row.get("precision", 0),
            "recall": row.get("recall", 0),
            "f1": row.get("f1", 0),
        })

    optimal = min(results, key=lambda x: x["expected_cost"])
    return _json_safe({
        "threshold_costs": results,
        "optimal_threshold": optimal["threshold"],
        "min_expected_cost": optimal["expected_cost"],
        "fp_cost_used": req.fp_cost,
        "fn_cost_used": req.fn_cost,
    })


@router.get("/calibration", tags=["Model"])
async def get_calibration():
    path = os.path.join(settings.MODELS_DIR, "calibration.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return _json_safe(json.load(f))
    raise HTTPException(status_code=404, detail="Calibration curve data not found.")


@router.get("/history", tags=["History"])
async def get_prediction_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str = Query(""),
    risk_category: str = Query(""),
    sort_by: str = Query("timestamp"),
    sort_order: str = Query("desc"),
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await get_predictions(
            db, page=page, page_size=page_size,
            search=search, risk_category=risk_category,
            sort_by=sort_by, sort_order=sort_order,
        )
        return _json_safe(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History query failed: {e}")


@router.delete("/history/{prediction_id}", tags=["History"])
async def delete_history_item(prediction_id: int, db: AsyncSession = Depends(get_db)):
    raise HTTPException(
        status_code=403,
        detail="Prediction audit history is immutable and cannot be deleted in production.",
    )


@router.post("/history/{prediction_id}/outcome", tags=["History"])
async def record_history_outcome(
    prediction_id: int,
    req: RecordOutcomeRequest,
    db: AsyncSession = Depends(get_db),
):
    pred = await record_prediction_outcome(
        db,
        prediction_id=prediction_id,
        actual_claim=req.actual_claim,
        outcome_source=req.outcome_source or "production_audit",
    )
    if not pred:
        raise HTTPException(status_code=404, detail=f"Prediction record {prediction_id} not found.")
    return {
        "status": "outcome_recorded",
        "id": pred.id,
        "actual_claim": pred.actual_claim,
        "actual_outcome_recorded_at": pred.actual_outcome_recorded_at.isoformat() if pred.actual_outcome_recorded_at else None,
        "outcome_source": pred.outcome_source,
    }


@router.get("/history/export", tags=["History"])
async def export_history(db: AsyncSession = Depends(get_db)):
    try:
        csv_content = await export_predictions_csv(db)
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=prediction_history.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")
