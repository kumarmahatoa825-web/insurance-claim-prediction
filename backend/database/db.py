import os
import io
import csv
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, select, func, delete, text

from backend.core.settings import settings

# Use centralized settings — do NOT read os.getenv() directly here.
# This ensures the database URL is controlled by one source of truth.
DATABASE_URL = settings.DATABASE_URL
# Ensure aiosqlite dialect for SQLite
if DATABASE_URL.startswith("sqlite:///") and not DATABASE_URL.startswith("sqlite+aiosqlite"):
    DATABASE_URL = DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


# ── ORM Models ───────────────────────────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    policy_id = Column(String, index=True, nullable=True, default="manual")
    claim_probability = Column(Float, nullable=False)
    risk_score = Column(Integer, nullable=False)
    risk_category = Column(String, nullable=False)
    prediction = Column(String, nullable=False)
    model_version = Column(String, nullable=True)
    feature_schema_version = Column(String, nullable=True, default="vehicle-v1")
    threshold_used = Column(Float, nullable=True)
    model_release_hash = Column(String, nullable=True)
    request_id = Column(String, nullable=True)
    actual_claim = Column(Integer, nullable=True)
    actual_outcome_recorded_at = Column(DateTime, nullable=True)
    outcome_source = Column(String, nullable=True)
    top_factors_json = Column(Text, nullable=True)
    input_data_json = Column(Text, nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    version = Column(String, unique=True, index=True)
    model_name = Column(String)
    training_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    dataset_size = Column(Integer)
    feature_count = Column(Integer)
    validation_metrics_json = Column(Text)
    test_metrics_json = Column(Text)


# ── Lifecycle ────────────────────────────────────────────────────────────────

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for col_def in [
            "input_data_json TEXT",
            "feature_schema_version TEXT DEFAULT 'vehicle-v1'",
            "threshold_used REAL",
            "model_release_hash TEXT",
            "request_id TEXT",
            "actual_claim INTEGER",
            "actual_outcome_recorded_at TIMESTAMP",
            "outcome_source TEXT",
        ]:
            try:
                await conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {col_def}"))
            except Exception:
                pass


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# ── CRUD Operations ──────────────────────────────────────────────────────────

async def save_prediction(db: AsyncSession, data: dict) -> Prediction:
    input_payload = data.get("input_data") or data.get("input_data_json")
    if isinstance(input_payload, dict):
        input_payload_str = json.dumps(input_payload)
    elif isinstance(input_payload, str):
        input_payload_str = input_payload
    else:
        input_payload_str = "{}"

    factors = data.get("top_risk_factors", [])
    if isinstance(factors, list) and factors:
        factors_str = json.dumps(factors)
    else:
        factors_str = data.get("top_factors_json", "[]")

    pred = Prediction(
        policy_id=data.get("policy_id", "manual"),
        claim_probability=float(data.get("claim_probability", 0)),
        risk_score=int(data.get("risk_score", 0)),
        risk_category=data.get("risk_category", "Unknown"),
        prediction=data.get("prediction", "Unknown"),
        model_version=data.get("model_version"),
        feature_schema_version=data.get("feature_schema_version", "vehicle-v1"),
        threshold_used=data.get("threshold_used"),
        model_release_hash=data.get("model_release_hash"),
        request_id=data.get("request_id"),
        actual_claim=data.get("actual_claim"),
        top_factors_json=factors_str,
        input_data_json=input_payload_str,
    )
    db.add(pred)
    await db.commit()
    await db.refresh(pred)
    return pred


async def get_predictions(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    search: str = "",
    risk_category: str = "",
    sort_by: str = "timestamp",
    sort_order: str = "desc",
) -> dict:
    stmt = select(Prediction)

    if search:
        stmt = stmt.where(Prediction.policy_id.ilike(f"%{search}%"))
    if risk_category:
        stmt = stmt.where(Prediction.risk_category == risk_category)

    # Total count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0

    # Sorting
    sort_col = getattr(Prediction, sort_by, Prediction.timestamp)
    if sort_order.lower() == "desc":
        stmt = stmt.order_by(sort_col.desc())
    else:
        stmt = stmt.order_by(sort_col.asc())

    # Pagination
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size)

    result = await db.execute(stmt)
    items = result.scalars().all()

    return {
        "items": [
            {
                "id": p.id,
                "timestamp": p.timestamp.isoformat() if p.timestamp else None,
                "policy_id": p.policy_id,
                "claim_probability": p.claim_probability,
                "risk_score": p.risk_score,
                "risk_category": p.risk_category,
                "prediction": p.prediction,
                "model_version": p.model_version,
                "threshold_used": p.threshold_used,
                "model_release_hash": p.model_release_hash,
                "request_id": p.request_id,
                "actual_claim": p.actual_claim,
                "actual_outcome_recorded_at": p.actual_outcome_recorded_at.isoformat() if getattr(p, "actual_outcome_recorded_at", None) else None,
                "outcome_source": getattr(p, "outcome_source", None),
                "top_factors": json.loads(p.top_factors_json) if p.top_factors_json else [],
                "input_data": json.loads(p.input_data_json) if getattr(p, "input_data_json", None) else {},
            }
            for p in items
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
    }


async def delete_prediction(db: AsyncSession, prediction_id: int) -> bool:
    result = await db.execute(select(Prediction).where(Prediction.id == prediction_id))
    pred = result.scalar_one_or_none()
    if pred is None:
        return False
    await db.delete(pred)
    await db.commit()
    return True


async def record_prediction_outcome(
    db: AsyncSession,
    prediction_id: int,
    actual_claim: int,
    outcome_source: str = "manual_audit",
) -> Optional[Prediction]:
    result = await db.execute(select(Prediction).where(Prediction.id == prediction_id))
    pred = result.scalar_one_or_none()
    if pred is None:
        return None
    pred.actual_claim = int(actual_claim)
    pred.actual_outcome_recorded_at = datetime.now(timezone.utc)
    pred.outcome_source = str(outcome_source)
    await db.commit()
    await db.refresh(pred)
    return pred


async def export_predictions_csv(db: AsyncSession) -> str:
    result = await db.execute(select(Prediction).order_by(Prediction.timestamp.desc()))
    items = result.scalars().all()

    output = io.StringIO()
    fieldnames = [
        "id", "timestamp", "policy_id", "claim_probability",
        "risk_score", "risk_category", "prediction", "model_version"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for p in items:
        writer.writerow({
            "id": p.id,
            "timestamp": p.timestamp.isoformat() if p.timestamp else "",
            "policy_id": p.policy_id,
            "claim_probability": p.claim_probability,
            "risk_score": p.risk_score,
            "risk_category": p.risk_category,
            "prediction": p.prediction,
            "model_version": p.model_version,
        })
    return output.getvalue()


async def save_model_version(db: AsyncSession, metadata: dict) -> ModelVersion:
    mv = ModelVersion(
        version=metadata.get("version", "v1.0"),
        model_name=metadata.get("model_name", "Unknown"),
        dataset_size=metadata.get("dataset_size", 0),
        feature_count=metadata.get("feature_count", 0),
        validation_metrics_json=json.dumps(metadata.get("cv_metrics", {})),
        test_metrics_json=json.dumps(metadata.get("test_metrics", {})),
    )
    db.add(mv)
    await db.commit()
    await db.refresh(mv)
    return mv
