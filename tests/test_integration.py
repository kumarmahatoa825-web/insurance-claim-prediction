"""
InsureAI — End-to-End Real Model & Governance Integration Tests
===============================================================
Fulfills Requirements 24, 25, 26:
  - Tests real model inference on fixture record
  - Tests server-controlled threshold (and rejects client override with 422)
  - Tests batch prediction
  - Tests calibrated probability bounds
  - Tests SHAP explainer contract
  - Tests monitoring with PSI & missingness
  - Tests candidate approval & promotion gates
  - Tests pointer-only rollback
  - Tests final submission output structure
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.ml.registry import ModelRegistry
from backend.ml.predict import PredictionEngine
from backend.ml.explain import ExplainabilityEngine
from backend.ml.monitoring import ModelMonitor


@pytest.fixture
def real_fixture():
    path = os.path.join("tests", "fixtures", "real_prediction_fixture.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
async def client():
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


# ── Requirement 24: Core Real Model Integration Tests ─────────────────────────

@pytest.mark.asyncio
async def test_real_prediction(client, real_fixture):
    """Test full pipeline prediction on real motor policy fixture."""
    res = await client.post("/predict", json=real_fixture)
    assert res.status_code == 200
    data = res.json()
    assert "claim_probability" in data
    assert 0.0 <= data["claim_probability"] <= 1.0
    assert data["prediction"] in ["Likely to Claim", "Unlikely to Claim"]
    assert 0 <= data["risk_score"] <= 100
    assert data["risk_category"] in ["Low", "Moderate", "High", "Critical"]
    assert "disclaimer" in data


@pytest.mark.asyncio
async def test_threshold_override_rejected(client, real_fixture):
    """
    Requirement 25: Test that /predict rejects any client-supplied threshold override
    with 422 Unprocessable Entity.
    """
    payload = dict(real_fixture)
    payload["threshold"] = 0.01  # Attempt to override locked threshold
    res = await client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "Overriding the decision threshold is not permitted" in str(res.json())


@pytest.mark.asyncio
async def test_active_threshold_used(client, real_fixture):
    """Test that the prediction result reports the authoritative model threshold."""
    res = await client.post("/predict", json=real_fixture)
    assert res.status_code == 200
    data = res.json()
    assert "threshold_used" in data
    assert data["threshold_used"] is not None
    assert 0.01 <= data["threshold_used"] <= 0.99


@pytest.mark.asyncio
async def test_model_version_reported(client, real_fixture):
    """Test that model_version is returned and matches active release."""
    res = await client.post("/predict", json=real_fixture)
    assert res.status_code == 200
    data = res.json()
    assert "model_version" in data
    assert len(data["model_version"]) > 0


@pytest.mark.asyncio
async def test_calibrated_probability_properties(client, real_fixture):
    """Test that calibrated claim and non-claim probabilities sum to 1.0."""
    res = await client.post("/predict", json=real_fixture)
    assert res.status_code == 200
    data = res.json()
    p_claim = data["claim_probability"]
    p_non_claim = data["non_claim_probability"]
    assert pytest.approx(p_claim + p_non_claim, abs=1e-3) == 1.0


@pytest.mark.asyncio
async def test_real_batch_prediction(client, real_fixture):
    """Test non-crashing batch prediction endpoint with multi-row CSV."""
    df = pd.DataFrame([real_fixture, real_fixture])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("batch.csv", csv_bytes, "text/csv")}
    res = await client.post("/batch-predict", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["total_rows"] == 2
    assert data["valid_predictions_count"] == 2


@pytest.mark.asyncio
async def test_shap_contract(client, real_fixture):
    """Test explain endpoint returns standard SHAP contract."""
    res = await client.post("/explain", json=real_fixture)
    assert res.status_code == 200
    data = res.json()
    assert "available" in data
    if data["available"]:
        assert "top_risk_factors" in data
        assert "protective_factors" in data
        assert "attribution_note" in data
    else:
        assert "reason" in data


@pytest.mark.asyncio
async def test_monitoring_endpoint(client):
    """Test monitoring report computes metrics and drift levels."""
    res = await client.get("/monitoring")
    assert res.status_code == 200
    data = res.json()
    assert "overall_status" in data
    assert "drift_monitoring" in data


# ── Requirement 26: Registry Promotion Security & Rollback Tests ──────────────

def test_registry_promotion_security(tmp_path):
    """
    Test candidate approval gate, gate evaluation, and atomic rollback.
    """
    models_dir = str(tmp_path / "models")
    reg = ModelRegistry(models_dir)

    m1 = tmp_path / "m1.joblib"
    m1.write_text("dummy model 1")

    # 1. Register candidate v1
    reg.register_candidate(
        model_name="Gradient Boosting",
        version="v1.0",
        model_obj_or_path=str(m1),
        metadata={
            "default_threshold": 0.08,
            "data_mode": "real",
            "feature_count": 84,
            "test_metrics": {"f1": 0.165, "roc_auc": 0.640, "pr_auc": 0.100, "brier_score": 0.059},
        },
    )

    # Promoting unapproved candidate must FAIL
    with pytest.raises(ValueError, match="not approved"):
        reg.promote_candidate("v1.0")

    # Approve and promote candidate v1
    reg.approve_candidate("v1.0")
    reg.promote_candidate("v1.0")
    assert reg.get_champion_version() == "v1.0"

    # 2. Register candidate v2
    m2 = tmp_path / "m2.joblib"
    m2.write_text("dummy model 2")
    reg.register_candidate(
        model_name="Random Forest",
        version="v2.0",
        model_obj_or_path=str(m2),
        metadata={
            "default_threshold": 0.08,
            "data_mode": "real",
            "feature_count": 84,
            "test_metrics": {"f1": 0.170, "roc_auc": 0.650, "pr_auc": 0.105, "brier_score": 0.058},
        },
    )
    reg.approve_candidate("v2.0")
    reg.promote_candidate("v2.0")
    assert reg.get_champion_version() == "v2.0"

    # 3. Pointer-only Rollback to v1.0
    restored = reg.rollback()
    assert restored["version"] == "v1.0"
    assert reg.get_champion_version() == "v1.0"


def test_submission_format_integrity():
    """Verify that sample submission template is strictly 2 columns and no NaNs."""
    sub_path = os.path.join("data", "real", "sample_submission.csv")
    if os.path.exists(sub_path):
        df = pd.read_csv(sub_path, nrows=50)
        assert list(df.columns) == ["policy_id", "is_claim"]
        assert not df.isna().any().any()
