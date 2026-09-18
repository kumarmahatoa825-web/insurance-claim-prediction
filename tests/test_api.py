import pytest
import io
import pandas as pd
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.mark.asyncio
async def test_health_endpoint_returns_200():
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "model_version" in data

@pytest.mark.asyncio
async def test_data_mode_endpoint():
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/data-mode")
            assert response.status_code == 200
            assert response.json()["data_mode"] in ("real", "synthetic")

@pytest.mark.asyncio
async def test_predict_valid_input_returns_200(sample_real_input):
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/predict", json=sample_real_input)
            assert response.status_code == 200
            data = response.json()
            assert "claim_probability" in data
            assert 0.0 <= data["claim_probability"] <= 1.0
            assert 0 <= data["risk_score"] <= 100
            assert data["risk_category"] in ["Low", "Moderate", "High", "Critical"]
            assert "disclaimer" in data

@pytest.mark.asyncio
async def test_predict_missing_required_field_returns_422(sample_real_input):
    # Remove required field policy_tenure
    del sample_real_input["policy_tenure"]
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/predict", json=sample_real_input)
            assert response.status_code == 422

@pytest.mark.asyncio
async def test_predict_target_leakage_rejected_with_422(sample_real_input):
    # Pass target column 'is_claim' to test leakage detection
    sample_real_input["is_claim"] = 1
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/predict", json=sample_real_input)
            assert response.status_code == 422

@pytest.mark.asyncio
async def test_model_info_endpoint_returns_200():
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/model-info")
            assert response.status_code == 200
            assert "model_name" in response.json()

@pytest.mark.asyncio
async def test_model_comparison_endpoint_returns_200():
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/model-comparison")
            assert response.status_code == 200

@pytest.mark.asyncio
async def test_analytics_endpoint_returns_200():
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/analytics")
            assert response.status_code == 200
            data = response.json()
            assert "total_analyzed" in data
            assert "actual_claim_rate" in data
            assert "risk_category_distribution" in data

@pytest.mark.asyncio
async def test_monitoring_endpoint_returns_200():
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/monitoring")
            assert response.status_code == 200
            data = response.json()
            assert "overall_status" in data
            assert "performance_metrics" in data
            assert data["performance_metrics"]["test_roc_auc"] > 0

@pytest.mark.asyncio
async def test_registry_endpoint_returns_200():
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/registry")
            assert response.status_code == 200
            data = response.json()
            assert "champion" in data

@pytest.mark.asyncio
async def test_batch_predict_valid_csv(sample_real_input):
    df = pd.DataFrame([sample_real_input, sample_real_input])
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/batch-predict",
                files={"file": ("batch.csv", io.BytesIO(csv_bytes), "text/csv")}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["successful_predictions"] == 2

@pytest.mark.asyncio
async def test_cost_analysis_endpoint():
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/cost-analysis", json={"fp_cost": 500.0, "fn_cost": 5000.0})
            assert response.status_code == 200
            data = response.json()
            assert "optimal_threshold" in data
            assert "min_expected_cost" in data
