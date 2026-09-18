import os
import pytest
from backend.ml.predict import PredictionEngine
from backend.core.settings import settings

@pytest.fixture(scope="module")
def loaded_engine():
    engine = PredictionEngine()
    model_path = settings.CHAMPION_MODEL_PATH if os.path.exists(settings.CHAMPION_MODEL_PATH) else settings.LEGACY_MODEL_PATH
    if not os.path.exists(model_path):
        pytest.skip("No trained model found. Run training first.")
    engine.load_model(model_path)
    meta_path = settings.CHAMPION_METADATA_PATH if os.path.exists(settings.CHAMPION_METADATA_PATH) else os.path.join(settings.MODELS_DIR, "model_metadata.json")
    engine.load_metadata(meta_path)
    return engine

def test_predict_single_returns_required_fields(loaded_engine, sample_real_input):
    result = loaded_engine.predict_single(sample_real_input)
    required = ["prediction", "claim_probability", "risk_score", "risk_category", "risk_profile", "recommendations", "model_version"]
    for req in required:
        assert req in result

def test_predict_batch_returns_list(loaded_engine, sample_df):
    result = loaded_engine.predict_batch(sample_df)
    assert isinstance(result, list)
    assert len(result) == len(sample_df)

def test_non_claim_probability_sums_to_one(loaded_engine, sample_real_input):
    result = loaded_engine.predict_single(sample_real_input)
    assert abs(result["claim_probability"] + result["non_claim_probability"] - 1.0) < 1e-4

def test_risk_profile_is_string(loaded_engine, sample_real_input):
    result = loaded_engine.predict_single(sample_real_input)
    assert isinstance(result["risk_profile"], str)

def test_recommendations_is_list(loaded_engine, sample_real_input):
    result = loaded_engine.predict_single(sample_real_input)
    assert isinstance(result["recommendations"], list)

def test_model_version_present(loaded_engine, sample_real_input):
    result = loaded_engine.predict_single(sample_real_input)
    assert "model_version" in result
    assert isinstance(result["model_version"], str)

