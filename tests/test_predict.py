import os
import pytest
from backend.ml.predict import PredictionEngine
from backend.core.settings import settings

@pytest.fixture(scope="module")
def loaded_engine():
    from backend.ml.registry import ModelRegistry
    reg = ModelRegistry(settings.MODELS_DIR)
    try:
        model_path, meta_path = reg.get_active_champion_paths()
    except Exception as e:
        pytest.skip(f"No active champion found in registry: {e}")
    engine = PredictionEngine()
    engine.load_model(model_path)
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

