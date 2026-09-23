import os
import json
import pytest
from backend.ml.registry import ModelRegistry, MANDATORY_PROMOTION_REQUIREMENTS


@pytest.fixture
def temp_registry(tmp_path):
    reg_dir = str(tmp_path / "models")
    return ModelRegistry(reg_dir)


def test_initial_registry_empty(temp_registry):
    assert temp_registry.get_champion() is None
    assert len(temp_registry.get_candidates()) == 0
    assert temp_registry.get_current_champion_version() is None


def test_register_candidate_pending(temp_registry, tmp_path):
    dummy_model = tmp_path / "dummy.joblib"
    dummy_model.write_text("dummy")

    cand = temp_registry.register_candidate(
        model_name="Random Forest",
        version="v3.1-cand",
        model_obj_or_path=str(dummy_model),
        metadata={
            "default_threshold": 0.08,
            "test_metrics": {"f1": 0.162, "roc_auc": 0.639, "brier_score": 0.059},
        },
    )
    assert cand["status"] == "candidate"
    assert cand["approval_status"] == "pending"
    assert len(temp_registry.get_candidates()) == 1
    assert temp_registry.get_champion() is None


def test_promotion_requires_approval(temp_registry, tmp_path):
    dummy_model = tmp_path / "dummy.joblib"
    dummy_model.write_text("dummy")

    temp_registry.register_candidate(
        model_name="Random Forest",
        version="v3.1-cand",
        model_obj_or_path=str(dummy_model),
        metadata={
            "default_threshold": 0.08,
            "test_metrics": {"f1": 0.162, "roc_auc": 0.639, "brier_score": 0.059},
        },
    )

    # Promoting without approval must raise ValueError
    with pytest.raises(ValueError, match="not approved"):
        temp_registry.promote_candidate("v3.1-cand")


def test_promotion_gate_enforcement(temp_registry, tmp_path):
    dummy_model = tmp_path / "dummy.joblib"
    dummy_model.write_text("dummy")

    # Register candidate that fails F1 gate (min 0.10)
    temp_registry.register_candidate(
        model_name="Bad Model",
        version="v_bad",
        model_obj_or_path=str(dummy_model),
        metadata={
            "default_threshold": 0.08,
            "test_metrics": {"f1": 0.02, "roc_auc": 0.50, "brier_score": 0.25},
        },
    )
    temp_registry.approve_candidate("v_bad")

    with pytest.raises(ValueError, match="mandatory promotion checks"):
        temp_registry.promote_candidate("v_bad")


def test_approve_promote_and_atomic_pointer(temp_registry, tmp_path):
    m1 = tmp_path / "m1.joblib"
    m1.write_text("model 1")

    # Fairness metadata is required (fail-closed gate) — include it in all test candidates
    fairness_meta = {"gate_passed": True, "demographic_parity_disparity": 0.05, "eo_disparity": 0.04}

    # Initial champion via register_candidate -> approve -> promote
    temp_registry.register_candidate(
        model_name="Model A",
        version="v1.0",
        model_obj_or_path=str(m1),
        metadata={"default_threshold": 0.08, "data_mode": "real", "feature_count": 86,
                  "test_metrics": {"f1": 0.15, "roc_auc": 0.60, "pr_auc": 0.09, "brier_score": 0.06},
                  "fairness": fairness_meta},
    )
    temp_registry.approve_candidate("v1.0")
    c1 = temp_registry.promote_candidate("v1.0")
    assert temp_registry.get_current_champion_version() == "v1.0"
    m_path, meta_path = temp_registry.get_active_champion_paths()
    assert os.path.exists(m_path)
    assert os.path.exists(meta_path)

    # Candidate 2
    m2 = tmp_path / "m2.joblib"
    m2.write_text("model 2")
    temp_registry.register_candidate(
        model_name="Model B",
        version="v2.0",
        model_obj_or_path=str(m2),
        metadata={"default_threshold": 0.08, "data_mode": "real", "feature_count": 86,
                  "test_metrics": {"f1": 0.18, "roc_auc": 0.65, "pr_auc": 0.10, "brier_score": 0.055},
                  "fairness": fairness_meta},
    )

    temp_registry.approve_candidate("v2.0")
    c2 = temp_registry.promote_candidate("v2.0")
    assert c2["version"] == "v2.0"
    assert temp_registry.get_current_champion_version() == "v2.0"

    # Rollback to v1.0
    restored = temp_registry.rollback()
    assert restored["version"] == "v1.0"
    assert temp_registry.get_current_champion_version() == "v1.0"
