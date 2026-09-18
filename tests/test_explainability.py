import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from backend.ml.explain import ExplainabilityEngine


def test_uninitialized_explainer_returns_available_false():
    engine = ExplainabilityEngine()
    # Uninitialized engine must return available=False without heuristics
    result = engine.explain_single({"feature1": 10})
    assert result["available"] is False
    assert "unavailable" in result.get("reason", "").lower() or "not initialized" in result.get("reason", "").lower()
    assert result["shap_values"] == {}
    assert result["top_risk_factors"] == []
    assert result["protective_factors"] == []


def test_tree_explainer_computes_valid_factors():
    # Train a minimal tree model
    X = pd.DataFrame({
        "feat_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feat_b": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    })
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)

    engine = ExplainabilityEngine()
    engine.setup(tree, X, feature_names=["feat_a", "feat_b"])

    assert engine.explainer is not None

    # Test single prediction explanation
    input_sample = {"feat_a": 9, "feat_b": 2}
    result = engine.explain_single(input_sample)

    assert result["available"] is True
    assert "shap_values" in result
    assert len(result["shap_values"]) == 2
    # Total factors should match
    total_factors = len(result["top_risk_factors"]) + len(result["protective_factors"])
    assert total_factors > 0


def test_global_importance_separation():
    X = pd.DataFrame({
        "feat_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feat_b": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    })
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)

    engine = ExplainabilityEngine()
    engine.setup(tree, X, feature_names=["feat_a", "feat_b"])

    # Global MDI/Gini importance from tree step
    mdi_imp = engine.get_global_model_importance(tree, ["feat_a", "feat_b"])
    assert len(mdi_imp) == 2
    assert "feature" in mdi_imp[0]
    assert "importance" in mdi_imp[0]
