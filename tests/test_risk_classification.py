import pytest
from backend.ml.predict import PredictionEngine

def test_risk_score_low():
    score = PredictionEngine.compute_risk_score(0.15)
    category = PredictionEngine.get_risk_category(score)
    assert score == 15
    assert category == 'Low'

def test_risk_score_moderate():
    score = PredictionEngine.compute_risk_score(0.35)
    category = PredictionEngine.get_risk_category(score)
    assert score == 35
    assert category == 'Moderate'

def test_risk_score_high():
    score = PredictionEngine.compute_risk_score(0.65)
    category = PredictionEngine.get_risk_category(score)
    assert score == 65
    assert category == 'High'

def test_risk_score_critical():
    score = PredictionEngine.compute_risk_score(0.85)
    category = PredictionEngine.get_risk_category(score)
    assert score == 85
    assert category == 'Critical'

@pytest.mark.parametrize("prob,expected_cat", [
    (0.25, 'Low'),
    (0.26, 'Moderate'),
    (0.50, 'Moderate'),
    (0.51, 'High'),
    (0.75, 'High'),
    (0.76, 'Critical')
])
def test_boundary_values(prob, expected_cat):
    score = PredictionEngine.compute_risk_score(prob)
    category = PredictionEngine.get_risk_category(score)
    assert category == expected_cat

def test_prediction_label_above_threshold():
    label = PredictionEngine.get_prediction_label(0.6, 0.5)
    assert label == "Likely to Claim"

def test_prediction_label_below_threshold():
    label = PredictionEngine.get_prediction_label(0.4, 0.5)
    assert label == "Unlikely to Claim"

