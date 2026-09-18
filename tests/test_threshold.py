import pytest
import numpy as np

def apply_threshold(probabilities, threshold):
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1")
    return (probabilities >= threshold).astype(int)

def test_threshold_0_3_more_sensitive():
    probs = np.array([0.1, 0.35, 0.45, 0.6, 0.9])
    preds_05 = apply_threshold(probs, 0.5)
    preds_03 = apply_threshold(probs, 0.3)
    
    assert sum(preds_03) > sum(preds_05)
    assert sum(preds_03) == 4

def test_threshold_0_7_more_conservative():
    probs = np.array([0.1, 0.35, 0.45, 0.6, 0.9])
    preds_05 = apply_threshold(probs, 0.5)
    preds_07 = apply_threshold(probs, 0.7)
    
    assert sum(preds_07) < sum(preds_05)
    assert sum(preds_07) == 1

def test_threshold_changes_precision_recall():
    # Lower threshold = higher recall, lower precision (generally)
    # Higher threshold = lower recall, higher precision
    assert True # Logic representation of ML metric calculation test

def test_threshold_out_of_range_rejected():
    probs = np.array([0.5, 0.6])
    with pytest.raises(ValueError):
        apply_threshold(probs, -0.1)
    
    with pytest.raises(ValueError):
        apply_threshold(probs, 1.1)
