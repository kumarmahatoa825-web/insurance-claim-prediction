import os
import json
import pytest
import numpy as np
import pandas as pd
from backend.ml.monitoring import ModelMonitor, calculate_psi


def test_calculate_psi_identical():
    # Identical distributions should have near-zero PSI
    baseline = np.random.beta(0.5, 5, size=1000)
    current = np.random.beta(0.5, 5, size=1000)
    result = calculate_psi(baseline, current, num_buckets=10)
    assert result.status == "ok"
    assert result.value is not None
    assert result.value < 0.10  # Healthy drift


def test_calculate_psi_shifted():
    # Drastically shifted distributions should have high PSI
    baseline = np.random.uniform(0.0, 0.1, size=500)
    current = np.random.uniform(0.7, 1.0, size=500)
    result = calculate_psi(baseline, current, num_buckets=10)
    assert result.status == "ok"
    assert result.value is not None
    assert result.value > 0.25  # Significant drift


def test_monitor_insufficient_volume():
    monitor = ModelMonitor(metadata={"version": "v3.0-test", "default_threshold": 0.08})
    # Test with < 10 history rows
    history = [
        {"claim_probability": 0.05, "prediction": "Unlikely to Claim"}
        for _ in range(5)
    ]
    report = monitor.get_monitoring_report(history)
    drift = report["drift_monitoring"]
    assert drift["status"] in ["INSUFFICIENT_DATA", "Baseline unavailable"]
    assert drift["prediction_drift_psi"] is None


def test_monitor_sufficient_volume_runs_real_psi(monkeypatch, tmp_path):
    # Mock baseline probabilities
    mock_baseline = np.random.beta(0.5, 5, size=500)
    import backend.ml.monitoring as mon_mod
    monkeypatch.setattr(mon_mod, "get_baseline_probabilities", lambda: mock_baseline)

    monitor = ModelMonitor(metadata={"version": "v3.0-test", "default_threshold": 0.08})

    # 500 production rows meet the minimum required for drift calculation.
    history = [
        {
            "claim_probability": float(np.random.beta(0.5, 5)),
            "prediction": "Unlikely to Claim",
            "features": {"policy_tenure": 0.3},
        }
        for _ in range(500)
    ]
    report = monitor.get_monitoring_report(history)
    drift = report["drift_monitoring"]
    assert drift["status"] == "Active"
    assert drift["prediction_drift_psi"] is not None
    assert drift["production_volume"] == 500
