"""
test_active_champion_integration.py
===================================
End-to-end integration and immutability tests for the active Champion model release.
Locks in:
  - Active version pointer (v4.1.0-20260918-6640e9b)
  - Release artifact cryptographic hash integrity (checksums.json)
  - Single source of truth decision threshold (0.080)
  - Cross-fitted calibration standards (Brier score < 0.060)
  - Final test submission volume and hash verification
"""

import os
import json
import hashlib
import pandas as pd
import pytest

from backend.ml.registry import ModelRegistry
from backend.ml.predict import PredictionEngine
from backend.core.settings import settings


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry()


@pytest.fixture(scope="module")
def active_release_info(registry):
    version = registry.get_champion_version()
    assert version is not None
    model_path, meta_path = registry.get_active_champion_paths()
    release_dir = os.path.dirname(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(release_dir, "checksums.json"), "r", encoding="utf-8") as f:
        checksums = json.load(f)
    return {
        "version": version,
        "model_path": model_path,
        "meta_path": meta_path,
        "release_dir": release_dir,
        "metadata": meta,
        "checksums": checksums,
    }


def test_active_champion_version_pointer(active_release_info):
    """Verify active Champion version pointer is v4.1.0-20260918-6640e9b."""
    assert active_release_info["version"] == "v4.1.0-20260918-6640e9b"


def test_active_release_sha256_checksums(active_release_info):
    """Verify all release files match their exact SHA-256 checksums."""
    release_dir = active_release_info["release_dir"]
    checksums = active_release_info["checksums"]

    for fname, expected_hash in checksums.items():
        fpath = os.path.join(release_dir, fname)
        assert os.path.exists(fpath), f"Release artifact missing: {fpath}"
        with open(fpath, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        norm_expected = expected_hash.replace("sha256:", "")
        assert actual_hash == norm_expected, f"Hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}"


def test_active_threshold_governance(active_release_info):
    """Verify authoritative threshold is 0.080 and defined in metadata."""
    meta = active_release_info["metadata"]
    assert meta.get("default_threshold") == 0.08
    if "threshold" in meta and isinstance(meta["threshold"], dict):
        assert meta["threshold"].get("selected") == 0.08


def test_calibration_brier_standards(active_release_info):
    """Verify cross-fitted Platt calibration achieves Brier < 0.060 and beats baseline."""
    meta = active_release_info["metadata"]
    cal = meta.get("calibration", {})
    assert cal.get("cross_fitted") is True
    assert cal.get("brier_score") < 0.060
    assert cal.get("brier_score") < cal.get("baseline_brier")


def test_prediction_engine_uses_authoritative_threshold(sample_real_input):
    """Verify PredictionEngine evaluates prediction using active 0.080 threshold."""
    engine = PredictionEngine()
    engine.load_model()
    result = engine.predict_single(sample_real_input)
    assert result["threshold_used"] == 0.08
    assert result["model_version"] == "v4.1.0-20260918-6640e9b"
    assert "claim_probability" in result
    assert 0.0 <= result["claim_probability"] <= 1.0


def test_final_submission_manifest_integrity():
    """Verify final_submission.csv has 39,063 rows and matches manifest hash."""
    sub_path = os.path.join("data", "processed", "final_submission.csv")
    manifest_path = os.path.join("data", "processed", "final_submission_manifest.json")

    assert os.path.exists(sub_path), "final_submission.csv does not exist"
    assert os.path.exists(manifest_path), "final_submission_manifest.json does not exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(sub_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    expected_hash = manifest.get("sha256") or manifest.get("submission_sha256")
    norm_expected = expected_hash.replace("sha256:", "")
    assert actual_hash == norm_expected

    df = pd.read_csv(sub_path)
    assert len(df) == 39063
    assert list(df.columns) == ["policy_id", "is_claim"]
    assert set(df["is_claim"].unique()).issubset({0, 1})
