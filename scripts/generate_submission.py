"""
InsureAI — Final Test Set Submission Generator
===============================================
Executes final test set inference on data/real/test.csv (39,063 unseen policies)
using the locked Champion model from the authoritative ModelRegistry (current.json).
Outputs strictly compliant submission to data/processed/final_submission.csv.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SubmissionGenerator")


def generate_submission(
    test_path: str = None,
    output_path: str = None,
    threshold: float = None,
) -> pd.DataFrame:
    test_path = test_path or os.path.join(settings.BASE_DIR, "data", "real", "test.csv")
    output_path = output_path or os.path.join(settings.BASE_DIR, "data", "processed", "final_submission.csv")

    logger.info(f"Loading test dataset from {test_path}...")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    df_test = pd.read_csv(test_path)
    logger.info(f"Loaded {len(df_test):,} rows with {df_test.shape[1]} columns.")

    if "policy_id" not in df_test.columns:
        raise ValueError("test.csv must contain 'policy_id' column")

    policy_ids = df_test["policy_id"]
    X_test = df_test.drop(columns=["policy_id"])
    if "is_claim" in X_test.columns:
        X_test = X_test.drop(columns=["is_claim"])

    # Load Champion model & metadata strictly via Registry current.json pointer
    from backend.ml.registry import ModelRegistry
    reg = ModelRegistry(os.path.join(settings.BASE_DIR, "models"))
    model_path, meta_path = reg.get_active_champion_paths()

    logger.info(f"Loading Champion model from {model_path}...")
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict):
        preprocessor = loaded.get("preprocessor")
        classifier = loaded.get("model")
    elif hasattr(loaded, "__getitem__"):
        preprocessor = loaded[:-1]
        classifier = loaded[-1]
    else:
        preprocessor = None
        classifier = loaded

    if threshold is None:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        threshold = meta.get("default_threshold")
        if threshold is None and isinstance(meta.get("threshold"), dict):
            threshold = meta["threshold"].get("selected")
        if threshold is None:
            raise RuntimeError("Active model threshold is missing from metadata.")

    logger.info(f"Using locked decision threshold: {float(threshold):.3f}")

    logger.info("Computing prediction probabilities on unseen test dataset...")
    try:
        # CalibratedClassifierCV wrapping base pipeline expects raw DataFrame
        probabilities = classifier.predict_proba(X_test)[:, 1]
    except Exception:
        logger.info("Direct inference failed; transforming test dataset through preprocessor...")
        if preprocessor is not None:
            X_test_proc = preprocessor.transform(X_test)
        else:
            X_test_proc = X_test

        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba(X_test_proc)[:, 1]
        elif hasattr(classifier, "decision_function"):
            raw_scores = classifier.decision_function(X_test_proc)
            probabilities = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            probabilities = classifier.predict(X_test_proc).astype(float)

    # Apply decision threshold to get binary claim predictions
    is_claim_preds = (probabilities >= threshold).astype(int)

    # Format dataframe to match sample_submission.csv exactly
    df_submission = pd.DataFrame({
        "policy_id": policy_ids,
        "is_claim": is_claim_preds
    })

    # Verification checks
    assert len(df_submission) == len(df_test), f"Row count mismatch: {len(df_submission)} vs {len(df_test)}"
    assert list(df_submission.columns) == ["policy_id", "is_claim"], f"Invalid columns: {df_submission.columns}"
    assert not df_submission.isna().any().any(), "Submission contains null/NaN values!"
    assert set(df_submission["is_claim"].unique()).issubset({0, 1}), "Invalid values in is_claim column!"

    # Create destination dir if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_submission.to_csv(output_path, index=False)
    logger.info(f"Successfully saved final submission to {output_path}")

    # Compute submission file hash
    import hashlib
    h = hashlib.sha256()
    with open(output_path, "rb") as f_sub:
        for chunk in iter(lambda: f_sub.read(65536), b""):
            h.update(chunk)
    sub_hash = f"sha256:{h.hexdigest()}"

    # Log summary statistics
    claim_count = int(is_claim_preds.sum())
    total_count = len(is_claim_preds)
    claim_pct = (claim_count / total_count) * 100.0
    avg_prob = float(np.mean(probabilities))

    logger.info("--- Submission Summary ---")
    logger.info(f"Total Policies: {total_count:,}")
    logger.info(f"Predicted Claims: {claim_count:,} ({claim_pct:.2f}%)")
    logger.info(f"Predicted Non-Claims: {total_count - claim_count:,} ({100.0 - claim_pct:.2f}%)")
    logger.info(f"Average Predicted Probability: {avg_prob:.4f}")
    logger.info(f"Submission SHA-256: {sub_hash}")
    logger.info("--------------------------")

    # Save manifest
    import datetime
    manifest_path = os.path.join(os.path.dirname(output_path), "final_submission_manifest.json")
    manifest = {
        "submission_file": os.path.basename(output_path),
        "total_rows": total_count,
        "predicted_claims": claim_count,
        "predicted_claim_rate": round(claim_pct / 100.0, 4),
        "average_predicted_probability": round(avg_prob, 4),
        "threshold_used": float(threshold),
        "active_champion_version": reg.get_current_champion_version(),
        "model_path": model_path,
        "sha256": sub_hash,
        "generated_at": datetime.datetime.now().isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f_m:
        json.dump(manifest, f_m, indent=2)
    logger.info(f"Saved submission manifest to {manifest_path}")

    return df_submission


if __name__ == "__main__":
    generate_submission()
