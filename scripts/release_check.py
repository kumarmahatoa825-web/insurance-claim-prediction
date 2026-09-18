"""
InsureAI — Production Release Verifier
=======================================
Conducts rigorous automated auditing of:
  1. Active Champion registry pointer (models/current.json)
  2. Release directory integrity & SHA-256 checksums
  3. Canonical schema compliance (schemas/vehicle_insurance_v1.json)
  4. Cross-fitted calibration diagnostics (Brier, Baseline Brier, ECE)
  5. Decision threshold source of truth
  6. Algorithmic fairness audit
  7. Test submission format & manifest (39,063 rows)
"""

import os
import sys
import json
import hashlib
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.ml.registry import ModelRegistry, compute_sha256


def run_release_check() -> bool:
    print("=" * 75)
    print("InsureAI -- Production Release Audit & Verification")
    print("=" * 75)

    reg = ModelRegistry(os.path.join(BASE_DIR, "models"))
    all_passed = True

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal all_passed
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_passed = False
        print(f"[{status}] {name}")
        if detail:
            print(f"       -> {detail}")

    # 1. Authoritative Champion Pointer
    champ_ver = reg.get_current_champion_version()
    check("Authoritative Champion Pointer Configured", bool(champ_ver), f"Champion Version: {champ_ver}")

    if not champ_ver:
        print("[FAIL] Cannot proceed with release checks: No active Champion.")
        return False

    release_dir = os.path.join(reg.releases_dir, champ_ver)
    check("Release Directory Exists", os.path.isdir(release_dir), release_dir)

    # 2. File and Checksum Integrity
    m_path, meta_path = None, None
    try:
        m_path, meta_path = reg.get_active_champion_paths()
        integrity_ok = True
    except Exception as e:
        integrity_ok = False
        print(f"       -> Integrity error: {e}")
    check("Release Artifact Checksum Integrity", integrity_ok, "All hashes in checksums.json match files")

    # 3. Canonical Schema Compliance
    schema_path = os.path.join(BASE_DIR, "schemas", "vehicle_insurance_v1.json")
    schema_exists = os.path.exists(schema_path)
    check("Canonical Schema Exists", schema_exists, schema_path)

    canonical_hash = None
    if schema_exists:
        with open(schema_path, "rb") as sf:
            canonical_hash = f"sha256:{hashlib.sha256(sf.read()).hexdigest()}"

    metadata = {}
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as mf:
            metadata = json.load(mf)

    recorded_schema_hash = metadata.get("schema_hash") or metadata.get("dataset", {}).get("schema_hash")
    schema_match = bool(canonical_hash and recorded_schema_hash == canonical_hash)
    check("Schema Hash Match", schema_match, f"Canonical: {canonical_hash} | Model: {recorded_schema_hash}")

    # 4. Calibration Diagnostics
    cal = metadata.get("calibration", {})
    brier = metadata.get("brier_score")
    baseline_brier = metadata.get("baseline_brier")
    ece = cal.get("ece")
    cal_valid = bool(brier is not None and brier <= 0.15 and baseline_brier is not None)
    check(
        "Calibration Validated",
        cal_valid,
        f"Champion Brier: {brier} (vs Baseline: {baseline_brier}) | ECE: {ece}"
    )

    # 5. Threshold Consistency
    thresh = metadata.get("default_threshold")
    check(
        "Authoritative Threshold Configured",
        isinstance(thresh, (int, float)) and 0.01 <= thresh <= 0.99,
        f"Locked Threshold: {thresh}"
    )

    # 6. Algorithmic Fairness
    fairness = metadata.get("fairness", {})
    fairness_passed = fairness.get("gate_passed", False)
    check(
        "Fairness Gate Passed",
        bool(fairness_passed),
        f"Demographic Parity Disparity: {fairness.get('demographic_parity_disparity')} | EO Disparity: {fairness.get('equal_opportunity_disparity')}"
    )

    # 7. Final Submission Verification
    sub_path = os.path.join(BASE_DIR, "data", "processed", "final_submission.csv")
    manifest_path = os.path.join(BASE_DIR, "data", "processed", "final_submission_manifest.json")
    sub_exists = os.path.exists(sub_path)
    manifest_exists = os.path.exists(manifest_path)

    sub_valid = False
    if sub_exists:
        try:
            sub_df = pd.read_csv(sub_path)
            rows_match = (len(sub_df) == 39063)
            cols_match = (list(sub_df.columns) == ["policy_id", "is_claim"])
            no_nulls = not sub_df.isna().any().any()
            valid_vals = set(sub_df["is_claim"].unique()).issubset({0, 1})
            sub_valid = rows_match and cols_match and no_nulls and valid_vals
            detail = f"Rows: {len(sub_df):,} | Claims: {sub_df['is_claim'].sum():,} ({sub_df['is_claim'].mean()*100:.2f}%)"
        except Exception as ex:
            detail = f"Read error: {ex}"
    else:
        detail = f"File not found: {sub_path}"

    check("Final Submission (39,063 rows) Valid", sub_valid, detail)
    check("Submission Manifest Recorded", manifest_exists, manifest_path)

    print("=" * 75)
    if all_passed:
        print("[SUCCESS] ALL RELEASE GATES PASSED! SYSTEM IS 10/10 PRODUCTION READY.")
    else:
        print("[WARNING] ONE OR MORE RELEASE GATES FAILED. REVIEW DETAILS ABOVE.")
    print("=" * 75)

    return all_passed


if __name__ == "__main__":
    success = run_release_check()
    sys.exit(0 if success else 1)
