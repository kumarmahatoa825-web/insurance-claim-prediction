# InsureAI — Model Governance, Registry Lifecycle & Compliance Framework

## 1. Single Source of Truth Architecture

InsureAI enforces rigorous architectural isolation to eliminate schema drift, metric ambiguity, and unauthorized configuration overrides.

```
                    ┌─────────────────────────────────────────┐
                    │    schemas/vehicle_insurance_v1.json    │
                    │         (Canonical Schema Spec)         │
                    └────────────────────┬────────────────────┘
                                         │ SHA-256 Hash
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │       schemas/schema_version.json       │
                    │   document_sha256 & feature_set_sha256  │
                    └────────────────────┬────────────────────┘
                                         │ Validated at Training & Inference
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │           models/current.json           │
                    │   Authoritative Active Champion Pointer │
                    │   {"champion_version": "v4.1.1-..."}    │
                    └────────────────────┬────────────────────┘
                                         │ Resolves to Immutable Release
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │   models/releases/v4.1.1-20260919.../   │
                    │   ├── model.joblib                      │
                    │   ├── metadata.json                     │
                    │   ├── calibration.json                  │
                    │   ├── global_shap_importance.json       │
                    │   ├── global_mdi_importance.json        │
                    │   ├── release_manifest.json             │
                    │   ├── roc_curve.json / pr_curve.json    │
                    │   └── checksums.json (SHA-256 Hashes)   │
                    └─────────────────────────────────────────┘
```

---

## 2. Model Lifecycle & Promotion Gates

Every model artifact must progress through deterministic, auditable promotion gates before serving live traffic:

### 2.1 Candidate Phase
1. **Training & Registration:** Models are trained using `backend.ml.train_real` and registered strictly as `candidate` status with `approval_status="pending"` inside `models/releases/<version>/`.
2. **Cryptographic Checksum Generation:** SHA-256 checksums are generated for all artifacts in the release directory and recorded in `checksums.json`.

### 2.2 Mandatory Promotion Quality Gates
To be eligible for promotion to Champion, a candidate must pass all 7 automated checks:
- **Artifact Integrity:** All files recorded in `checksums.json` exist and match their SHA-256 hashes.
- **Schema Compatibility:** Feature set SHA-256 matches canonical `schemas/schema_version.json`.
- **Calibration Standard:** Cross-fitted Brier Score $\le 0.150$ and below the no-skill baseline.
- **Discrimination Standards:** Out-of-Fold PR-AUC $\ge 0.080$ and ROC-AUC $\ge 0.550$.
- **Fairness Assessment:** Subgroup disparate impact and equal opportunity disparities within documented tolerance limits.
- **Non-Regression:** Candidate must meet or exceed the performance of the incumbent Champion on cross-validated development metrics.

### 2.3 Promotion & Rollback Protocol
- **Explicit Approval:** Requires human-in-the-loop sign-off:
  ```powershell
  python scripts/model_registry.py approve --version v4.1.1-20260919-7771382
  ```
- **Atomic Promotion:** Promotion atomically writes `models/current.json` to point to the new version and appends an entry to `deployment_history` in `models/registry.json`. No existing release files are ever modified or overwritten.
  ```powershell
  python scripts/model_registry.py promote --version v4.1.1-20260919-7771382
  ```
- **Pointer-Only Rollback:** Rollback updates `models/current.json` to the previous active champion in `deployment_history` without copying files:
  ```powershell
  python scripts/model_registry.py rollback
  ```

---

## 3. Decision Threshold Governance

- **Single Source of Truth:** The operating threshold (**`0.083`**) is determined via calibrated Out-of-Fold (OOF) F1 maximization on the development set and stored in `metadata.json["default_threshold"]`.
- **No Client Overrides:** The `/predict` and `/batch-predict` endpoints strictly reject client-supplied threshold parameters with HTTP 422.
- **Deterministic Evaluation:** All downstream decision labels (`Likely to Claim` vs `Unlikely to Claim`) are derived exclusively using the server-side authoritative threshold.

---

## 4. Multi-Feature Data Drift Monitoring

InsureAI monitors live production traffic against the sealed development baseline (`data/processed/drift_baseline.json`, 46,873 rows):

1. **Prediction Drift:** Population Stability Index (PSI) computed on predicted claim probabilities against the OOF prediction distribution.
   - $\text{PSI} < 0.10$: Stable / No significant drift.
   - $0.10 \le \text{PSI} < 0.25$: Moderate drift / Alert triggered.
   - $\text{PSI} \ge 0.25$: Severe drift / Action required.
2. **Continuous Feature Drift:** 6 key numerical features (`policy_tenure`, `age_of_car`, `age_of_policyholder`, `population_density`, `displacement`, `gross_weight`) evaluated against 5-quantile baseline bins using PSI.
3. **Categorical Frequency Shift:** 5 categorical attributes (`make`, `segment`, `fuel_type`, `transmission_type`, `area_cluster`) monitored via proportional divergence.
4. **Sensor Dropout & Missingness Tracking:** Real-time missingness rates compared per-feature against baseline missingness rates.
5. **Volume Gating:** Statistical drift calculation is gated to activate after $\ge 10$ live production predictions to avoid small-sample volatility.

---

## 5. Responsible AI & Underwriting Disclaimers

### 5.1 Human-in-the-Loop Requirement
Automated predictions from this system are probabilistic risk indicators designed to assist human underwriters, not replace them. Adverse underwriting decisions (such as policy declination or substantial surcharge) must never be executed solely on automated model outputs.

### 5.2 Educational & Portfolio Framing
This project is an advanced, production-oriented portfolio and academic demonstration system. While it demonstrates enterprise-grade MLOps architecture, real-world commercial insurance production requires proprietary carrier actuarial loss models, DOI regulatory approvals, and insured loss history data.
