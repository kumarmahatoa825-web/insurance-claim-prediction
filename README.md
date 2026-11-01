# InsureAI — Intelligent Insurance Claim Prediction & Risk Analysis System

> **Academic/Portfolio AI/ML System** trained on a **real-world vehicle insurance dataset** (58,592 labeled records, 39,063 official unseen test records) featuring nested cross-fitted probability calibration, Out-of-Fold (OOF) decision threshold optimization, SHAP tree explainability, immutable Model Registry with human-in-the-loop approval gates, and multi-feature Population Stability Index (PSI) drift monitoring.
>
> **⚠️ Scope**: This is an educational portfolio project demonstrating production ML engineering concepts. It is not deployed as a production insurance system. Predictions are probabilistic estimates only and must not be used as the sole basis for real insurance decisions. See [Known Limitations](#known-limitations) below.

---

## 🎯 System Architecture & Data Flow

```
                                  REAL-WORLD DATASET
                                (data/real/train.csv)
                                     58,592 rows
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
          Development Set (80%)                         Locked Holdout (20%)
              46,873 rows                                   11,719 rows
                    │                                           │
      ┌─────────────┴─────────────┐                             │
      ▼                           ▼                             │
Fold-Safe Preprocessing     5-Fold Stratified CV                │
 (RealInsurancePreprocessor) (Scale-Pos-Weight inner fit)       │
      │                           │                             │
      └─────────────┬─────────────┘                             │
                    ▼                                           │
         Nested Calibrated CV (cv=3)                            │
                    │                                           │
                    ▼                                           │
          Out-of-Fold (OOF) Probs                               │
                    │                                           │
                    ▼                                           │
          OOF Threshold Tuning                                  │
      (Optimal F1 on calibrated OOF)                            │
                    │                                           │
                    ▼                                           │
          TRAIN FINAL CHAMPION                                  │
       (All 46,873 development rows)                            │
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                               UNSEEN HOLDOUT EVALUATION
                                  (11,719 holdout rows)
                                          │
                                          ▼
                                    MODEL REGISTRY
                        (models/releases/v4.1.1-20260919-7771382/)
                                   Checksum Verification
                                  Approval Required Gate
                                          │
                                          ▼
                              ATOMIC POINTER PROMOTION
                                (models/current.json)
                                          │
                                          ▼
                               OFFICIAL TEST INFERENCE
                                 (data/real/test.csv)
                                     39,063 rows
                                          │
                                          ▼
                              FINAL SUBMISSION CSV
                     (data/processed/final_submission.csv)
```

---

## 🚀 Quick Start

### 0. Install dependencies (Windows PowerShell)
Use Python 3.10 for the saved model artifacts. Run these commands from the project root:
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 1. Backend (FastAPI + ML Model)
Run from the project root using the virtual environment's Python (do not use a global `uvicorn`):
```powershell
$env:PYTHONIOENCODING="utf-8"
$env:LOKY_MAX_CPU_COUNT="4"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000 --reload
```

### 2. Frontend (React + Vite + Tailwind)
```powershell
cd "insurance-claim-prediction/frontend"
npm run dev
```

### 3. Docker Compose (Production Environment)
```powershell
docker-compose up --build
```

### 4. Application URLs
| Service | URL | Description |
|---|---|---|
| **Web Dashboard** | `http://localhost:5173` | Interactive React application |
| **Backend API** | `http://localhost:8000` | REST API (FastAPI) |
| **Interactive Docs** | `http://localhost:8000/docs` | Swagger UI OpenAPI specification |
| **Health Liveness** | `http://localhost:8000/health/live` | Process liveness probe |
| **Health Readiness** | `http://localhost:8000/health/ready` | Model readiness probe |

---

## 📊 Dataset & Class Imbalance

| File | Records | Labels | Role |
|---|---|---|---|
| `data/real/train.csv` | 58,592 | `is_claim` (0 or 1) | ML Training + 5-Fold OOF CV (46,873 dev / 11,719 holdout) |
| `data/real/test.csv` | 39,063 | None (unlabelled) | Official unseen test set for submission |
| `data/real/sample_submission.csv` | 39,063 | Placeholder | Submission output template |
| `data/synthetic/insurance_claims.csv` | 5,000 | `claim_filed` | Synthetic baseline for stress-testing |

### Imbalance Characteristics:
- **Total Policies:** 58,592 labeled training rows
- **Claims Filed:** 3,748 (6.40% claim prevalence)
- **Imbalance Ratio:** 14.6 : 1
- **Handling Strategy:** Scale-Pos-Weight class-weighting applied **strictly inside training folds**, preventing data leakage or synthetic sample distortion.

---

## 🏆 Model Architecture & Probability Calibration

Cross-fitted sigmoid calibration using `CalibratedClassifierCV` with internal CV (`cv=3`), wrapping an inner pipeline of `RealInsurancePreprocessor` and `XGBClassifier(scale_pos_weight=14.63)`.
Cross-fitted sigmoid calibration fits sigmoid mapping on un-oversampled inner validation folds, guaranteeing that output probabilities reflect the natural 6.4% empirical prevalence.

### Model Selection Hierarchy:
1. **Primary Metric:** Out-of-Fold (OOF) PR-AUC (Precision-Recall Area Under Curve) $\ge 0.100$
2. **Secondary Metric:** OOF ROC-AUC
3. **Tertiary Metric:** OOF Brier Score
4. **Authoritative Threshold:** F1 score optimization is used exclusively for selecting the operating decision threshold on calibrated OOF probabilities (**0.083**), governed as a single source of truth in `metadata.json`.

### Final Holdout Evaluation (11,719 Sealed Rows) with 95% Bootstrap CIs:
- **ROC-AUC:** `0.6510` (95% CI: `[0.6319, 0.6707]`)
- **PR-AUC:** `0.1037` (95% CI: `[0.0928, 0.1157]`)
- **F1 Score:** `0.1672` (95% CI: `[0.1512, 0.1831]`)
- **Brier Score:** `0.0589` (vs Baseline: `0.0599`, ECE: `0.0052`)
- **Mean Probability:** `0.0640` (matches empirical prevalence `0.0640`)
- **Decile 1 Lift:** `1.97x` (Decile 1 claim rate 12.64% vs 6.40% prevalence)
- **Top 5% Triage:** Lift `2.03x` | Top 10% Triage: Lift `1.97x` | Top 20% Triage: Lift `1.76x`

---

## 📦 Test Set Submission Generator
Generate compliant predictions on unseen `data/real/test.csv` (39,063 records) using the authoritative Champion model:
```powershell
python scripts/generate_submission.py
```
Output saved to: `data/processed/final_submission.csv` (39,063 rows, verified against `sample_submission.csv` with cryptographic SHA-256 manifest).

---

## 🏛️ Model Governance & Registry Lifecycle

The Model Registry enforces strictly immutable releases and pointer-atomic state transitions:
1. **Immutable Releases:** All artifacts are stored in `models/releases/<version>/` (`model.joblib`, `metadata.json`, `calibration.json`, `global_shap_importance.json`, `global_mdi_importance.json`, `release_manifest.json`, and SHA-256 `checksums.json`).
2. **Mandatory Pointer:** `models/current.json` is the single source of truth for the active Champion (`v4.1.1-20260919-7771382`).
3. **Release Gate Verification Script:**
   ```powershell
   python scripts/release_check.py
   ```
4. **Approval & Promotion Commands:**
   ```powershell
   python scripts/model_registry.py approve --version <version>
   python scripts/model_registry.py promote --version <version>
   ```
5. **Pointer-Only Rollback:**
   ```powershell
   python scripts/model_registry.py rollback
   ```

---

## 📡 Production Data Drift Monitoring

Monitors 11 vehicle and policyholder features against the full 46,873 development record baseline (`data/processed/drift_baseline.json`):
- **Prediction Drift:** Population Stability Index (PSI) computed on predicted claim probabilities vs OOF baseline.
- **Continuous Feature Shift:** 6 features (`policy_tenure`, `age_of_car`, `age_of_policyholder`, `population_density`, `displacement`, `gross_weight`) evaluated against 5-quantile baseline bins.
- **Categorical Frequency Shift:** 5 features (`make`, `segment`, `fuel_type`, `transmission_type`, `area_cluster`) tracked by category distribution.
- **Real Missingness Tracking:** Real-time missingness rates compared per-feature against baseline missingness rates.
- **Volume Gating:** Statistical drift calculation activates after $\ge 10$ production predictions.

---

## 🔬 Explainable AI (SHAP & MDI)

The system clearly demarcates 3 distinct explainability perspectives:
1. **Global Model Feature Importance:** Mean Decrease in Impurity (Gini / MDI) from the base tree ensemble.
2. **Global SHAP Importance:** Precomputed mean absolute SHAP values across representative validation records.
3. **Local SHAP Attribution:** Individual policy waterfall explanations separating risk-increasing vs protective factors.
- **Attribution Note:** Local SHAP attributions explain the underlying base tree estimator, while the final displayed claim probability is produced by the cross-fitted calibrated model.
- **Zero Pseudo-SHAP:** If SHAP is unavailable, the API returns `{ "available": false }` without substituting heuristic proxies.

---

## 🧪 Verification & Automated Testing

Run the full automated test suite (83 unit, integration, and regression tests):
```powershell
pytest -v
```
Run release gate audit:
```powershell
python scripts/release_check.py
```
Verify frontend production build:
```powershell
cd frontend
npm run build
```
Verify active Champion:
```powershell
python -c "from backend.ml.registry import ModelRegistry; r=ModelRegistry(); print(r.get_champion_version())"
```

---

## Known Limitations

This is an **academic/portfolio project**. The following are intentional scope boundaries rather than defects:

| Area | Current State | What Real Production Would Need |
|------|--------------|--------------------------------|
| **Authentication** | Static API key for registry mutations | OAuth2 / JWT with user database and RBAC |
| **Database** | SQLite (sufficient for demo) | PostgreSQL with Alembic migrations |
| **Batch processing** | Synchronous in-request | Celery/Redis async job queue |
| **Model performance** | ROC-AUC 0.651 (insurance is inherently hard to predict) | Proprietary insurer data, telematics, claims history |
| **Frontend** | Vite dev server (`npm run dev`) | Production nginx with minified static build |
| **Fairness** | Diagnostic only — fairness groups use vehicle attributes (no protected demographics in dataset) | Requires actual demographic data + legal/compliance review |
| **Monitoring** | PSI drift detection against training baseline | Production outcome feedback loop with real claim labels |
| **Security** | CORS, input validation, API key | WAF, TLS termination, secret management (Vault) |

**What this project correctly demonstrates:**
- Real-dataset ML pipeline with proper train/holdout/unseen-test split discipline
- Nested cross-validated calibration and OOF threshold optimization
- SHAP TreeExplainer integration for per-prediction explanations
- Model Registry with immutable versioned releases and promotion gates
- Schema-validated prediction API with target leakage protection
- Multi-dimensional drift monitoring (PSI on 11 features)
- Release gate verification with end-to-end inference smoke test
