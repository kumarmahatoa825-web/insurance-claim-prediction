# InsureAI — Model Card: Motor Vehicle Claim Prediction

## 1. Model Details
- **Model Name:** InsureAI Motor Claim Classifier (Active Champion Release)
- **Model Version:** `v4.1.1-20260919-7771382`
- **Model Architecture:** Cross-Fitted Platt Calibrated Classifier (`CalibratedClassifierCV(method="sigmoid", cv=3)`) wrapping an inner pipeline of `RealInsurancePreprocessor` + `XGBClassifier(scale_pos_weight=14.63287, eval_metric="logloss", random_state=42)`.
- **Release Date:** September 2026
- **License:** Educational / Portfolio / Production-Oriented Demonstration
- **Canonical Schema:** `schemas/vehicle_insurance_v1.json`
  - `document_sha256`: `bf95ab9e3837a7cf4e7d6eb671c46e34143967bd882af2b95d2d5dd36d8a5dcd`
  - `feature_set_sha256`: `8fce36d22ec1ed83e1bfff466ba4f6c5165da84da34f79824429c0ba8521c635`
- **Release Directory:** `models/releases/v4.1.1-20260919-7771382/`
  - `model.joblib`: Serialized calibrated pipeline (`sha256:a67f887c86c9b7ea...`)
  - `metadata.json`: Immutable audit metadata (`sha256:b9d2a60de1cddd37...`)
  - `calibration.json`: Reliability diagram & ECE evaluation (`sha256:0cd7ec974dbe5bd2...`)
  - `global_shap_importance.json`: Precomputed global feature importances via genuine TreeExplainer (`sha256:ec3a46ba8093849f...`)
  - `global_mdi_importance.json`: Precomputed Gini MDI feature importances (`sha256:e498e834eb82c047...`)
  - `release_manifest.json`: Release manifest (`sha256:143011cf7c1535c5...`)
  - `roc_curve.json` & `pr_curve.json`: Downsampled evaluation curves (100 points)
  - `checksums.json`: Cryptographic integrity manifest for all release artifacts

---

## 2. Intended Use
- **Primary Use Case:** Motor vehicle underwriting risk stratification, early claim propensity triage, and portfolio health monitoring.
- **Audience:** Actuarial teams, underwriting risk officers, and quantitative ML auditors.
- **Out-of-Scope Uses:**
  - Fully automated, unsupervised claim denial or policy cancellation without manual underwriter review.
  - Transfer to non-vehicle lines of insurance (health, travel, property).
  - Use as an unauthorized substitute for consumer credit scoring.
  - Literal real-world insurance production without insurer-specific actuarial data, legal review, and production infrastructure.

---

## 3. Training & Evaluation Datasets
- **Dataset Source:** Real-world vehicle insurance claims dataset (`data/real/train.csv`, 58,592 records).
- **Target Variable:** `is_claim` (`1` = claim filed during policy tenure; `0` = no claim).
- **Empirical Prevalence:** 6.397% (14.63:1 class imbalance).
- **Development Split:** 46,873 records (80% stratified split), evaluated strictly via 5-Fold Stratified Out-of-Fold (OOF) cross-validation for architecture selection, hyperparameter validation, calibration, and threshold optimization.
- **Sealed Test Holdout:** 11,719 records (20% stratified holdout split), untouched throughout development, evaluated exactly once on the final promoted Champion.
- **Unseen Test Set:** 39,063 records (`data/real/test.csv`) for submission generation.

---

## 4. Pipeline & Leakage Protection
1. **Feature Engineering:**
   - Numerical parsing of torque (`torque_value`, `torque_rpm`) and power (`power_value`, `power_rpm`).
   - Vehicle interaction ratios: `power_to_weight`, `torque_to_weight`, `car_size`, `safety_score`, and `age_tenure_ratio`.
   - Categorical one-hot encoding for `make` (1–5), `model` (M1–M11), `area_cluster` (C1–C22), `segment`, `fuel_type`, `transmission_type`, and `engine_type`. Total 98 transformed features.
   - Binary indicator imputation (`SimpleImputer(strategy="most_frequent", add_indicator=True)`) for sensor columns (`is_esc`, `is_tpms`, etc.).
2. **Leak-Free Guarantee:**
   - All transformations are fitted strictly inside cross-validation folds.
   - `is_claim` is excluded from the feature space and flagged as fatal target leakage during API inference.
   - Duplicate policy IDs are rejected with HTTP 422.
3. **Cross-Fitted Probability Calibration:**
   - Base classifier trained with class-weighting (`scale_pos_weight=14.63`).
   - Sigmoid Platt calibrators cross-fitted via 3-fold inner CV (`CalibratedClassifierCV(cv=3)`).
   - Guarantees well-calibrated probabilities matching the natural 6.4% empirical prevalence without uncalibrated oversampling distortion.

---

## 5. Performance Metrics & Benchmarks

### 5.1 Final Holdout Evaluation (11,719 Sealed Rows) with 95% Bootstrap CIs
| Metric | Holdout Value | 95% Confidence Interval | Baseline / Benchmark |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.6510** | `[0.6319, 0.6707]` | Random: `0.5000` |
| **PR-AUC** | **0.1037** | `[0.0928, 0.1157]` | Baseline prevalence: `0.0640` |
| **F1 Score** | **0.1672** | `[0.1512, 0.1831]` | Zero-claim baseline: `0.0000` |
| **Brier Score** | **0.0589** | `[0.0552, 0.0624]` | Baseline reference: `0.0599` |
| **Expected Calibration Error (ECE)** | **0.0052** | — | Calibrated target: `< 0.015` |
| **Mean Probability** | **0.0640** | — | Observed Prevalence: `0.0640` |
| **Triage @ 5% (Precision / Recall / Lift)** | **12.99% / 10.13% / 2.03x** | — | Random: `6.40% / 5.0% / 1.00x` |
| **Triage @ 10% (Precision / Recall / Lift)** | **12.64% / 19.73% / 1.97x** | — | Random: `6.40% / 10.0% / 1.00x` |
| **Triage @ 20% (Precision / Recall / Lift)** | **11.27% / 35.20% / 1.76x** | — | Random: `6.40% / 20.0% / 1.00x` |

### 5.2 5-Fold Stratified Cross-Validation (46,873 Development Rows)
| Model Architecture | PR-AUC (OOF) | ROC-AUC (OOF) | F1 Score (OOF) | Brier Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Scale-Pos-Weight)** | **0.1002 ± 0.0042** | **0.6393 ± 0.0076** | **0.1594 ± 0.0033** | **0.0590 ± 0.0001** | **Promoted Champion** |
| Gradient Boosting (Natural) | 0.0996 ± 0.0051 | 0.6473 ± 0.0081 | 0.1568 ± 0.0040 | 0.0591 ± 0.0001 | Candidate |
| Gradient Boosting (SMOTE) | 0.1005 ± 0.0048 | 0.6400 ± 0.0069 | 0.1582 ± 0.0038 | 0.0592 ± 0.0001 | Candidate |
| Random Forest (Class-Weighted) | 0.0986 ± 0.0060 | 0.6429 ± 0.0089 | 0.1554 ± 0.0045 | 0.0590 ± 0.0001 | Candidate |
| Random Forest (SMOTE) | 0.0963 ± 0.0055 | 0.6382 ± 0.0078 | 0.1541 ± 0.0042 | 0.0591 ± 0.0001 | Candidate |
| LinearSVC (Class-Weighted) | 0.0911 ± 0.0041 | 0.6105 ± 0.0062 | 0.1492 ± 0.0039 | 0.0593 ± 0.0001 | Candidate |
| Logistic Regression (Class-Weighted) | 0.0910 ± 0.0040 | 0.6103 ± 0.0059 | 0.1488 ± 0.0038 | 0.0593 ± 0.0001 | Baseline |

---

## 6. Single Source of Truth Decision Threshold
- **Authoritative Value:** `0.083` (optimized on calibrated OOF predictions to maximize F1).
- **Governance:** Governed by `models/current.json` -> active release `metadata.json["default_threshold"]`.
- **Client Override Policy:** Client overrides via `/predict` or client-side sliders are strictly prohibited. The server enforces the authoritative threshold across all inferences.

---

## 7. Algorithmic Fairness Audit
- **Protected Attribute Audited:** Age brackets and Vehicle Segments.
- **Demographic Parity Disparity:** `0.1370` (selection rate range: `0.184` to `0.321`).
- **Equal Opportunity Disparity:** `0.2105` (true positive rate range: `0.251` to `0.462`).
- **Fairness Gate Result:** PASSED (documented subgroup variation tracked in monitoring).

---

## 8. Explainability & Attribution
- **Global MDI Importance:** Tree Mean Decrease in Impurity measuring feature contribution to tree splits across the ensemble (`global_mdi_importance.json`).
- **Global SHAP Importance:** Precomputed mean absolute SHAP values across representative validation records via genuine TreeExplainer (`global_shap_importance.json`). Top drivers: `policy_tenure` (0.3885), `age_of_car` (0.1904), `age_tenure_ratio` (0.0924), `age_of_policyholder` (0.0687), `population_density` (0.0542).
- **Local SHAP Attribution:** TreeExplainer provides directional risk increases and protective factors for individual policyholders.
- **Attribution Note:** Attributions explain the underlying base estimator; the final displayed claim probability is produced by the cross-fitted calibrated model.
