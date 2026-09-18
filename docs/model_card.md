# InsureAI — Model Card: Motor Vehicle Claim Prediction

## 1. Model Details
- **Model Name:** InsureAI Motor Claim Classifier (Active Champion Release)
- **Model Version:** `v4.1.0-20260918-6640e9b`
- **Model Architecture:** Cross-Fitted Platt Calibrated Classifier (`CalibratedClassifierCV(method="sigmoid", cv=3)`) wrapping an inner pipeline of `RealInsurancePreprocessor` + `XGBClassifier(scale_pos_weight=14.63, eval_metric="logloss", random_state=42)`.
- **Release Date:** September 2026
- **License:** Educational / Portfolio / Production Demonstration
- **Canonical Schema:** `schemas/vehicle_insurance_v1.json` (`sha256:b858ec67201eb6c8cc903132f518615d68947e1230dc3d1a83a93527162149a7`)
- **Release Directory:** `models/releases/v4.1.0-20260918-6640e9b/`
  - `model.joblib`: Serialized calibrated pipeline (`sha256:115905f2237414d2...`)
  - `metadata.json`: Immutable audit metadata (`sha256:77ff26b7727c62b5...`)
  - `calibration.json`: Reliability diagram & ECE evaluation (`sha256:13a5255476a8a4f9...`)
  - `global_shap_importance.json`: Precomputed global feature importances (`sha256:6e1841315b6d7733...`)
  - `checksums.json`: Cryptographic integrity manifest

---

## 2. Intended Use
- **Primary Use Case:** Motor vehicle underwriting risk stratification, early claim propensity triage, and portfolio health monitoring.
- **Audience:** Actuarial teams, underwriting risk officers, and quantitative ML auditors.
- **Out-of-Scope Uses:**
  - Fully automated, unsupervised claim denial or policy cancellation without manual underwriter intervention.
  - Transfer to non-vehicle lines of insurance (health, travel, property).
  - Use as an unauthorized substitute for regulated consumer credit scoring.

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
   - Categorical one-hot encoding for `make` (1–5), `model` (M1–M11), `area_cluster` (C1–C22), `segment`, `fuel_type`, `transmission_type`, and `engine_type`.
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
| **ROC-AUC** | **0.6498** | `[0.6304, 0.6694]` | Random: `0.5000` |
| **PR-AUC** | **0.1038** | `[0.0930, 0.1161]` | Baseline prevalence: `0.0640` |
| **F1 Score** | **0.1676** | `[0.1537, 0.1820]` | Natural baseline: `0.0000` |
| **Brier Score** | **0.0589** | `[0.0573, 0.0606]` | Baseline reference: `0.0599` |
| **Expected Calibration Error (ECE)** | **0.0045** | — | Calibrated target: `< 0.015` |
| **Mean Probability** | **0.0638** | — | Observed Prevalence: `0.0640` |
| **Decile 1 Lift (Top-10% Triage)** | **2.41x** | — | Random: `1.00x` |

### 5.2 5-Fold Stratified Cross-Validation (46,873 Development Rows)
| Model Architecture | PR-AUC (OOF) | ROC-AUC (OOF) | Brier Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Scale-Pos-Weight)** | **0.1006 ± 0.0044** | **0.6403 ± 0.0074** | **0.0590 ± 0.0001** | **Promoted Champion** |
| Gradient Boosting (Natural) | 0.0996 ± 0.0051 | 0.6473 ± 0.0081 | 0.0591 ± 0.0001 | Candidate |
| Gradient Boosting (SMOTE) | 0.1005 ± 0.0048 | 0.6400 ± 0.0069 | 0.0592 ± 0.0001 | Candidate |
| Random Forest (Class-Weighted) | 0.0986 ± 0.0060 | 0.6429 ± 0.0089 | 0.0590 ± 0.0001 | Candidate |
| Random Forest (SMOTE) | 0.0963 ± 0.0055 | 0.6382 ± 0.0078 | 0.0591 ± 0.0001 | Candidate |
| LinearSVC (Class-Weighted) | 0.0911 ± 0.0041 | 0.6105 ± 0.0062 | 0.0593 ± 0.0001 | Candidate |
| Logistic Regression (Class-Weighted) | 0.0910 ± 0.0040 | 0.6103 ± 0.0059 | 0.0593 ± 0.0001 | Baseline |

---

## 6. Single Source of Truth Decision Threshold
- **Authoritative Value:** `0.080` (optimized on calibrated OOF predictions to maximize F1).
- **Governance:** Governed by `models/current.json` -> active release `metadata.json["default_threshold"]`.
- **Client Override Policy:** Client overrides via `/predict` or client-side sliders are strictly prohibited. The server enforces the authoritative threshold across all inferences.

---

## 7. Algorithmic Fairness Audit
- **Protected Attribute Audited:** Vehicle Segment (`segment_A`, `segment_B1`, `segment_B2`, `segment_C1`, `segment_C2`).
- **Demographic Parity Disparity:** `0.1569` (selection rate range: `0.210` to `0.367`).
- **Equal Opportunity Disparity:** `0.1597` (true positive rate range: `0.335` to `0.493`).
- **Fairness Gate Result:** PASSED (disparity `< 0.20` gate limit).

---

## 8. Explainability & Attribution
- **Global MDI Importance:** Tree Mean Decrease in Impurity measuring feature contribution to tree splits across the ensemble. Top drivers: `gross_weight`, `displacement`, `cylinder`, `steering_type_Electric`, `policy_tenure`.
- **Global SHAP Importance:** Precomputed mean absolute SHAP values across representative validation records.
- **Local SHAP Attribution:** TreeExplainer provides directional risk increases (red) and protective factors (green) for individual policyholders.
- **Attribution Note:** Attributions explain the underlying base estimator; the final displayed claim probability is produced by the cross-fitted calibrated model.
