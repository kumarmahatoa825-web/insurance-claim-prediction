# InsureAI — Data Card: Vehicle Insurance Claim Dataset

## 1. Dataset Overview
- **Dataset Title:** Real-World Motor Vehicle Insurance Claims Dataset
- **Domain:** Automotive Insurance Underwriting & Claims Administration
- **Data Modality:** Tabular / Structured Records
- **Target Variable:** `is_claim` (0 = No claim filed, 1 = Claim filed)
- **Primary Source Files:**
  - `data/real/train.csv`: 58,592 rows × 44 columns (with `is_claim` label)
  - `data/real/test.csv`: 39,063 rows × 43 columns (unlabelled test policies)
  - `data/real/sample_submission.csv`: 39,063 rows × 2 columns (`policy_id`, `is_claim`)

---

## 2. Dataset Partitioning & Architecture
```
                         REAL-WORLD DATASET
                        (data/real/train.csv)
                             58,592 rows
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
  Development Set (80%)                         Internal Holdout (20%)
      46,873 rows                                     11,719 rows
          │                                               │
  5-Fold Stratified CV                                    │
  Nested Calibrated CV                                    │
  OOF Threshold Selection                                 │
  Final Calibrated Fit                                    │
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  ▼
                       UNSEEN TEST EVALUATION
                        (11,719 holdout rows)
```

---

## 3. Class Balance & Ratios
- **Total Labeled Policies:** 58,592
- **No Claim (`0`):** 54,844 policies (93.60%)
- **Claim Filed (`1`):** 3,748 policies (6.40%)
- **Imbalance Ratio:** 14.63 : 1
- **Underwriting Implication:** Severe natural class imbalance typical of personal auto lines, where the vast majority of policyholders do not experience an insurable collision or theft event during a policy term.

---

## 4. Feature Taxonomy
The dataset features span 4 core categories:

### A. Policy & Driver Profile
- `policy_id`: Unique identifier (excluded from ML features; preserved for tracking and submission)
- `policy_tenure`: Policy active duration in decimal years
- `age_of_policyholder`: Normalized age bracket of the primary policyholder
- `area_cluster`: Geographic demographic grouping (e.g., `C1` to `C22`)
- `population_density`: Population density of the policyholder's residential cluster

### B. Vehicle Specifications & Performance
- `make`: Vehicle manufacturer category code
- `segment`: Vehicle market tier (`A`, `B1`, `B2`, `C1`, `C2`, `Utility`)
- `model`: Specific model variant identifier (`M1` to `M11`)
- `fuel_type`: Fuel system (`Petrol`, `Diesel`, `CNG`)
- `displacement`: Engine displacement in cubic centimeters (cc)
- `cylinder`: Number of engine cylinders
- `transmission_type`: `Manual` vs `Automatic`
- `gear_box`: Gear count
- `max_torque` & `max_power`: Raw engine output strings parsed into numerical torque (Nm), torque RPM, power (bhp), and power RPM

### C. Physical Dimensions & Geometry
- `length`, `width`, `height`: Exterior vehicle dimensions in millimeters
- `gross_weight`: Gross vehicle weight rating in kilograms
- `turning_radius`: Turning circle radius in meters

### D. Safety Equipment & Telematics Features
- `airbags`: Installed airbag count
- Binary safety indicators (`is_esc`, `is_tpms`, `is_parking_sensors`, `is_parking_camera`, `is_brake_assist`, `is_power_steering`, etc.)
- `ncap_rating`: Standardized vehicle crash safety rating (0 to 5)

---

## 5. Missingness & Data Quality
- Baseline missingness across the 46,873 development records is recorded in `data/processed/drift_baseline.json`.
- Missing values in numerical features are imputed using training-fold medians.
- Missing values in categorical features are imputed using training-fold modes.
- Production inferences are continuously benchmarked against baseline missingness rates to detect sensor dropouts or API integration errors.

---

## 6. Train / Test Isolation & Leakage Prevention
- **Canonical Schema Enforcement:** Validated against `schemas/vehicle_insurance_v1.json` (`document_sha256: bf95ab9e3837a7cf4e7d6eb671c46e34143967bd882af2b95d2d5dd36d8a5dcd`, `feature_set_sha256: 8fce36d22ec1ed83e1bfff466ba4f6c5165da84da34f79824429c0ba8521c635`).
- **Strict Separation:** `test.csv` (39,063 records) is held completely unlabelled and unseen during all model development, cross-validation, probability calibration, and threshold optimization.
- **Holdout Protection:** The 20% internal holdout (`11,719` rows) is evaluated strictly once after candidate model training and threshold freezing.
- **Identifier Hygiene:** `policy_id` is excluded from all feature matrices and encoders, eliminating arbitrary ID memorization. Duplicate `policy_id` submissions in batch inference are strictly rejected.
- **Target Leakage Prohibition:** Any occurrence of `is_claim` in prediction payloads is rejected with HTTP 422.
