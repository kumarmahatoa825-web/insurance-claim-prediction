# InsureAI — Scientific Model Evaluation & Benchmark Report

## 1. Executive Summary & Evaluation Protocol

This document provides the definitive, scientifically defensible evaluation record for the **InsureAI Motor Vehicle Claim Prediction System** under the promoted Champion release (`v4.1.1-20260919-7771382`).

### 1.1 Integrity & Protocol Safeguards
- **Development / Holdout Partition:** 58,592 real-world records partitioned via stratified 80/20 split into 46,873 development records and 11,719 internal test holdout records.
- **Sealed Holdout Rule:** The 11,719 holdout records were strictly sealed and evaluated **exactly once** after model selection, probability calibration, and threshold optimization were finalized.
- **Official Unseen Test Set:** 39,063 unlabelled records (`data/real/test.csv`) remain completely unlabelled, used solely for final submission generation.
- **Cross-Validation Scheme:** 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- **Nested Probability Calibration:** Internal 3-fold cross-fitting (`CalibratedClassifierCV(method="sigmoid", cv=3)`) ensuring zero calibration data leakage.
- **Decision Threshold Optimization:** Operating decision threshold optimized strictly on **calibrated out-of-fold (OOF)** predictions on development data to maximize F1, frozen at **`0.083`**.

---

## 2. Multi-Model Candidate Benchmark (5-Fold Stratified OOF CV)

Evaluated across 46,873 development records across 7 distinct architectures under identical fold splits:

| Architecture | Handling Strategy | PR-AUC (OOF) | ROC-AUC (OOF) | F1 Score (OOF) | Brier Score | Selection Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **Scale-Pos-Weight (14.63)** | **0.1002 ± 0.0042** | **0.6393 ± 0.0076** | **0.1594 ± 0.0033** | **0.0590 ± 0.0001** | **PROMOTED CHAMPION** |
| Gradient Boosting | Natural Class Weights | 0.0996 ± 0.0051 | 0.6473 ± 0.0081 | 0.1568 ± 0.0040 | 0.0591 ± 0.0001 | Candidate (Evaluated) |
| Gradient Boosting | SMOTE (Inner Folds) | 0.1005 ± 0.0048 | 0.6400 ± 0.0069 | 0.1582 ± 0.0038 | 0.0592 ± 0.0001 | Candidate (Evaluated) |
| Random Forest | Balanced Weights | 0.0986 ± 0.0060 | 0.6429 ± 0.0089 | 0.1554 ± 0.0045 | 0.0590 ± 0.0001 | Candidate (Evaluated) |
| Random Forest | SMOTE (Inner Folds) | 0.0963 ± 0.0055 | 0.6382 ± 0.0078 | 0.1541 ± 0.0042 | 0.0591 ± 0.0001 | Candidate (Evaluated) |
| LinearSVC | Balanced Weights | 0.0911 ± 0.0041 | 0.6105 ± 0.0062 | 0.1492 ± 0.0039 | 0.0593 ± 0.0001 | Candidate (Evaluated) |
| Logistic Regression | Balanced Weights | 0.0910 ± 0.0040 | 0.6103 ± 0.0059 | 0.1488 ± 0.0038 | 0.0593 ± 0.0001 | Baseline (Evaluated) |

### Model Selection Rationale:
1. **Primary Metric (PR-AUC):** Under extreme class imbalance (6.40% claim prevalence), PR-AUC measures the true trade-off between precision and recall across all thresholds without being inflated by true negatives.
2. **Probability Calibration:** XGBoost with Scale-Pos-Weight coupled with nested 3-fold Platt calibration achieved the lowest Brier score (0.0590) and highest calibrated F1 score (0.1594).

---

## 3. Final Sealed Holdout Evaluation (11,719 Records)

Evaluated exactly once on the final promoted Champion pipeline using decision threshold `0.083`:

| Metric | Holdout Value | 95% Bootstrap Confidence Interval | Baseline Reference |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.6510** | `[0.6319, 0.6707]` | Random Guessing: `0.5000` |
| **PR-AUC** | **0.1037** | `[0.0928, 0.1157]` | Natural Claim Prevalence: `0.0640` |
| **F1 Score** | **0.1672** | `[0.1512, 0.1831]` | Trivial Zero-Claim: `0.0000` |
| **Precision** | **0.1046** | `[0.0935, 0.1160]` | Empirical Prevalence: `0.0640` |
| **Recall** | **0.4173** | `[0.3750, 0.4600]` | — |
| **Accuracy** | **0.7340** | `[0.7260, 0.7420]` | Majority Class: `0.9360` |
| **Brier Score** | **0.0589** | `[0.0552, 0.0624]` | Climatological Baseline: `0.0599` |
| **Expected Calibration Error (ECE)** | **0.0052** | — | Uncalibrated Target: `> 0.150` |
| **Mean Predicted Probability** | **0.0640** | — | Observed Holdout Prevalence: `0.0640` |

### Confusion Matrix (Holdout Set, Threshold = 0.083):
```
                       Predicted: No Claim (0)    Predicted: Claim (1)
Actual: No Claim (0)            8,289                    2,680
Actual: Claim (1)                 437                      313
```

---

## 4. Underwriting Triage & Empirical Decile Lift Analysis

In motor insurance underwriting, models serve primarily as **risk ranking and triaging engines**. The table below documents empirical claim concentration across predicted risk tiers on the sealed holdout set:

### 4.1 Triage @ Top-K Capacity
| Operating Tier | Cumulative Policies Evaluated | Actual Claims Captured | Precision | Recall | Empirical Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 5% of Portfolio** | 585 | 76 | 12.99% | 10.13% | **2.03x** |
| **Top 10% of Portfolio** | 1,171 | 148 | 12.64% | 19.73% | **1.97x** |
| **Top 20% of Portfolio** | 2,343 | 264 | 11.27% | 35.20% | **1.76x** |

### 4.2 Complete Decile Lift Table
| Decile | Policy Count | Actual Claims | Observed Claim Rate | Decile Lift vs Portfolio Base |
| :---: | :---: | :---: | :---: | :---: |
| **1 (Highest Risk)** | 1,171 | 148 | **12.64%** | **1.97x** |
| **2** | 1,171 | 116 | 9.91% | 1.55x |
| **3** | 1,171 | 99 | 8.45% | 1.32x |
| **4** | 1,171 | 89 | 7.60% | 1.19x |
| **5** | 1,171 | 76 | 6.49% | 1.01x |
| **6** | 1,171 | 72 | 6.15% | 0.96x |
| **7** | 1,171 | 49 | 4.18% | 0.65x |
| **8** | 1,171 | 47 | 4.01% | 0.63x |
| **9** | 1,171 | 32 | 2.73% | 0.43x |
| **10 (Lowest Risk)** | 1,180 | 22 | **1.86%** | **0.29x** |

> **Underwriting Interpretation:** Inspecting the top 10% of predicted policies yields a claim frequency of 12.64%, nearly double the baseline portfolio risk (1.97x lift). Conversely, policies placed in Decile 10 exhibit a negligible 1.86% claim rate (0.29x lift).

---

## 5. Algorithmic Fairness & Subgroup Equity Audit

Conducted across age cohorts and vehicle market segments:

| Subgroup | Group Size | Selection Rate (Threshold 0.083) | True Positive Rate (Recall) |
| :--- | :--- | :--- | :--- |
| **Age: Young (<= 35)** | 23,768 | 21.11% | 36.34% |
| **Age: Mature (> 35)** | 23,105 | 29.72% | 45.35% |
| **Segment: A** | 13,794 | 19.31% | 38.64% |
| **Segment: B1** | 3,335 | 18.41% | 35.26% |
| **Segment: B2** | 14,702 | 32.11% | 46.19% |
| **Segment: C1** | 2,851 | 19.19% | 25.14% |
| **Segment: C2** | 11,214 | 28.11% | 43.14% |

- **Demographic Parity Disparity:** `0.1370` (Difference between highest and lowest selection rate).
- **Equal Opportunity Disparity:** `0.2105` (Difference between highest and lowest true positive rate).
- **Fairness Assessment:** Passed with active monitoring requirements for segment C1 and young driver subgroups.

---

## 6. Scientific Disclaimer & Empirical Boundaries

> [!IMPORTANT]
> **Empirical Ranking, Not Causal Fact:** All reported metrics, feature importances, and SHAP values reflect statistical correlations within this observational motor insurance dataset. They do not establish causal relationships (e.g., policy tenure does not causally prevent collisions).
>
> **Production-Oriented Scope:** This system demonstrates advanced production-grade machine learning engineering, data hygiene, and MLOps discipline for academic, portfolio, and demonstration purposes. Real-world commercial deployment requires proprietary insurer loss cost modeling, actuarial ratemaking approval, and regulatory compliance.
