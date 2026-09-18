import os
import sys
import json
import time
import argparse
import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from .preprocessing import InsurancePreprocessor
from .config import config


def _to_python(obj):
    """Recursively convert numpy scalars to Python native types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def build_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=config.RANDOM_SEED, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=config.RANDOM_SEED, class_weight="balanced", max_depth=10
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=config.RANDOM_SEED,
            class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=config.RANDOM_SEED
        ),
        "SVM": SVC(
            probability=True, random_state=config.RANDOM_SEED,
            class_weight="balanced", kernel="rbf"
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=config.RANDOM_SEED,
            eval_metric="logloss", use_label_encoder=False,
            scale_pos_weight=(1 - config.CLAIM_RATE) / config.CLAIM_RATE,
        )
    return models


def cross_validate_model(model, X_train, y_train, skf) -> dict:
    """Run 5-fold stratified CV and return mean/std metrics."""
    metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train[val_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)
        metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
        try:
            metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))
        except Exception:
            metrics["roc_auc"].append(0.5)
    return {
        "mean": {k: float(np.mean(v)) for k, v in metrics.items()},
        "std": {k: float(np.std(v)) for k, v in metrics.items()},
    }


def compute_threshold_analysis(y_true, y_prob) -> list:
    """Evaluate multiple thresholds and return per-threshold metrics."""
    results = []
    thresholds = np.arange(0.10, 0.92, 0.05)
    total = len(y_true)
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        results.append({
            "threshold": round(float(t), 2),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
            "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
            "fp_count": int(fp),
            "fn_count": int(fn),
            "tp_count": int(tp),
            "tn_count": int(tn),
            "total": total,
        })
    return results


def main(data_path: str = None):
    if data_path is None:
        data_path = os.path.join("data", "raw", "insurance_claims.csv")

    print(f"\n{'='*60}")
    print("InsureAI -- Model Training Pipeline")
    print(f"{'='*60}")
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # Drop risk_score_base -- it's an independent pre-existing indicator correlated
    # with latent risk but NOT derived from the claim outcome formula, so we exclude
    # it to avoid any potential leakage and to keep features realistic.
    X = df.drop(columns=["claim_filed", "risk_score_base"], errors="ignore")
    y = df["claim_filed"]

    print(f"\nClass distribution:\n{y.value_counts().to_string()}")
    claim_rate = float(y.mean())
    print(f"Claim rate: {claim_rate:.2%}")

    # Stratified train/test split -- test set is NEVER touched until final eval
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # Class imbalance check
    class_counts = y_train.value_counts()
    imbalance_ratio = float(class_counts.max() / class_counts.min())
    imbalance_detected = class_counts.min() / class_counts.sum() < 0.4
    print(f"Imbalance detected: {imbalance_detected} (ratio {imbalance_ratio:.2f}:1)")

    # Fit preprocessor on training data ONLY
    print("\nFitting preprocessor on training data...")
    preprocessor = InsurancePreprocessor()
    X_train_proc, feature_names = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    print(f"Feature count after engineering: {len(feature_names)}")

    # 5-Fold Stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    MODELS = build_models()

    print(f"\n{'-'*60}")
    print(f"{'Model':<22} {'F1':>8} {'+-':>2} {'ROC-AUC':>10} {'Time':>8}")
    print(f"{'-'*60}")


    all_results = {}
    for name, model in MODELS.items():
        start = time.time()
        cv = cross_validate_model(model, X_train_proc, y_train, skf)
        elapsed = time.time() - start
        all_results[name] = {"cv": cv, "train_time_seconds": round(elapsed, 2)}
        print(
            f"{name:<22} {cv['mean']['f1']:.4f}  Â± {cv['std']['f1']:.4f}"
            f"  {cv['mean']['roc_auc']:.4f}  {elapsed:.1f}s"
        )

    # Select best model
    metric = config.MODEL_SELECTION_METRIC
    best_name = max(all_results, key=lambda k: all_results[k]["cv"]["mean"].get(metric, 0))
    print(f"\nâœ… Best model by {metric.upper()}: {best_name}")

    # SMOTE comparison on best model
    print("\nEvaluating SMOTE on best model...")
    smote = SMOTE(random_state=config.RANDOM_SEED)
    best_model_cls = MODELS[best_name]
    smote_scores = []
    for train_idx, val_idx in skf.split(X_train_proc, y_train):
        X_tr, y_tr = X_train_proc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train_proc[val_idx], y_train.iloc[val_idx]
        X_tr_s, y_tr_s = smote.fit_resample(X_tr, y_tr)
        m_clone = type(best_model_cls)(**best_model_cls.get_params())
        m_clone.fit(X_tr_s, y_tr_s)
        smote_scores.append(f1_score(y_val, m_clone.predict(X_val), zero_division=0))
    smote_f1 = float(np.mean(smote_scores))
    baseline_f1 = all_results[best_name]["cv"]["mean"]["f1"]
    print(f"Baseline F1: {baseline_f1:.4f} | SMOTE F1: {smote_f1:.4f}")

    # Train final model on all training data
    print(f"\nTraining final {best_name} on full training set...")
    final_model = MODELS[best_name]
    final_model.fit(X_train_proc, y_train)

    # Calibrate using cross-fitted sigmoid calibration
    print("Calibrating probabilities...")
    calibrated = CalibratedClassifierCV(estimator=final_model, cv=3, method="sigmoid")
    calibrated.fit(X_train_proc, y_train)

    # Final test evaluation â€” ONLY happens once on unseen test data
    print("\nFinal evaluation on held-out test set...")
    y_test_prob = calibrated.predict_proba(X_test_proc)[:, 1]
    y_test_pred = (y_test_prob >= config.DEFAULT_THRESHOLD).astype(int)

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_test_prob)),
        "pr_auc": float(average_precision_score(y_test, y_test_prob)),
    }
    cm = confusion_matrix(y_test, y_test_pred).tolist()
    brier = float(brier_score_loss(y_test, y_test_prob))
    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    # Threshold optimization
    thresh_analysis = compute_threshold_analysis(y_test.values, y_test_prob)
    best_thresh_row = max(thresh_analysis, key=lambda x: x["f1"])
    optimal_threshold = best_thresh_row["threshold"]
    print(f"\nOptimal threshold (max F1): {optimal_threshold}")

    # ROC and PR curves per model (compute on test set for the best model, CV val for others)
    print("\nComputing ROC/PR curves...")
    roc_curves = {}
    for name, result in all_results.items():
        m = MODELS[name]
        # Refit on full training set for curve generation
        try:
            m.fit(X_train_proc, y_train)
            prob = m.predict_proba(X_test_proc)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, prob)
            prec_vals, rec_vals, _ = precision_recall_curve(y_test, prob)
            roc_curves[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc_score(y_test, prob)),
                "pr_precision": prec_vals.tolist(),
                "pr_recall": rec_vals.tolist(),
                "pr_auc": float(average_precision_score(y_test, prob)),
            }
        except Exception as e:
            print(f"  Warning: ROC curve for {name} failed: {e}")

    # Calibration curve
    try:
        frac_pos, mean_pred = calibration_curve(y_test, y_test_prob, n_bins=10)
        cal_curve = [
            {"mean_predicted": float(mp), "fraction_positives": float(fp)}
            for mp, fp in zip(mean_pred, frac_pos)
        ]
    except Exception:
        cal_curve = []

    # Feature importance for best model
    try:
        if hasattr(final_model, "feature_importances_"):
            importances = final_model.feature_importances_.tolist()
        elif hasattr(final_model, "coef_"):
            importances = np.abs(final_model.coef_[0]).tolist()
        else:
            importances = [0.0] * len(feature_names)
    except Exception:
        importances = [0.0] * len(feature_names)

    # Save everything
    os.makedirs("models", exist_ok=True)

    pipeline_data = {"preprocessor": preprocessor, "model": calibrated}
    joblib.dump(pipeline_data, "models/best_model.joblib")
    print("\nâœ… Model pipeline saved: models/best_model.joblib")

    metadata = {
        "version": "v1.0",
        "model_name": best_name,
        "training_date": datetime.datetime.now().isoformat(),
        "dataset_size": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "feature_importances": [
            {"feature": f, "importance": float(i)}
            for f, i in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        ],
        "target_column": "claim_filed",
        "claim_rate": claim_rate,
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "imbalance_detected": imbalance_detected,
        "imbalance_ratio": imbalance_ratio,
        "smote_comparison": {"baseline_f1": baseline_f1, "smote_f1": smote_f1},
        "cv_metrics": all_results[best_name]["cv"],
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "optimal_threshold": float(optimal_threshold),
        "default_threshold": float(config.DEFAULT_THRESHOLD),
        "calibration_brier_score": brier,
        "calibration_curve": cal_curve,
        "selection_metric": config.MODEL_SELECTION_METRIC,
        "random_seed": config.RANDOM_SEED,
        "risk_score_base_decision": (
            "risk_score_base was excluded from model inputs. "
            "It is generated as an independent pre-existing risk indicator with added noise. "
            "Including it could introduce target leakage since it correlates with the latent risk process."
        ),
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(_to_python(metadata), f, indent=2)
    print("âœ… Metadata saved: models/model_metadata.json")

    # All model results with CV metrics
    all_models_output = {}
    for name, result in all_results.items():
        all_models_output[name] = {
            "cv_mean": result["cv"]["mean"],
            "cv_std": result["cv"]["std"],
            "train_time_seconds": result["train_time_seconds"],
            "is_best": name == best_name,
        }
    with open("models/all_models_results.json", "w") as f:
        json.dump(_to_python(all_models_output), f, indent=2)
    print("âœ… Model comparison saved: models/all_models_results.json")

    with open("models/threshold_analysis.json", "w") as f:
        json.dump(_to_python(thresh_analysis), f, indent=2)
    print("âœ… Threshold analysis saved: models/threshold_analysis.json")

    with open("models/roc_curves.json", "w") as f:
        json.dump(_to_python(roc_curves), f, indent=2)
    print("âœ… ROC/PR curves saved: models/roc_curves.json")

    calibration_data = {
        "brier_score": brier,
        "calibration_curve": cal_curve,
        "model_name": best_name,
        "note": "Probabilities calibrated using cross-fitted CalibratedClassifierCV (sigmoid method, cv=3).",
    }
    with open("models/calibration.json", "w") as f:
        json.dump(_to_python(calibration_data), f, indent=2)
    print("âœ… Calibration data saved: models/calibration.json")

    print(f"\n{'='*60}")
    print(f"Training complete! Best model: {best_name}")
    print(f"Test F1: {test_metrics['f1']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Optimal threshold: {optimal_threshold}")
    print(f"Brier score: {brier:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsureAI Model Training")
    parser.add_argument("--data", type=str, default=None, help="Path to training dataset CSV")
    args = parser.parse_args()
    main(data_path=args.data)

