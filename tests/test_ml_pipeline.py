import pytest
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from backend.ml.real_preprocessing import RealInsurancePreprocessor
from backend.ml.train_real import tune_threshold_oof

def test_pipeline_fold_safe_execution(sample_real_input):
    df_rows = []
    for i in range(30):
        row = sample_real_input.copy()
        row["policy_tenure"] = 0.1 * (i + 1)
        row["is_claim"] = 1 if i % 5 == 0 else 0
        df_rows.append(row)
    df = pd.DataFrame(df_rows)

    X = df.drop(columns=["is_claim"])
    y = df["is_claim"]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = y.iloc[tr_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        pipe = ImbPipeline([
            ("preprocessor", RealInsurancePreprocessor()),
            ("smote", SMOTE(k_neighbors=2, random_state=42)),
            ("classifier", LogisticRegression(max_iter=200)),
        ])

        # Fitting on training fold only (NO LEAKAGE)
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_val)

        assert probs.shape[0] == len(y_val)
        assert probs.shape[1] == 2
        assert not np.isnan(probs).any()

def test_tune_threshold_oof():
    y_true = pd.Series([0, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    oof_probs = np.array([0.05, 0.08, 0.12, 0.25, 0.07, 0.35, 0.04, 0.09, 0.40, 0.02])

    result = tune_threshold_oof(y_true, oof_probs)
    assert "selected_threshold" in result
    assert "threshold_results" in result
    assert 0.05 <= result["selected_threshold"] <= 0.55


def test_generate_submission_logic(sample_real_input, tmp_path):
    from scripts.generate_submission import generate_submission
    # Create a small dummy test csv
    dummy_rows = []
    for i in range(5):
        row = sample_real_input.copy()
        row["policy_id"] = f"TEST_{i:04d}"
        dummy_rows.append(row)
    df_dummy = pd.DataFrame(dummy_rows)
    test_csv = tmp_path / "test_sample.csv"
    df_dummy.to_csv(test_csv, index=False)

    out_csv = tmp_path / "submission_sample.csv"
    df_sub = generate_submission(
        test_path=str(test_csv),
        output_path=str(out_csv),
        threshold=0.50
    )

    assert len(df_sub) == 5
    assert list(df_sub.columns) == ["policy_id", "is_claim"]
    assert df_sub["policy_id"].tolist() == [f"TEST_{i:04d}" for i in range(5)]
    assert df_sub["is_claim"].isin([0, 1]).all()
    assert out_csv.exists()

