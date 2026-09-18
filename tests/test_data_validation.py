import pytest
import pandas as pd
from backend.validation.data_validator import DataValidator

def test_valid_real_input_accepted(sample_real_input):
    is_valid, errors, warnings = DataValidator.validate_prediction_input(sample_real_input, data_mode="real")
    assert is_valid is True
    assert len(errors) == 0

def test_missing_required_column_rejected(sample_real_input):
    del sample_real_input["policy_tenure"]
    is_valid, errors, warnings = DataValidator.validate_prediction_input(sample_real_input, data_mode="real")
    assert is_valid is False
    assert any("policy_tenure" in e for e in errors)

def test_target_leakage_detected(sample_real_input):
    sample_real_input["is_claim"] = 1
    is_valid, errors, warnings = DataValidator.validate_prediction_input(sample_real_input, data_mode="real")
    assert is_valid is False
    assert any("leakage" in e.lower() for e in errors)

def test_nan_or_inf_detected(sample_real_input):
    sample_real_input["policy_tenure"] = float("nan")
    is_valid, errors, warnings = DataValidator.validate_prediction_input(sample_real_input, data_mode="real")
    assert is_valid is False
    assert any("NaN" in e for e in errors)

def test_out_of_range_numerical_warning(sample_real_input):
    sample_real_input["airbags"] = 25  # realistic range is 0-12
    is_valid, errors, warnings = DataValidator.validate_prediction_input(sample_real_input, data_mode="real")
    assert any("airbags" in w for w in warnings)

def test_batch_validation_non_crashing(sample_real_input):
    row_good = sample_real_input.copy()
    row_bad = sample_real_input.copy()
    row_bad["policy_tenure"] = None  # missing required

    df = pd.DataFrame([row_good, row_bad])
    valid_df, failed_df, summary = DataValidator.validate_batch_dataframe(df, data_mode="real")

    # Should separate valid and failed rows without raising (Requirement 265)
    assert len(valid_df) == 1
    assert len(failed_df) == 1
    assert summary["valid_rows"] == 1
    assert summary["failed_rows"] == 1

def test_critical_negative_tenure_rejected(sample_real_input):
    sample_real_input["policy_tenure"] = -0.5
    is_valid, errors, warnings = DataValidator.validate_prediction_input(sample_real_input, data_mode="real")
    assert is_valid is False
    assert any("physical constraint" in e.lower() for e in errors)

def test_critical_negative_age_rejected(sample_real_input):
    sample_real_input["age_of_policyholder"] = -0.1
    is_valid, errors, warnings = DataValidator.validate_prediction_input(sample_real_input, data_mode="real")
    assert is_valid is False
    assert any("physical constraint" in e.lower() for e in errors)

