import pytest
import pandas as pd
import numpy as np
from backend.ml.real_preprocessing import RealInsurancePreprocessor, VehicleFeatureEngineer

def test_feature_engineering_extracts_torque_and_power(sample_real_input):
    df = pd.DataFrame([sample_real_input])
    engineer = VehicleFeatureEngineer()
    eng_df = engineer.transform(df)

    assert "torque_value" in eng_df.columns
    assert "torque_rpm" in eng_df.columns
    assert "power_value" in eng_df.columns
    assert "power_rpm" in eng_df.columns
    assert "safety_score" in eng_df.columns
    assert eng_df["torque_value"].iloc[0] == 60.0
    assert eng_df["torque_rpm"].iloc[0] == 3500.0

def test_binary_features_mapped_to_0_1(sample_real_input):
    df = pd.DataFrame([sample_real_input])
    engineer = VehicleFeatureEngineer()
    eng_df = engineer.transform(df)

    assert eng_df["is_parking_sensors"].iloc[0] == 1
    assert eng_df["is_esc"].iloc[0] == 0

def test_no_target_leakage(sample_real_input):
    sample_real_input["is_claim"] = 1
    sample_real_input["policy_id"] = "POL123"
    df = pd.DataFrame([sample_real_input])
    engineer = VehicleFeatureEngineer()
    eng_df = engineer.transform(df)

    # Policy ID should be dropped
    assert "policy_id" not in eng_df.columns

def test_real_preprocessor_fit_transform_consistency(sample_real_input):
    df_train = pd.DataFrame([sample_real_input] * 5)
    preprocessor = RealInsurancePreprocessor()

    X_proc, feature_names = preprocessor.fit_transform_legacy(df_train)
    assert len(feature_names) > 0
    assert X_proc.shape[0] == 5
    assert X_proc.shape[1] == len(feature_names)

    # Test single transform
    single_proc = preprocessor.transform(pd.DataFrame([sample_real_input]))
    assert single_proc.shape[1] == len(feature_names)
    assert not np.isnan(single_proc).any()

def test_handles_unknown_categories(sample_real_input):
    df_train = pd.DataFrame([sample_real_input] * 3)
    df_test = pd.DataFrame([sample_real_input])
    df_test.loc[0, "fuel_type"] = "UnknownFuel"

    preprocessor = RealInsurancePreprocessor()
    preprocessor.fit(df_train)
    res = preprocessor.transform(df_test)
    assert res is not None
    assert not np.isnan(res).any()
