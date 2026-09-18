"""
test_schema_regression.py
=========================
Regression tests to prevent data leakage, schema drifting, and improper encoding:
  - Validates full area cluster support (C1-C22)
  - Validates categorical encoding of make (1-5) and model (M1-M11)
  - Validates rejection of duplicate policy_id
  - Validates rejection of target leakage (is_claim)
"""

import pandas as pd
import pytest

from backend.validation.data_validator import DataValidator
from backend.ml.real_preprocessing import RealInsurancePreprocessor


def test_all_area_clusters_accepted(sample_real_input):
    """Ensure all 22 area clusters C1-C22 are accepted without fatal schema errors."""
    for i in range(1, 23):
        cluster = f"C{i}"
        payload = sample_real_input.copy()
        payload["area_cluster"] = cluster
        is_valid, errors, _ = DataValidator.validate_prediction_input(payload, data_mode="real")
        assert is_valid, f"Area cluster {cluster} failed validation: {errors}"


def test_invalid_area_cluster_generates_warning(sample_real_input):
    """Ensure undefined area clusters are flagged with category warnings."""
    payload = sample_real_input.copy()
    payload["area_cluster"] = "C99"
    is_valid, errors, warnings = DataValidator.validate_prediction_input(payload, data_mode="real")
    assert any("area_cluster" in w for w in warnings)


def test_target_leakage_is_claim_rejected(sample_real_input):
    """Ensure presence of is_claim in prediction payload is rejected with fatal error."""
    payload = sample_real_input.copy()
    payload["is_claim"] = 0
    is_valid, errors, _ = DataValidator.validate_prediction_input(payload, data_mode="real")
    assert not is_valid
    assert any("leakage" in e.lower() and "is_claim" in e.lower() for e in errors)


def test_duplicate_policy_id_rejected_in_batch():
    """Ensure batch validation rejects files containing duplicate policy_id values."""
    data = [
        {"policy_id": "POL_001", "policy_tenure": 0.5, "age_of_car": 0.1, "age_of_policyholder": 0.4},
        {"policy_id": "POL_001", "policy_tenure": 0.8, "age_of_car": 0.2, "age_of_policyholder": 0.5},
    ]
    df = pd.DataFrame(data)
    valid_df, failed_df, summary = DataValidator.validate_batch_dataframe(df, data_mode="real")
    assert len(summary["errors"]) > 0
    assert any("duplicate" in e.lower() and "policy_id" in e.lower() for e in summary["errors"])


def test_make_and_model_one_hot_encoded():
    """Verify make and model are treated as categorical features and one-hot encoded."""
    preprocessor = RealInsurancePreprocessor()
    data = [
        {
            "policy_id": "POL_1", "policy_tenure": 0.5, "age_of_car": 0.1, "age_of_policyholder": 0.4,
            "area_cluster": "C1", "population_density": 5000, "make": 1, "segment": "A", "model": "M1",
            "fuel_type": "Petrol", "max_torque": "100Nm@4000rpm", "max_power": "80bhp@5000rpm",
            "engine_type": "1.2 L K Series Engine", "airbags": 2, "is_esc": "No", "is_adjustable_steering": "No",
            "is_tpms": "No", "is_parking_sensors": "Yes", "is_parking_camera": "No", "rear_brakes_type": "Drum",
            "displacement": 1200, "cylinder": 4, "transmission_type": "Manual", "gear_box": 5,
            "steering_type": "Power", "turning_radius": 5.0, "length": 3800, "width": 1600, "height": 1500,
            "gross_weight": 1200, "is_front_fog_lights": "No", "is_rear_window_wiper": "No",
            "is_rear_window_washer": "No", "is_rear_window_defogger": "No", "is_brake_assist": "No",
            "is_power_door_locks": "Yes", "is_central_locking": "Yes", "is_power_steering": "Yes",
            "is_driver_seat_height_adjustable": "No", "is_day_night_rear_view_mirror": "No",
            "is_ecw": "No", "is_speed_alert": "Yes", "ncap_rating": 2,
        },
        {
            "policy_id": "POL_2", "policy_tenure": 0.8, "age_of_car": 0.2, "age_of_policyholder": 0.5,
            "area_cluster": "C2", "population_density": 8000, "make": 3, "segment": "B2", "model": "M4",
            "fuel_type": "Diesel", "max_torque": "200Nm@2000rpm", "max_power": "100bhp@4000rpm",
            "engine_type": "1.5 L U2 CRDi", "airbags": 6, "is_esc": "Yes", "is_adjustable_steering": "Yes",
            "is_tpms": "Yes", "is_parking_sensors": "Yes", "is_parking_camera": "Yes", "rear_brakes_type": "Disc",
            "displacement": 1500, "cylinder": 4, "transmission_type": "Automatic", "gear_box": 6,
            "steering_type": "Electric", "turning_radius": 5.2, "length": 4000, "width": 1750, "height": 1600,
            "gross_weight": 1500, "is_front_fog_lights": "Yes", "is_rear_window_wiper": "Yes",
            "is_rear_window_washer": "Yes", "is_rear_window_defogger": "Yes", "is_brake_assist": "Yes",
            "is_power_door_locks": "Yes", "is_central_locking": "Yes", "is_power_steering": "Yes",
            "is_driver_seat_height_adjustable": "Yes", "is_day_night_rear_view_mirror": "Yes",
            "is_ecw": "Yes", "is_speed_alert": "Yes", "ncap_rating": 5,
        }
    ]
    df = pd.DataFrame(data)
    transformed = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names()

    # Verify categorical dummy expansion
    make_features = [f for f in feature_names if f.startswith("make_") or "make" in f]
    model_features = [f for f in feature_names if f.startswith("model_") or "model" in f]

    assert len(make_features) > 1, f"Make should be one-hot encoded: {make_features}"
    assert len(model_features) > 1, f"Model should be one-hot encoded: {model_features}"
    assert "is_claim" not in feature_names, "Target leakage detected in preprocessor features"
