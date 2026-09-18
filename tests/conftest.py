import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_real_input():
    return {
        "policy_tenure": 0.52,
        "age_of_car": 0.05,
        "age_of_policyholder": 0.45,
        "area_cluster": "C1",
        "population_density": 4990,
        "make": 1,
        "segment": "A",
        "model": "M1",
        "fuel_type": "CNG",
        "max_torque": "60Nm@3500rpm",
        "max_power": "40.36bhp@6000rpm",
        "engine_type": "F8D Petrol Engine",
        "airbags": 2,
        "is_esc": "No",
        "is_adjustable_steering": "No",
        "is_tpms": "No",
        "is_parking_sensors": "Yes",
        "is_parking_camera": "No",
        "rear_brakes_type": "Drum",
        "displacement": 796,
        "cylinder": 3,
        "transmission_type": "Manual",
        "gear_box": 5,
        "steering_type": "Power",
        "turning_radius": 4.6,
        "length": 3445,
        "width": 1515,
        "height": 1475,
        "gross_weight": 1185,
        "is_front_fog_lights": "No",
        "is_rear_window_wiper": "No",
        "is_rear_window_washer": "No",
        "is_rear_window_defogger": "No",
        "is_brake_assist": "No",
        "is_power_door_locks": "No",
        "is_central_locking": "No",
        "is_power_steering": "Yes",
        "is_driver_seat_height_adjustable": "No",
        "is_day_night_rear_view_mirror": "No",
        "is_ecw": "No",
        "is_speed_alert": "Yes",
        "ncap_rating": 0,
    }

@pytest.fixture
def sample_input(sample_real_input):
    return sample_real_input.copy()

@pytest.fixture
def sample_synthetic_input():
    return {
        "age": 35,
        "gender": "Male",
        "income": 600000.0,
        "occupation": "Salaried",
        "location": "Urban",
        "policy_type": "Vehicle",
        "policy_tenure": 5,
        "premium": 25000.0,
        "coverage_amount": 1000000.0,
        "vehicle_age": 2,
        "num_dependents": 2,
        "previous_claims": 1,
        "payment_history": 0.85,
        "renewal_count": 4,
        "health_score": 0.75,
    }

@pytest.fixture
def sample_df(sample_real_input):
    return pd.DataFrame([sample_real_input])

@pytest.fixture
def sample_prediction_result():
    return {
        "prediction": "Unlikely to Claim",
        "claim_probability": 0.08,
        "non_claim_probability": 0.92,
        "risk_score": 8,
        "risk_category": "Low",
        "threshold_used": 0.10,
        "model_name": "Random Forest",
        "model_version": "v3.0-real",
        "recommendations": ["Maintain consistent policy renewal."],
        "disclaimer": "Project-scale estimation."
    }
