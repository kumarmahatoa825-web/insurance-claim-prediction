"""
InsureAI — End-to-End API Smoke Test Script
===========================================
Executes synchronous/async smoke tests against all core API endpoints
using a realistic sample vehicle insurance record.
"""

import os
import sys
import asyncio
from httpx import AsyncClient, ASGITransport

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.main import app


async def run_smoke():
    print("=" * 75)
    print("InsureAI — End-to-End API Smoke Tests with Demo Data")
    print("=" * 75)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # 1. Live & Ready
            r = await ac.get("/health/live")
            assert r.status_code == 200, f"Live failed: {r.text}"
            print(f"  [PASS] GET /health/live -> {r.json()['status']}")

            r = await ac.get("/health/ready")
            assert r.status_code == 200, f"Ready failed: {r.text}"
            ready_data = r.json()
            assert ready_data["status"] == "ready"
            assert ready_data["model_loaded"] is True
            print(f"  [PASS] GET /health/ready -> status=ready, champion={ready_data['model_version']}")

            # 2. Model Info
            r = await ac.get("/model-info")
            assert r.status_code == 200
            mi = r.json()
            print(f"  [PASS] GET /model-info -> model={mi['model_name']} | threshold={mi['default_threshold']}")

            # 3. Analytics
            r = await ac.get("/analytics")
            assert r.status_code == 200
            an = r.json()
            print(f"  [PASS] GET /analytics -> total_analyzed={an['total_analyzed']} | avg_prob={an['avg_claim_probability']}")

            # 4. Monitoring
            r = await ac.get("/monitoring")
            assert r.status_code == 200
            mo = r.json()
            print(f"  [PASS] GET /monitoring -> overall_status={mo['overall_status']} | test_roc_auc={mo['performance_metrics']['test_roc_auc']}")

            # 5. Predict with sample real payload
            sample_payload = {
                "policy_id": "POLICY_DEMO_001",
                "policy_tenure": 0.528794,
                "age_of_car": 0.08,
                "age_of_policyholder": 0.644231,
                "area_cluster": "C8",
                "population_density": 8794,
                "make": 1,
                "segment": "B2",
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
                "ncap_rating": 0
            }

            r = await ac.post("/predict", json=sample_payload)
            assert r.status_code == 200, f"Predict failed: {r.text}"
            pred = r.json()
            print(f"  [PASS] POST /predict -> prob={pred['claim_probability']} | risk_score={pred['risk_score']} | category={pred['risk_category']} | label={pred.get('prediction')}")

            # 6. Explain endpoint
            r = await ac.post("/explain", json=sample_payload)
            assert r.status_code == 200, f"Explain failed: {r.text}"
            exp = r.json()
            has_shap = exp.get("available", False)
            print(f"  [PASS] POST /explain -> available={has_shap} | factors count={len(exp.get('top_risk_factors', []))}")

            # 7. Target leakage rejection (is_claim)
            leakage_payload = sample_payload.copy()
            leakage_payload["is_claim"] = 1
            r = await ac.post("/predict", json=leakage_payload)
            assert r.status_code == 422, f"Expected 422 on target leakage, got {r.status_code}"
            print("  [PASS] POST /predict with is_claim -> properly rejected with HTTP 422 (target leakage prevented)")

            # 8. Client threshold override rejection
            override_payload = sample_payload.copy()
            override_payload["threshold"] = 0.50
            r = await ac.post("/predict", json=override_payload)
            assert r.status_code == 422, f"Expected 422 on client threshold override, got {r.status_code}"
            print("  [PASS] POST /predict with threshold -> properly rejected with HTTP 422 (client override prevented)")

            # 9. Real test set batch inference
            r = await ac.post("/batch-predict-real-test")
            assert r.status_code == 200, f"Real test batch failed: {r.text}"
            bt = r.json()
            print(f"  [PASS] POST /batch-predict-real-test -> returned {bt.get('total_predictions')} predictions")

    print("=" * 75)
    print("ALL API SMOKE TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_smoke())
