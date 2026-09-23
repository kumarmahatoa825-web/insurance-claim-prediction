"""
InsureAI — Live Production Server Verification Script
====================================================
Tests live network connections to:
  1. FastAPI Backend on http://127.0.0.1:8000
  2. Vite React Frontend on http://localhost:5173
"""

import urllib.request
import json
import time

BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://localhost:5173"

def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "InsureAI-Verifier"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.read().decode("utf-8")

def http_post(url, data_dict):
    payload = json.dumps(data_dict).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "InsureAI-Verifier"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))

def run_live_checks():
    print("=" * 80)
    print("INSUREAI — LIVE SYSTEM INTEGRATION & ENDPOINT VERIFICATION")
    print("=" * 80)
    
    # 1. Check Frontend
    print("\n[1/12] Testing React Frontend (http://localhost:5173)...")
    status, html = http_get(FRONTEND_URL)
    assert status == 200 and "<div id=\"root\">" in html, f"Frontend unexpected response: status={status}"
    print(f"  --> [PASS] Frontend HTTP 200 OK | Title/Root confirmed ({len(html)} bytes)")

    # 2. Check Backend Liveness
    print("\n[2/12] Testing Backend Liveness (/health/live)...")
    status, body = http_get(f"{BACKEND_URL}/health/live")
    data = json.loads(body)
    assert status == 200 and data.get("status") == "alive"
    print(f"  --> [PASS] Backend Liveness HTTP 200 OK | Status: '{data.get('status')}'")

    # 3. Check Backend Readiness
    print("\n[3/12] Testing Backend Readiness (/health/ready)...")
    status, body = http_get(f"{BACKEND_URL}/health/ready")
    data = json.loads(body)
    assert status == 200 and data.get("status") == "ready" and data.get("model_loaded") is True
    print(f"  --> [PASS] Backend Readiness HTTP 200 OK | Active Champion: '{data.get('model_version')}' | Checksums: {data.get('checksums_verified')}")

    # 4. Check Model Info
    print("\n[4/12] Testing Model Info (/model-info)...")
    status, body = http_get(f"{BACKEND_URL}/model-info")
    data = json.loads(body)
    assert status == 200
    print(f"  --> [PASS] Model Info HTTP 200 OK | Model: '{data.get('model_name')}' | Default Threshold: {data.get('default_threshold')}")

    # 5. Check Analytics
    print("\n[5/12] Testing Analytics Aggregation (/analytics)...")
    status, body = http_get(f"{BACKEND_URL}/analytics")
    data = json.loads(body)
    assert status == 200 and data.get("total_analyzed", 0) > 0
    print(f"  --> [PASS] Analytics HTTP 200 OK | Total Analyzed: {data.get('total_analyzed'):,} | Avg Claim Prob: {data.get('avg_claim_probability')} | Actual Claim Rate: {data.get('actual_claim_rate')}")

    # 6. Check Monitoring
    print("\n[6/12] Testing Monitoring & Drift Engine (/monitoring)...")
    status, body = http_get(f"{BACKEND_URL}/monitoring")
    data = json.loads(body)
    assert status == 200
    perf = data.get("performance_metrics", {})
    print(f"  --> [PASS] Monitoring HTTP 200 OK | Overall: '{data.get('overall_status')}' | Holdout ROC-AUC: {perf.get('test_roc_auc')} | Holdout PR-AUC: {perf.get('test_pr_auc')}")

    # 7. Check Model Registry Endpoint
    print("\n[7/12] Testing Model Registry (/registry)...")
    status, body = http_get(f"{BACKEND_URL}/registry")
    data = json.loads(body)
    assert status == 200 and "champion" in data
    champ = data.get("champion", {})
    print(f"  --> [PASS] Registry HTTP 200 OK | Champion: '{champ.get('version')}' | Status: '{champ.get('status')}'")

    # Sample realistic vehicle payload
    sample_policy = {
        "policy_id": "POL_LIVE_DEMO_999",
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

    # 8. Single Prediction
    print("\n[8/12] Testing Single Prediction (/predict)...")
    status, pred = http_post(f"{BACKEND_URL}/predict", sample_policy)
    assert status == 200 and "claim_probability" in pred
    print(f"  --> [PASS] Prediction HTTP 200 OK | Probability: {pred.get('claim_probability')} | Risk Score: {pred.get('risk_score')}/100 | Tier: '{pred.get('risk_category')}' | Decision: '{pred.get('prediction')}'")

    # 9. Explainability & SHAP Waterfall
    print("\n[9/12] Testing Explainability & SHAP Attribution (/explain)...")
    status, exp = http_post(f"{BACKEND_URL}/explain", sample_policy)
    assert status == 200 and exp.get("available") is True
    factors = exp.get("top_risk_factors", [])
    top_factor = factors[0] if factors else {}
    print(f"  --> [PASS] Explainability HTTP 200 OK | SHAP Available: {exp.get('available')} | Risk Factors: {len(factors)} (Top Driver: '{top_factor.get('feature')}', Impact: {top_factor.get('impact_percentage')}%)")

    # 10. Cost Analysis
    print("\n[10/12] Testing Cost Optimization Engine (/cost-analysis)...")
    status, cost_res = http_post(f"{BACKEND_URL}/cost-analysis", {"fp_cost": 500.0, "fn_cost": 5000.0})
    assert status == 200 and "optimal_threshold" in cost_res
    print(f"  --> [PASS] Cost Analysis HTTP 200 OK | Optimal Cost Threshold: {cost_res.get('optimal_threshold')} | Min Expected Cost: ${cost_res.get('min_expected_cost'):,.2f}")

    # 11. Official Test Set Batch Prediction
    print("\n[11/12] Testing Unseen Test Set Batch Demo (/batch-predict-real-test)...")
    status, batch_res = http_post(f"{BACKEND_URL}/batch-predict-real-test", {})
    assert status == 200
    sample_count = len(batch_res.get('predictions', []))
    avg_prob = batch_res.get('avg_claim_probability')
    print(f"  --> [PASS] Batch Predict Real Test HTTP 200 OK | Sample Inferences Executed: {sample_count} | Avg Batch Prob: {avg_prob}")

    # 12. Security & Target Leakage Guardrails
    print("\n[12/12] Testing Guardrails (Target Leakage & Client Threshold Override)...")
    
    # Target Leakage check
    bad_leakage = sample_policy.copy()
    bad_leakage["is_claim"] = 1
    status, leak_res = http_post(f"{BACKEND_URL}/predict", bad_leakage)
    assert status == 422, f"Target leakage was not rejected with 422! Got status {status}"
    print(f"  --> [PASS] Target Leakage ('is_claim') rejected with HTTP 422 ({leak_res.get('detail')})")

    # Client Override check
    bad_override = sample_policy.copy()
    bad_override["threshold"] = 0.50
    status, over_res = http_post(f"{BACKEND_URL}/predict", bad_override)
    assert status == 422, f"Client threshold override was not rejected with 422! Got status {status}"
    print(f"  --> [PASS] Client Threshold Override rejected with HTTP 422 ({over_res.get('detail')})")

    print("\n" + "=" * 80)
    print("ALL 12 LIVE CHECKS PASSED PERFECTLY! THE SYSTEM IS 100% OPERATIONAL.")
    print("=" * 80)

if __name__ == "__main__":
    run_live_checks()
