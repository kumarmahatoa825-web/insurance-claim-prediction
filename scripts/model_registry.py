"""
InsureAI — Model Registry CLI
=============================
Authoritative CLI interface for Model Registry operations:
  - approve: mark candidate as approved
  - promote: evaluate gates and atomically promote candidate to Champion
  - rollback: atomically rollback to previous Champion
  - status: display active Champion and registered candidates
  - list: list all releases in models/releases/

Usage:
  python scripts/model_registry.py approve --version v4.0-real
  python scripts/model_registry.py promote --version v4.0-real
  python scripts/model_registry.py rollback
  python scripts/model_registry.py status
"""

import os
import sys
import argparse
import logging

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.ml.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_registry_cli")


def main():
    parser = argparse.ArgumentParser(description="InsureAI Model Registry CLI")
    subparsers = parser.add_subparsers(dest="action", help="Action to execute")

    # approve
    p_approve = subparsers.add_parser("approve", help="Approve a candidate model release.")
    p_approve.add_argument("--version", type=str, required=True, help="Candidate version string.")
    p_approve.add_argument("--models-dir", type=str, default="models", help="Models directory path.")

    # promote
    p_promote = subparsers.add_parser("promote", help="Promote an approved candidate to Active Champion.")
    p_promote.add_argument("--version", type=str, required=True, help="Candidate version string.")
    p_promote.add_argument("--models-dir", type=str, default="models", help="Models directory path.")

    # rollback
    p_rollback = subparsers.add_parser("rollback", help="Rollback Champion to previous release.")
    p_rollback.add_argument("--models-dir", type=str, default="models", help="Models directory path.")

    # status
    p_status = subparsers.add_parser("status", help="Show current Champion and candidate versions.")
    p_status.add_argument("--models-dir", type=str, default="models", help="Models directory path.")

    args = parser.parse_args()
    if not args.action:
        parser.print_help()
        sys.exit(1)

    models_dir = getattr(args, "models_dir", "models")
    reg = ModelRegistry(models_dir)

    if args.action == "approve":
        try:
            reg.approve_candidate(args.version)
            logger.info("[OK] Candidate %s approved successfully.", args.version)
        except Exception as e:
            logger.error("[FAIL] Approval failed: %s", e)
            sys.exit(1)

    elif args.action == "promote":
        # 1. Evaluate gates
        try:
            comp = reg.compare_with_champion(args.version)
            logger.info("=== Promotion Gate Evaluation for %s ===", args.version)
            for check, passed in comp["checks"].items():
                status_icon = "PASS" if passed else "FAIL"
                logger.info("  [%s] %s", status_icon, check)
            logger.info("Gate Recommendation: %s", comp["recommendation"].upper())
        except Exception as e:
            logger.warning("Gate comparison warning: %s", e)

        # 2. Promote
        try:
            champ = reg.promote_candidate(args.version)
            logger.info("[OK] Successfully promoted %s to Active Champion!", args.version)
            logger.info("Active Champion Pointer: %s", reg.get_champion_version())
            logger.info("Model Path: %s", champ.get("model_path"))
        except Exception as e:
            logger.error("[FAIL] Promotion failed: %s", e)
            sys.exit(1)

    elif args.action == "rollback":
        try:
            restored = reg.rollback()
            logger.info("[OK] Successfully rolled back to: %s", restored.get("version"))
            logger.info("Active Champion Pointer: %s", reg.get_champion_version())
        except Exception as e:
            logger.error("[FAIL] Rollback failed: %s", e)
            sys.exit(1)

    elif args.action == "status":
        curr = reg.get_current_champion_version()
        print("\n" + "=" * 50)
        print(f"Authoritative Champion (current.json): {curr or 'None (Not configured)'}")
        candidates = reg.get_candidates()
        print(f"Registered Candidates ({len(candidates)}):")
        for c in candidates:
            print(f"  - {c['version']} | {c['model_name']} | Status: {c.get('approval_status')}")
        dep_hist = reg.get_deployment_history()
        print(f"Deployment Activations ({len(dep_hist)}):")
        for d in dep_hist:
            print(f"  - {d.get('version')} ({d.get('reason')}) at {d.get('activated_at')}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
