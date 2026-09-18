"""
InsureAI — Model Promotion Script
===================================
Approves a candidate model in the registry and atomically promotes it to Champion.

Usage:
  python scripts/promote_model.py --version VERSION
  python scripts/promote_model.py --latest
"""

import sys
import os
import argparse
import logging

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.ml.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("promote_model")


def main():
    parser = argparse.ArgumentParser(description="Approve and promote a candidate model to Champion.")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Specific candidate version to promote.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Automatically select the most recent candidate model in registry.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Path to models directory (default: models).",
    )

    args = parser.parse_args()
    reg = ModelRegistry(args.models_dir)

    target_version = args.version
    if not target_version:
        candidates = reg.get_candidates()
        if not candidates:
            logger.error("No candidate models found in registry. Train a model first via `python -m backend.ml.train_real`.")
            sys.exit(1)
        if args.latest or len(candidates) == 1:
            target_version = candidates[-1]["version"]
            logger.info("Selected candidate: %s", target_version)
        else:
            logger.error("Multiple candidates found. Specify --version <ver> or use --latest.")
            for c in candidates:
                print(f"  - {c['version']} ({c['model_name']}) registered at {c.get('registered_at')}")
            sys.exit(1)

    # 1. Compare with current champion
    try:
        comparison = reg.compare_with_champion(target_version)
        logger.info("=== Promotion Gate Evaluation for %s ===", target_version)
        for check, passed in comparison["checks"].items():
            status_str = "PASS" if passed else "FAIL"
            logger.info("  [%s] %s", status_str, check)
        logger.info("Recommendation: %s", comparison["recommendation"].upper())
    except Exception as e:
        logger.warning("Could not run champion comparison: %s", e)

    # 2. Approve candidate
    try:
        reg.approve_candidate(target_version)
        logger.info("Candidate %s successfully approved.", target_version)
    except Exception as e:
        logger.error("Failed to approve candidate: %s", e)
        sys.exit(1)

    # 3. Promote candidate to champion
    try:
        champ = reg.promote_candidate(target_version)
        logger.info("Successfully promoted candidate %s to Active Champion!", target_version)
        logger.info("Champion Pointer: %s", reg.get_current_champion_version())
        logger.info("Model Path: %s", champ.get("model_path"))
        logger.info("Metadata Path: %s", champ.get("metadata_path"))
    except Exception as e:
        logger.error("Failed to promote candidate: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
