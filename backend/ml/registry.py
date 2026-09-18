"""
InsureAI — Production Model Registry
====================================
Manages candidate and champion model lifecycles with strictly immutable releases,
SHA-256 checksum integrity verification, mandatory atomic current.json pointers,
explicit promotion quality gates, and pointer-only rollbacks.

Directory Layout:
  models/
  ├── registry.json           # Catalog of champion, candidates, and deployment history
  ├── current.json            # Authoritative atomic pointer: {"champion_version": "v4.0-real"}
  └── releases/
      ├── v4.0-real/
      │   ├── model.joblib
      │   ├── metadata.json
      │   └── checksums.json  # SHA-256 hashes of model and metadata
      └── ...

Champion/Candidate Workflow:
  1. Training scripts register models strictly as CANDIDATES in models/releases/<version>/.
     Initial state: status="candidate", approval_status="pending".
  2. Candidate is evaluated against explicit promotion gates (artifact integrity, schema,
     PR-AUC, ROC-AUC, Brier score, fairness, and non-regression).
  3. Candidate is explicitly approved via approve_candidate().
  4. Candidate is promoted via promote_candidate() -> atomically updates current.json pointer.
  5. rollback() atomically points current.json to the previous champion in deployment_history.
"""

import os
import json
import hashlib
import logging
import datetime
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

REGISTRY_FILE = "registry.json"
CURRENT_POINTER_FILE = "current.json"
RELEASES_SUBDIR = "releases"

# Mandatory quality gates for a candidate to be eligible for promotion
MANDATORY_PROMOTION_REQUIREMENTS = {
    "min_roc_auc": 0.55,
    "min_f1": 0.10,
    "max_brier": 0.15,
    "min_pr_auc": 0.08,
}


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash formatted as sha256:<hex>."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


class ModelRegistry:
    """
    Production-grade Model Registry using strictly immutable releases and atomic pointers.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.registry_path = os.path.join(models_dir, REGISTRY_FILE)
        self.current_pointer_path = os.path.join(models_dir, CURRENT_POINTER_FILE)
        self.releases_dir = os.path.join(models_dir, RELEASES_SUBDIR)

        os.makedirs(self.releases_dir, exist_ok=True)

    def _load(self) -> dict:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to read registry file: %s", e)
        return {
            "champion": None,
            "candidates": [],
            "history": [],
            "deployment_history": [],
            "last_updated": None,
        }

    def _save(self, data: dict) -> None:
        data["last_updated"] = datetime.datetime.now().isoformat()
        tmp_path = self.registry_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.registry_path)
        logger.info("Registry saved atomically: %s", self.registry_path)

    # ── Checksum & Integrity Management ──────────────────────────────────────

    def generate_checksums(self, release_dir: str) -> Dict[str, str]:
        """Generate SHA-256 checksums for model and metadata in a release directory."""
        model_path = os.path.join(release_dir, "model.joblib")
        meta_path = os.path.join(release_dir, "metadata.json")
        checksums = {}
        if os.path.exists(model_path):
            checksums["model.joblib"] = compute_sha256(model_path)
        if os.path.exists(meta_path):
            checksums["metadata.json"] = compute_sha256(meta_path)

        csum_path = os.path.join(release_dir, "checksums.json")
        with open(csum_path, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)
        return checksums

    def verify_release_integrity(self, release_dir: str) -> bool:
        """
        Verify release directory files exist and match their recorded SHA-256 hashes.
        Raises RuntimeError on integrity failure.
        """
        csum_path = os.path.join(release_dir, "checksums.json")
        if not os.path.exists(csum_path):
            raise RuntimeError(f"Model integrity failure: missing 'checksums.json' in {release_dir}")

        with open(csum_path, "r", encoding="utf-8") as f:
            expected = json.load(f)

        for filename, expected_hash in expected.items():
            fpath = os.path.join(release_dir, filename)
            if not os.path.exists(fpath):
                raise RuntimeError(f"Model integrity failure: missing file '{filename}' in {release_dir}")
            actual_hash = compute_sha256(fpath)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Model integrity failure: hash mismatch for '{filename}' in {release_dir} "
                    f"(expected {expected_hash}, got {actual_hash})"
                )
        return True

    # ── Mandatory Pointer Management ─────────────────────────────────────────

    def get_current_pointer(self) -> Optional[dict]:
        """Read authoritative current.json pointer."""
        if os.path.exists(self.current_pointer_path):
            try:
                with open(self.current_pointer_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to read current.json: %s", e)
        return None

    def get_current_champion_version(self) -> Optional[str]:
        """Return active champion version string from current.json."""
        ptr = self.get_current_pointer()
        if ptr:
            return ptr.get("champion_version")
        return None

    def get_champion_version(self) -> str:
        """
        Return authoritative active champion version.
        Raises RuntimeError if no active champion is configured in current.json.
        """
        version = self.get_current_champion_version()
        if not version:
            raise RuntimeError("No active Champion is configured.")
        return version

    def _set_current_champion_pointer(self, version: str) -> None:
        """Atomically update current.json pointer using os.replace."""
        tmp_path = self.current_pointer_path + ".tmp"
        payload = {
            "champion_version": version,
            "updated_at": datetime.datetime.now().isoformat(),
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.current_pointer_path)
        logger.info("Atomically updated current.json pointer to: %s", version)

    def get_active_champion_paths(self) -> Tuple[str, str]:
        """
        Resolve active model.joblib and metadata.json file paths strictly
        from current.json pointer. Verifies checksum integrity before returning.
        Raises RuntimeError if current.json is missing or corrupted.
        """
        champ_ver = self.get_current_champion_version()
        if not champ_ver:
            raise RuntimeError("No active Champion is configured.")

        release_dir = os.path.join(self.releases_dir, champ_ver)
        if not os.path.isdir(release_dir):
            raise RuntimeError(f"Configured Champion release directory not found: {release_dir}")

        self.verify_release_integrity(release_dir)

        m_path = os.path.join(release_dir, "model.joblib")
        meta_path = os.path.join(release_dir, "metadata.json")

        if not os.path.exists(m_path):
            raise RuntimeError(f"Model artifact not found: {m_path}")
        if not os.path.exists(meta_path):
            raise RuntimeError(f"Model metadata not found: {meta_path}")

        return m_path, meta_path

    # ── Candidate Registration ───────────────────────────────────────────────

    def register_candidate(
        self,
        model_name: str,
        version: str,
        model_obj_or_path: Any,
        metadata: dict,
    ) -> dict:
        """
        Register a new trained model as an immutable release in models/releases/<version>/.
        Initial status: candidate / pending approval.
        Computes SHA-256 checksums and stores in checksums.json.
        Raises ValueError if version already exists (releases are strictly immutable).
        """
        import joblib
        import shutil

        release_dir = os.path.join(self.releases_dir, version)
        if os.path.exists(release_dir):
            raise ValueError(f"Release directory already exists for version '{version}'. Releases are immutable.")
        os.makedirs(release_dir, exist_ok=False)

        target_model_path = os.path.join(release_dir, "model.joblib")
        target_meta_path = os.path.join(release_dir, "metadata.json")

        if isinstance(model_obj_or_path, str) and os.path.exists(model_obj_or_path):
            shutil.copy2(model_obj_or_path, target_model_path)
        else:
            joblib.dump(model_obj_or_path, target_model_path)

        meta_to_store = dict(metadata)
        meta_to_store["version"] = version
        meta_to_store["model_status"] = "candidate"
        with open(target_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_to_store, f, indent=2)

        # Generate checksums.json immediately
        self.generate_checksums(release_dir)

        reg = self._load()

        entry = {
            "version": version,
            "model_name": model_name,
            "model_path": target_model_path,
            "metadata_path": target_meta_path,
            "status": "candidate",
            "approval_status": "pending",
            "registered_at": datetime.datetime.now().isoformat(),
            "metrics": {
                "cv_f1_mean": metadata.get("cv_metrics", {}).get("mean", {}).get("f1") or metadata.get("test_metrics", {}).get("f1") or metadata.get("f1", 0),
                "cv_roc_auc_mean": metadata.get("cv_metrics", {}).get("mean", {}).get("roc_auc") or metadata.get("test_metrics", {}).get("roc_auc") or metadata.get("roc_auc", 0),
                "cv_pr_auc_mean": metadata.get("cv_metrics", {}).get("mean", {}).get("pr_auc") or metadata.get("test_metrics", {}).get("pr_auc") or metadata.get("pr_auc", 0),
                "test_f1": metadata.get("test_metrics", {}).get("f1") or metadata.get("holdout_metrics", {}).get("f1", 0),
                "test_roc_auc": metadata.get("test_metrics", {}).get("roc_auc") or metadata.get("holdout_metrics", {}).get("roc_auc", 0),
                "test_pr_auc": metadata.get("test_metrics", {}).get("pr_auc") or metadata.get("holdout_metrics", {}).get("pr_auc", 0),
                "brier_score": metadata.get("brier_score") or metadata.get("holdout_metrics", {}).get("brier_score", 0),
            },
            "threshold": metadata.get("default_threshold") or (
                metadata.get("threshold", {}).get("selected") if isinstance(metadata.get("threshold"), dict) else None
            ),
            "training_date": metadata.get("training_date", ""),
            "data_mode": metadata.get("data_mode", "real"),
        }

        reg["candidates"] = [c for c in reg.get("candidates", []) if c["version"] != version]
        reg["candidates"].append(entry)
        self._save(reg)
        logger.info("Registered immutable candidate release: %s (%s)", model_name, version)
        return entry

    # ── Comparison & Gate Verification ───────────────────────────────────────

    def compare_with_champion(self, candidate_version: str) -> dict:
        """
        Evaluate candidate against explicit promotion gates using 5-fold CV development metrics:
          1. artifact_integrity: model, metadata, and checksums valid
          2. schema_compatible: canonical schema hash matches schemas/vehicle_insurance_v1.json
          3. calibration_valid: Brier score <= 0.15
          4. pr_auc_valid: CV PR-AUC >= 0.08
          5. roc_auc_valid: CV ROC-AUC >= 0.55
          6. fairness_check: fairness assessment gate passed
          7. candidate_not_materially_worse_than_champion: non-regression vs active champion on CV
        """
        reg = self._load()
        candidate = next(
            (c for c in reg.get("candidates", []) if c["version"] == candidate_version),
            None,
        )
        if not candidate:
            raise ValueError(f"Candidate '{candidate_version}' not found in registry.")

        release_dir = os.path.join(self.releases_dir, candidate_version)
        artifact_ok = False
        try:
            artifact_ok = self.verify_release_integrity(release_dir)
        except Exception as e:
            logger.warning("Artifact integrity check failed for %s: %s", candidate_version, e)

        c_metrics = candidate.get("metrics", {})
        cv_roc = c_metrics.get("cv_roc_auc_mean", 0)
        cv_pr = c_metrics.get("cv_pr_auc_mean", 0)
        cv_f1 = c_metrics.get("cv_f1_mean", 0)
        brier = c_metrics.get("brier_score", 1.0)

        # Read metadata for schema and fairness checks
        meta_path = candidate.get("metadata_path", "")
        meta_dict = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)

        # Verify against canonical schema hash
        canonical_hash = None
        schema_path = os.path.abspath(os.path.join(self.models_dir, "..", "schemas", "vehicle_insurance_v1.json"))
        if os.path.exists(schema_path):
            with open(schema_path, "rb") as sf:
                canonical_hash = f"sha256:{hashlib.sha256(sf.read()).hexdigest()}"

        candidate_schema_hash = meta_dict.get("dataset", {}).get("schema_hash") or meta_dict.get("schema_hash")
        feature_count = meta_dict.get("feature_count", 0)
        data_mode = meta_dict.get("data_mode", "real")

        if canonical_hash and candidate_schema_hash:
            schema_ok = (candidate_schema_hash == canonical_hash)
        else:
            schema_ok = (data_mode == "real" and feature_count >= 80)

        fairness_meta = meta_dict.get("fairness", {})
        fairness_ok = fairness_meta.get("gate_passed", True) if fairness_meta else True

        checks = {
            "artifact_integrity": bool(artifact_ok),
            "schema_compatible": bool(schema_ok),
            "calibration_valid": bool(brier <= MANDATORY_PROMOTION_REQUIREMENTS["max_brier"]),
            "pr_auc_valid": bool(cv_pr >= MANDATORY_PROMOTION_REQUIREMENTS["min_pr_auc"]),
            "roc_auc_valid": bool(cv_roc >= MANDATORY_PROMOTION_REQUIREMENTS["min_roc_auc"]),
            "fairness_check": bool(fairness_ok),
        }

        # Check against champion (using development cross-validation to protect holdout)
        champ_ver = self.get_current_champion_version()
        champ_metrics = {}
        if champ_ver:
            champ = reg.get("champion")
            if champ:
                champ_metrics = champ.get("metrics", {})
                champ_roc = champ_metrics.get("cv_roc_auc_mean") or champ_metrics.get("test_roc_auc", 0)
                champ_f1 = champ_metrics.get("cv_f1_mean") or champ_metrics.get("test_f1", 0)
                checks["candidate_not_materially_worse_than_champion"] = (
                    cv_f1 >= (champ_f1 - 0.02) and cv_roc >= (champ_roc - 0.02)
                )
            else:
                checks["candidate_not_materially_worse_than_champion"] = True
        else:
            checks["candidate_not_materially_worse_than_champion"] = True

        mandatory_passed = all(checks.values())
        recommendation = "promote" if mandatory_passed else "reject"

        return {
            "candidate_version": candidate_version,
            "champion_version": champ_ver,
            "checks": checks,
            "mandatory_passed": mandatory_passed,
            "recommendation": recommendation,
            "champion_metrics": champ_metrics,
            "candidate_metrics": c_metrics,
        }

    # ── Approval & Promotion ─────────────────────────────────────────────────

    def approve_candidate(self, candidate_version: str) -> None:
        """Mark candidate as approved (explicit human-in-the-loop gate)."""
        reg = self._load()
        for cand in reg.get("candidates", []):
            if cand["version"] == candidate_version:
                cand["approval_status"] = "approved"
                cand["approved_at"] = datetime.datetime.now().isoformat()
                self._save(reg)
                logger.info("Candidate explicitly approved: %s", candidate_version)
                return
        raise ValueError(f"Candidate '{candidate_version}' not found.")

    def promote_candidate(self, candidate_version: str) -> dict:
        """
        Promote an approved candidate to champion.
        Requires approval_status='approved' and all mandatory quality gates passed.
        Atomically updates current.json pointer and records in deployment_history.
        Release directory remains 100% immutable (no metadata edits or file copies).
        """
        reg = self._load()

        candidate = next(
            (c for c in reg.get("candidates", []) if c["version"] == candidate_version),
            None,
        )
        if not candidate:
            raise ValueError(f"Candidate '{candidate_version}' not found.")

        if candidate.get("approval_status") != "approved":
            raise ValueError(f"Candidate '{candidate_version}' is not approved. Must call approve_candidate() first.")

        comparison = self.compare_with_champion(candidate_version)
        if not comparison["mandatory_passed"]:
            failed = [k for k, v in comparison["checks"].items() if not v]
            raise ValueError(f"Candidate failed mandatory promotion checks: {failed}")

        # Archive old champion in registry history
        old_champ = reg.get("champion")
        if old_champ:
            old = old_champ.copy()
            old["archived_at"] = datetime.datetime.now().isoformat()
            old["status"] = "archived"
            reg.setdefault("history", []).append(old)

        candidate["status"] = "champion"
        candidate["promoted_at"] = datetime.datetime.now().isoformat()
        candidate["promotion_gates"] = comparison["checks"]

        # Atomically update current.json pointer
        self._set_current_champion_pointer(candidate_version)

        # Record in deployment_history
        if "deployment_history" not in reg:
            reg["deployment_history"] = []
        reg["deployment_history"].append({
            "version": candidate_version,
            "activated_at": datetime.datetime.now().isoformat(),
            "reason": "promotion",
        })

        reg["champion"] = candidate
        reg["candidates"] = [c for c in reg.get("candidates", []) if c["version"] != candidate_version]
        self._save(reg)
        logger.info("Successfully promoted candidate %s to Active Champion via current.json pointer.", candidate_version)
        return candidate

    # ── Pointer-Only Rollback ────────────────────────────────────────────────

    def rollback(self) -> dict:
        """
        Pointer-only rollback to the previous active champion.
        Atomically updates current.json pointer to the previous release in deployment_history.
        Zero file copies.
        """
        reg = self._load()
        dep_history = reg.get("deployment_history", [])
        curr_ver = self.get_current_champion_version()

        prev_version = None
        # Walk backwards through deployment history to find previous distinct version
        for entry in reversed(dep_history):
            if entry.get("version") != curr_ver:
                prev_version = entry.get("version")
                break

        # Fallback to history array
        if not prev_version and reg.get("history"):
            prev_version = reg["history"][-1].get("version")

        if not prev_version:
            raise ValueError("No previous champion in history to rollback to.")

        prev_release_dir = os.path.join(self.releases_dir, prev_version)
        if not os.path.isdir(prev_release_dir):
            raise RuntimeError(f"Release directory for {prev_version} not found: {prev_release_dir}")

        # Verify integrity of target release
        self.verify_release_integrity(prev_release_dir)

        # Archive current champion
        if reg.get("champion"):
            curr = reg["champion"].copy()
            curr["archived_at"] = datetime.datetime.now().isoformat()
            curr["status"] = "rolled_back"
            reg.setdefault("history", []).append(curr)

        # Update pointer atomically
        self._set_current_champion_pointer(prev_version)

        # Record rollback in deployment_history
        dep_history.append({
            "version": prev_version,
            "activated_at": datetime.datetime.now().isoformat(),
            "reason": "rollback",
        })
        reg["deployment_history"] = dep_history

        # Read restored metadata
        prev_meta_path = os.path.join(prev_release_dir, "metadata.json")
        prev_meta = {}
        if os.path.exists(prev_meta_path):
            with open(prev_meta_path, "r", encoding="utf-8") as f:
                prev_meta = json.load(f)

        restored_entry = {
            "version": prev_version,
            "model_name": prev_meta.get("model_name", "Unknown"),
            "model_path": os.path.join(prev_release_dir, "model.joblib"),
            "metadata_path": prev_meta_path,
            "status": "champion",
            "approval_status": "approved",
            "metrics": {
                "cv_f1_mean": prev_meta.get("cv_metrics", {}).get("mean", {}).get("f1", 0),
                "cv_roc_auc_mean": prev_meta.get("cv_metrics", {}).get("mean", {}).get("roc_auc", 0),
                "cv_pr_auc_mean": prev_meta.get("cv_metrics", {}).get("mean", {}).get("pr_auc", 0),
                "test_f1": prev_meta.get("test_metrics", {}).get("f1") or prev_meta.get("holdout_metrics", {}).get("f1", 0),
                "test_roc_auc": prev_meta.get("test_metrics", {}).get("roc_auc") or prev_meta.get("holdout_metrics", {}).get("roc_auc", 0),
                "test_pr_auc": prev_meta.get("test_metrics", {}).get("pr_auc") or prev_meta.get("holdout_metrics", {}).get("pr_auc", 0),
                "brier_score": prev_meta.get("brier_score", 0),
            },
            "threshold": prev_meta.get("default_threshold"),
            "training_date": prev_meta.get("training_date", ""),
            "restored_at": datetime.datetime.now().isoformat(),
        }

        reg["champion"] = restored_entry
        self._save(reg)
        logger.info("Successfully rolled back champion pointer to: %s", prev_version)
        return restored_entry

    # ── Accessors ────────────────────────────────────────────────────────────

    def get_champion(self) -> Optional[dict]:
        """Return current champion entry."""
        reg = self._load()
        champ = reg.get("champion")
        if champ:
            c_ver = champ.get("version")
            if c_ver:
                rel_dir = os.path.join(self.releases_dir, c_ver)
                m_path = os.path.join(rel_dir, "model.joblib")
                if os.path.exists(m_path):
                    champ["model_path"] = m_path
        return champ

    def get_candidates(self) -> List[dict]:
        """Return all registered candidates."""
        reg = self._load()
        return reg.get("candidates", [])

    def get_history(self) -> List[dict]:
        """Return archived champion history."""
        reg = self._load()
        return reg.get("history", [])

    def get_deployment_history(self) -> List[dict]:
        """Return audit trail of all pointer activations."""
        reg = self._load()
        return reg.get("deployment_history", [])

    def to_dict(self) -> dict:
        return self._load()
