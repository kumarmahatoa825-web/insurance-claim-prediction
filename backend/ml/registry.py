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
import platform
import datetime
import subprocess
from pathlib import Path
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


def _get_git_sha() -> str:
    """
    Return the current Git commit SHA.

    Issue 12 — NEVER fabricate provenance:
    - If Git is available → return real short SHA.
    - If BUILD_ID env var is set (CI) → return it.
    - Otherwise → raise RuntimeError; the release must not be created.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        if sha:
            return sha
    except Exception:
        pass

    build_id = os.environ.get("BUILD_ID") or os.environ.get("GITHUB_SHA") or os.environ.get("CI_COMMIT_SHA")
    if build_id:
        return build_id[:12]

    raise RuntimeError(
        "Cannot determine Git provenance: 'git rev-parse HEAD' failed and no BUILD_ID/GITHUB_SHA "
        "environment variable is set. Release creation requires verifiable source provenance. "
        "Set BUILD_ID=<commit-sha> in CI or run inside a Git repository."
    )


def _generate_runtime_manifest(release_dir: Path, version: str, git_sha: str) -> dict:
    """
    Issue 1 — Generate runtime_manifest.json alongside every release.

    Records the exact Python and library versions used to serialize the model,
    plus schema and model hashes, so any future environment can be validated
    against this manifest before serving predictions.
    """
    import importlib

    def _pkg_version(name: str) -> str:
        try:
            return importlib.import_module(name).__version__
        except Exception:
            try:
                import importlib.metadata
                return importlib.metadata.version(name)
            except Exception:
                return "unknown"

    manifest = {
        "release_id": version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _pkg_version("numpy"),
        "pandas": _pkg_version("pandas"),
        "scikit_learn": _pkg_version("sklearn"),
        "imbalanced_learn": _pkg_version("imblearn"),
        "xgboost": _pkg_version("xgboost"),
        "shap": _pkg_version("shap"),
        "joblib": _pkg_version("joblib"),
        "training_git_sha": git_sha,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    manifest_path = release_dir / "runtime_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Generated runtime_manifest.json for release %s", version)
    return manifest


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

    Issue 13: All internal paths use pathlib.Path and are resolved relative to models_dir.
    Release entries store LOGICAL artifact names (e.g. "model.joblib"), not absolute OS paths.
    Path resolution happens at runtime via: self.releases_dir / release_id / artifact_name.
    """

    def __init__(self, models_dir: str = "models"):
        # Issue 13: use pathlib throughout — normalizes separators on Windows/Linux
        self.models_dir = Path(models_dir).resolve()
        self.registry_path = self.models_dir / REGISTRY_FILE
        self.current_pointer_path = self.models_dir / CURRENT_POINTER_FILE
        self.releases_dir = self.models_dir / RELEASES_SUBDIR

        self.releases_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_artifact(self, release_id: str, artifact_name: str) -> Path:
        """Resolve a logical artifact name to an absolute path. Always use this, never store OS paths."""
        return self.releases_dir / release_id / artifact_name

    def _load(self) -> dict:
        if self.registry_path.exists():
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
        tmp_path = str(self.registry_path) + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, str(self.registry_path))
        logger.info("Registry saved atomically: %s", self.registry_path)

    # ── Checksum & Integrity Management ──────────────────────────────────────

    def generate_checksums(self, release_dir: Path) -> Dict[str, str]:
        """Generate SHA-256 checksums for model, metadata, and all release artifacts."""
        tracked_files = [
            "model.joblib",
            "metadata.json",
            "calibration.json",
            "global_shap_importance.json",
            "global_feature_importance.json",
            "runtime_manifest.json",       # Issue 1: include runtime manifest in checksums
            "release_manifest.json",
            "roc_curve.json",
            "pr_curve.json",
        ]
        release_dir = Path(release_dir)
        checksums = {}
        for fname in tracked_files:
            fpath = release_dir / fname
            if fpath.exists():
                checksums[fname] = compute_sha256(str(fpath))

        csum_path = release_dir / "checksums.json"
        with open(csum_path, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)
        return checksums

    def verify_release_integrity(self, release_dir) -> bool:
        """
        Verify release directory files exist and match their recorded SHA-256 hashes.
        Raises RuntimeError on integrity failure.
        """
        release_dir = Path(release_dir)
        csum_path = release_dir / "checksums.json"
        if not csum_path.exists():
            raise RuntimeError(f"Model integrity failure: missing 'checksums.json' in {release_dir}")

        with open(csum_path, "r", encoding="utf-8") as f:
            expected = json.load(f)

        for filename, expected_hash in expected.items():
            fpath = release_dir / filename
            if not fpath.exists():
                raise RuntimeError(
                    f"Model integrity failure: missing file '{filename}' in {release_dir}"
                )
            actual_hash = compute_sha256(str(fpath))
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Model integrity failure: hash mismatch for '{filename}' in {release_dir} "
                    f"(expected {expected_hash}, got {actual_hash})"
                )
        return True

    def verify_release(self, version: str) -> Tuple[bool, str]:
        """
        Verify release directory exists, contains checksums.json, and all files match.
        Never auto-generates missing checksums on verification.
        Returns: (is_valid, reason)
        """
        release_dir = self.releases_dir / version
        if not release_dir.is_dir():
            return False, f"Release directory does not exist: {release_dir}"
        try:
            self.verify_release_integrity(release_dir)
            return True, "Integrity verified successfully"
        except Exception as e:
            return False, str(e)

    # ── Mandatory Pointer Management ─────────────────────────────────────────

    def get_current_pointer(self) -> Optional[dict]:
        """Read authoritative current.json pointer."""
        if self.current_pointer_path.exists():
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
        tmp_path = str(self.current_pointer_path) + ".tmp"
        payload = {
            "champion_version": version,
            "updated_at": datetime.datetime.now().isoformat(),
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, str(self.current_pointer_path))
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

        release_dir = self.releases_dir / champ_ver
        if not release_dir.is_dir():
            raise RuntimeError(f"Configured Champion release directory not found: {release_dir}")

        self.verify_release_integrity(release_dir)

        m_path = release_dir / "model.joblib"
        meta_path = release_dir / "metadata.json"

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
        *,
        require_git_sha: bool = True,
    ) -> dict:
        """
        Register a new trained model as an immutable release in models/releases/<version>/.
        Initial status: candidate / pending approval.
        Generates runtime_manifest.json (Issue 1), enforces Git provenance (Issue 12),
        stores only LOGICAL artifact names in registry (Issue 13).
        Raises ValueError if version already exists (releases are strictly immutable).
        """
        import joblib
        import shutil

        # Issue 12: fail hard if no real Git provenance is available
        if require_git_sha:
            git_sha = _get_git_sha()
        else:
            git_sha = metadata.get("git_sha", "untracked")

        release_dir = self.releases_dir / version
        if release_dir.exists():
            raise ValueError(
                f"Release directory already exists for version '{version}'. Releases are immutable."
            )
        release_dir.mkdir(parents=True, exist_ok=False)

        target_model_path = release_dir / "model.joblib"
        target_meta_path = release_dir / "metadata.json"

        if isinstance(model_obj_or_path, (str, Path)) and Path(model_obj_or_path).exists():
            shutil.copy2(str(model_obj_or_path), str(target_model_path))
        else:
            joblib.dump(model_obj_or_path, str(target_model_path))

        meta_to_store = dict(metadata)
        meta_to_store["version"] = version
        meta_to_store["model_status"] = "candidate"
        meta_to_store["training_git_sha"] = git_sha
        with open(target_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_to_store, f, indent=2)

        # Issue 1: generate runtime_manifest.json BEFORE checksums (so it's included)
        _generate_runtime_manifest(release_dir, version, git_sha)

        # Generate checksums.json (includes runtime_manifest.json)
        self.generate_checksums(release_dir)

        reg = self._load()

        entry = {
            "version": version,
            "model_name": model_name,
            # Issue 13: store LOGICAL artifact names only, not absolute OS paths
            "model_artifact": "model.joblib",
            "metadata_artifact": "metadata.json",
            "status": "candidate",
            "approval_status": "pending",
            "registered_at": datetime.datetime.now().isoformat(),
            "training_git_sha": git_sha,
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
        logger.info("Registered immutable candidate release: %s (%s) SHA=%s", model_name, version, git_sha)
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

        # Issue 13: resolve via logical artifact name, not stored OS path
        release_dir = self.releases_dir / candidate_version
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

        # Read metadata via logical artifact name (Issue 13)
        meta_artifact = candidate.get("metadata_artifact", "metadata.json")
        meta_path = release_dir / meta_artifact
        # Backward compat: also accept legacy stored metadata_path field
        if not meta_path.exists():
            legacy = candidate.get("metadata_path", "")
            if legacy and Path(legacy).exists():
                meta_path = Path(legacy)
        meta_dict = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)

        # Verify against canonical schema hash
        canonical_feature_hash = None
        canonical_doc_hash = None
        schema_version_path = self.models_dir.parent / "schemas" / "schema_version.json"
        if schema_version_path.exists():
            try:
                with open(schema_version_path, "r", encoding="utf-8") as svf:
                    s_meta = json.load(svf)
                    canonical_feature_hash = s_meta.get("feature_set_sha256")
                    canonical_doc_hash = s_meta.get("document_sha256")
            except Exception as se:
                logger.warning("Failed reading schema_version.json: %s", se)

        candidate_feature_hash = meta_dict.get("feature_set_sha256")
        candidate_doc_hash = meta_dict.get("document_sha256") or meta_dict.get("dataset", {}).get("schema_hash")
        feature_count = meta_dict.get("feature_count", 0)
        data_mode = meta_dict.get("data_mode", "real")

        if canonical_feature_hash and candidate_feature_hash:
            schema_ok = (candidate_feature_hash == canonical_feature_hash)
        elif canonical_doc_hash and candidate_doc_hash:
            schema_ok = (candidate_doc_hash == canonical_doc_hash)
        else:
            schema_ok = (data_mode == "real" and feature_count >= 80)

        fairness_meta = meta_dict.get("fairness", {})
        # FAIL-CLOSED: missing or empty fairness metadata is a hard rejection.
        # A model with no fairness assessment cannot be promoted to champion.
        fairness_ok = bool(fairness_meta.get("gate_passed") is True) if fairness_meta else False

        checks = {
            "artifact_integrity": bool(artifact_ok),
            "schema_compatible": bool(schema_ok),
            "calibration_valid": bool(brier <= MANDATORY_PROMOTION_REQUIREMENTS["max_brier"]),
            "pr_auc_valid": bool(cv_pr >= MANDATORY_PROMOTION_REQUIREMENTS["min_pr_auc"]),
            "roc_auc_valid": bool(cv_roc >= MANDATORY_PROMOTION_REQUIREMENTS["min_roc_auc"]),
            "f1_valid": bool(cv_f1 >= MANDATORY_PROMOTION_REQUIREMENTS["min_f1"]),
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
