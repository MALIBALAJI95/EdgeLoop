"""Model registry, safe promotion gatekeeper, and rollback manager."""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from edgeloop.common.config import get_config
from edgeloop.common.database import get_db
from edgeloop.training.evaluator import (
    ModelEvaluator,
    EvaluationReport,
    ComparisonReport
)

logger = logging.getLogger("edgeloop.registry")


class PromotionRejectionError(RuntimeError):
    """Raised when a candidate fails governance gates or regression criteria."""
    pass


class ModelEntry(BaseModel):
    id: int
    version: str
    model_name: str
    checkpoint_path: str
    stage: str
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    trash_f1: Optional[float] = None
    metrics_json: Optional[Dict[str, Any]] = None
    approved_by: Optional[str] = None
    promotion_reason: Optional[str] = None
    created_at: str
    updated_at: str


class ModelRegistry:
    """Manages model artifacts, stage transitions, and atomic champion promotions/rollbacks."""

    VALID_STAGES = ["candidate", "evaluated", "approved", "champion", "archived"]

    def __init__(self):
        self.config = get_config()
        self.db = get_db()
        self.evaluator = ModelEvaluator()

        self.champion_dir = self.config.champion_dir
        self.champion_dir.mkdir(parents=True, exist_ok=True)
        self.champion_ckpt_path = self.champion_dir / "mobilenetv2_champion.pth"
        self.previous_champion_ckpt_path = self.champion_dir / "mobilenetv2_previous_champion.pth"

        # Ensure champion is initialized
        self._ensure_baseline_champion_seeded()

    def _ensure_baseline_champion_seeded(self) -> None:
        """Seed baseline checkpoint as initial champion if no champion is registered."""
        champ = self.db.get_champion_model()
        if champ is None:
            baseline_p = self.config.baseline_checkpoint_path
            if baseline_p.exists():
                # Copy baseline to champion checkpoint path
                if not self.champion_ckpt_path.exists():
                    shutil.copy2(str(baseline_p), str(self.champion_ckpt_path))

                # Load baseline metrics
                metrics_p = Path(self.config.root_dir) / "research" / "edgeloop" / "baseline_holdout_metrics.json"
                metrics_data = {}
                if metrics_p.exists():
                    with open(metrics_p, "r") as f:
                        metrics_data = json.load(f)

                acc = metrics_data.get("accuracy", self.config.get("governance.baseline_accuracy", 0.913043))
                f1 = metrics_data.get("macro_f1", self.config.get("governance.baseline_macro_f1", 0.894340))
                trash_f1 = metrics_data.get("classification_report", {}).get("trash", {}).get("f1-score", self.config.get("governance.baseline_trash_f1", 0.765957))

                self.db.register_model(
                    version="mobilenetv2_baseline",
                    model_name="MobileNetV2",
                    checkpoint_path=str(self.champion_ckpt_path),
                    stage="champion",
                    accuracy=acc,
                    macro_f1=f1,
                    trash_f1=trash_f1,
                    metrics_json=metrics_data,
                    approved_by="System Initializer",
                    promotion_reason="Initial baseline champion from Kaggle research holdout evaluation"
                )
                logger.info("Seeded initial champion: mobilenetv2_baseline (F1: %.4f)", f1)

    def list_models(self) -> List[ModelEntry]:
        """Fetch all models in the registry ordered by ID descending."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_registry ORDER BY id DESC")
            rows = cursor.fetchall()
            results = []
            for r in rows:
                m_json = json.loads(r["metrics_json"]) if r["metrics_json"] else None
                results.append(
                    ModelEntry(
                        id=r["id"],
                        version=r["version"],
                        model_name=r["model_name"],
                        checkpoint_path=r["checkpoint_path"],
                        stage=r["stage"],
                        accuracy=r["accuracy"],
                        macro_f1=r["macro_f1"],
                        trash_f1=r["trash_f1"],
                        metrics_json=m_json,
                        approved_by=r["approved_by"],
                        promotion_reason=r["promotion_reason"],
                        created_at=r["created_at"],
                        updated_at=r["updated_at"]
                    )
                )
            return results

    def get_champion(self) -> Optional[ModelEntry]:
        """Retrieve current champion metadata."""
        champ_dict = self.db.get_champion_model()
        if not champ_dict:
            return None
        m_json = json.loads(champ_dict["metrics_json"]) if champ_dict["metrics_json"] else None
        return ModelEntry(
            id=champ_dict["id"],
            version=champ_dict["version"],
            model_name=champ_dict["model_name"],
            checkpoint_path=champ_dict["checkpoint_path"],
            stage=champ_dict["stage"],
            accuracy=champ_dict["accuracy"],
            macro_f1=champ_dict["macro_f1"],
            trash_f1=champ_dict["trash_f1"],
            metrics_json=m_json,
            approved_by=champ_dict["approved_by"],
            promotion_reason=champ_dict["promotion_reason"],
            created_at=champ_dict["created_at"],
            updated_at=champ_dict["updated_at"]
        )

    def register_candidate(
        self,
        version: str,
        checkpoint_path: Union[str, Path],
        model_name: str = "MobileNetV2"
    ) -> ModelEntry:
        """Register a new candidate in the registry in 'candidate' stage."""
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_p}")

        self.db.register_model(
            version=version,
            model_name=model_name,
            checkpoint_path=str(ckpt_p),
            stage="candidate"
        )
        model_data = self.db.get_model_by_version(version)
        return ModelEntry(**model_data)

    def evaluate_candidate(
        self,
        candidate_version: str,
        eval_manifest_path: Union[str, Path],
        max_samples: Optional[int] = None
    ) -> ComparisonReport:
        """
        Evaluate candidate and compare against active champion.
        Transitions candidate stage to 'evaluated' if successful.
        """
        candidate_row = self.db.get_model_by_version(candidate_version)
        if not candidate_row:
            raise ValueError(f"Candidate {candidate_version} not found in registry.")

        champion = self.get_champion()
        if not champion:
            raise RuntimeError("No active champion found to compare against.")

        # Run candidate evaluation
        candidate_eval = self.evaluator.evaluate_checkpoint(
            checkpoint_path=candidate_row["checkpoint_path"],
            eval_manifest_or_df=eval_manifest_path,
            model_version=candidate_version,
            max_samples=max_samples
        )

        # Run champion evaluation on identical dataset
        champion_eval = self.evaluator.evaluate_checkpoint(
            checkpoint_path=champion.checkpoint_path,
            eval_manifest_or_df=eval_manifest_path,
            model_version=champion.version,
            max_samples=max_samples
        )

        # Compare metrics
        comparison = self.evaluator.compare_against_champion(
            candidate_metrics=candidate_eval,
            champion_metrics=champion_eval
        )

        # Update candidate record in DB
        trash_f1 = candidate_eval.per_class.get("trash").f1_score if "trash" in candidate_eval.per_class else None
        self.db.register_model(
            version=candidate_version,
            model_name=candidate_row["model_name"],
            checkpoint_path=candidate_row["checkpoint_path"],
            stage="evaluated",
            accuracy=candidate_eval.accuracy,
            macro_f1=candidate_eval.macro_f1,
            trash_f1=trash_f1,
            metrics_json=candidate_eval.model_dump()
        )

        return comparison

    def promote_candidate(
        self,
        candidate_version: str,
        approver: str,
        promotion_reason: str,
        force: bool = False
    ) -> ModelEntry:
        """
        Safely promote candidate to champion.
        Fails and aborts if:
        - Candidate has not been evaluated
        - Candidate failed regression gates (unless force=True with explicit justification)
        - Approver is empty
        - Candidate is one of the known defective retraining candidates
        """
        if not approver or not approver.strip():
            raise PromotionRejectionError("Promotion rejected: Approver identity cannot be blank.")

        if not promotion_reason or not promotion_reason.strip():
            raise PromotionRejectionError("Promotion rejected: Promotion justification reason is required.")

        # Safeguard: Reject known defective candidates
        if candidate_version in ["uncertainty_candidate", "unique_aware_candidate"]:
            raise PromotionRejectionError(
                f"Promotion rejected: Candidate '{candidate_version}' scored below the champion baseline on clean holdout metrics (Research Governance Rule)."
            )

        candidate = self.db.get_model_by_version(candidate_version)
        if not candidate:
            raise ValueError(f"Candidate {candidate_version} not found in registry.")

        champion = self.get_champion()

        if candidate["stage"] not in ["evaluated", "approved"] and not force:
            raise PromotionRejectionError(
                f"Promotion rejected: Candidate stage is '{candidate['stage']}'. Must be 'evaluated' or 'approved' prior to promotion."
            )

        # Check governance gates if metrics available
        if champion and candidate["macro_f1"] is not None and champion.macro_f1 is not None and not force:
            if candidate["macro_f1"] <= champion.macro_f1:
                raise PromotionRejectionError(
                    f"Promotion rejected: Candidate Macro-F1 ({candidate['macro_f1']:.4f}) does not exceed Champion ({champion.macro_f1:.4f})."
                )
            if candidate["trash_f1"] is not None and champion.trash_f1 is not None:
                max_drop = float(self.config.get("governance.max_per_class_f1_drop", 0.02))
                if candidate["trash_f1"] < champion.trash_f1 - max_drop:
                    raise PromotionRejectionError(
                        f"Promotion rejected: Candidate 'trash' class F1 ({candidate['trash_f1']:.4f}) dropped beyond tolerance compared to champion ({champion.trash_f1:.4f})."
                    )

        # Execute safe atomic file swap
        candidate_src = Path(candidate["checkpoint_path"])
        if not candidate_src.exists():
            raise FileNotFoundError(f"Candidate checkpoint missing: {candidate_src}")

        # 1. Backup current champion checkpoint for rollback
        if self.champion_ckpt_path.exists():
            shutil.copy2(str(self.champion_ckpt_path), str(self.previous_champion_ckpt_path))

        # 2. Copy candidate to champion checkpoint location
        temp_target = self.champion_dir / "mobilenetv2_champion.tmp"
        shutil.copy2(str(candidate_src), str(temp_target))
        os.replace(str(temp_target), str(self.champion_ckpt_path))

        # 3. Demote current champion in database
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if champion:
                cursor.execute(
                    "UPDATE model_registry SET stage = 'archived', updated_at = ? WHERE id = ?",
                    (now, champion.id)
                )

            # 4. Promote candidate to champion in database
            cursor.execute(
                """
                UPDATE model_registry SET
                    stage = 'champion',
                    approved_by = ?,
                    promotion_reason = ?,
                    checkpoint_path = ?,
                    updated_at = ?
                WHERE version = ?
                """,
                (approver.strip(), promotion_reason.strip(), str(self.champion_ckpt_path), now, candidate_version)
            )

        logger.info("Successfully promoted candidate %s to active champion by %s.", candidate_version, approver)
        return self.get_champion()

    def rollback_champion(self, approver: str, reason: str) -> ModelEntry:
        """
        Safely roll back active champion to previously approved model.
        Restores backup checkpoint file and updates database registry.
        """
        if not self.previous_champion_ckpt_path.exists():
            raise RuntimeError("Rollback failed: No previous champion checkpoint backup exists.")

        current_champion = self.get_champion()
        if not current_champion:
            raise RuntimeError("Rollback failed: No active champion exists.")

        # Find the most recently archived champion
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM model_registry WHERE stage = 'archived' ORDER BY updated_at DESC LIMIT 1"
            )
            prev_row = cursor.fetchone()

        if not prev_row:
            raise RuntimeError("Rollback failed: No archived previous champion found in registry database.")

        prev_version = prev_row["version"]

        # Restore checkpoint file atomically
        temp_target = self.champion_dir / "mobilenetv2_champion.tmp"
        shutil.copy2(str(self.previous_champion_ckpt_path), str(temp_target))
        os.replace(str(temp_target), str(self.champion_ckpt_path))

        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # Demote current champion to archived
            cursor.execute(
                "UPDATE model_registry SET stage = 'archived', updated_at = ? WHERE id = ?",
                (now, current_champion.id)
            )
            # Re-promote previous champion
            cursor.execute(
                """
                UPDATE model_registry SET
                    stage = 'champion',
                    approved_by = ?,
                    promotion_reason = ?,
                    updated_at = ?
                WHERE version = ?
                """,
                (f"Rollback by {approver}", f"Rollback reason: {reason}", now, prev_version)
            )

        logger.info("Successfully rolled back champion from %s to %s.", current_champion.version, prev_version)
        return self.get_champion()
