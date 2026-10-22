"""Human review workflow, audit trail, and dataset provenance manager."""

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field

from edgeloop.common.config import get_config
from edgeloop.common.database import get_db

logger = logging.getLogger("edgeloop.review")


class ReviewItem(BaseModel):
    queue_id: int
    inference_id: int
    sample_id: str
    image_path: str
    predicted_class: str
    confidence: float
    entropy: float
    model_version: str
    condition: str
    strategy: str
    upload_reason: str
    is_drift: bool = False
    triggered_features: List[str] = Field(default_factory=list)


class ReviewSubmission(BaseModel):
    queue_id: int
    true_label: str
    reviewer: str
    notes: Optional[str] = None


class DatasetVersionInfo(BaseModel):
    version: str
    record_count: int
    manifest_path: str
    created_at: str
    description: Optional[str] = None


class ReviewService:
    """Manages human-in-the-loop review and provenance-backed dataset curation."""

    VALID_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    def __init__(self):
        self.config = get_config()
        self.db = get_db()
        self.reviewed_data_dir = Path(self.config.root_dir) / "data" / "reviewed"
        self.reviewed_data_dir.mkdir(parents=True, exist_ok=True)

    def get_pending_queue(self, limit: int = 50) -> List[ReviewItem]:
        """Fetch pending review tasks joined with inference and drift metadata."""
        items = self.db.get_pending_review_items(limit=limit)
        results = []

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            for it in items:
                inf_id = it["inference_id"]
                cursor.execute(
                    "SELECT is_drift, triggered_features FROM drift_records WHERE inference_id = ? ORDER BY id DESC LIMIT 1",
                    (inf_id,)
                )
                drift_row = cursor.fetchone()
                is_drift = bool(drift_row["is_drift"]) if drift_row else False
                trig = json.loads(drift_row["triggered_features"]) if (drift_row and drift_row["triggered_features"]) else []

                results.append(
                    ReviewItem(
                        queue_id=it["queue_id"],
                        inference_id=it["inference_id"],
                        sample_id=it["sample_id"],
                        image_path=it["image_path"],
                        predicted_class=it["predicted_class"],
                        confidence=it["confidence"],
                        entropy=it["entropy"],
                        model_version=it["model_version"],
                        condition=it["condition"],
                        strategy=it["strategy"],
                        upload_reason=it["upload_reason"],
                        is_drift=is_drift,
                        triggered_features=trig
                    )
                )

        return results

    def submit_review(self, submission: ReviewSubmission) -> int:
        """Process reviewer verification, record audit trail, and persist ground truth."""
        if submission.true_label not in self.VALID_CLASSES:
            raise ValueError(f"Invalid label '{submission.true_label}'. Must be one of {self.VALID_CLASSES}")

        if not submission.reviewer or not submission.reviewer.strip():
            raise ValueError("Reviewer identity cannot be blank.")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT sq.id as queue_id, sq.inference_id, sq.sample_id, sq.image_path,
                       inf.predicted_class
                FROM selection_queue sq
                JOIN inferences inf ON sq.inference_id = inf.id
                WHERE sq.id = ? AND sq.status = 'pending'
                """,
                (submission.queue_id,)
            )
            item = cursor.fetchone()
            if not item:
                raise ValueError(f"Queue item {submission.queue_id} not found or already reviewed.")

            inference_id = item["inference_id"]
            sample_id = item["sample_id"]
            image_path = item["image_path"]
            predicted_class = item["predicted_class"]

        # Atomic commit to review_records and update selection_queue status
        review_id = self.db.submit_review(
            queue_id=submission.queue_id,
            inference_id=inference_id,
            sample_id=sample_id,
            image_path=image_path,
            predicted_class=predicted_class,
            true_label=submission.true_label,
            reviewer=submission.reviewer.strip(),
            notes=submission.notes
        )

        logger.info(
            "Review submitted: queue_id=%d, sample=%s, pred=%s, true=%s, reviewer=%s",
            submission.queue_id, sample_id, predicted_class, submission.true_label, submission.reviewer
        )
        return review_id

    def export_dataset_version(self, description: Optional[str] = None) -> DatasetVersionInfo:
        """
        Export all human-reviewed ground truth into an immutable versioned dataset.
        Ensures unreviewed predictions never enter training sets.
        """
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        version = f"trashnet_reviewed_v_{timestamp_str}"

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, inference_id, sample_id, image_path,
                       predicted_class, true_label, is_corrected,
                       reviewer, notes, reviewed_at
                FROM review_records
                ORDER BY id ASC
                """
            )
            rows = cursor.fetchall()

        if not rows:
            raise ValueError("Cannot export dataset version: No reviewed records found.")

        manifest_records = []
        for r in rows:
            fpath = Path(r["image_path"])
            sha256 = None
            if fpath.exists() and fpath.is_file():
                h = hashlib.sha256()
                with open(fpath, "rb") as fp:
                    while chunk := fp.read(65536):
                        h.update(chunk)
                sha256 = h.hexdigest()

            manifest_records.append({
                "review_id": r["id"],
                "sample_id": r["sample_id"],
                "image_path": str(fpath),
                "label": r["true_label"],
                "original_prediction": r["predicted_class"],
                "is_corrected": bool(r["is_corrected"]),
                "reviewer": r["reviewer"],
                "reviewed_at": r["reviewed_at"],
                "sha256": sha256
            })

        manifest_path = self.reviewed_data_dir / f"manifest_{version}.csv"
        df = pd.DataFrame(manifest_records)
        df.to_csv(manifest_path, index=False)

        # Register version in database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO dataset_versions (version, record_count, manifest_path, description, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (version, len(manifest_records), str(manifest_path), description or "Human-reviewed dataset version", now.isoformat())
            )

        logger.info("Successfully exported dataset %s with %d verified records.", version, len(manifest_records))
        return DatasetVersionInfo(
            version=version,
            record_count=len(manifest_records),
            manifest_path=str(manifest_path),
            created_at=now.isoformat(),
            description=description
        )
