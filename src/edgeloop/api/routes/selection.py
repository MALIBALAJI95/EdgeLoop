"""Selective upload and budget management endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from edgeloop.common.database import get_db
from edgeloop.selection.policies import (
    SelectionEngine,
    StreamRecord,
    SelectionReport
)

router = APIRouter(prefix="/selection", tags=["Selection"])


class RunSelectionRequest(BaseModel):
    strategy: str = "unique_aware"
    budget_records: Optional[int] = 203
    budget_fraction: Optional[float] = None


@router.get("/queue")
def get_selection_queue(limit: int = Query(50, ge=1, le=500)):
    """Fetch current items waiting in the upload selection queue."""
    db = get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT sq.*, inf.predicted_class, inf.confidence, inf.condition
            FROM selection_queue sq
            JOIN inferences inf ON sq.inference_id = inf.id
            ORDER BY sq.id DESC
            LIMIT ?
            """,
            (limit,)
        )
        return [dict(r) for r in cursor.fetchall()]


@router.post("/run", response_model=SelectionReport)
def run_selection_on_stream(req: RunSelectionRequest):
    """Run selective upload policy on un-uploaded inferences and populate selection queue."""
    db = get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT inf.id, inf.sample_id, inf.image_path, inf.predicted_class,
                   inf.confidence, inf.entropy, inf.condition,
                   dr.is_drift
            FROM inferences inf
            LEFT JOIN drift_records dr ON inf.id = dr.inference_id
            WHERE inf.uploaded = 0
            """
        )
        rows = cursor.fetchall()

    if not rows:
        raise HTTPException(status_code=400, detail="No un-uploaded inference records found in stream.")

    records = [
        StreamRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            predicted_class=r["predicted_class"],
            confidence=r["confidence"],
            entropy=r["entropy"],
            condition=r["condition"] or "unknown",
            drift_flag=bool(r["is_drift"]),
            file_size_bytes=25000
        )
        for r in rows
    ]

    engine = SelectionEngine()
    selected, report = engine.run_selection(
        records=records,
        strategy=req.strategy,
        budget_records=req.budget_records,
        budget_fraction=req.budget_fraction
    )

    # Insert selected items into selection queue and mark uploaded in database
    for rec in selected:
        # Find matching inference ID
        inf_id = next(r["id"] for r in rows if r["sample_id"] == rec.sample_id and r["image_path"] == rec.image_path)
        db.add_to_selection_queue(
            inference_id=inf_id,
            sample_id=rec.sample_id,
            image_path=rec.image_path,
            strategy=req.strategy,
            upload_reason=report.strategy,
            entropy=rec.entropy
        )

    return report
