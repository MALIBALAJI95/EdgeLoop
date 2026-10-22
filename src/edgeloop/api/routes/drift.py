"""Drift monitoring API endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import pandas as pd

from edgeloop.common.database import get_db
from edgeloop.drift.detector import DriftDetector

router = APIRouter(prefix="/drift", tags=["Drift"])


@router.get("/status")
def get_drift_status():
    """Retrieve summary of recent drift events and feature trigger frequencies."""
    db = get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM drift_records")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as drift_count FROM drift_records WHERE is_drift = 1")
        drift_count = cursor.fetchone()["drift_count"]

        cursor.execute("SELECT triggered_features FROM drift_records WHERE is_drift = 1")
        rows = cursor.fetchall()

    trigger_counts: Dict[str, int] = {}
    import json
    for r in rows:
        if r["triggered_features"]:
            try:
                feats = json.loads(r["triggered_features"])
                for f in feats:
                    trigger_counts[f] = trigger_counts.get(f, 0) + 1
            except Exception:
                pass

    detector = DriftDetector()
    return {
        "total_monitored_records": total,
        "drift_alerts_count": drift_count,
        "drift_alert_rate": float(drift_count / total) if total > 0 else 0.0,
        "feature_trigger_counts": trigger_counts,
        "reference_version": detector.reference_version,
        "reference_stats": detector.ref_stats
    }


@router.get("/records")
def get_recent_drift_records(limit: int = Query(50, ge=1, le=500)):
    """Fetch recent drift metric logs."""
    db = get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT dr.*, inf.sample_id, inf.predicted_class, inf.condition
            FROM drift_records dr
            LEFT JOIN inferences inf ON dr.inference_id = inf.id
            ORDER BY dr.id DESC
            LIMIT ?
            """,
            (limit,)
        )
        return [dict(r) for r in cursor.fetchall()]
