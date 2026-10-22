"""Human review queue and label verification endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from edgeloop.review.service import (
    ReviewService,
    ReviewItem,
    ReviewSubmission,
    DatasetVersionInfo
)

router = APIRouter(prefix="/review", tags=["Review"])
_service_instance: Optional[ReviewService] = None


def get_review_service() -> ReviewService:
    global _service_instance
    if _service_instance is None:
        _service_instance = ReviewService()
    return _service_instance


class ExportDatasetRequest(BaseModel):
    description: Optional[str] = "Human-reviewed ground truth dataset version"


@router.get("/pending", response_model=List[ReviewItem])
def get_pending_review_items(limit: int = Query(50, ge=1, le=500)):
    """Fetch pending review tasks."""
    svc = get_review_service()
    return svc.get_pending_queue(limit=limit)


@router.post("/submit")
def submit_review_decision(submission: ReviewSubmission):
    """Submit human label confirmation or correction."""
    svc = get_review_service()
    try:
        rev_id = svc.submit_review(submission)
        return {"status": "success", "review_id": rev_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit review: {e}")


@router.post("/export", response_model=DatasetVersionInfo)
def export_dataset_version(req: ExportDatasetRequest):
    """Export all reviewed labels into an immutable versioned dataset."""
    svc = get_review_service()
    try:
        return svc.export_dataset_version(description=req.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")
