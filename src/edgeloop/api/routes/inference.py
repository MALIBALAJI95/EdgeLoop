"""Inference and prediction history endpoints."""

import io
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel

from edgeloop.common.config import get_config
from edgeloop.common.database import get_db
from edgeloop.drift.detector import DriftDetector
from edgeloop.drift.features import DriftFeatureExtractor
from edgeloop.inference.engine import (
    InferenceEngine,
    PredictionResult,
    InvalidImageError
)

router = APIRouter(tags=["Inference"])

# Singleton engines
_engine_instance: Optional[InferenceEngine] = None
_drift_detector: Optional[DriftDetector] = None
_feature_extractor: Optional[DriftFeatureExtractor] = None


def get_inference_engine() -> InferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = InferenceEngine()
    return _engine_instance


def get_drift_detector() -> DriftDetector:
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector


def get_feature_extractor() -> DriftFeatureExtractor:
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = DriftFeatureExtractor()
    return _feature_extractor


class PathInferenceRequest(BaseModel):
    image_path: str
    sample_id: Optional[str] = None
    condition: str = "unknown"


@router.post("/infer", response_model=PredictionResult)
async def infer_uploaded_image(
    file: UploadFile = File(...),
    sample_id: Optional[str] = None,
    condition: str = "unknown"
):
    """Run inference on an uploaded image file."""
    engine = get_inference_engine()
    detector = get_drift_detector()
    extractor = get_feature_extractor()

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        sid = sample_id or Path(file.filename or "upload.jpg").stem

        # Save incoming image to data/incoming for persistent reference
        config = get_config()
        incoming_dir = Path(config.root_dir) / "data" / "incoming"
        incoming_dir.mkdir(parents=True, exist_ok=True)
        saved_path = incoming_dir / f"{sid}_{file.filename}"
        with open(saved_path, "wb") as f:
            f.write(contents)

        # 1. Run inference
        result = engine.predict_image(
            image_input=pil_img,
            sample_id=sid,
            condition=condition
        )
        result.image_path = str(saved_path)

        # 2. Extract drift features and log
        feats = extractor.extract_from_image(pil_img)
        # Find latest inference ID
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM inferences ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            inf_id = row["id"] if row else None

        detector.check_sample(feats, inference_id=inf_id, log_to_db=True)

        return result
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference processing error: {e}")


@router.post("/infer/path", response_model=PredictionResult)
def infer_image_path(req: PathInferenceRequest):
    """Run inference on an existing local image path."""
    engine = get_inference_engine()
    detector = get_drift_detector()
    extractor = get_feature_extractor()

    try:
        p = Path(req.image_path)
        result = engine.predict_image(
            image_input=p,
            sample_id=req.sample_id,
            condition=req.condition
        )

        # Drift check
        feats = extractor.extract_from_image(p)
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM inferences ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            inf_id = row["id"] if row else None

        detector.check_sample(feats, inference_id=inf_id, log_to_db=True)

        return result
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@router.get("/history")
def get_inference_history(limit: int = Query(50, ge=1, le=500)):
    """Retrieve recent inference records with drift status."""
    db = get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT inf.id, inf.sample_id, inf.image_path, inf.predicted_class,
                   inf.confidence, inf.entropy, inf.latency_ms, inf.model_version,
                   inf.condition, inf.uploaded, inf.created_at,
                   dr.is_drift, dr.triggered_features
            FROM inferences inf
            LEFT JOIN drift_records dr ON inf.id = dr.inference_id
            ORDER BY inf.id DESC
            LIMIT ?
            """,
            (limit,)
        )
        return [dict(r) for r in cursor.fetchall()]
