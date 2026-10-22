"""Model registry, candidate evaluation, safe promotion, and rollback endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from edgeloop.api.auth import verify_admin_token
from edgeloop.registry.manager import (
    ModelRegistry,
    ModelEntry,
    PromotionRejectionError
)
from edgeloop.training.evaluator import ComparisonReport

router = APIRouter(prefix="/registry", tags=["Model Registry"])
_registry_instance: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance


class RegisterCandidateRequest(BaseModel):
    version: str
    checkpoint_path: str
    model_name: str = "MobileNetV2"


class EvaluateCandidateRequest(BaseModel):
    candidate_version: str
    eval_manifest_path: str


class PromoteCandidateRequest(BaseModel):
    candidate_version: str
    approver: str
    promotion_reason: str
    force: bool = False


class RollbackChampionRequest(BaseModel):
    approver: str
    reason: str


@router.get("/models", response_model=List[ModelEntry])
def list_models():
    """List all models and stages registered in EdgeLoop."""
    reg = get_model_registry()
    return reg.list_models()


@router.get("/champion", response_model=Optional[ModelEntry])
def get_current_champion():
    """Get active champion model."""
    reg = get_model_registry()
    champ = reg.get_champion()
    if not champ:
        raise HTTPException(status_code=404, detail="No active champion found.")
    return champ


@router.post("/candidate", response_model=ModelEntry, dependencies=[Depends(verify_admin_token)])
def register_candidate(req: RegisterCandidateRequest):
    """Register a new candidate checkpoint."""
    reg = get_model_registry()
    try:
        return reg.register_candidate(
            version=req.version,
            checkpoint_path=req.checkpoint_path,
            model_name=req.model_name
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration error: {e}")


@router.post("/evaluate", response_model=ComparisonReport, dependencies=[Depends(verify_admin_token)])
def evaluate_candidate(req: EvaluateCandidateRequest):
    """Evaluate candidate model on holdout dataset and compare to active champion."""
    reg = get_model_registry()
    try:
        return reg.evaluate_candidate(
            candidate_version=req.candidate_version,
            eval_manifest_path=req.eval_manifest_path
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/promote", response_model=ModelEntry, dependencies=[Depends(verify_admin_token)])
def promote_candidate(req: PromoteCandidateRequest):
    """
    Safely promote a candidate model to champion.
    Enforces governance gates (Macro-F1 improvement, sensitive trash F1 checks, approval audit trail).
    """
    reg = get_model_registry()
    try:
        return reg.promote_candidate(
            candidate_version=req.candidate_version,
            approver=req.approver,
            promotion_reason=req.promotion_reason,
            force=req.force
        )
    except PromotionRejectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {e}")


@router.post("/rollback", response_model=ModelEntry, dependencies=[Depends(verify_admin_token)])
def rollback_champion(req: RollbackChampionRequest):
    """Safely roll back the active champion to the previous version."""
    reg = get_model_registry()
    try:
        return reg.rollback_champion(
            approver=req.approver,
            reason=req.reason
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rollback failed: {e}")
