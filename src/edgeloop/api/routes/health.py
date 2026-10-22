"""Health and readiness endpoints."""

from fastapi import APIRouter
from edgeloop.common.config import get_config
from edgeloop.common.database import get_db
from edgeloop.registry.manager import ModelRegistry

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    """System health check and component status."""
    config = get_config()
    db = get_db()
    reg = ModelRegistry()
    champ = reg.get_champion()

    return {
        "status": "healthy",
        "project": config.get("project.name", "EdgeLoop"),
        "version": config.get("project.version", "1.0.0"),
        "database": "connected" if db.db_path.exists() else "uninitialized",
        "champion_model": champ.version if champ else "none",
        "champion_macro_f1": champ.macro_f1 if champ else None
    }


@router.get("/readiness")
def readiness_check():
    """Readiness probe indicating if inference model is loaded and ready."""
    reg = ModelRegistry()
    champ = reg.get_champion()
    if not champ or not reg.champion_ckpt_path.exists():
        return {"ready": False, "reason": "No champion checkpoint available."}
    return {"ready": True, "champion_version": champ.version}
