"""FastAPI main application entrypoint for EdgeLoop."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from edgeloop.api.routes import (
    health,
    inference,
    drift,
    selection,
    review,
    registry
)
from edgeloop.common.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("edgeloop.api")

config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting EdgeLoop FastAPI Service...")
    yield
    logger.info("Stopping EdgeLoop FastAPI Service...")


app = FastAPI(
    title=config.get("project.name", "EdgeLoop"),
    description="Drift-Aware Edge-to-Cloud Machine Learning System for Waste Classification",
    version=config.get("project.version", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router)
app.include_router(inference.router)
app.include_router(drift.router)
app.include_router(selection.router)
app.include_router(review.router)
app.include_router(registry.router)
