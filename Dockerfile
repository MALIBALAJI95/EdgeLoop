# Multi-stage Dockerfile for EdgeLoop MLOps System
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY configs/ ./configs/
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY data/ ./data/
COPY research/ ./research/
COPY artifacts/ ./artifacts/
COPY README.md ./

# Install package in editable mode
RUN pip install --no-cache-dir -e . --no-deps

# Create runtime directories
RUN mkdir -p artifacts/champion artifacts/candidates artifacts/registry data/incoming data/reviewed

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    EDGELOOP_HOST=0.0.0.0 \
    EDGELOOP_PORT=8000 \
    DASHBOARD_PORT=8501 \
    EDGELOOP_DEVICE=cpu

EXPOSE 8000 8501

# Default command: launch FastAPI server
CMD ["uvicorn", "edgeloop.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
