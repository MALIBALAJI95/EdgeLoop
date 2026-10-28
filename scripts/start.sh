#!/bin/bash
# ==============================================================================
# EdgeLoop — Unified One-Click Application Startup
# Launches FastAPI Backend + Streamlit Dashboard + Auto-opens browser
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "================================================================="
echo "♻️  EdgeLoop — Self-Adaptive Edge AI for Waste Sorting"
echo "================================================================="
echo "Working Directory: $PROJECT_ROOT"

# 1. Environment check
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment .venv not found in $PROJECT_ROOT"
    exit 1
fi

PYTHON=".venv/bin/python"
UVICORN=".venv/bin/uvicorn"
STREAMLIT=".venv/bin/streamlit"

# 2. Kill any stale processes on 8000 and 8501
echo "🧹 Checking for stale processes on ports 8000 and 8501..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

# 3. Clean shutdown trap
API_PID=""
cleanup() {
    echo ""
    echo "🛑 Shutting down EdgeLoop services..."
    if [ -n "$API_PID" ]; then
        kill "$API_PID" 2>/dev/null || true
    fi
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:8501 | xargs kill -9 2>/dev/null || true
    echo "✅ Shutdown complete. All processes terminated cleanly."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 4. Start FastAPI Backend
echo "⚡ Starting FastAPI backend service on http://127.0.0.1:8000..."
$UVICORN edgeloop.api.app:app --host 127.0.0.1 --port 8000 --log-level warning &
API_PID=$!

# Wait for API readiness
echo "⏳ Verifying backend API readiness..."
for i in {1..10}; do
    if $PYTHON -c "from fastapi.testclient import TestClient; from edgeloop.api.app import app; r = TestClient(app).get('/health'); exit(0 if r.status_code == 200 else 1)" 2>/dev/null; then
        echo "✅ Backend API is ONLINE and HEALTHY (200 OK)."
        break
    fi
    sleep 0.5
done

# 5. Automatically open browser to localhost (Secure Context for Camera)
echo "🌐 Launching browser to http://localhost:8501 (Secure Context for Camera)..."
(sleep 2 && (open "http://localhost:8501" 2>/dev/null || xdg-open "http://localhost:8501" 2>/dev/null || true)) &

# 6. Start Streamlit Dashboard
echo "🚀 Starting EdgeLoop Streamlit Dashboard..."
echo "-----------------------------------------------------------------"
echo "Live Sorting Dashboard: http://localhost:8501"
echo "REST API & Docs:        http://127.0.0.1:8000/docs"
echo "Press Ctrl+C to stop all services cleanly."
echo "-----------------------------------------------------------------"

exec $STREAMLIT run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless false \
    --browser.gatherUsageStats false
