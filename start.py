#!/usr/bin/env python3
"""EdgeLoop Unified Launcher (Python).

Starts FastAPI backend, Streamlit dashboard, and opens the browser.
Clean shutdown on Ctrl+C.
"""

import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def main():
    print("=================================================================")
    print("♻️  EdgeLoop — Self-Adaptive Edge AI for Waste Sorting")
    print("=================================================================")
    os.chdir(ROOT_DIR)

    python_bin = sys.executable

    # Check for streamlit and uvicorn in venv
    venv_py = ROOT_DIR / ".venv" / "bin" / "python"
    if venv_py.exists():
        python_bin = str(venv_py)

    api_process = None
    dashboard_process = None

    def shutdown(sig, frame):
        print("\n🛑 Shutting down EdgeLoop services cleanly...")
        if api_process:
            api_process.terminate()
        if dashboard_process:
            dashboard_process.terminate()
        print("✅ All services stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 1. Start FastAPI Backend
    print("⚡ Starting FastAPI backend service on http://127.0.0.1:8000...")
    api_cmd = [
        python_bin, "-m", "uvicorn",
        "edgeloop.api.app:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--log-level", "warning"
    ]
    api_process = subprocess.Popen(api_cmd)
    time.sleep(1.5)

    # 2. Open Browser
    print("🌐 Opening dashboard at http://localhost:8501 (Secure Context for Camera)...")
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass

    # 3. Start Streamlit Dashboard
    print("🚀 Launching EdgeLoop Streamlit Dashboard...")
    dash_cmd = [
        python_bin, "-m", "streamlit", "run",
        "dashboard/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ]
    dashboard_process = subprocess.Popen(dash_cmd)

    print("-----------------------------------------------------------------")
    print("Live Dashboard: http://localhost:8501")
    print("FastAPI Docs:   http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to stop all services.")
    print("-----------------------------------------------------------------")

    dashboard_process.wait()


if __name__ == "__main__":
    main()
