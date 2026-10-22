# macOS Apple Silicon (M-Series) Setup Guide

This guide describes how to run EdgeLoop natively on macOS with Apple Silicon (M1/M2/M3/M4) leveraging Metal Performance Shaders (MPS) hardware acceleration.

---

## 1. System Requirements
- macOS 13.0 (Ventura) or newer (macOS 14/15/26 supported).
- Apple Silicon Mac (M1, M2, M3, M4 chip).
- Python 3.11 or Python 3.12 (installed via Homebrew or official installer).

---

## 2. Installation Steps

### Step 1: Install Homebrew and Python 3.12 (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12
```

### Step 2: Clone and Setup Virtual Environment
```bash
cd /path/to/mlops-project
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Core Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

---

## 3. MPS Hardware Acceleration Verification

PyTorch supports GPU acceleration on Apple Silicon via `mps` (Metal Performance Shaders).

Verify that MPS is available:
```bash
python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('MPS Available:', torch.backends.mps.is_available())
print('MPS Built:', torch.backends.mps.is_built())
"
```

Expected output:
```text
PyTorch Version: 2.14.x
MPS Available: True
MPS Built: True
```

---

## 4. Automatic Device Resolution in EdgeLoop

EdgeLoop automatically resolves the fastest available device:
1. If `torch.backends.mps.is_available()` is True, it selects `mps`.
2. If `torch.cuda.is_available()` is True, it selects `cuda`.
3. Otherwise, it gracefully falls back to `cpu`.

To force CPU mode (for deterministic debugging):
```bash
export EDGELOOP_DEVICE=cpu
```

---

## 5. Running the System

```bash
# 1. Run Smoke Test
./scripts/smoke_test.sh

# 2. Start FastAPI REST Backend
uvicorn edgeloop.api.app:app --host 0.0.0.0 --port 8000 --reload

# 3. Start Streamlit Operational Dashboard (in another terminal)
streamlit run dashboard/app.py
```
Navigate to `http://localhost:8501` to access the Streamlit UI, or `http://localhost:8000/docs` for the interactive OpenAPI documentation.
