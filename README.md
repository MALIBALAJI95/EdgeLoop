# ♻️ EdgeLoop: Self-Adaptive Edge AI for Intelligent Waste Detection & Sorting

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Pytest-45%20Passing-success?logo=pytest&logoColor=white)](https://github.com/MALIBALAJI95/EdgeLoop/actions)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/MALIBALAJI95/EdgeLoop/actions)

EdgeLoop is an end-to-end, production-grade edge artificial intelligence system designed for automated waste sorting facilities, reverse vending machines (RVMs), and intelligent recycling kiosks. 

Built with **PyTorch (SSDLite320 + MobileNetV2)**, **FastAPI**, **Streamlit**, and **SQLite WAL**, EdgeLoop delivers real-time multi-object detection, spatial IoU deduplication, dynamic class registration on the edge, human-in-the-loop active learning, strict statistical model governance gates, and automated GitHub Actions CI/CD deployment.

---

## 🌟 Key System Highlights

1. **Two-Stage Edge Architecture (< 50ms Latency)**:
   * **Stage 1 (Localizer)**: PyTorch SSDLite320 (MobileNetV3 backbone) proposes candidate non-living object bounding boxes across the full visible scene.
   * **Stage 2 (Classifier)**: Preserved MobileNetV2 Champion classifies localized crops into waste materials (`paper`, `plastic`, `glass`, `metal`, `cardboard`, `electronic_waste`, `trash`).
2. **Unified 3 Camera Operating Modes**:
   * **Mode 1 — Terminal Live HD Stream**: Native AVFoundation widescreen camera at 30 FPS (`scripts/live_camera.py`), automatically streaming synchronized frames, bounding boxes, and metadata directly to the web dashboard.
   * **Mode 2 — Browser Snapshot Input**: Instant camera capture directly inside the web browser via WebRTC (`st.camera_input`).
   * **Mode 3 — Multi-Image Batch Inspection**: Drag-and-drop batch analyzer for testing photo collections and dataset archives.
3. **Spatial IoU Tracking & Proximity Prioritization**:
   * Counts each physical item **only once** while in view, preventing duplicate counting across video frames.
   * Automatically prioritizes items closest to the camera (occupying the largest foreground area) for robotic sorting arm and bin routing.
   * Sleek UI overlays with corner-bracket accents and anti-collision floating pills.
4. **Non-Living Focus & Smart Human Suppression**:
   * Distinguishes human beings from objects: people and faces are strictly excluded from waste labels, but objects held by or placed near a person (e.g. bottles, books, phones, cups) are detected immediately.
5. **Dynamic Edge Vocabulary**:
   * Operators can register novel waste categories (e.g., `electronic_waste`, Bioplastics, Tetra Paks) on the fly without wiping existing baseline weights.
6. **Human-in-the-Loop Active Learning & Dataset Provenance**:
   * Genuinely uncertain items are buffered for operator verification.
   * Operators can edit labels, assign bins, manually add unlocalized items, or delete/purge noisy samples.
   * 1-Click candidate retraining with real-time epoch progress curves.
7. **Strict Model Governance & CI/CD Push**:
   * Candidates are evaluated on holdout test benchmarks (506 images). A candidate is promoted only if holdout Macro-F1 improves with zero sensitive class regressions.
   * Single-click authorized deployment atomically swaps champion weights, commits changes, and triggers GitHub Actions CI/CD.

---

## 🏗️ System Architecture

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         STAGE 1: EDGE INGESTION & LOCALIZATION              │
 │                                                                             │
 │   Camera Modes: Terminal HD (Mode 1) | Browser (Mode 2) | Batch (Mode 3)    │
 │                                    │                                        │
 │                        SSDLite320 Object Localizer                          │
 │         (Spatial IoU Tracking, Anti-Collision Pills, Human Suppression)      │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     STAGE 2: CLASSIFICATION & QUALITY DRIFT                 │
 │                                                                             │
 │                   MobileNetV2 Champion Classifier (7 Classes)               │
 │        [Cardboard, Glass, Metal, Paper, Plastic, Trash, Electronic Waste]   │
 │                                    │                                        │
 │                  Kolmogorov-Smirnov Sensor Drift Telemetry                  │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                    STAGE 3: ACTIVE AUDIT & GROUND-TRUTH STORE               │
 │                                                                             │
 │                 SQLite WAL Persistent Selection Review Queue                │
 │                  Human Verification & Dataset Provenance Store              │
 │                     (Inspect, Delete, Purge Noisy Samples)                  │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      STAGE 4: CONTINUOUS RETRAINING & GATES                 │
 │                                                                             │
 │                 Leak-Free Group Stratified Train/Val Fine-Tuning            │
 │                                    │                                        │
 │                      Statistical Model Governance Gate                      │
 │                 • Candidate Holdout Macro-F1 Improvement                    │
 │                 • Sensitive Class Zero-Regression Verification              │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      STAGE 5: DEPLOYMENT & CI/CD PIPELINE                   │
 │                                                                             │
 │             • Atomic Champion Checkpoint Swap & Fast Rollback               │
 │             • FastAPI REST Service (`http://127.0.0.1:8000`)                 │
 │             • Streamlit Kiosk Dashboard (`http://localhost:8501`)           │
 │             • Automated Git Commit & Push -> GitHub Actions CI/CD Matrix    │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Verified Holdout Benchmarks (Empirical Results)

Evaluated on 506 holdout test images across all waste classes:

| Class | Champion F1 Score | Target Recycling Bin | Bin Color Tag |
| :--- | :---: | :--- | :--- |
| **Cardboard** | **0.9231** | Blue Bin (Cardboard & Paper) | 🔵 Blue |
| **Glass** | **0.8485** | Green Bin (Glass & Bottles) | 🟢 Green |
| **Metal** | **0.8205** | Grey Bin (Metals & Cans) | ⚪ Grey |
| **Paper** | **0.8182** | Blue Bin (Paper & Books) | 🔵 Blue |
| **Plastic** | **0.7568** | Yellow Bin (Plastics & Packaging) | 🟡 Yellow |
| **Trash** | **0.7660** | Black Bin (General Refuse) | ⚫ Black |
| **Electronic Waste** | **Dynamic** | Red/Orange Bin (E-Waste & Electronics) | 🔴 Red/Orange |

**Overall Baseline Holdout Accuracy**: **91.30%** | **Macro-F1**: **0.8943**

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* macOS (Apple Silicon Metal / Intel), Linux (Ubuntu 22.04+), or Windows (WSL2).
* Python 3.11 or 3.12 installed.

### 2. Installation & Setup
```bash
# Clone the repository
git clone https://github.com/MALIBALAJI95/EdgeLoop.git
cd EdgeLoop

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Tests (45 Passing Suites)
```bash
pytest tests/ -v
```

### 4. Launch the Interactive Web Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 5. Launch the Live AI Edge Camera
In a separate terminal tab, run:
```bash
.venv/bin/python scripts/live_camera.py
```
* **Controls in Camera Window**:
  * `SPACE`: Flag the currently localized object for human verification.
  * `Q`: Exit the camera feed.
* Video frames and bounding boxes synchronize with the dashboard in real-time.

### 6. Launch the FastAPI REST Service
```bash
uvicorn edgeloop.api.app:app --host 127.0.0.1 --port 8000
```
* Interactive Swagger Docs: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
* Health Check: `curl http://127.0.0.1:8000/health`

---

## 🖥️ Operational Dashboard Tour

The dashboard is structured into 5 dedicated, presentation-ready panels:

1. **🚀 Overview & Quickstart**:
   Problem statement, EdgeLoop solution, system uniqueness, live system KPIs, and device profile.
2. **🎯 Live Detection & Sorting**:
   Unified 3 camera modes, real-time FPS/latency monitor, closest object material, recommended recycling bin, and "Stop & Review" audit button.
3. **👥 Human Review & Active Learning**:
   Audit uncertain items, manually add unlocalized objects, manage/prune ground-truth records, retrain candidate models with live epoch progress, and review governance scorecards.
4. **📊 Model Performance**:
   Holdout accuracy (91.30%), Macro-F1 (0.8943), 6x6 confusion matrix, per-class metrics, and loss/accuracy training history curves.
5. **🚢 Deployment & CI/CD Status**:
   Active champion version, REST API health status, live Git deployment trigger (`git push`), and GitHub Actions workflow telemetry console.

---

## ⚡ FastAPI REST Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness health probe (`200 OK`) |
| `GET` | `/ready` | Readiness probe verifying loaded champion weights |
| `POST` | `/infer` | Multi-format image inference (multipart upload or JSON path) |
| `GET` | `/history` | Recent inference history and confidence logs |
| `GET` | `/drift/status` | Kolmogorov-Smirnov sensor drift summary |
| `GET` | `/registry/models` | List of all candidate, evaluated, and champion models |
| `POST` | `/registry/promote` | Promote candidate to champion with audit log |
| `POST` | `/registry/rollback` | Roll back to previous champion model |

---

## 📁 Repository Structure

```text
EdgeLoop/
├── .github/workflows/          # GitHub Actions CI/CD pipeline (tests, lint, governance, docker)
├── artifacts/
│   ├── champion/               # Active champion weights (mobilenetv2_champion.pth)
│   ├── candidates/             # Retrained fine-tuned candidate checkpoints
│   └── registry/               # SQLite WAL database (edgeloop.db)
├── configs/
│   ├── config.yaml             # System thresholds, drift windows, and paths
│   └── class_mapping.json      # Dynamic waste classes, bins, and color codes
├── dashboard/
│   └── app.py                  # Streamlit modern 5-stage operating kiosk UI
├── data/
│   ├── incoming/               # Buffered frames flagged for review
│   ├── reviewed/               # Provenance-backed ground-truth training images
│   └── live_stream/            # Real-time terminal camera synchronization bridge
├── notebooks/                  # Kaggle baseline research notebooks
├── reports/                    # Model governance cards, audit logs, and limitations
├── research/                   # Clean holdout test benchmarks and corruption datasets
├── scripts/
│   ├── live_camera.py          # Native widescreen HD AI camera with dashboard sync
│   ├── init_registry.py        # Database and champion initialization
│   └── train_candidate.py      # Standalone headless candidate retraining
├── src/edgeloop/
│   ├── api/                    # FastAPI routes, schemas, and app
│   ├── common/                 # Database, config, and dynamic class manager
│   ├── drift/                  # Sobel/Laplacian feature extraction & KS test
│   ├── inference/              # SSDLite localizer, MobileNetV2 classifier & IoU tracker
│   ├── registry/               # Model promotion, governance comparison, rollback
│   ├── review/                 # Human audit service, dataset export & provenance
│   ├── selection/              # Active learning acquisition policies
│   └── training/               # Leak-free trainer, holdout evaluator, splitters
└── tests/                      # 45 Pytest unit, integration, and drift tests
```

---

## 🛡️ License & Ethics

Distributed under the MIT License. Developed for automated recycling facilities and environmental sustainability. Excludes human beings and living organisms from waste classification by design.
