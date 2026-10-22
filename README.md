# EdgeLoop: Drift-Aware Edge-to-Cloud ML System for Waste Classification

EdgeLoop is an end-to-end, production-grade MLOps system simulating drift-aware edge-to-cloud machine learning operations for waste classification. Built with PyTorch, MobileNetV2, FastAPI, Streamlit, and SQLite, EdgeLoop enforces budget-constrained edge-to-cloud sample selection, human-in-the-loop review, and strict model governance.

---

## 1. System Architecture

```text
       [ Edge Inference Node ]
                  │
        (1) Image Ingestion & Input Validation
                  │
        (2) MobileNetV2 Baseline Inference & Shannon Entropy
                  │
        (3) Image Quality Drift Feature Extraction (Sobel, Laplacian, Luminance)
                  │
        (4) Kolmogorov-Smirnov Drift Detector
                  │
        (5) Budget-Constrained Selective Upload
         ├── Policies: Random, Uncertainty, Unique-Aware, Drift-Aware
         └── Dedupes variants of identical source images
                  │
                  ▼
       [ Cloud / Central Server ]
                  │
        (6) SQLite Audit Database & Model Registry
                  │
        (7) Human-in-the-Loop Review Queue (Immutable Ground Truth)
                  │
        (8) Candidate Model Retraining & Evaluation Gates
                  │
        (9) Safe Promotion Engine (Holdout F1 Check + Class Regression Gate)
                  │
        (10) FastAPI REST API & Streamlit Operational Dashboard
```

---

## 2. Research Evidence & Governance Baseline

From initial Kaggle experiments on the 6-class TrashNet dataset (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`):

| Model / Experiment | Clean Holdout Accuracy | Clean Holdout Macro-F1 | `trash` Class F1 | Promotion Status |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2 Baseline** | **0.913043** | **0.894340** | **0.765957** | **Active Champion** |
| Uncertainty Candidate | 0.905138 | 0.885502 | 0.745098 | Rejected (Regression) |
| Unique-Aware Candidate | 0.903162 | 0.880525 | 0.716981 | Rejected (Regression) |

> [!IMPORTANT]
> **Governance Rule**: The baseline model is the current champion. Retraining experiments on synthetic corruption streams captured errors but failed to improve clean holdout performance. EdgeLoop strictly prohibits automatic promotion of candidates that do not exceed the champion on holdout Macro-F1 or that cause regressions in sensitive classes such as `trash`.

---

## 3. Technology Stack

- **ML Framework**: PyTorch 2.x, Torchvision
- **Backend API**: FastAPI, Pydantic v2, Uvicorn
- **Dashboard**: Streamlit
- **Data & Feature Engineering**: NumPy, Pandas, SciPy, Pillow, Scikit-learn
- **Database & Storage**: SQLite, JSON metadata, atomic file management
- **Testing & Containerization**: Pytest, Docker, Docker Compose

---

## 4. Quickstart

### Prerequisites
- Python 3.11 or 3.12 (Apple Silicon or Linux/Windows x86_64)

### Local Setup
```bash
# 1. Clone repository and set up virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize champion checkpoint and registry
python scripts/init_registry.py

# 4. Run automated tests
pytest tests/
```

### Running the System
```bash
# Start FastAPI backend service (port 8000)
uvicorn src.edgeloop.api.app:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit operational dashboard (port 8501)
streamlit run dashboard/app.py
```

---

## 5. Project Structure

```text
EdgeLoop/
├── EdgeLoop_Kaggle_Research_v0.5.zip  # Read-only source research archive
├── research/                         # Extracted research evidence & manifests
│   ├── archive_manifest.json
│   └── edgeloop/
├── artifacts/
│   ├── champion/                     # Active champion weights
│   ├── candidates/                   # Retrained candidate checkpoints
│   └── registry/                     # SQLite database (edgeloop.db)
├── configs/
│   ├── config.yaml                   # Global system configuration
│   └── class_mapping.json            # Class dictionary & indices
├── data/
│   ├── incoming/                     # Edge stream incoming images
│   └── reviewed/                     # Human-verified ground-truth dataset
├── src/
│   └── edgeloop/
│       ├── api/                      # FastAPI endpoints
│       ├── inference/                # MobileNetV2 inference engine
│       ├── drift/                    # Kolmogorov-Smirnov drift detector
│       ├── selection/                # Upload budget & active selection policies
│       ├── review/                   # Human review & dataset provenance
│       ├── training/                 # Leakage-free training & evaluation
│       ├── registry/                 # Safe model promotion & rollback
│       ├── monitoring/               # Metric aggregation & reporting
│       └── common/                   # Config, logging, SQLite database
├── dashboard/                        # Streamlit monitoring & review UI
├── scripts/                          # Smoke tests, CLI utilities, seeding
├── tests/                            # Automated Pytest suite
└── reports/                          # Governance, model cards, and audit logs
```

---

## 6. License & Limitations

All synthetic corruption streams (blur, low-light, noise) represent simulated edge degradation and do not replace field trials in active waste sorting facilities. See [reports/limitations.md](reports/limitations.md) for detailed operational constraints.
