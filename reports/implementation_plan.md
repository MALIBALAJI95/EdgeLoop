# EdgeLoop: Implementation Plan

## Drift-Aware Edge-to-Cloud Machine Learning System for Waste Classification

### 1. Executive Summary & Architecture Overview

EdgeLoop is an edge-to-cloud MLOps system for waste classification using PyTorch MobileNetV2 trained on the 6-class TrashNet dataset (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`). The system addresses edge network constraints, input distribution shifts (low-light, blur, sensor noise), active learning sample selection under budget constraints, human-in-the-loop review, and strict model governance with explicit promotion gates.

```
       [ Edge Device / Stream ]
                  │
        (1) Image Ingestion & Validation
                  │
        (2) Baseline MobileNetV2 Inference
         ├── Predicted Class, Probs, Entropy, Latency
         └── (3) Feature Extraction (Brightness, Contrast, Edges, High-Freq)
                  │
        (4) Drift Detector (Kolmogorov-Smirnov vs Reference)
                  │
        (5) Budget-Constrained Selective Upload
         ├── Random | Uncertainty | Unique-Aware | Drift-Aware
         └── Dedupes repeated variants of source images
                  │  (Selected records uploaded)
                  ▼
      [ Cloud / Server Central Registry ]
                  │
        (6) SQLite Persistent Store & Audit Log
                  │
        (7) Human-in-the-Loop Review Workflow
         ├── Verify/Correct predictions
         └── Provenance-tracked ground-truth datasets
                  │
        (8) Candidate Model Retraining & Evaluation
         ├── Validation metrics vs Champion
         └── Class-level regression checks (esp. 'trash')
                  │
        (9) Safe Model Registry & Promotion Gate
         ├── Strict performance criteria (Macro-F1 + No class drops)
         ├── Explicit Human Approval
         └── Safe Rollback Mechanism
                  │
        (10) FastAPI REST API + Streamlit Operational Dashboard
```

---

### 2. Research Evidence & Model Governance Baseline

The baseline results from `EdgeLoop_Kaggle_Research_v0.5.zip` establish the initial champion metrics:
- **Architecture**: MobileNetV2 (`classifier[1] = Linear(1280, 6)`)
- **Holdout Test Set**: 506 clean images from TrashNet
- **Baseline Accuracy**: `0.913043` (91.30%)
- **Baseline Macro-F1**: `0.894340` (89.43%)
- **Per-Class Support & F1**:
  - `cardboard`: F1 `0.968553` (support 81)
  - `glass`: F1 `0.884422` (support 100)
  - `metal`: F1 `0.895705` (support 82)
  - `paper`: F1 `0.950413` (support 119)
  - `plastic`: F1 `0.900990` (support 97)
  - `trash`: F1 `0.765957` (support 27) — *Known weak class; strict regression checks required.*

#### Governance Rule on Retraining Candidates
In Kaggle research experiments, two candidates were trained on 203 selected synthetic stream samples:
1. `uncertainty_candidate`: Holdout Accuracy `0.905138`, Macro-F1 `0.885502`
2. `unique_aware_candidate`: Holdout Accuracy `0.903162`, Macro-F1 `0.880525`

**Governance Mandate**: Neither candidate outperformed the baseline champion (`0.894340`). Under no circumstance may either candidate be automatically promoted. The baseline remains the champion until a candidate meets all statistical gates.

---

### 3. Implementation Phases

#### Phase 1: Research Verification & Foundation Scaffolding
- [x] Inspect workspace and unpack `EdgeLoop_Kaggle_Research_v0.5.zip` to `research/`.
- [x] Inspect state dictionary format, classes, and normalization parameters.
- [x] Generate `research/archive_manifest.json`.
- [x] Create project layout, virtual environment, and dependency specifications.
- [x] Run baseline model forward pass smoke test.

#### Phase 2: Inference Engine & Local Storage
- [ ] Implement `src/edgeloop/inference/engine.py` for MobileNetV2 inference.
- [ ] Implement image preprocessing, validation (format, dimensions, max file size), softmax probabilities, and Shannon entropy uncertainty computation.
- [ ] Implement `src/edgeloop/common/database.py` with SQLite schema for inferences, drift events, review queue, model registry, and dataset manifests.
- [ ] Implement CLI tools for single-image and batch inference.
- [ ] Unit tests for inference, corrupted image handling, and output schemas.

#### Phase 3: Drift Detection & Image Quality Feature Monitoring
- [ ] Implement `src/edgeloop/drift/features.py` computing:
  - Brightness (mean grayscale luminance)
  - Contrast (standard deviation of luminance)
  - Edge strength (Sobel filter gradient magnitude)
  - High-frequency standard deviation (Laplacian high-pass filter)
  - High-frequency mean absolute magnitude
- [ ] Implement `src/edgeloop/drift/detector.py` using two-sample Kolmogorov-Smirnov test against reference distribution.
- [ ] Support sliding window drift detection, alert generation, and feature contribution logging.
- [ ] Unit tests for feature extraction and drift alerting.

#### Phase 4: Selective Upload & Budget Management
- [ ] Implement `src/edgeloop/selection/policies.py`:
  - `RandomSelectionPolicy`: Uniform random baseline.
  - `UncertaintySelectionPolicy`: Top entropy / least confidence.
  - `UniqueAwareSelectionPolicy`: Deduplicates multiple corruptions/variants of the same source image.
  - `DriftAwareSelectionPolicy`: Prioritizes images triggering drift alerts combined with uncertainty.
- [ ] Implement budget management (record count or transfer volume budget).
- [ ] Implement offline evaluation utility comparing selection policies on research stream.

#### Phase 5: Human Review & Dataset Provenance
- [ ] Implement `src/edgeloop/review/service.py` managing the review queue.
- [ ] Allow reviewers to confirm/correct labels, provide notes, and record timestamps.
- [ ] Guarantee provenance: raw predictions never overwrite labels; only human-verified images enter training datasets.
- [ ] Version reviewed datasets with immutable manifests.

#### Phase 6: Training Pipeline & Evaluation
- [ ] Implement `src/edgeloop/training/trainer.py` for MobileNetV2 fine-tuning on reviewed datasets.
- [ ] Enforce leak-free data splitting (grouping by unique source image).
- [ ] Compute comprehensive evaluation metrics (Accuracy, Macro-F1, Weighted-F1, per-class P/R/F1, confusion matrix, inference latency).
- [ ] Enforce class-level regression check (especially for `trash` class).

#### Phase 7: Model Registry & Safe Promotion Gates
- [ ] Implement `src/edgeloop/registry/manager.py` managing lifecycle:
  `candidate -> evaluated -> approved -> champion`
- [ ] Promotion requirements:
  1. Holdout Macro-F1 strictly exceeds Champion.
  2. No class drops below tolerance (e.g. `trash` F1 within -0.02).
  3. Complete training provenance attached.
  4. Explicit authorized approval recorded.
- [ ] Rollback functionality to safely revert to previous champion with atomic file swaps.

#### Phase 8: REST API & Streamlit Dashboard
- [ ] Implement FastAPI backend in `src/edgeloop/api/`:
  - Inference endpoints (`/infer`, `/infer/batch`)
  - Drift status (`/drift/status`, `/drift/features`)
  - Selection & Review endpoints (`/selection/queue`, `/review/pending`, `/review/submit`)
  - Model registry & promotion endpoints (`/registry/models`, `/registry/promote`, `/registry/rollback`)
- [ ] Build interactive Streamlit dashboard (`dashboard/app.py`):
  - Model status & active champion
  - Live prediction feed & class distribution
  - Drift trends & feature radar/time-series
  - Upload budget utilization
  - Interactive human review queue
  - Candidate comparison & safe promotion console

#### Phase 9: Packaging, Testing & Documentation
- [ ] Comprehensive test suite covering unit tests, API tests, promotion rejection, and rollback.
- [ ] Dockerfile and Docker Compose configuration.
- [ ] CLI smoke test script `scripts/smoke_test.sh`.
- [ ] Complete engineering reports in `reports/`.
