# EdgeLoop System Limitations & Operational Boundaries

This document provides transparent disclosure of EdgeLoop's assumptions, theoretical limitations, and operational boundaries.

---

## 1. Synthetic vs. Real-World Corruptions

The experimental results and drift evaluations rely on synthetic corruptions applied to TrashNet source images:
- **Gaussian Blur**: Simulates conveyor motion blur or optical defocus.
- **Low-Light Dimming**: Simulates poor facility illumination or sensor under-exposure.
- **Additive Noise**: Simulates high-ISO sensor gain or thermal electrical noise.

### Crucial Distinction:
- These synthetic perturbations represent **controlled mathematical proxies** for environmental degradation.
- They do **not** capture complex physical phenomena encountered in real-world municipal waste recovery facilities (MRFs):
  - Direct sun glints and specular reflections on wet plastic.
  - Crushed, torn, or highly soiled cardboard soaked with liquids.
  - Nested or occluded items (e.g. an aluminum can inside a transparent plastic bag).
  - High conveyor vibration and dynamic multi-axis rotations.

> [!CAUTION]
> EdgeLoop has **not** been validated in operating commercial waste sorting facilities. Claims of real-world deployment readiness must be withheld until extensive physical on-belt trials are performed.

---

## 2. Dataset Distribution & Bias

1. **TrashNet Benchmark Limitations**:
   - TrashNet was collected by photographing individual items against white poster-board backgrounds under controlled laboratory conditions.
   - The dataset lacks background clutter, conveyor textures, and multi-object scenes.
2. **`trash` Class Imbalance**:
   - The holdout evaluation set contains only 27 `trash` class instances out of 506 (5.3%).
   - While representing general refuse, this class has high visual ambiguity, leading to the lowest baseline F1 score (76.60%).

---

## 3. Drift Detector Sensitivity & Condition Bias

1. **Condition Skew in Naive Thresholding**:
   - As observed in `checkpoint_v0.3.json`, naive threshold drift detectors flagged 100% of low-light samples while flagging only 6.5% of blur and 3.4% of noise.
   - Using naive drift detection to drive upload selection caused extreme condition starvation (184 low-light samples out of 203 selected).
2. **Kolmogorov-Smirnov Window Latency**:
   - The two-sample KS test requires an observation window of at least 20-50 samples to achieve statistical power ($p < 0.01$). It does not provide instantaneous single-frame alerting with the same statistical confidence.

---

## 4. Retraining Generalization Gap

Fine-tuning MobileNetV2 directly on active-learning-selected corrupted images resulted in negative transfer:
- Both candidates had lower clean holdout accuracy and Macro-F1.
- Active learning selection captures edge errors, but naive empirical risk minimization on these errors without regularization damages representations of clean data.

---

## 5. Local Cloud Simulation vs. Production Cloud Infrastructure

- In this release, cloud synchronization and registry operations run locally via SQLite, file system atomic operations, and FastAPI REST endpoints.
- Production scale deployment will require object storage (e.g. S3 / GCS), message queues (e.g. Kafka / Celery), and managed relational databases (PostgreSQL).
