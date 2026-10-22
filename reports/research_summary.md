# EdgeLoop: Research Summary & Baseline Empirical Evidence

## 1. Executive Summary

EdgeLoop originates from empirical Kaggle research investigating edge-to-cloud machine learning loops for automated waste sorting under environmental degradations. The study evaluated MobileNetV2 on the 6-class TrashNet dataset (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`) subjected to edge-stream image corruptions (blur, low-light, sensor noise).

Treating the archived artifacts in `EdgeLoop_Kaggle_Research_v0.5.zip` as read-only source evidence, this document summarizes the baseline performance, active learning selection strategies, and retraining candidate evaluations.

---

## 2. Baseline Model Performance

The champion model is a fine-tuned **MobileNetV2** classifier evaluated on a fixed, clean holdout set of 506 TrashNet images.

- **Holdout Accuracy**: `0.913043` (91.30%)
- **Holdout Macro-F1**: `0.894340` (89.43%)
- **Evaluation Set Size**: 506 images

### Per-Class Classification Report (Holdout Set)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `cardboard` | 0.9872 | 0.9506 | **0.9686** | 81 |
| `glass` | 0.8889 | 0.8800 | **0.8844** | 100 |
| `metal` | 0.9012 | 0.8902 | **0.8957** | 82 |
| `paper` | 0.9350 | 0.9664 | **0.9504** | 119 |
| `plastic` | 0.8667 | 0.9381 | **0.9010** | 97 |
| `trash` | 0.9000 | 0.6667 | **0.7660** | 27 |
| **Macro Average** | **0.9132** | **0.8820** | **0.8943** | **506** |
| **Weighted Average** | **0.9138** | **0.9130** | **0.9121** | **506** |

> [!NOTE]
> The `trash` class exhibits the lowest baseline recall (66.67%) and F1-score (76.60%), attributable to high within-class visual variance and lower representation (support: 27). This makes `trash` a primary regression canary in candidate promotion governance.

---

## 3. Selective Upload Experiment on Synthetic Corruptions

Edge devices operate under severe bandwidth and compute constraints. A stream of 2,024 images was evaluated across 4 conditions: `clean`, `blur`, `low_light`, and `noise` (506 images per condition, originating from the same 506 source physical objects).

With an upload budget constrained to **203 records (10% of stream)**, four selection policies were compared:

| Strategy | Selected Records | Unique Source Images | Captured Errors | Error Capture Rate | Unique Error Sources |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Baseline** | 203 | 173 | 46 | 22.66% | 43 |
| **Uncertainty-Based** | 203 | 134 | 120 | 59.11% | 89 |
| **Unique-Aware** | 203 | 203 | 102 | 50.25% | 102 |
| **Drift-Aware** | 203 | 188 | 83 | 40.89% | 79 |

### Key Empirical Findings:
1. **Uncertainty Capture vs. Diversity**: Standard uncertainty selection captured the highest raw error count (120 errors), but redundant corruptions of the same object consumed the budget, resulting in only 134 unique images.
2. **Unique-Aware Policy**: By capping representation to one variant per physical sample before taking duplicates, unique-aware selection captured 102 errors across 102 distinct physical items.
3. **Drift Detector Bias**: The naive threshold-based drift detector flagged 100% of `low_light` images while missing subtle blur and noise. Consequently, drift-aware selection allocated 184 of its 203 slots to low-light samples. Kolmogorov-Smirnov two-sample testing resolved this condition bias.

---

## 4. Retraining Candidate Evaluation and Model Governance

Two candidate models were retrained using 203 selected samples and evaluated against the identical fixed clean holdout set:

| Model Version | Holdout Accuracy | Holdout Macro-F1 | `trash` Class F1 | Governance Decision |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2 Baseline** | **0.913043** | **0.894340** | **0.765957** | **Active Champion** |
| Uncertainty Candidate | 0.905138 (-0.79%) | 0.885502 (-0.88%) | 0.745098 (-2.09%) | **Rejected (Regression)** |
| Unique-Aware Candidate | 0.903162 (-0.99%) | 0.880525 (-1.38%) | 0.716981 (-4.90%) | **Rejected (Regression)** |

### Governance Decision Rationale:
Both retrained candidates exhibited regressions on the clean holdout benchmark:
1. Macro-F1 fell by 0.88% (Uncertainty) and 1.38% (Unique-Aware).
2. The sensitive `trash` class regressed significantly from 0.7660 down to 0.7451 and 0.7170.

**Enforced Governance Rule**: Neither candidate is permitted to be promoted. High error-capture during active learning does not automatically translate to improved clean holdout generalization. EdgeLoop strictly enforces baseline retention.
