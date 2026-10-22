# EdgeLoop Experiment Results & Empirical Analysis

## 1. Baseline Model vs. Retrained Candidates

In the research experiments, candidate models were fine-tuned using the 203 records selected by each active learning policy from the 2,024-image corrupted edge stream. Both candidates were evaluated against the preserved clean benchmark holdout (506 images).

### Overall Metrics Comparison

| Model | Checkpoint File | Accuracy | Macro-F1 | Weighted-F1 | Promotion Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2 Baseline (Champion)** | `mobilenetv2_baseline.pth` | **0.913043** | **0.894340** | **0.912093** | **ACTIVE CHAMPION** |
| Uncertainty Candidate | `uncertainty_candidate.pth` | 0.905138 | 0.885502 | 0.903914 | **REJECTED** |
| Unique-Aware Candidate | `unique_aware_candidate.pth` | 0.903162 | 0.880525 | 0.901847 | **REJECTED** |

---

## 2. Per-Class F1-Score Breakdown

| Class | Support | Champion Baseline F1 | Uncertainty Candidate F1 | Unique-Aware Candidate F1 | Delta (Uncertainty vs Champion) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cardboard` | 81 | 0.9686 | 0.9620 | 0.9620 | -0.0066 |
| `glass` | 100 | 0.8844 | 0.8824 | 0.8713 | -0.0020 |
| `metal` | 82 | 0.8957 | 0.8846 | 0.8889 | -0.0111 |
| `paper` | 119 | 0.9504 | 0.9417 | 0.9456 | -0.0087 |
| `plastic` | 97 | 0.9010 | 0.8974 | 0.8984 | -0.0036 |
| `trash` | 27 | **0.7660** | **0.7451** | **0.7170** | **-0.0209** |

### Detailed Analysis:
1. **Universal Drop in Macro-F1**: Every single class experienced an F1 drop in both candidates compared to the baseline.
2. **Severe Degradation in `trash`**: The `trash` class suffered the worst regression:
   - Uncertainty Candidate: dropped from 76.60% to 74.51% (-2.09%)
   - Unique-Aware Candidate: dropped from 76.60% to 71.70% (-4.90%)
3. **Reason for Retraining Failure**: Fine-tuning directly on synthetic corruptions without distribution regularization causes the model to over-index on artifact features (e.g. noise speckles, blur edges) at the expense of clean visual discriminators.

---

## 3. Selective Upload Policy Performance

Evaluated on the full 2,024-image corrupted stream with budget = 203 records (10%):

| Strategy | Records Selected | Unique Physical Objects | Errors Captured | Selection Error Rate | Unique Error Sources |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Baseline** | 203 | 173 | 46 | 22.66% | 43 |
| **Uncertainty Policy** | 203 | 134 | 120 | 59.11% | 89 |
| **Unique-Aware Policy** | 203 | 203 | 102 | 50.25% | 102 |
| **Drift-Aware Policy** | 203 | 188 | 83 | 40.89% | 79 |

### Visual / Condition Allocation:
- **Random**: Low-light 64, Clean 55, Blur 44, Noise 40
- **Uncertainty**: Noise 95, Low-light 54, Blur 42, Clean 12
- **Unique-Aware**: Noise 112, Low-light 46, Blur 35, Clean 10
- **Drift-Aware**: Low-light 184, Blur 9, Clean 5, Noise 5

### Conclusion:
- **Uncertainty** maximizes raw error harvesting on the stream.
- **Unique-Aware** provides the best diversity coverage across unique source objects.
- **Drift-Aware** effectively detects edge anomalies, but naive thresholding can lead to starvation of non-dominant corruption modes.
