# EdgeLoop Reproducibility Guide

This guide ensures complete deterministic reproducibility of EdgeLoop research benchmarks, selective upload evaluations, and model governance experiments.

---

## 1. Random Seeds & Determinism

EdgeLoop enforces fixed pseudo-random seeds across all stochastic operations:
- **Baseline Seed**: `42`
- **PyTorch**: `torch.manual_seed(42)`
- **NumPy**: `np.random.default_rng(42)`
- **Python**: `random.seed(42)`

---

## 2. Reproducing the Selective Upload Benchmark

The Kaggle research experiment evaluated 4 active learning selection policies on a 2,024-image corrupted edge stream under a strict 10% (203-record) budget.

To reproduce the exact error capture numbers:
```bash
python scripts/compare_selection_policies.py
```

Expected Output Table:
```text
    strategy  selected_records  unique_source_images  errors_captured error_rate  unique_error_images
      random               203                   173               46     22.66%                   43
 uncertainty               203                   134              120     59.11%                   89
unique_aware               203                   203              102     50.25%                  102
 drift_aware               203                   188               83     40.89%                   79
```

---

## 3. Reproducing the Candidate-vs-Champion Holdout Benchmark

To reproduce the holdout evaluation metrics comparing the MobileNetV2 Baseline against the two retraining candidates:
```bash
python -c "
import json
with open('research/edgeloop/retraining_holdout_detailed.json') as f:
    data = json.load(f)
for m in data:
    print(f\"{m['model']}: Accuracy={m['accuracy']:.4f}, Macro-F1={m['macro_f1']:.4f}, Trash F1={m['classification_report']['trash']['f1-score']:.4f}\")
"
```

Expected Output:
```text
baseline: Accuracy=0.9130, Macro-F1=0.8943, Trash F1=0.7660
uncertainty_candidate: Accuracy=0.9051, Macro-F1=0.8855, Trash F1=0.7451
unique_aware_candidate: Accuracy=0.9032, Macro-F1=0.8805, Trash F1=0.7170
```

---

## 4. Reproducing Automated Verification

Run all unit tests, integration tests, and promotion rejection tests:
```bash
pytest tests/ -v
```

Expected Result:
- 34 passed tests with 0 failures.
