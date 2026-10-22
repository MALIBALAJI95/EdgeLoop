# EdgeLoop Data & Split Policy

## 1. Principles of Data Provenance and Split Hygiene

In edge-to-cloud machine learning systems with repeated synthetic or real-world image variants, data leakage is the most common cause of inflated validation scores and catastrophic production failure. EdgeLoop establishes strict architectural data split rules:

1. **Group-Based Partitioning by Physical Source ID**:
   - Multiple views, corruptions (clean, blur, low_light, noise), or time frames of the same physical item share a root `sample_id` (e.g., `glass_glass320`).
   - All variants of a given `sample_id` MUST reside exclusively on one side of a split (either in Train, Validation, or Holdout).
   - Random record-level splitting across variants is strictly prohibited.

2. **Fixed Benchmark Holdout Set**:
   - The 506-sample clean TrashNet holdout set is strictly preserved and read-only.
   - It is used solely for final candidate-versus-champion comparative evaluation, never for hyperparameter tuning or early stopping.

3. **Ground-Truth Integrity Barrier**:
   - Edge predictions, regardless of confidence, are classified as unverified telemetry.
   - Unverified predictions are NEVER ingested into training datasets.
   - Training manifests are constructed exclusively from human-verified entries in `review_records` with cryptographic SHA-256 image hashes.

---

## 2. Dataset Composition & Splits

### Baseline Dataset: TrashNet (6 Classes)
- **Total Images**: 2,527 images
- **Train Split**: 1,515 images (60%)
- **Validation Split**: 506 images (20%)
- **Test Holdout Split**: 506 images (20%)

### Class Distribution in Benchmark Holdout:
- `cardboard`: 81
- `glass`: 100
- `metal`: 82
- `paper`: 119
- `plastic`: 97
- `trash`: 27

---

## 3. Human Review & Versioning Workflow

```
[ Edge Prediction ] 
        │ (Selected under budget)
        ▼
[ Selection Queue ] 
        │ (Human Reviewer audits)
        ▼
[ review_records ] (SQLite: reviewer, notes, timestamp, is_corrected)
        │
        ▼
[ export_dataset_version() ]
        │
        ▼
[ data/reviewed/manifest_trashnet_reviewed_v_YYYYMMDD_HHMMSS.csv ]
(Includes sample_id, image_path, label, is_corrected, reviewer, sha256)
```

## 4. Leakage Prevention Checks in Code

The `CandidateTrainer.group_train_val_split()` function implements grouping by `sample_id`:

```python
unique_samples = df["sample_id"].unique()
val_count = int(round(len(unique_samples) * val_fraction))
val_samples = set(unique_samples[:val_count])
train_samples = set(unique_samples[val_count:])

train_df = df[df["sample_id"].isin(train_samples)]
val_df = df[df["sample_id"].isin(val_samples)]
```

Unit tests verify that `set(train_df["sample_id"]).intersection(set(val_df["sample_id"])) == set()`.
