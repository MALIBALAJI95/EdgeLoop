# EdgeLoop Model Card: MobileNetV2 Waste Classifier

## Model Details
- **Model Name**: MobileNetV2 Waste Classifier
- **Architecture**: `torchvision.models.mobilenet_v2` with fine-tuned linear head (`classifier[1] = Linear(1280, 6)`)
- **Version**: `mobilenetv2_baseline` (Active Champion)
- **Framework**: PyTorch 2.x
- **Input Resolution**: $224 \times 224 \times 3$ RGB
- **Input Normalization**: ImageNet standard (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Target Classes**: 6 categories (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`)
- **Intended Deployment**: Edge devices (e.g. Raspberry Pi, Jetson Nano, industrial sorting cameras) and cloud aggregation servers.

---

## Intended Use
- **Primary Use**: Real-time classification of recyclable and municipal solid waste on moving conveyor belts or sorting stations.
- **Secondary Use**: Image quality feature extraction and Shannon entropy uncertainty estimation for selective edge-to-cloud upload.
- **Out-of-Scope**: Autonomous sorting of hazardous materials, medical waste, or explosive items. Not validated for clinical, chemical, or radioactive waste.

---

## Performance Metrics

Measured on fixed, clean holdout dataset (506 images):

| Metric | Measured Value |
| :--- | :--- |
| **Overall Accuracy** | 91.30% (`0.913043`) |
| **Macro-Averaged F1** | 89.43% (`0.894340`) |
| **Weighted-Averaged F1** | 91.21% (`0.912093`) |
| **Inference Latency (CPU)** | ~12 - 25 ms |
| **Inference Latency (MPS / GPU)** | ~4 - 8 ms |

### Per-Class F1 Scores
- `cardboard`: 96.86%
- `glass`: 88.44%
- `metal`: 89.57%
- `paper`: 95.04%
- `plastic`: 90.10%
- `trash`: 76.60% (Underrepresented, highest misclassification risk)

---

## Ethical Considerations & Governance
- **Data Provenance**: Training datasets originated from the TrashNet benchmark. No personal, identifiable, or biometric data is captured or processed.
- **Model Governance Gate**: Retrained candidate models must beat the active champion on holdout Macro-F1 without causing `trash` class F1 to drop by more than 0.02.
- **Human-in-the-Loop Safeguard**: Unreviewed edge predictions are never automatically treated as ground truth. Only explicitly verified labels enter future training manifests.

---

## Limitations & Disclosures
1. **Synthetic vs. Real-World Corruptions**: The corruption stream evaluated in the research study (Gaussian blur, synthetic low-light dimming, additive noise) simulates edge degradation but does not fully represent variable real-world factors such as direct sunlight glare, water condensation on camera lenses, soiled items, or tangled materials.
2. **Dataset Class Imbalance**: The original TrashNet dataset contains relatively few `trash` class samples (27 in the holdout set), limiting the statistical confidence in `trash` precision and recall.
3. **No Field Facility Validation**: The system has been validated on benchmark datasets and simulated streams; it has not yet completed long-term operational field trials in active commercial waste recovery facilities.
