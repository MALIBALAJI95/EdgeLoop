#!/usr/bin/env bash
# ==============================================================================
# EdgeLoop End-to-End Smoke Test Script
# Verifies the full pipeline: Inference -> Drift -> Selection -> Review -> Registry
# ==============================================================================

set -euo pipefail

echo "================================================================="
echo "♻️  Starting EdgeLoop Comprehensive System Smoke Test"
echo "================================================================="

# Locate Python binary
if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
    PYTEST=".venv/bin/pytest"
else
    PYTHON="python3"
    PYTEST="pytest"
fi

echo "Using Python binary: $("$PYTHON" --version) at $PYTHON"

# 1. Environment & Checkpoint Verification
echo -e "\n[1/7] Verifying research archive and baseline model..."
if [[ ! -f "research/edgeloop/mobilenetv2_baseline.pth" ]]; then
    echo "❌ Error: Baseline checkpoint missing at research/edgeloop/mobilenetv2_baseline.pth"
    exit 1
fi
echo "✅ Baseline checkpoint verified."

# 2. Database Initialization & Seeding
echo -e "\n[2/7] Initializing SQLite Registry and Seeding Champion..."
"$PYTHON" -c "
from edgeloop.registry.manager import ModelRegistry
reg = ModelRegistry()
champ = reg.get_champion()
print(f'Active Champion: {champ.version} (Holdout Macro-F1: {champ.macro_f1:.4f})')
"
echo "✅ Registry initialized."

# 3. Model Inference Smoke Test
echo -e "\n[3/7] Testing MobileNetV2 Inference Engine on sample image..."
"$PYTHON" scripts/run_inference.py --image research/edgeloop/corrupted_edge/blur/glass_glass1.jpg
echo "✅ Inference test passed."

# 4. Drift Feature Extraction
echo -e "\n[4/7] Testing Drift Feature Extractor..."
"$PYTHON" -c "
from edgeloop.drift.features import DriftFeatureExtractor
from edgeloop.drift.detector import DriftDetector
from PIL import Image

extractor = DriftFeatureExtractor()
detector = DriftDetector()

img = Image.open('research/edgeloop/corrupted_edge/low_light/glass_glass1.jpg')
feats = extractor.extract_from_image(img)
is_drift, trig = detector.check_sample(feats, log_to_db=False)
print(f'Low-light test: is_drift={is_drift}, triggered_features={trig}')
assert is_drift is True, 'Expected drift alert on low-light sample'
"
echo "✅ Drift detection test passed."

# 5. Selective Upload Policy Comparison
echo -e "\n[5/7] Running Selective Upload Engine..."
"$PYTHON" scripts/compare_selection_policies.py
echo "✅ Selection policies verified."

# 6. Model Governance Gate Verification
echo -e "\n[6/7] Verifying Safe Promotion Rejection on Candidate Regression..."
"$PYTHON" -c "
from edgeloop.registry.manager import ModelRegistry, PromotionRejectionError

reg = ModelRegistry()
try:
    reg.promote_candidate(
        candidate_version='uncertainty_candidate',
        approver='Test',
        promotion_reason='Must be blocked'
    )
    raise AssertionError('Governance failure: Candidate was not rejected!')
except PromotionRejectionError as e:
    print('✅ Governance Gate verified: Successfully rejected inferior candidate.')
"

# 7. Run Full Automated Pytest Suite
echo -e "\n[7/7] Running Pytest Suite (34 Tests)..."
"$PYTEST" tests/ -q
echo "✅ Automated test suite passed."

echo -e "\n================================================================="
echo "🎉 ALL EDGELOOP SMOKE TESTS PASSED SUCCESSFULLY!"
echo "================================================================="
