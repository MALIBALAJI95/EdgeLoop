"""Unit tests for EdgeLoop drift detection and feature extraction."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from PIL import Image

from edgeloop.drift.features import DriftFeatureExtractor
from edgeloop.drift.detector import DriftDetector


@pytest.fixture
def extractor():
    return DriftFeatureExtractor()


@pytest.fixture
def clean_image():
    # Synthetic clean image
    arr = np.full((100, 100, 3), 180, dtype=np.uint8)
    # Add high-contrast pattern
    arr[20:80, 20:80] = 50
    return Image.fromarray(arr)


@pytest.fixture
def low_light_image(clean_image):
    # Darken image
    arr = np.array(clean_image, dtype=np.float32) * 0.2
    return Image.fromarray(arr.astype(np.uint8))


def test_feature_extractor_outputs(extractor, clean_image):
    feats = extractor.extract_from_image(clean_image)
    required = ["brightness", "contrast", "edge_strength", "high_frequency_std", "high_frequency_mean_abs"]
    for key in required:
        assert key in feats
        assert isinstance(feats[key], float)
        assert feats[key] >= 0.0


def test_low_light_drift_detection(extractor, clean_image, low_light_image):
    clean_feats = extractor.extract_from_image(clean_image)
    dark_feats = extractor.extract_from_image(low_light_image)

    # Brightness and contrast should decrease in low-light
    assert dark_feats["brightness"] < clean_feats["brightness"]
    assert dark_feats["edge_strength"] < clean_feats["edge_strength"]


def test_ks_window_drift():
    # Build clean reference distribution
    np.random.seed(42)
    n = 100
    ref_df = pd.DataFrame({
        "brightness": np.random.normal(0.65, 0.05, n),
        "contrast": np.random.normal(0.25, 0.03, n),
        "edge_strength": np.random.normal(0.18, 0.02, n),
        "high_frequency_std": np.random.normal(10.0, 1.5, n),
        "high_frequency_mean_abs": np.random.normal(3.5, 0.5, n)
    })

    detector = DriftDetector(reference_data=ref_df, ks_p_value_threshold=0.01)

    # 1. Clean window from same distribution should NOT drift
    clean_window = pd.DataFrame({
        "brightness": np.random.normal(0.65, 0.05, 50),
        "contrast": np.random.normal(0.25, 0.03, 50),
        "edge_strength": np.random.normal(0.18, 0.02, 50),
        "high_frequency_std": np.random.normal(10.0, 1.5, 50),
        "high_frequency_mean_abs": np.random.normal(3.5, 0.5, 50)
    })
    res_clean = detector.check_window_drift(clean_window)
    assert res_clean["is_drift"] is False
    assert len(res_clean["triggered_features"]) == 0

    # 2. Shifted window (low-light simulation) should trigger drift
    drift_window = pd.DataFrame({
        "brightness": np.random.normal(0.20, 0.03, 50),  # Marked drop
        "contrast": np.random.normal(0.08, 0.02, 50),
        "edge_strength": np.random.normal(0.05, 0.01, 50),
        "high_frequency_std": np.random.normal(2.0, 0.5, 50),
        "high_frequency_mean_abs": np.random.normal(1.0, 0.2, 50)
    })
    res_drift = detector.check_window_drift(drift_window)
    assert res_drift["is_drift"] is True
    assert "brightness" in res_drift["triggered_features"]
    assert "edge_strength" in res_drift["triggered_features"]


def test_single_sample_outlier_detection():
    detector = DriftDetector()
    extreme_features = {
        "brightness": 0.01,  # extreme dark
        "contrast": 0.01,
        "edge_strength": 0.001,
        "high_frequency_std": 0.1,
        "high_frequency_mean_abs": 0.1
    }
    is_drift, triggered = detector.check_sample(extreme_features, log_to_db=False)
    assert is_drift is True
    assert len(triggered) > 0
