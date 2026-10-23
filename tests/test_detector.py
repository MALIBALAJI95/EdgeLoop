"""Unit tests for multi-object waste detector and bin recommendation."""

from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from edgeloop.inference.detector import WasteObjectDetector, DetectionFrameResult, BIN_MAPPINGS


@pytest.fixture(scope="module")
def detector():
    return WasteObjectDetector(device="cpu")


@pytest.fixture(scope="module")
def sample_image():
    p = Path("research/edgeloop/corrupted_edge/blur/glass_glass1.jpg")
    if not p.exists():
        pytest.skip("Test image not found")
    return Image.open(p).convert("RGB")


def test_detector_inference(detector, sample_image):
    result = detector.detect_and_classify(sample_image)
    assert isinstance(result, DetectionFrameResult)
    assert result.total_objects >= 1
    assert result.dominant_class in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    assert result.dominant_bin is not None
    assert result.processing_time_ms > 0

    first_obj = result.objects[0]
    assert len(first_obj.box) == 4
    assert 0 <= first_obj.waste_confidence <= 1.0
    assert first_obj.suggested_bin in [v["bin_name"] for v in BIN_MAPPINGS.values()]


def test_detector_draw_overlays(detector, sample_image):
    result = detector.detect_and_classify(sample_image)
    annotated = detector.draw_overlays(sample_image, result)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape[:2] == (sample_image.height, sample_image.width)
    assert annotated.shape[2] == 3


def test_bin_mapping_consistency():
    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    for c in classes:
        assert c in BIN_MAPPINGS
        assert "bin_name" in BIN_MAPPINGS[c]
        assert "color_rgb" in BIN_MAPPINGS[c]
        assert "color_bgr" in BIN_MAPPINGS[c]
        assert "guidance" in BIN_MAPPINGS[c]
