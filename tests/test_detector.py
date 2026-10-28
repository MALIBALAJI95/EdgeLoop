"""Unit tests for multi-object waste detector and bin recommendation."""

from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from edgeloop.inference.detector import WasteObjectDetector, DetectionFrameResult, BIN_MAPPINGS
from edgeloop.inference.tracker import StreamObjectTracker


@pytest.fixture(scope="module")
def detector():
    return WasteObjectDetector(device="cpu")


@pytest.fixture(scope="module")
def sample_image():
    for candidate in [
        Path("tests/fixtures/sample_waste.jpg"),
        Path("research/edgeloop/corrupted_edge/blur/glass_glass1.jpg"),
    ]:
        if candidate.exists():
            return Image.open(candidate).convert("RGB")
    # Synthetic fallback
    return Image.new("RGB", (224, 224), (120, 150, 180))


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


def test_tracker_deduplication(detector, sample_image):
    tracker = StreamObjectTracker(iou_threshold=0.35, min_hits_to_confirm=2)
    res = detector.detect_and_classify(sample_image)

    # Frame 1
    active1 = tracker.update(res.objects, full_image=sample_image)
    assert len(active1) == len(res.objects)
    track_ids_1 = [t.track_id for t in active1]

    # Frame 2 with same detections (simulating next video frame)
    active2 = tracker.update(res.objects, full_image=sample_image)
    track_ids_2 = [t.track_id for t in active2]

    # Must match same track IDs without creating duplicates
    assert track_ids_1 == track_ids_2

    # Session unique objects must count only once
    unique_objs = tracker.get_session_unique_objects()
    assert len(unique_objs) == len(res.objects)
