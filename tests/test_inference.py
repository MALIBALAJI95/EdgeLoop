"""Unit tests for EdgeLoop inference engine."""

import os
import tempfile
from pathlib import Path
import pytest
from PIL import Image
import numpy as np

from edgeloop.inference.engine import (
    InferenceEngine,
    InvalidImageError,
    CheckpointLoadError,
    PredictionResult,
)


@pytest.fixture(scope="module")
def valid_image_path():
    path = Path("research/edgeloop/corrupted_edge/blur/glass_glass1.jpg")
    if not path.exists():
        pytest.skip("Test image not found in research artifacts")
    return path


@pytest.fixture(scope="module")
def engine():
    return InferenceEngine(
        checkpoint_path="research/edgeloop/mobilenetv2_baseline.pth",
        device="cpu",
        log_predictions=False
    )


def test_valid_image_inference(engine, valid_image_path):
    res = engine.predict_image(valid_image_path)
    assert isinstance(res, PredictionResult)
    assert res.predicted_class in engine.classes
    assert 0.0 <= res.confidence <= 1.0
    assert res.entropy >= 0.0
    assert res.inference_time_ms > 0.0
    assert len(res.class_probabilities) == 6

    # Verify probability normalization
    total_prob = sum(res.class_probabilities.values())
    assert pytest.approx(total_prob, rel=1e-3) == 1.0
    assert pytest.approx(res.class_probabilities[res.predicted_class], rel=1e-3) == res.confidence


def test_corrupted_file_handling(engine, tmp_path):
    corrupt_file = tmp_path / "corrupted.jpg"
    corrupt_file.write_bytes(b"JUNK_NOT_A_JPEG_FILE_1234567890")

    with pytest.raises(InvalidImageError):
        engine.predict_image(corrupt_file)


def test_empty_file_handling(engine, tmp_path):
    empty_file = tmp_path / "empty.jpg"
    empty_file.write_bytes(b"")

    with pytest.raises(InvalidImageError):
        engine.predict_image(empty_file)


def test_incorrect_extension_handling(engine, tmp_path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("hello world")

    with pytest.raises(InvalidImageError):
        engine.predict_image(text_file)


def test_missing_checkpoint():
    with pytest.raises(CheckpointLoadError):
        InferenceEngine(
            checkpoint_path="non_existent_model_checkpoint.pth",
            device="cpu",
            log_predictions=False
        )


def test_prediction_output_consistency(engine, valid_image_path):
    res1 = engine.predict_image(valid_image_path)
    res2 = engine.predict_image(valid_image_path)

    assert res1.predicted_class == res2.predicted_class
    assert pytest.approx(res1.confidence, rel=1e-5) == res2.confidence
    assert pytest.approx(res1.entropy, rel=1e-5) == res2.entropy
    for cls in engine.classes:
        assert pytest.approx(res1.class_probabilities[cls], rel=1e-5) == res2.class_probabilities[cls]


def test_batch_inference(engine, valid_image_path):
    images = [valid_image_path, valid_image_path]
    batch_res = engine.predict_batch(images)
    assert len(batch_res) == 2
    assert batch_res[0].predicted_class == batch_res[1].predicted_class
