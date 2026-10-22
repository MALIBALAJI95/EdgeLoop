"""Unit tests for EdgeLoop human review service and provenance."""

import tempfile
from pathlib import Path
import pytest

from edgeloop.common.database import Database
from edgeloop.review.service import ReviewService, ReviewSubmission


@pytest.fixture
def review_env(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_review.db"
    test_db = Database(test_db_path)

    # Patch global database
    import edgeloop.common.database as db_mod
    monkeypatch.setattr(db_mod, "_db_instance", test_db)

    svc = ReviewService()
    svc.reviewed_data_dir = tmp_path / "reviewed"
    svc.reviewed_data_dir.mkdir(parents=True, exist_ok=True)

    # Seed an inference and selection queue item
    inf_id = test_db.log_inference(
        sample_id="glass_test_01",
        image_path=str(tmp_path / "glass_test_01.jpg"),
        predicted_class="glass",
        confidence=0.85,
        entropy=0.45,
        latency_ms=12.5,
        model_version="mobilenetv2_baseline",
        condition="clean"
    )
    # Add dummy file for checksum test
    (tmp_path / "glass_test_01.jpg").write_bytes(b"FAKE_IMAGE_DATA")

    queue_id = test_db.add_to_selection_queue(
        inference_id=inf_id,
        sample_id="glass_test_01",
        image_path=str(tmp_path / "glass_test_01.jpg"),
        strategy="uncertainty",
        upload_reason="uncertainty_entropy",
        entropy=0.45
    )

    return svc, queue_id, inf_id, tmp_path


def test_get_pending_queue(review_env):
    svc, queue_id, inf_id, _ = review_env
    pending = svc.get_pending_queue()
    assert len(pending) == 1
    assert pending[0].queue_id == queue_id
    assert pending[0].predicted_class == "glass"
    assert pending[0].sample_id == "glass_test_01"


def test_submit_valid_review_with_correction(review_env):
    svc, queue_id, inf_id, _ = review_env

    # Correct prediction: glass -> metal
    sub = ReviewSubmission(
        queue_id=queue_id,
        true_label="metal",
        reviewer="annotator_alice",
        notes="Actually made of tin"
    )
    rev_id = svc.submit_review(sub)
    assert rev_id > 0

    # Pending queue should now be empty
    assert len(svc.get_pending_queue()) == 0

    # Dataset export should succeed and contain 1 record
    version_info = svc.export_dataset_version("First test export")
    assert version_info.record_count == 1
    assert Path(version_info.manifest_path).exists()


def test_submit_invalid_label(review_env):
    svc, queue_id, _, _ = review_env
    sub = ReviewSubmission(
        queue_id=queue_id,
        true_label="radioactive_waste",  # Invalid
        reviewer="annotator_alice"
    )
    with pytest.raises(ValueError, match="Invalid label"):
        svc.submit_review(sub)


def test_submit_empty_reviewer(review_env):
    svc, queue_id, _, _ = review_env
    sub = ReviewSubmission(
        queue_id=queue_id,
        true_label="glass",
        reviewer="   "  # Blank
    )
    with pytest.raises(ValueError, match="Reviewer identity cannot be blank"):
        svc.submit_review(sub)
