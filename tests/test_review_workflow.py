"""Tests for EdgeLoop automatic detection, review workflow, and candidate learning gate."""

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from edgeloop.common.database import Database
from edgeloop.inference.detector import WasteObjectDetector, DetectedWasteObject, BIN_MAPPINGS
from edgeloop.inference.tracker import StreamObjectTracker, TrackedWasteObject
from edgeloop.registry.manager import ModelRegistry, PromotionRejectionError
from edgeloop.training.trainer import CandidateTrainer


@pytest.fixture
def test_env(tmp_path, monkeypatch):
    test_db = Database(tmp_path / "test_workflow.db")
    import edgeloop.common.database as db_mod
    monkeypatch.setattr(db_mod, "_db_instance", test_db)

    # Mock baseline checkpoint
    dummy_ckpt = tmp_path / "dummy_base.pth"
    dummy_ckpt.write_bytes(b"DUMMY_WEIGHTS")

    reg = ModelRegistry()
    reg.champion_dir = tmp_path / "champion"
    reg.champion_dir.mkdir(parents=True, exist_ok=True)
    reg.champion_ckpt_path = reg.champion_dir / "mobilenetv2_champion.pth"
    reg.previous_champion_ckpt_path = reg.champion_dir / "mobilenetv2_previous_champion.pth"

    test_db.register_model(
        version="mobilenetv2_baseline",
        model_name="MobileNetV2",
        checkpoint_path=str(dummy_ckpt),
        stage="champion",
        accuracy=0.913043,
        macro_f1=0.894340,
        trash_f1=0.765957,
        approved_by="System",
        promotion_reason="Initial Champion Baseline"
    )
    shutil.copy2(str(dummy_ckpt), str(reg.champion_ckpt_path))

    return test_db, reg, tmp_path


def test_tracker_temporal_smoothing_and_uncertainty():
    tracker = StreamObjectTracker(iou_threshold=0.35, min_hits_to_confirm=1)

    det_uncertain = DetectedWasteObject(
        box=[10, 10, 100, 100],
        detector_label="book",
        detector_confidence=0.85,
        waste_class="paper",
        waste_confidence=0.60,  # Below 0.65 threshold -> uncertain
        entropy=0.85,            # Above 0.80 -> uncertain
        suggested_bin="Blue Bin (Paper & Books)",
        bin_code="PAPER_BLUE",
        bin_guidance="Clean paper and books."
    )

    test_img = Image.new("RGB", (200, 200), (200, 200, 200))
    active = tracker.update([det_uncertain], full_image=test_img)

    assert len(active) == 1
    track = active[0]
    assert track.track_id == 1
    assert track.detector_label == "book"
    assert track.waste_class == "paper"
    assert track.is_uncertain is True
    assert track.crop_image is not None

    # Update with shifted box and high confidence
    det_certain = DetectedWasteObject(
        box=[12, 12, 102, 102],
        detector_label="book",
        detector_confidence=0.90,
        waste_class="paper",
        waste_confidence=0.95,
        entropy=0.20,
        suggested_bin="Blue Bin (Paper & Books)",
        bin_code="PAPER_BLUE",
        bin_guidance="Clean paper and books."
    )
    active2 = tracker.update([det_certain], full_image=test_img)
    assert len(active2) == 1
    assert active2[0].track_id == 1
    assert active2[0].hit_streak == 2
    assert active2[0].waste_confidence == 0.95
    assert active2[0].is_uncertain is False


def test_session_review_and_database_provenance(test_env):
    db, reg, tmp_path = test_env

    # 1. Simulate verified session commit
    sample_id = "test_item_1"
    img_path = tmp_path / f"{sample_id}.jpg"
    Image.new("RGB", (224, 224), (100, 150, 200)).save(img_path)

    inf_id = db.log_inference(
        sample_id=sample_id,
        image_path=str(img_path),
        predicted_class="plastic",
        confidence=0.72,
        entropy=0.68,
        latency_ms=38.5,
        model_version="mobilenetv2_baseline",
        condition="camera_live",
        uploaded=1
    )

    q_id = db.add_to_selection_queue(
        inference_id=inf_id,
        sample_id=sample_id,
        image_path=str(img_path),
        strategy="human_verified",
        upload_reason="session_audit",
        entropy=0.68
    )

    rev_id = db.submit_review(
        queue_id=q_id,
        inference_id=inf_id,
        sample_id=sample_id,
        image_path=str(img_path),
        predicted_class="plastic",
        true_label="glass",  # Operator corrected plastic -> glass
        reviewer="field_tester",
        notes="Corrected transparent bottle to glass"
    )

    # Verify review records
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM review_records WHERE id = ?", (rev_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row["sample_id"] == sample_id
        assert row["predicted_class"] == "plastic"
        assert row["true_label"] == "glass"
        assert row["is_corrected"] == 1
        assert row["reviewer"] == "field_tester"

        cursor.execute("SELECT status FROM selection_queue WHERE id = ?", (q_id,))
        assert cursor.fetchone()["status"] == "reviewed"


def test_group_train_val_split_small_dataset():
    trainer = CandidateTrainer(device="cpu")
    df = pd.DataFrame([
        {"sample_id": "img1", "label": "cardboard", "image_path": "p1"},
        {"sample_id": "img2", "label": "glass", "image_path": "p2"},
        {"sample_id": "img3", "label": "paper", "image_path": "p3"},
        {"sample_id": "img4", "label": "plastic", "image_path": "p4"}
    ])

    train_df, val_df = trainer.group_train_val_split(df, val_fraction=0.20)
    assert len(train_df) >= 1
    assert len(val_df) >= 1
    assert len(train_df) + len(val_df) == 4

    # Assert zero overlap between train and val sample IDs
    train_ids = set(train_df["sample_id"])
    val_ids = set(val_df["sample_id"])
    assert train_ids.isdisjoint(val_ids)
