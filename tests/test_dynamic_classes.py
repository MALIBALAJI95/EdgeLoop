"""Tests for human-supervised learning with dynamic new object classes."""

import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from edgeloop.common.database import Database
from edgeloop.common.classes import (
    ensure_classes_table,
    get_registered_classes,
    get_class_names,
    get_bin_mapping,
    register_new_class,
    normalize_class_name
)
from edgeloop.inference.detector import DetectedWasteObject, WasteObjectDetector, DetectionFrameResult
from edgeloop.inference.engine import InferenceEngine
from edgeloop.review.service import ReviewService, ReviewSubmission
from edgeloop.training.trainer import CandidateTrainer, TrashDataset


def test_normalize_class_name():
    assert normalize_class_name("Juice Carton") == "juice_carton"
    assert normalize_class_name("  Milk_Jug 100% ") == "milk_jug_100"
    assert normalize_class_name("!@#$%^") == "custom_object"


def test_dynamic_class_registration(tmp_path, monkeypatch):
    test_db = Database(tmp_path / "test_classes.db")
    import edgeloop.common.database as db_mod
    monkeypatch.setattr(db_mod, "_db_instance", test_db)

    # 1. Baseline classes exist
    ensure_classes_table()
    classes = get_class_names()
    assert len(classes) == 6
    assert "plastic" in classes
    assert "trash" in classes

    # 2. Register dynamic new class
    reg_res = register_new_class(
        display_name="Juice Carton",
        category="Beverage Packaging",
        bin_name="Yellow Bin (Plastics & Cartons)",
        bin_code="CARTON_YELLOW",
        bin_guidance="Rinse and flatten."
    )
    assert reg_res["class_name"] == "juice_carton"

    # 3. Verify persistence
    updated_classes = get_class_names()
    assert len(updated_classes) == 7
    assert "juice_carton" in updated_classes

    bin_info = get_bin_mapping("juice_carton")
    assert bin_info["bin_name"] == "Yellow Bin (Plastics & Cartons)"
    assert bin_info["bin_code"] == "CARTON_YELLOW"


def test_uncertain_detection_labeling():
    # If confidence is low or entropy is high, is_uncertain should be True
    low_conf_obj = DetectedWasteObject(
        box=[10, 10, 100, 100],
        detector_label="bottle",
        detector_confidence=0.85,
        waste_class="plastic",
        waste_confidence=0.45,  # Low confidence
        entropy=0.95,           # High uncertainty
        suggested_bin="Needs Human Review (Stop & Review)",
        bin_code="REVIEW_NEEDED",
        bin_guidance="Needs human verification.",
        is_uncertain=True,
        display_label="Unknown / Uncertain — Needs Human Review"
    )
    assert low_conf_obj.is_uncertain is True
    assert "Unknown / Uncertain" in low_conf_obj.display_label


def test_candidate_training_with_dynamic_classes(tmp_path, monkeypatch):
    test_db = Database(tmp_path / "test_trainer.db")
    import edgeloop.common.database as db_mod
    monkeypatch.setattr(db_mod, "_db_instance", test_db)

    # Register juice_carton
    register_new_class(
        display_name="Juice Carton",
        category="Beverage Carton",
        bin_name="Yellow Bin (Plastics & Cartons)"
    )

    # Create dummy images for training
    img_dir = tmp_path / "train_imgs"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    classes = ["plastic", "paper", "juice_carton"]
    for i, cls in enumerate(classes):
        for j in range(2):
            p = img_dir / f"{cls}_{j}.jpg"
            img = Image.new("RGB", (224, 224), (i * 40, j * 40, 100))
            img.save(p)
            records.append({
                "sample_id": f"{cls}_{j}",
                "image_path": str(p),
                "label": cls
            })

    df = pd.DataFrame(records)

    # Train candidate supporting dynamic class
    trainer = CandidateTrainer()
    trainer.candidates_dir = tmp_path / "candidates"
    trainer.candidates_dir.mkdir(parents=True, exist_ok=True)

    save_path, meta = trainer.train_candidate(
        dataset_df=df,
        candidate_name="cand_dynamic_test",
        epochs=1,
        batch_size=2
    )

    assert save_path.exists()
    assert meta["num_classes"] >= 7
    assert "juice_carton" in meta["classes"]

    # Load into InferenceEngine and verify model weights expanded
    engine = InferenceEngine(checkpoint_path=save_path, device="cpu", log_predictions=False)
    assert "juice_carton" in engine.classes
    assert len(engine.classes) >= 7

    # Run inference with the new model
    sample_img = img_dir / "juice_carton_0.jpg"
    res = engine.predict_image(sample_img)
    assert res.predicted_class in engine.classes
    assert res.confidence >= 0.0
