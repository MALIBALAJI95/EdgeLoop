"""Unit tests for EdgeLoop model registry, safe promotion gates, and rollback."""

import shutil
from pathlib import Path
import pytest

from edgeloop.common.database import Database
from edgeloop.registry.manager import ModelRegistry, PromotionRejectionError


@pytest.fixture
def registry_env(tmp_path, monkeypatch):
    test_db = Database(tmp_path / "test_registry.db")

    import edgeloop.common.database as db_mod
    monkeypatch.setattr(db_mod, "_db_instance", test_db)

    # Mock baseline checkpoint path
    dummy_ckpt = tmp_path / "dummy_baseline.pth"
    dummy_ckpt.write_bytes(b"DUMMY_BASELINE_WEIGHTS")

    reg = ModelRegistry()
    reg.champion_dir = tmp_path / "champion"
    reg.champion_dir.mkdir(parents=True, exist_ok=True)
    reg.champion_ckpt_path = reg.champion_dir / "mobilenetv2_champion.pth"
    reg.previous_champion_ckpt_path = reg.champion_dir / "mobilenetv2_previous_champion.pth"

    # Seed initial champion
    test_db.register_model(
        version="mobilenetv2_baseline",
        model_name="MobileNetV2",
        checkpoint_path=str(dummy_ckpt),
        stage="champion",
        accuracy=0.913043,
        macro_f1=0.894340,
        trash_f1=0.765957,
        approved_by="System",
        promotion_reason="Initial Baseline"
    )
    shutil.copy2(str(dummy_ckpt), str(reg.champion_ckpt_path))

    return reg, tmp_path, test_db


def test_initial_champion_presence(registry_env):
    reg, _, _ = registry_env
    champ = reg.get_champion()
    assert champ is not None
    assert champ.version == "mobilenetv2_baseline"
    assert champ.stage == "champion"
    assert pytest.approx(champ.macro_f1, rel=1e-3) == 0.894340


def test_reject_archived_retraining_candidates(registry_env):
    reg, _, _ = registry_env
    for bad_cand in ["uncertainty_candidate", "unique_aware_candidate"]:
        with pytest.raises(PromotionRejectionError, match="Research Governance Rule"):
            reg.promote_candidate(
                candidate_version=bad_cand,
                approver="Lead Engineer",
                promotion_reason="Testing governance block"
            )


def test_reject_unevaluated_candidate(registry_env):
    reg, tmp_path, test_db = registry_env
    cand_path = tmp_path / "candidate_uneval.pth"
    cand_path.write_bytes(b"CANDIDATE_WEIGHTS")

    reg.register_candidate("candidate_uneval", cand_path)

    with pytest.raises(PromotionRejectionError, match="Must be 'evaluated' or 'approved'"):
        reg.promote_candidate(
            candidate_version="candidate_uneval",
            approver="Lead Engineer",
            promotion_reason="Trying to promote before evaluation"
        )


def test_reject_inferior_macro_f1(registry_env):
    reg, tmp_path, test_db = registry_env
    cand_path = tmp_path / "inferior_model.pth"
    cand_path.write_bytes(b"INFERIOR_MODEL_WEIGHTS")

    test_db.register_model(
        version="inferior_model",
        model_name="MobileNetV2",
        checkpoint_path=str(cand_path),
        stage="evaluated",
        accuracy=0.890000,
        macro_f1=0.875000,  # Below champion's 0.894340
        trash_f1=0.720000
    )

    with pytest.raises(PromotionRejectionError, match="does not exceed Champion"):
        reg.promote_candidate(
            candidate_version="inferior_model",
            approver="Lead Engineer",
            promotion_reason="Testing gate failure"
        )


def test_reject_sensitive_trash_class_regression(registry_env):
    reg, tmp_path, test_db = registry_env
    cand_path = tmp_path / "regression_model.pth"
    cand_path.write_bytes(b"REGRESSION_MODEL_WEIGHTS")

    test_db.register_model(
        version="regression_model",
        model_name="MobileNetV2",
        checkpoint_path=str(cand_path),
        stage="evaluated",
        accuracy=0.920000,
        macro_f1=0.900000,  # Improved overall macro-F1
        trash_f1=0.700000   # Regressed significantly from 0.765957 (>0.02 drop)
    )

    with pytest.raises(PromotionRejectionError, match="dropped beyond tolerance"):
        reg.promote_candidate(
            candidate_version="regression_model",
            approver="Lead Engineer",
            promotion_reason="Testing trash regression check"
        )


def test_successful_promotion_and_rollback(registry_env):
    reg, tmp_path, test_db = registry_env
    cand_path = tmp_path / "better_model.pth"
    cand_path.write_bytes(b"BETTER_MODEL_WEIGHTS_V2")

    test_db.register_model(
        version="better_model",
        model_name="MobileNetV2",
        checkpoint_path=str(cand_path),
        stage="evaluated",
        accuracy=0.925000,
        macro_f1=0.910000,  # Higher than champion
        trash_f1=0.780000   # Higher than champion
    )

    # 1. Promote
    new_champ = reg.promote_candidate(
        candidate_version="better_model",
        approver="MLOps Lead",
        promotion_reason="Strictly improved holdout Macro-F1 with no class regression"
    )
    assert new_champ.version == "better_model"
    assert new_champ.stage == "champion"
    assert reg.champion_ckpt_path.read_bytes() == b"BETTER_MODEL_WEIGHTS_V2"
    assert reg.previous_champion_ckpt_path.read_bytes() == b"DUMMY_BASELINE_WEIGHTS"

    # 2. Rollback
    reverted_champ = reg.rollback_champion(
        approver="MLOps Lead",
        reason="Emergency incident rollback"
    )
    assert reverted_champ.version == "mobilenetv2_baseline"
    assert reverted_champ.stage == "champion"
    assert reg.champion_ckpt_path.read_bytes() == b"DUMMY_BASELINE_WEIGHTS"
