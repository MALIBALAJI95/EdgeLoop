"""Integration tests for EdgeLoop FastAPI service."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from edgeloop.api.app import app
from edgeloop.common.database import Database


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Setup temporary database for API tests
    test_db_dir = tmp_path_factory.mktemp("api_db")
    test_db = Database(test_db_dir / "api_test.db")

    import edgeloop.common.database as db_mod
    db_mod._db_instance = test_db

    test_db.log_inference(
        sample_id="history-seed",
        image_path="tests/fixtures/history-seed.jpg",
        predicted_class="glass",
        confidence=0.99,
        entropy=0.01,
        latency_ms=10.0,
        model_version="test_model"
    )

    # Create dummy champion file if not exists
    champ_p = Path("artifacts/champion/mobilenetv2_champion.pth")
    champ_p.parent.mkdir(parents=True, exist_ok=True)
    if not champ_p.exists():
        base_p = Path("research/edgeloop/mobilenetv2_baseline.pth")
        if base_p.exists():
            import shutil
            shutil.copy2(str(base_p), str(champ_p))

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"


def test_readiness_endpoint(client):
    res = client.get("/readiness")
    assert res.status_code == 200
    assert res.json()["ready"] is True


def test_infer_image_path(client):
    img_p = Path("research/edgeloop/corrupted_edge/blur/glass_glass1.jpg")
    if not img_p.exists():
        pytest.skip("Test image not found")

    res = client.post(
        "/infer/path",
        json={"image_path": str(img_p.resolve()), "condition": "blur"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] == "glass"
    assert data["confidence"] > 0.5
    assert data["entropy"] >= 0.0


def test_history_endpoint(client):
    res = client.get("/history?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_drift_status_endpoint(client):
    res = client.get("/drift/status")
    assert res.status_code == 200
    data = res.json()
    assert "total_monitored_records" in data
    assert "reference_version" in data


def test_registry_endpoints(client):
    res = client.get("/registry/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) >= 1
    assert any(m["stage"] == "champion" for m in models)

    res_champ = client.get("/registry/champion")
    assert res_champ.status_code == 200
    assert res_champ.json()["stage"] == "champion"


def test_api_promotion_rejection_governance(client):
    # Attempting to promote uncertainty candidate must be rejected with 400
    res = client.post(
        "/registry/promote",
        headers={"X-EdgeLoop-Token": "edgeloop-dev-secret-token"},
        json={
            "candidate_version": "uncertainty_candidate",
            "approver": "Test Approver",
            "promotion_reason": "Testing API rejection"
        }
    )
    assert res.status_code == 400
    assert "Research Governance Rule" in res.json()["detail"]
