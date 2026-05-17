from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.config import MODELS_DIR, SPLIT_META_PATH
from src.modeling.final_model import FINAL_MODEL_NAME

MODEL_PATH = MODELS_DIR / FINAL_MODEL_NAME
SVD_PATH = MODELS_DIR / "text_tfidf_svd_artifacts.joblib"


def _artifacts_ready() -> bool:
    return MODEL_PATH.exists() and SVD_PATH.exists() and SPLIT_META_PATH.exists()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


@pytest.mark.skipif(not _artifacts_ready(), reason="Нужны артефакты модели")
def test_predict_endpoint(client: TestClient) -> None:
    r = client.post(
        "/predict",
        json={
            "title": "Scientists Discover Surprising Link Between Sleep and Memory",
            "channel": "tech",
            "weekday": "wednesday",
            "threshold": 0.5,
        },
    )
    if r.status_code == 503:
        pytest.skip("Модель не загружена в TestClient lifespan")
    assert r.status_code == 200
    data = r.json()
    assert 0.0 < data["probability"] < 1.0
    assert "is_popular" in data


def test_predict_empty_title_422(client: TestClient) -> None:
    r = client.post(
        "/predict",
        json={"title": "", "channel": "tech", "weekday": "monday"},
    )
    assert r.status_code == 422


def test_predict_bad_channel(client: TestClient) -> None:
    r = client.post(
        "/predict",
        json={
            "title": "Some headline",
            "channel": "not_a_channel",
            "weekday": "monday",
        },
    )
    assert r.status_code == 422


def test_predict_bad_weekday(client: TestClient) -> None:
    r = client.post(
        "/predict",
        json={
            "title": "Some headline",
            "channel": "tech",
            "weekday": "not_a_day",
        },
    )
    assert r.status_code == 422


@pytest.mark.skipif(not _artifacts_ready(), reason="Нужны артефакты модели")
def test_version_endpoint(client: TestClient) -> None:
    r = client.get("/version")
    if r.status_code == 503:
        pytest.skip("Модель не загружена")
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "LightGBM"
    assert "popularity_threshold_shares" in data
    assert "git_sha" in data
    assert "version" in data


@pytest.mark.skipif(not _artifacts_ready(), reason="Нужны артефакты модели")
def test_predict_batch_endpoint(client: TestClient) -> None:
    r = client.post(
        "/predict_batch",
        json={
            "items": [
                {"title": "AI Revolution in Healthcare", "channel": "tech", "weekday": "monday"},
                {"title": "New Study Shows Benefits of Exercise", "channel": "lifestyle", "weekday": "friday"},
            ]
        },
    )
    if r.status_code == 503:
        pytest.skip("Модель не загружена")
    assert r.status_code == 200
    data = r.json()
    assert len(data["predictions"]) == 2
    assert all(0.0 <= p["probability"] <= 1.0 for p in data["predictions"])
    assert data["model"] == "LightGBM"


def test_predict_batch_empty_items_422(client: TestClient) -> None:
    r = client.post("/predict_batch", json={"items": []})
    assert r.status_code == 422


def test_predict_title_too_long_422(client: TestClient) -> None:
    r = client.post(
        "/predict",
        json={"title": "A" * 600, "channel": "tech", "weekday": "monday"},
    )
    assert r.status_code == 422
