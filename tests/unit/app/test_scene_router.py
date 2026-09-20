"""Unit tests for app/routers/scene.py — POST /query/scene.

Uses the FastAPI test client with mocked DB pool and Qdrant.
No live PostgreSQL or GPU required.
"""

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Stub out app.state before any router import to avoid torchvision circular import
_state_stub = MagicMock()
_state_stub.clip_model = MagicMock()
_state_stub.qdrant_store = MagicMock()
_state_stub.dino_model = None
sys.modules.setdefault("selfsuvis.app.state", _state_stub)


# ── App / route fixture ───────────────────────────────────────────────────────


@pytest.fixture
def client():
    """Return a TestClient for the FastAPI app with auth bypassed."""
    from fastapi import FastAPI

    from selfsuvis.app.deps import rate_limit, require_api_key
    from selfsuvis.app.routers.scene import router

    # Minimal app — just the scene router with auth/rate-limit overridden
    app = FastAPI()

    async def _no_auth():
        return "test-key"

    async def _no_rate():
        pass

    app.dependency_overrides[require_api_key] = _no_auth
    app.dependency_overrides[rate_limit] = _no_rate
    app.include_router(router)
    return TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_db_row(**kwargs) -> dict[str, Any]:
    """Build a fake asyncpg row-like dict."""
    defaults = {
        "frame_id": "frame_001",
        "mission_id": "mission_abc",
        "frame_path": "/data/frames/frame_001.jpg",
        "t_sec": 42.0,
        "caption": "Two trucks on a wet road.",
        "frame_facts_json": {
            "vehicle_groups": [{"type": "truck", "count": 2}],
            "road_surface": "asphalt",
            "road_condition": "wet",
            "scene_summary": "Two trucks on a wet road.",
        },
        "gps_json": {"lat": 47.123, "lon": 8.456},
        "qdrant_id": None,
    }
    defaults.update(kwargs)
    return defaults


async def _fake_fetch(sql, *params):
    """Return one fake row for any SQL query."""
    return [_make_db_row()]


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_query_scene_no_filters(client):
    """POST /query/scene with no filters returns results."""
    fake_pool = MagicMock()
    fake_pool.fetch = AsyncMock(return_value=[_make_db_row()])

    with patch("selfsuvis.app.routers.scene.get_db_pool", return_value=fake_pool):
        resp = client.post("/query/scene", json={"top_k": 5})

    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "total_matched" in data
    assert "filters_applied" in data


def test_query_scene_with_road_condition_filter(client):
    """road_condition filter is applied and reflected in filters_applied."""
    fake_pool = MagicMock()
    fake_pool.fetch = AsyncMock(return_value=[_make_db_row()])

    with patch("selfsuvis.app.routers.scene.get_db_pool", return_value=fake_pool):
        resp = client.post("/query/scene", json={"road_condition": "wet"})

    assert resp.status_code == 200
    data = resp.json()
    assert "road_condition" in data["filters_applied"]


def test_query_scene_with_gps_bbox(client):
    """gps_bbox filter is reflected in filters_applied."""
    fake_pool = MagicMock()
    fake_pool.fetch = AsyncMock(return_value=[_make_db_row()])

    payload = {
        "gps_bbox": {
            "min_lat": 47.0,
            "max_lat": 47.5,
            "min_lon": 8.0,
            "max_lon": 8.5,
        }
    }
    with patch("selfsuvis.app.routers.scene.get_db_pool", return_value=fake_pool):
        resp = client.post("/query/scene", json=payload)

    assert resp.status_code == 200
    assert "gps_bbox" in resp.json()["filters_applied"]


def test_query_scene_with_time_range(client):
    """time_range filter is reflected in filters_applied."""
    fake_pool = MagicMock()
    fake_pool.fetch = AsyncMock(return_value=[_make_db_row()])

    with patch("selfsuvis.app.routers.scene.get_db_pool", return_value=fake_pool):
        resp = client.post("/query/scene", json={"time_range": {"start_sec": 0, "end_sec": 120}})

    assert resp.status_code == 200
    assert "time_range" in resp.json()["filters_applied"]


def test_query_scene_vehicle_count_filters(client):
    fake_pool = MagicMock()
    fake_pool.fetch = AsyncMock(return_value=[_make_db_row()])

    with patch("selfsuvis.app.routers.scene.get_db_pool", return_value=fake_pool):
        resp = client.post("/query/scene", json={"vehicle_count_min": 1, "vehicle_count_max": 5})

    assert resp.status_code == 200
    data = resp.json()
    assert "vehicle_count_min" in data["filters_applied"]
    assert "vehicle_count_max" in data["filters_applied"]


def test_query_scene_result_fields(client):
    """Each result has the expected fields."""
    fake_pool = MagicMock()
    fake_pool.fetch = AsyncMock(return_value=[_make_db_row()])

    with patch("selfsuvis.app.routers.scene.get_db_pool", return_value=fake_pool):
        resp = client.post("/query/scene", json={})

    results = resp.json()["results"]
    assert len(results) == 1
    r = results[0]
    assert r["frame_id"] == "frame_001"
    assert r["mission_id"] == "mission_abc"
    assert r["road_condition"] == "wet"
    assert r["vehicle_count"] == 2
    assert r["lat"] == pytest.approx(47.123)
    assert r["lon"] == pytest.approx(8.456)
    assert r["caption"] == "Two trucks on a wet road."


def test_query_scene_empty_db_result(client):
    """No matching frames → empty results list."""
    fake_pool = MagicMock()
    fake_pool.fetch = AsyncMock(return_value=[])

    with patch("selfsuvis.app.routers.scene.get_db_pool", return_value=fake_pool):
        resp = client.post("/query/scene", json={})

    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []
    assert data["total_matched"] == 0


def test_query_scene_top_k_respected(client):
    """top_k limits the number of returned results."""
    # Return 5 rows from DB
    rows = [_make_db_row(frame_id=f"frame_{i:03d}") for i in range(5)]
    fake_pool = MagicMock()
    fake_pool.fetch = AsyncMock(return_value=rows)

    with patch("selfsuvis.app.routers.scene.get_db_pool", return_value=fake_pool):
        resp = client.post("/query/scene", json={"top_k": 2})

    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_query_scene_db_error_returns_503(client):
    """DB failure raises 503."""
    fake_pool = MagicMock()
    fake_pool.fetch = AsyncMock(side_effect=Exception("connection refused"))

    with patch("selfsuvis.app.routers.scene.get_db_pool", return_value=fake_pool):
        resp = client.post("/query/scene", json={})

    assert resp.status_code == 503


def test_query_scene_invalid_gps_bbox_rejected(client):
    """min_lat > max_lat is a validation error (422)."""
    payload = {"gps_bbox": {"min_lat": 48.0, "max_lat": 47.0, "min_lon": 8.0, "max_lon": 8.5}}
    fake_pool = MagicMock()
    with patch("selfsuvis.app.routers.scene.get_db_pool", return_value=fake_pool):
        resp = client.post("/query/scene", json=payload)
    assert resp.status_code == 422


def test_query_scene_invalid_time_range_rejected(client):
    """start_sec > end_sec is a validation error (422)."""
    payload = {"time_range": {"start_sec": 100, "end_sec": 50}}
    fake_pool = MagicMock()
    with patch("selfsuvis.app.routers.scene.get_db_pool", return_value=fake_pool):
        resp = client.post("/query/scene", json=payload)
    assert resp.status_code == 422


def test_query_scene_text_filter_listed_in_filters(client):
    """text query adds 'text_rerank' to filters_applied."""
    fake_pool = MagicMock()
    fake_pool.fetch = AsyncMock(return_value=[_make_db_row()])

    fake_clip = MagicMock()
    fake_clip.encode_texts.return_value = [[0.1] * 512]

    with (
        patch("selfsuvis.app.routers.scene.get_db_pool", return_value=fake_pool),
        patch("selfsuvis.app.routers.scene.clip_model", fake_clip),
    ):
        resp = client.post("/query/scene", json={"text": "military convoy"})

    assert resp.status_code == 200
    assert "text_rerank" in resp.json()["filters_applied"]
