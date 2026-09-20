"""Unit tests for video /health endpoint."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(pool=None):
    from selfsuvis.app.routers.health import router

    app = FastAPI()
    app.include_router(router)
    app.state.db_pool = pool
    return app


def test_health_postgres_ok():
    pool = AsyncMock()
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=1)

    @asynccontextmanager
    async def _acq():
        yield conn

    pool.acquire = _acq
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock()
    mock_redis.aclose = AsyncMock()

    with (
        patch("selfsuvis.app.routers.health.settings") as ms,
        patch("redis.asyncio.from_url", return_value=mock_redis),
        patch("selfsuvis.app.state.app_state") as mock_state,
    ):
        ms.HEALTH_REDIS_URL = "redis://localhost"
        mock_state.store.client.get_collections = MagicMock()
        client = TestClient(_make_app(pool))
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["postgres"] == "ok"
        assert resp.json()["redis"] == "ok"


def test_health_redis_unreachable():
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(side_effect=ConnectionError("refused"))

    with (
        patch("selfsuvis.app.routers.health.settings") as ms,
        patch("redis.asyncio.from_url", return_value=mock_redis),
        patch("selfsuvis.app.state.app_state") as mock_state,
    ):
        ms.HEALTH_REDIS_URL = "redis://localhost"
        mock_state.store.client.get_collections = MagicMock()
        client = TestClient(_make_app())
        resp = client.get("/health")
        assert resp.json()["redis"] == "error"
        assert resp.json()["status"] == "down"
        assert resp.status_code == 503
