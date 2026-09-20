"""Unit tests for GET /admin/caption-eval and caption_null_rate in automation-roi."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def admin_client():
    """TestClient for the admin router with auth bypassed."""
    from fastapi import FastAPI

    from selfsuvis.app.deps import rate_limit, require_api_key
    from selfsuvis.app.routers.admin import router

    app = FastAPI()

    async def _no_auth():
        return "test-key"

    async def _no_rate():
        pass

    app.dependency_overrides[require_api_key] = _no_auth
    app.dependency_overrides[rate_limit] = _no_rate
    app.include_router(router)
    return TestClient(app)


def _make_pool(agg_row=None, model_rows=None):
    """Build a mock asyncpg pool for caption-eval queries."""
    if agg_row is None:
        agg_row = {
            "total_frames": 100,
            "captioned_frames": 90,
            "skipped_frames": 5,
            "null_caption_frames": 5,
            "mean_confidence": 0.82,
            "p95_confidence": 0.95,
        }
    if model_rows is None:
        model_rows = [
            {"caption_model": "gemma-api:gemma4:e4b", "cnt": 70},
            {"caption_model": "florence-2-large:v1:fp16", "cnt": 20},
        ]

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=agg_row)
    conn.fetch = AsyncMock(return_value=model_rows)

    pool = MagicMock()
    pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    return pool, conn


# ── _compute_caption_null_rate ────────────────────────────────────────────────


@pytest.mark.anyio
async def test_compute_caption_null_rate_typical():
    """5 null out of 95 captionable → rate ≈ 0.0526."""
    from selfsuvis.app.routers.admin import _compute_caption_null_rate

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        return_value={
            "null_count": 5,
            "captionable_total": 95,
        }
    )
    rate = await _compute_caption_null_rate(conn)
    assert rate == pytest.approx(5 / 95, rel=1e-5)


@pytest.mark.anyio
async def test_compute_caption_null_rate_zero_captionable():
    """No captionable frames → returns None."""
    from selfsuvis.app.routers.admin import _compute_caption_null_rate

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"null_count": 0, "captionable_total": 0})
    rate = await _compute_caption_null_rate(conn)
    assert rate is None


@pytest.mark.anyio
async def test_compute_caption_null_rate_all_captioned():
    """0 null captions → rate = 0.0."""
    from selfsuvis.app.routers.admin import _compute_caption_null_rate

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"null_count": 0, "captionable_total": 50})
    rate = await _compute_caption_null_rate(conn)
    assert rate == pytest.approx(0.0)


@pytest.mark.anyio
async def test_compute_caption_null_rate_all_null():
    """All captionable frames are null → rate = 1.0."""
    from selfsuvis.app.routers.admin import _compute_caption_null_rate

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"null_count": 20, "captionable_total": 20})
    rate = await _compute_caption_null_rate(conn)
    assert rate == pytest.approx(1.0)


# ── GET /admin/caption-eval ───────────────────────────────────────────────────


def test_caption_eval_returns_expected_fields(admin_client):
    pool, conn = _make_pool()
    with patch("selfsuvis.app.routers.admin.get_db_pool", return_value=pool):
        resp = admin_client.get("/admin/caption-eval")

    assert resp.status_code == 200
    data = resp.json()
    assert "caption_null_rate" in data
    assert "mean_confidence" in data
    assert "p95_confidence" in data
    assert "total_frames" in data
    assert "captioned_frames" in data
    assert "skipped_frames" in data
    assert "null_caption_frames" in data
    assert "model_breakdown" in data


def test_caption_eval_null_rate_calculation(admin_client):
    """caption_null_rate = null_caption_frames / (total - skipped)."""
    agg = {
        "total_frames": 100,
        "captioned_frames": 90,
        "skipped_frames": 5,
        "null_caption_frames": 5,
        "mean_confidence": 0.80,
        "p95_confidence": 0.92,
    }
    pool, _ = _make_pool(agg_row=agg)
    with patch("selfsuvis.app.routers.admin.get_db_pool", return_value=pool):
        resp = admin_client.get("/admin/caption-eval")

    data = resp.json()
    # captionable = 100 - 5 = 95; null = 5; rate = 5/95 ≈ 0.0526
    assert data["caption_null_rate"] == pytest.approx(5 / 95, rel=1e-3)


def test_caption_eval_model_breakdown(admin_client):
    pool, _ = _make_pool(
        model_rows=[
            {"caption_model": "gemma-api:gemma4:e4b", "cnt": 80},
            {"caption_model": "florence-2-large:v1:fp16", "cnt": 10},
        ]
    )
    with patch("selfsuvis.app.routers.admin.get_db_pool", return_value=pool):
        resp = admin_client.get("/admin/caption-eval")

    breakdown = resp.json()["model_breakdown"]
    assert breakdown.get("gemma-api:gemma4:e4b") == 80
    assert breakdown.get("florence-2-large:v1:fp16") == 10


def test_caption_eval_db_error_returns_error_key(admin_client):
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=Exception("db down"))
    with patch("selfsuvis.app.routers.admin.get_db_pool", return_value=pool):
        resp = admin_client.get("/admin/caption-eval")

    assert resp.status_code == 200
    assert "error" in resp.json()


def test_caption_eval_confidence_rounded(admin_client):
    """mean/p95 confidence is rounded to 4 decimal places."""
    agg = {
        "total_frames": 50,
        "captioned_frames": 50,
        "skipped_frames": 0,
        "null_caption_frames": 0,
        "mean_confidence": 0.8123456,
        "p95_confidence": 0.9876543,
    }
    pool, _ = _make_pool(agg_row=agg)
    with patch("selfsuvis.app.routers.admin.get_db_pool", return_value=pool):
        resp = admin_client.get("/admin/caption-eval")

    data = resp.json()
    # Should be rounded to 4 decimal places
    assert str(data["mean_confidence"]).find(".") != -1
    assert len(str(data["mean_confidence"]).split(".")[1]) <= 4


# ── automation-roi includes caption_null_rate ─────────────────────────────────


def test_automation_roi_includes_caption_null_rate(admin_client):
    """GET /admin/automation-roi response includes caption_null_rate field."""
    annotated_row = {"first_at": None, "last_at": None}

    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=0)
    conn.fetchrow = AsyncMock(
        side_effect=[
            annotated_row,  # timeline
            {"null_count": 3, "captionable_total": 100},  # caption null rate
        ]
    )
    conn.fetch = AsyncMock(
        side_effect=[
            [],  # al_tag counts
            [],  # finetune job rows
        ]
    )

    conn.close = AsyncMock()

    with patch("selfsuvis.app.routers.admin.asyncpg") as mock_asyncpg:
        mock_asyncpg.connect = AsyncMock(return_value=conn)
        with patch("selfsuvis.app.routers.admin.settings") as mock_settings:
            mock_settings.DATABASE_URL = "postgresql://fake/db"
            resp = admin_client.get("/admin/automation-roi")

    assert resp.status_code == 200
    data = resp.json()
    assert "caption_null_rate" in data
