"""Unit tests for the reembed worker and admin /reembed-all endpoint."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def run(coro):
    return asyncio.run(coro)


# ── Helpers ──────────────────────────────────────────────────────────────────


class _AsyncCtx:
    def __init__(self, obj):
        self._obj = obj

    async def __aenter__(self):
        return self._obj

    async def __aexit__(self, *_):
        pass


def _make_request(pool=None):
    req = MagicMock()
    req.app = SimpleNamespace(state=SimpleNamespace(db_pool=pool))
    return req


# ── _load_batch_images: unreadable file → WARNING + continue ─────────────────


def test_load_batch_images_skips_unreadable_with_warning(caplog, tmp_path):
    from selfsuvis.worker.handlers.reembed import _load_batch_images

    good_path = tmp_path / "good.jpg"
    from PIL import Image

    Image.new("RGB", (4, 4)).save(str(good_path))

    batch = [
        {"id": "f1", "frame_path": str(good_path)},
        {"id": "f2", "frame_path": "/nonexistent/missing.jpg"},
    ]

    logger = logging.getLogger("test_reembed")
    with caplog.at_level(logging.WARNING, logger="test_reembed"):
        images, valid_rows = _load_batch_images(batch, logger)

    assert len(images) == 1
    assert len(valid_rows) == 1
    assert valid_rows[0]["id"] == "f1"
    assert "f2" in caplog.text
    assert "skipping" in caplog.text.lower()


def test_load_batch_images_all_unreadable(caplog, tmp_path):
    from selfsuvis.worker.handlers.reembed import _load_batch_images

    batch = [
        {"id": "x1", "frame_path": "/missing/a.jpg"},
        {"id": "x2", "frame_path": "/missing/b.jpg"},
    ]

    logger = logging.getLogger("test_reembed2")
    with caplog.at_level(logging.WARNING, logger="test_reembed2"):
        images, valid_rows = _load_batch_images(batch, logger)

    assert images == []
    assert valid_rows == []
    assert caplog.text.count("skipping") >= 2


# ── _load_reembed_cursor: restores cursor on restart ─────────────────────────


@pytest.mark.asyncio
async def test_load_reembed_cursor_restores_from_progress():
    from selfsuvis.worker.handlers.reembed import _load_reembed_cursor

    stored_progress = json.dumps(
        {"last_cursor": [1234567890.0, "abc123"], "frames_reembedded": 512}
    )
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"progress_json": stored_progress})

    cursor, frames_done = await _load_reembed_cursor(conn, "job-xyz")

    assert cursor == (1234567890.0, "abc123")
    assert frames_done == 512


@pytest.mark.asyncio
async def test_load_reembed_cursor_returns_none_when_no_progress():
    from selfsuvis.worker.handlers.reembed import _load_reembed_cursor

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"progress_json": None})

    cursor, frames_done = await _load_reembed_cursor(conn, "job-new")

    assert cursor is None
    assert frames_done == 0


@pytest.mark.asyncio
async def test_load_reembed_cursor_empty_job_row():
    from selfsuvis.worker.handlers.reembed import _load_reembed_cursor

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)

    cursor, frames_done = await _load_reembed_cursor(conn, "job-missing")

    assert cursor is None
    assert frames_done == 0


# ── _run_reembed: progress checkpoint written per batch ──────────────────────


@pytest.mark.asyncio
async def test_run_reembed_checkpoints_after_each_batch(tmp_path):
    """After each batch, update_job is called with updated cursor and frame count."""
    import datetime

    from selfsuvis.worker.handlers.reembed import _run_reembed

    ts = datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC)
    frame_rows = [
        {
            "id": f"f{i}",
            "mission_id": "m1",
            "frame_path": str(tmp_path / f"f{i}.jpg"),
            "qdrant_id": f"q{i}",
            "created_at": ts,
        }
        for i in range(3)
    ]

    from PIL import Image

    for row in frame_rows:
        Image.new("RGB", (4, 4)).save(row["frame_path"])

    conn = AsyncMock()
    # _load_reembed_cursor returns (None, 0)
    conn.fetchrow = AsyncMock(return_value={"progress_json": None})
    # list_frames_after: first call returns 3 rows, second call returns empty
    conn.execute = AsyncMock()

    update_job_calls = []

    async def _fake_update_job(c, jid, **kw):
        update_job_calls.append(kw)

    import numpy as np

    dino = MagicMock()
    dino.encode_images = MagicMock(return_value=np.zeros((3, 4), dtype=np.float32))

    clip = MagicMock()
    clip.encode_images = MagicMock(return_value=np.zeros((3, 4), dtype=np.float32))

    qdrant = MagicMock()
    qdrant.upsert_points = MagicMock()

    with (
        patch(
            "selfsuvis.worker.handlers.reembed.list_frames_after", new_callable=AsyncMock
        ) as mock_lfa,
        patch("selfsuvis.worker.handlers.reembed.update_job", side_effect=_fake_update_job),
        patch(
            "selfsuvis.worker.handlers.reembed._load_reembed_cursor",
            new_callable=AsyncMock,
            return_value=(None, 0),
        ),
    ):
        mock_lfa.side_effect = [frame_rows, []]  # one batch then done

        total = await _run_reembed(
            conn, "job-1", dino, clip, qdrant, 256, logging.getLogger("test")
        )

    assert total == 3
    # Checkpoint call after the batch + final finished call
    assert any(c.get("progress", {}).get("frames_reembedded") == 3 for c in update_job_calls)
    assert any(c.get("status") == "finished" for c in update_job_calls)


# ── POST /admin/reembed-all: 409 when job already active ────────────────────


@pytest.mark.asyncio
async def test_reembed_all_returns_409_when_job_active():
    from fastapi import HTTPException

    from selfsuvis.app.routers.admin import reembed_all

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"id": "running-job-id"})

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncCtx(conn))

    request = _make_request(pool)

    with pytest.raises(HTTPException) as exc_info:
        await reembed_all(request)

    assert exc_info.value.status_code == 409
    assert "running-job-id" in exc_info.value.detail


@pytest.mark.asyncio
async def test_reembed_all_enqueues_when_no_active_job():
    from selfsuvis.app.routers.admin import reembed_all

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)  # no active job
    conn.execute = AsyncMock()

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncCtx(conn))

    request = _make_request(pool)

    with patch("selfsuvis.pipeline.storage.jobs.create_job", new_callable=AsyncMock) as mock_create:
        result = await reembed_all(request)

    assert "job_id" in result
    mock_create.assert_awaited_once()
