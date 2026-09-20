"""Unit tests for CVAT webhook finetune trigger and label fetch-back logic."""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub app.state before the cvat router is imported to avoid live Qdrant connect.
# Always install a real asyncio.Lock for _finetune_lock: an earlier test file
# (test_automation_roi.py) may have already set app.state to a MagicMock whose
# .locked() returns a truthy MagicMock, causing _maybe_trigger_finetune to
# return early and never call create_job.
if "selfsuvis.app.state" not in sys.modules:
    sys.modules["selfsuvis.app.state"] = types.SimpleNamespace()
sys.modules["selfsuvis.app.state"]._finetune_lock = asyncio.Lock()


def run(coro):
    return asyncio.run(coro)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_pool(fetchval_returns=None, fetchrow_returns=None, execute_side_effect=None):
    """Build a minimal asyncpg pool mock for _maybe_trigger_finetune."""
    conn = AsyncMock()
    conn.fetchval = AsyncMock(side_effect=fetchval_returns or [0])
    conn.fetchrow = AsyncMock(return_value=fetchrow_returns)
    conn.execute = AsyncMock(side_effect=execute_side_effect)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncCtx(conn))
    return pool, conn


class _AsyncCtx:
    def __init__(self, obj):
        self._obj = obj

    async def __aenter__(self):
        return self._obj

    async def __aexit__(self, *_):
        pass


# ── _maybe_trigger_finetune tests ────────────────────────────────────────────


@patch("selfsuvis.pipeline.core.config.settings")
def test_finetune_not_triggered_when_disabled(mock_settings):
    from selfsuvis.app.routers.cvat import _maybe_trigger_finetune

    mock_settings.SUP_AUTO_TRIGGER = False
    pool, _ = _make_pool()
    run(_maybe_trigger_finetune(pool))
    pool.acquire.assert_not_called()


@patch("selfsuvis.pipeline.core.config.settings")
def test_finetune_not_triggered_below_threshold(mock_settings):
    from selfsuvis.app.routers.cvat import _maybe_trigger_finetune

    mock_settings.SUP_AUTO_TRIGGER = True
    mock_settings.MIN_ANNOTATED_FRAMES = 100
    mock_settings.MIN_NEW_ANNOTATED_SINCE_RETRAIN = 10

    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=5)  # below MIN_ANNOTATED_FRAMES

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncCtx(conn))

    run(_maybe_trigger_finetune(pool))

    # No create_job call — returned early at threshold check
    conn.fetchrow.assert_not_awaited()


@patch("selfsuvis.pipeline.storage.jobs.create_job", new_callable=AsyncMock)
@patch("selfsuvis.pipeline.core.config.settings")
def test_finetune_triggered_when_threshold_met(mock_settings, mock_create_job):
    from selfsuvis.app.routers.cvat import _maybe_trigger_finetune

    mock_settings.SUP_AUTO_TRIGGER = True
    mock_settings.MIN_ANNOTATED_FRAMES = 10
    mock_settings.MIN_NEW_ANNOTATED_SINCE_RETRAIN = 5

    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=20)
    conn.fetchrow = AsyncMock(
        side_effect=[
            {"value": "10"},  # last_retrain_watermark=10 → delta=10 >= 5
            None,  # no existing finetune job
        ]
    )
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncCtx(conn))

    run(_maybe_trigger_finetune(pool))

    mock_create_job.assert_awaited_once()
    call_kwargs = mock_create_job.call_args
    assert call_kwargs.kwargs.get("job_type") == "supervised_finetune"


@patch("selfsuvis.pipeline.storage.jobs.create_job", new_callable=AsyncMock)
@patch("selfsuvis.pipeline.core.config.settings")
def test_finetune_not_triggered_when_job_already_active(mock_settings, mock_create_job):
    from selfsuvis.app.routers.cvat import _maybe_trigger_finetune

    mock_settings.SUP_AUTO_TRIGGER = True
    mock_settings.MIN_ANNOTATED_FRAMES = 10
    mock_settings.MIN_NEW_ANNOTATED_SINCE_RETRAIN = 5

    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=20)
    conn.fetchrow = AsyncMock(
        side_effect=[
            {"value": "10"},
            {"id": "existing-job-id"},  # existing job found → skip
        ]
    )
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncCtx(conn))

    run(_maybe_trigger_finetune(pool))

    mock_create_job.assert_not_awaited()


@patch("selfsuvis.pipeline.storage.jobs.create_job", new_callable=AsyncMock)
@patch("selfsuvis.pipeline.core.config.settings")
def test_finetune_lock_prevents_duplicate_enqueue(mock_settings, mock_create_job):
    """When _finetune_lock is already held, _maybe_trigger_finetune returns early."""
    from selfsuvis.app.routers.cvat import _maybe_trigger_finetune

    mock_settings.SUP_AUTO_TRIGGER = True
    mock_settings.MIN_ANNOTATED_FRAMES = 1
    mock_settings.MIN_NEW_ANNOTATED_SINCE_RETRAIN = 1

    pool = MagicMock()
    finetune_lock = sys.modules["selfsuvis.app.state"]._finetune_lock

    async def _run_with_lock():
        async with finetune_lock:
            await _maybe_trigger_finetune(pool)

    run(_run_with_lock())
    mock_create_job.assert_not_awaited()
    pool.acquire.assert_not_called()


# ── _fetch_cvat_labels tests ─────────────────────────────────────────────────


@patch("selfsuvis.app.routers.cvat.settings")
def test_fetch_labels_no_token_returns_empty_with_warning(mock_settings, caplog):
    from selfsuvis.app.routers.cvat import _fetch_cvat_labels

    mock_settings.CVAT_API_TOKEN = ""

    import logging

    with caplog.at_level(logging.WARNING, logger="selfsuvis.app.routers.cvat"):
        result = run(_fetch_cvat_labels(42))

    assert result == {}
    assert "CVAT_API_TOKEN" in caplog.text


@patch("selfsuvis.app.routers.cvat.settings")
def test_fetch_labels_http_401_returns_empty_with_warning(mock_settings, caplog):
    from selfsuvis.app.routers.cvat import _fetch_cvat_labels

    mock_settings.CVAT_API_TOKEN = "mytoken"
    mock_settings.CVAT_URL = "http://cvat.local"

    import logging

    mock_response = MagicMock()
    mock_response.status_code = 401

    with patch("selfsuvis.app.routers.cvat.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        with caplog.at_level(logging.WARNING, logger="selfsuvis.app.routers.cvat"):
            result = run(_fetch_cvat_labels(99))

    assert result == {}
    assert "401" in caplog.text


@patch("selfsuvis.app.routers.cvat.settings")
def test_fetch_labels_http_200_stores_labels(mock_settings):
    from selfsuvis.app.routers.cvat import _fetch_cvat_labels

    mock_settings.CVAT_API_TOKEN = "mytoken"
    mock_settings.CVAT_URL = "http://cvat.local"

    xml_content = b"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><labels>
    <label><name>car</name></label>
    <label><name>person</name></label>
  </labels></task></meta>
  <image id="0" name="data/frame_0001.jpg" width="1920" height="1080">
    <box label="car" xtl="0" ytl="0" xbr="10" ybr="10"/>
  </image>
  <image id="1" name="data/frame_0002.jpg" width="1920" height="1080">
    <box label="person" xtl="0" ytl="0" xbr="10" ybr="10"/>
  </image>
</annotations>"""

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = xml_content

    with patch("selfsuvis.app.routers.cvat.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = run(_fetch_cvat_labels(7))

    assert result.get("frame_0001.jpg") == "car"
    assert result.get("frame_0002.jpg") == "person"


# ── _mark_frames_annotated label-storage tests ───────────────────────────────


@pytest.mark.asyncio
async def test_mark_frames_stores_cvat_labels():
    from selfsuvis.app.routers.cvat import _mark_frames_annotated

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="UPDATE 2")
    conn.fetch = AsyncMock(
        return_value=[
            {"id": "f1", "frame_path": "/data/frames/frame_0001.jpg"},
            {"id": "f2", "frame_path": "/data/frames/frame_0002.jpg"},
        ]
    )
    conn.executemany = AsyncMock()

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncCtx(conn))

    labels = {"frame_0001.jpg": "car", "frame_0002.jpg": "person"}
    count = await _mark_frames_annotated(["f1", "f2"], pool, labels)

    assert count == 2
    conn.executemany.assert_awaited_once()
    updates_arg = conn.executemany.call_args[0][1]
    assert ("car", "f1") in updates_arg
    assert ("person", "f2") in updates_arg


@pytest.mark.asyncio
async def test_mark_frames_no_labels_skips_executemany():
    from selfsuvis.app.routers.cvat import _mark_frames_annotated

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="UPDATE 1")
    conn.fetch = AsyncMock(return_value=[{"id": "f1", "frame_path": "/data/frame_0001.jpg"}])
    conn.executemany = AsyncMock()

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncCtx(conn))

    count = await _mark_frames_annotated(["f1"], pool, {})

    assert count == 1
    conn.executemany.assert_not_awaited()
