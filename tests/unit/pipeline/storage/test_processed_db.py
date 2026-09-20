"""Unit tests for pipeline.storage.processed."""

import asyncio
import importlib
import sys
import types
from datetime import datetime, timezone
from unittest.mock import AsyncMock

if "asyncpg" not in sys.modules:
    _asyncpg = types.ModuleType("asyncpg")
    _asyncpg.connect = AsyncMock()
    _asyncpg.Pool = AsyncMock()  # type: ignore[attr-defined]
    sys.modules["asyncpg"] = _asyncpg

mod = sys.modules.get("selfsuvis.pipeline.storage.processed")
if mod is not None and not hasattr(mod, "aget_by_hash"):
    del sys.modules["selfsuvis.pipeline.storage.processed"]

processed_db = importlib.import_module("selfsuvis.pipeline.storage.processed")
aget_by_hash = processed_db.aget_by_hash
aget_by_size = processed_db.aget_by_size
aget_by_url = processed_db.aget_by_url
aupsert = processed_db.aupsert


class _Row(dict):
    pass


async def _test_get_by_hash_happy_path():
    row = _Row(
        file_hash="abc123",
        video_id="vid1",
        path="/tmp/v.mp4",
        size_bytes=1000,
        mtime=1.5,
        status="done",
        meta_json={"source": "upload"},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=row)

    rec = await aget_by_hash("abc123", conn=conn)
    assert rec is not None
    assert rec["file_hash"] == "abc123"
    assert rec["video_id"] == "vid1"
    assert rec["meta"] == {"source": "upload"}


async def _test_get_by_hash_miss_returns_none():
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    assert await aget_by_hash("missing", conn=conn) is None


async def _test_get_by_url_happy_path():
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        return_value=_Row(
            file_hash="url1",
            video_id="v1",
            path="/tmp/v.mp4",
            size_bytes=100,
            mtime=1.0,
            status="done",
            meta_json={"url": "http://example.com/vid.mp4"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )
    rec = await aget_by_url("http://example.com/vid.mp4", conn=conn)
    assert rec is not None
    assert rec["file_hash"] == "url1"


async def _test_get_by_size_multiple_returns_most_recent():
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        return_value=_Row(
            file_hash="new",
            video_id="v-new",
            path="/new.mp4",
            size_bytes=512,
            mtime=2.0,
            status="done",
            meta_json={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )
    rec = await aget_by_size(512, conn=conn)
    assert rec is not None
    assert rec["file_hash"] == "new"


async def _test_upsert_executes_jsonb_upsert():
    conn = AsyncMock()
    await aupsert("dup1", "vid1", "/tmp/v.mp4", 123, 1.0, "done", {"url": "http://x"}, conn=conn)
    conn.execute.assert_awaited_once()
    args = conn.execute.await_args.args
    assert args[1] == "dup1"
    assert args[2] == "vid1"
    assert args[6] == "done"
    assert '"url": "http://x"' in args[7]


def run(coro):
    return asyncio.run(coro)


def test_get_by_hash_happy_path_sync():
    run(_test_get_by_hash_happy_path())


def test_get_by_hash_miss_returns_none_sync():
    run(_test_get_by_hash_miss_returns_none())


def test_get_by_url_happy_path_sync():
    run(_test_get_by_url_happy_path())


def test_get_by_size_multiple_returns_most_recent_sync():
    run(_test_get_by_size_multiple_returns_most_recent())


def test_upsert_executes_jsonb_upsert_sync():
    run(_test_upsert_executes_jsonb_upsert())
