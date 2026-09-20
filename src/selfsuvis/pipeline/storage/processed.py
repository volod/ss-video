import asyncio
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from selfsuvis.pipeline.core import datetime_to_ts, settings, utcnow
from selfsuvis.pipeline.storage.common import decoded_json, jsonb

_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS processed_files (
        file_hash   TEXT PRIMARY KEY,
        video_id    TEXT NOT NULL,
        path        TEXT,
        size_bytes  BIGINT,
        mtime       DOUBLE PRECISION,
        status      TEXT NOT NULL DEFAULT 'done',
        meta_json   JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
"""


@asynccontextmanager
async def _borrow_conn(conn=None):
    if conn is not None:
        yield conn
        return
    if not settings.DATABASE_URL:
        yield None
        return
    owned = await asyncpg.connect(settings.DATABASE_URL, timeout=5)
    try:
        yield owned
    finally:
        await owned.close()


async def init_db_conn(conn) -> None:
    await conn.execute(_CREATE_TABLE_SQL)


async def aget_by_hash(file_hash: str, conn=None) -> dict[str, Any] | None:
    async with _borrow_conn(conn) as db:
        if db is None:
            return None
        row = await db.fetchrow(
            "SELECT * FROM processed_files WHERE file_hash = $1",
            file_hash,
        )
        return _row_to_dict(row) if row else None


async def aget_by_url(url: str, conn=None) -> dict[str, Any] | None:
    async with _borrow_conn(conn) as db:
        if db is None:
            return None
        row = await db.fetchrow(
            "SELECT * FROM processed_files WHERE meta_json ->> 'url' = $1",
            url,
        )
        return _row_to_dict(row) if row else None


async def aget_by_size(size_bytes: int, conn=None) -> dict[str, Any] | None:
    async with _borrow_conn(conn) as db:
        if db is None:
            return None
        row = await db.fetchrow(
            """
            SELECT *
            FROM processed_files
            WHERE size_bytes = $1
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            size_bytes,
        )
        return _row_to_dict(row) if row else None


async def aupsert(
    file_hash: str,
    video_id: str,
    path: str,
    size_bytes: int,
    mtime: float,
    status: str,
    meta: dict[str, Any],
    conn=None,
) -> None:
    async with _borrow_conn(conn) as db:
        if db is None:
            return
        now = utcnow()
        await db.execute(
            """
            INSERT INTO processed_files
                (file_hash, video_id, path, size_bytes, mtime, status, meta_json, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9)
            ON CONFLICT (file_hash) DO UPDATE SET
                video_id = EXCLUDED.video_id,
                path = EXCLUDED.path,
                size_bytes = EXCLUDED.size_bytes,
                mtime = EXCLUDED.mtime,
                status = EXCLUDED.status,
                meta_json = EXCLUDED.meta_json,
                updated_at = EXCLUDED.updated_at
            """,
            file_hash,
            video_id,
            path,
            size_bytes,
            mtime,
            status,
            jsonb(meta),
            now,
            now,
        )


async def ainit_db() -> None:
    if not settings.DATABASE_URL:
        return
    conn = await asyncpg.connect(settings.DATABASE_URL, timeout=5)
    try:
        await init_db_conn(conn)
    finally:
        await conn.close()


def init_db() -> None:
    asyncio.run(ainit_db())


def get_by_hash(file_hash: str) -> dict[str, Any] | None:
    return asyncio.run(aget_by_hash(file_hash))


def get_by_url(url: str) -> dict[str, Any] | None:
    return asyncio.run(aget_by_url(url))


def get_by_size(size_bytes: int) -> dict[str, Any] | None:
    return asyncio.run(aget_by_size(size_bytes))


def upsert(
    file_hash: str,
    video_id: str,
    path: str,
    size_bytes: int,
    mtime: float,
    status: str,
    meta: dict[str, Any],
) -> None:
    asyncio.run(aupsert(file_hash, video_id, path, size_bytes, mtime, status, meta))


def _row_to_dict(row) -> dict[str, Any]:
    meta = decoded_json(row["meta_json"], default={})
    return {
        "file_hash": row["file_hash"],
        "video_id": row["video_id"],
        "path": row["path"],
        "size_bytes": row["size_bytes"],
        "mtime": row["mtime"],
        "status": row["status"],
        "meta": dict(meta),
        "created_at": datetime_to_ts(row["created_at"]),
        "updated_at": datetime_to_ts(row["updated_at"]),
    }
