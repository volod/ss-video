"""asyncpg-backed PostgreSQL job queue.

Replaces the SQLite job_db.py for production use with PostgreSQL 16.
The worker claims jobs with SELECT FOR UPDATE SKIP LOCKED for concurrency-safe
polling — no external lock manager required.

All functions accept an asyncpg Connection (or pool) as their first argument
so callers control connection lifecycle and transaction boundaries.
"""

from typing import Any

from selfsuvis.pipeline.core import datetime_to_ts, get_logger, to_utc_datetime, utcnow
from selfsuvis.pipeline.storage.common import decoded_json, jsonb

logger = get_logger(__name__)

# Columns that update_job is allowed to touch (prevents injection of arbitrary column names)
_UPDATE_JOB_COLUMNS = frozenset(
    {"status", "progress", "payload", "started_at", "finished_at", "error", "type"}
)

_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS jobs (
        id            TEXT PRIMARY KEY,
        status        TEXT NOT NULL DEFAULT 'pending',
        type          TEXT,
        progress_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        payload_json  JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        started_at    TIMESTAMPTZ,
        finished_at   TIMESTAMPTZ,
        error         TEXT
    )
"""


async def init_db(conn) -> None:
    """Create the jobs table if it does not already exist."""
    await conn.execute(_CREATE_TABLE_SQL)


async def create_job(
    conn, job_id: str, payload: dict[str, Any], job_type: str | None = None
) -> None:
    """Insert a new job in 'pending' state."""
    now = utcnow()
    await conn.execute(
        "INSERT INTO jobs (id, status, type, progress_json, payload_json, created_at)"
        " VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6)",
        job_id,
        "pending",
        job_type,
        "{}",
        jsonb(payload),
        now,
    )


async def update_job(conn, job_id: str, **kwargs: Any) -> None:
    """Update whitelisted job fields.

    Unknown keyword arguments are silently ignored to prevent column-name injection.
    """
    allowed = {k: v for k, v in kwargs.items() if k in _UPDATE_JOB_COLUMNS}
    if not allowed:
        return

    parts = []
    values: list = []
    placeholder = 1
    for k, v in allowed.items():
        if k in {"progress", "payload"}:
            col = f"{k}_json"
            parts.append(f"{col} = ${placeholder}::jsonb")
            values.append(jsonb(v))
            placeholder += 1
            continue
        elif k in {"started_at", "finished_at"}:
            col = k
            v = to_utc_datetime(v)
        else:
            col = k
        parts.append(f"{col} = ${placeholder}")
        values.append(v)
        placeholder += 1

    values.append(job_id)
    await conn.execute(
        f"UPDATE jobs SET {', '.join(parts)} WHERE id = ${placeholder}",
        *values,
    )


async def fetch_job(conn, job_id: str) -> dict[str, Any] | None:
    """Fetch a single job by id. Returns None if not found."""
    row = await conn.fetchrow("SELECT * FROM jobs WHERE id = $1", job_id)
    return _row_to_dict(row) if row else None


async def fetch_and_claim_next_pending(conn) -> dict[str, Any] | None:
    """Atomically claim the oldest pending job.

    Uses SELECT FOR UPDATE SKIP LOCKED so multiple workers can poll concurrently
    without stepping on each other. Must be called inside a transaction.
    Returns None if no pending jobs exist.
    """
    row = await conn.fetchrow(
        """
        SELECT * FROM jobs
        WHERE  status = 'pending'
        ORDER  BY created_at ASC
        LIMIT  1
        FOR UPDATE SKIP LOCKED
        """
    )
    if not row:
        return None

    now = utcnow()
    await conn.execute(
        "UPDATE jobs SET status = 'running', started_at = $1 WHERE id = $2",
        now,
        row["id"],
    )
    d = _row_to_dict(row)
    d["status"] = "running"
    d["started_at"] = now
    return d


async def fetch_queue_depth(conn) -> int:
    """Return the number of pending jobs."""
    return await conn.fetchval("SELECT COUNT(*) FROM jobs WHERE status = 'pending'")


def _row_to_dict(row) -> dict[str, Any]:
    progress = decoded_json(row["progress_json"], default={})
    payload = decoded_json(row["payload_json"], default={})
    return {
        "id": row["id"],
        "status": row["status"],
        "type": row["type"],
        "progress": progress,
        "payload": payload,
        "created_at": datetime_to_ts(row["created_at"]),
        "started_at": datetime_to_ts(row["started_at"]),
        "finished_at": datetime_to_ts(row["finished_at"]),
        "error": row["error"],
    }
