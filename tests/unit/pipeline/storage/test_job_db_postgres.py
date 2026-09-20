"""Unit tests for pipeline.job_db_pg (asyncpg-backed PostgreSQL job queue).

Uses a synchronous in-memory MockConn so these tests run without a live database
or the asyncpg package installed.  asyncio.run() drives the coroutines.
"""

import asyncio

import pytest

from selfsuvis.pipeline.storage.jobs import (
    _UPDATE_JOB_COLUMNS,
    create_job,
    fetch_and_claim_next_pending,
    fetch_job,
    fetch_queue_depth,
    update_job,
)

# ── Minimal asyncpg Connection mock ──────────────────────────────────────────


class _Row(dict):
    """dict subclass that supports both row["col"] and row.col access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class MockConn:
    """In-memory simulation of an asyncpg Connection.

    Implements execute / fetchrow / fetchval as coroutines so they can be
    awaited by the production code under test.
    """

    def __init__(self):
        self._jobs: dict[str, dict] = {}

    # ── asyncpg coroutine stubs ───────────────────────────────────────────

    async def execute(self, query: str, *args) -> None:
        q = query.strip().upper()

        if "INSERT INTO JOBS" in q:
            job_id, status, job_type, progress_json, payload_json, created_at = args
            self._jobs[job_id] = _Row(
                id=job_id,
                status=status,
                type=job_type,
                progress_json=progress_json,
                payload_json=payload_json,
                created_at=created_at,
                started_at=None,
                finished_at=None,
                error=None,
            )

        elif "UPDATE JOBS SET STATUS" in q and "'RUNNING'" in q:
            # fetch_and_claim UPDATE: (started_at, job_id)
            started_at, job_id = args
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "running"
                self._jobs[job_id]["started_at"] = started_at

        elif "UPDATE JOBS SET" in q:
            # update_job: positional args, last is job_id
            *set_vals, job_id = args
            if job_id not in self._jobs:
                return
            # Parse column names from the SET clause
            set_clause = query[
                query.upper().index("SET") + 3 : query.upper().index("WHERE")
            ].strip()
            col_names = [part.split("=")[0].strip() for part in set_clause.split(",")]
            for col, val in zip(col_names, set_vals):
                self._jobs[job_id][col] = val

    async def fetchrow(self, query: str, *args) -> _Row | None:
        q = query.strip().upper()

        if "WHERE ID = " in q and args:
            return self._jobs.get(args[0])

        if "WHERE" in q and "STATUS = 'PENDING'" in q:
            pending = [j for j in self._jobs.values() if j["status"] == "pending"]
            if not pending:
                return None
            return min(pending, key=lambda j: j["created_at"])

        return None

    async def fetchval(self, query: str, *args):
        if "COUNT(*)" in query.upper():
            return sum(1 for j in self._jobs.values() if j["status"] == "pending")
        return None


# ── helpers ───────────────────────────────────────────────────────────────────


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def conn():
    return MockConn()


# ── whitelist constant ────────────────────────────────────────────────────────


def test_update_job_columns_whitelist():
    """Whitelist contains exactly the expected columns."""
    assert "status" in _UPDATE_JOB_COLUMNS
    assert "progress" in _UPDATE_JOB_COLUMNS
    assert "payload" in _UPDATE_JOB_COLUMNS
    assert "started_at" in _UPDATE_JOB_COLUMNS
    assert "finished_at" in _UPDATE_JOB_COLUMNS
    assert "error" in _UPDATE_JOB_COLUMNS
    assert "id" not in _UPDATE_JOB_COLUMNS
    assert "created_at" not in _UPDATE_JOB_COLUMNS


# ── create_job / fetch_job ────────────────────────────────────────────────────


def test_create_job_and_fetch(conn):
    run(create_job(conn, "j1", {"video_id": "v1"}))
    job = run(fetch_job(conn, "j1"))
    assert job is not None
    assert job["id"] == "j1"
    assert job["status"] == "pending"
    assert job["payload"] == {"video_id": "v1"}
    assert job["progress"] == {}
    assert job["created_at"] is not None
    assert job["started_at"] is None
    assert job["finished_at"] is None
    assert job["error"] is None


def test_fetch_job_nonexistent_returns_none(conn):
    assert run(fetch_job(conn, "missing")) is None


def test_create_job_multiple(conn):
    run(create_job(conn, "a", {"n": 1}))
    run(create_job(conn, "b", {"n": 2}))
    assert run(fetch_job(conn, "a"))["payload"] == {"n": 1}
    assert run(fetch_job(conn, "b"))["payload"] == {"n": 2}


# ── update_job ────────────────────────────────────────────────────────────────


def test_update_job_status(conn):
    run(create_job(conn, "j1", {}))
    run(update_job(conn, "j1", status="done"))
    assert run(fetch_job(conn, "j1"))["status"] == "done"


def test_update_job_ignores_unknown_columns(conn):
    run(create_job(conn, "j1", {}))
    run(update_job(conn, "j1", status="done", __sql_injection__="ignored"))
    job = run(fetch_job(conn, "j1"))
    assert job["status"] == "done"
    assert "__sql_injection__" not in job


def test_update_job_empty_kwargs_is_noop(conn):
    run(create_job(conn, "j1", {}))
    run(update_job(conn, "j1", unknown_key="x"))
    assert run(fetch_job(conn, "j1"))["status"] == "pending"


def test_update_job_progress_serialised(conn):
    run(create_job(conn, "j1", {}))
    run(update_job(conn, "j1", progress={"frames": 42}))
    job = run(fetch_job(conn, "j1"))
    assert job["progress"] == {"frames": 42}


def test_update_job_error_field(conn):
    run(create_job(conn, "j1", {}))
    run(update_job(conn, "j1", status="error", error="ffmpeg crash"))
    job = run(fetch_job(conn, "j1"))
    assert job["error"] == "ffmpeg crash"


# ── fetch_and_claim_next_pending ──────────────────────────────────────────────


def test_claim_oldest_pending(conn):
    """Oldest job (by created_at) is claimed first."""
    run(create_job(conn, "early", {}))
    # bump created_at of 'late' slightly so ordering is deterministic
    conn._jobs["early"]["created_at"] = 1.0
    run(create_job(conn, "late", {}))
    conn._jobs["late"]["created_at"] = 2.0

    claimed = run(fetch_and_claim_next_pending(conn))
    assert claimed is not None
    assert claimed["id"] == "early"
    assert claimed["status"] == "running"
    assert claimed["started_at"] is not None


def test_claim_marks_job_running(conn):
    run(create_job(conn, "j1", {}))
    run(fetch_and_claim_next_pending(conn))
    assert run(fetch_job(conn, "j1"))["status"] == "running"


def test_claim_returns_none_when_no_pending(conn):
    assert run(fetch_and_claim_next_pending(conn)) is None


def test_claim_skips_running_jobs(conn):
    """After one job is claimed, fetch_and_claim returns the next pending one."""
    run(create_job(conn, "a", {}))
    conn._jobs["a"]["created_at"] = 1.0
    run(create_job(conn, "b", {}))
    conn._jobs["b"]["created_at"] = 2.0

    first = run(fetch_and_claim_next_pending(conn))
    assert first["id"] == "a"
    second = run(fetch_and_claim_next_pending(conn))
    assert second["id"] == "b"
    none = run(fetch_and_claim_next_pending(conn))
    assert none is None


# ── fetch_queue_depth ─────────────────────────────────────────────────────────


def test_queue_depth_empty(conn):
    assert run(fetch_queue_depth(conn)) == 0


def test_queue_depth_counts_pending(conn):
    run(create_job(conn, "a", {}))
    run(create_job(conn, "b", {}))
    assert run(fetch_queue_depth(conn)) == 2


def test_queue_depth_decrements_on_claim(conn):
    run(create_job(conn, "a", {}))
    run(create_job(conn, "b", {}))
    run(fetch_and_claim_next_pending(conn))
    assert run(fetch_queue_depth(conn)) == 1


# ── full lifecycle ────────────────────────────────────────────────────────────


def test_full_lifecycle(conn):
    """pending → running → done via claim + update."""
    run(create_job(conn, "lc", {"video_id": "v1"}))
    assert run(fetch_job(conn, "lc"))["status"] == "pending"

    claimed = run(fetch_and_claim_next_pending(conn))
    assert claimed["id"] == "lc"
    assert run(fetch_job(conn, "lc"))["status"] == "running"

    run(update_job(conn, "lc", status="done", progress={"frames": 99}, finished_at=9999.0))
    done = run(fetch_job(conn, "lc"))
    assert done["status"] == "done"
    assert done["progress"] == {"frames": 99}
    assert done["finished_at"] == pytest.approx(9999.0)
