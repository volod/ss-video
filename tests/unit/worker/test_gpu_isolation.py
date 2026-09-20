"""Unit tests for worker.main._gpu_checkin and _gpu_checkout.

asyncpg is not installed in the test venv; it is stubbed in sys.modules so
that `import asyncpg` inside worker.main succeeds.  The cv2/skimage stubs
are also required because worker.main transitively imports pipeline.indexer.

All asyncpg.connect calls are mocked per-test — no live PostgreSQL required.
"""

import sys
import time
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Stub asyncpg ──────────────────────────────────────────────────────────────
if "asyncpg" not in sys.modules:
    _asyncpg = types.ModuleType("asyncpg")
    _asyncpg.connect = MagicMock()  # patch.object requires attribute to pre-exist
    _asyncpg.Pool = MagicMock()  # type: ignore[attr-defined]
    sys.modules["asyncpg"] = _asyncpg

_asyncpg_mod = sys.modules["asyncpg"]

# ── Import worker.gpu ────────────────────────────────────────────────────────
# worker.gpu only imports asyncpg, pipeline.core, and worker._run — it does
# not transitively import cv2 or skimage, so no stubs are needed for those.
import selfsuvis.worker.gpu as wm  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_conn(active_count=0, evicted_str="DELETE 0"):
    conn = AsyncMock()
    conn.close = AsyncMock()

    async def _execute(query, *args):
        if "DELETE FROM gpu_jobs WHERE started_at" in query:
            return evicted_str
        return "OK"

    conn.execute = AsyncMock(side_effect=_execute)
    conn.fetchval = AsyncMock(return_value=active_count)
    conn.fetchrow = AsyncMock(return_value=None)
    return conn


@pytest.fixture
def logger():
    return MagicMock()


@pytest.fixture
def conn_url():
    return "postgresql://fake/db"


# ── _gpu_checkin ──────────────────────────────────────────────────────────────


class TestGpuCheckin:
    def _checkin(
        self,
        conn,
        logger,
        conn_url,
        job_id="job-1",
        job_type="finetune",
        timeout_sec=3600,
        worker_id="worker-1",
    ):
        settings_mock = MagicMock()
        settings_mock.GPU_JOB_TIMEOUT_SEC = timeout_sec
        settings_mock.WORKER_ID = worker_id
        with patch.object(_asyncpg_mod, "connect", new=AsyncMock(return_value=conn)):
            with patch.object(wm, "settings", settings_mock):
                return wm._gpu_checkin(job_id, job_type, conn_url, logger)

    def test_returns_true_when_no_contention(self, logger, conn_url):
        conn = _make_conn(active_count=0)
        result = self._checkin(conn, logger, conn_url)
        assert result is True

    def test_inserts_row_on_checkin(self, logger, conn_url):
        conn = _make_conn(active_count=0)
        self._checkin(conn, logger, conn_url, job_id="job-42")
        insert_calls = [c for c in conn.execute.call_args_list if "INSERT INTO gpu_jobs" in str(c)]
        assert len(insert_calls) == 1
        assert "job-42" in str(insert_calls[0])

    def test_returns_true_with_contention_logs_warning(self, logger, conn_url):
        """Even with another active job, checkin succeeds (advisory, not hard lock)."""
        holder = MagicMock()
        holder.__getitem__ = lambda s, k: "job-0" if k == "job_id" else "reembed"

        conn = _make_conn(active_count=1)
        conn.fetchrow = AsyncMock(return_value=holder)
        result = self._checkin(conn, logger, conn_url)

        assert result is True
        logger.warning.assert_called_once()
        assert "busy" in logger.warning.call_args[0][0].lower()

    def test_evicts_stale_entries_and_logs(self, logger, conn_url):
        conn = _make_conn(active_count=0, evicted_str="DELETE 2")
        self._checkin(conn, logger, conn_url)
        logger.info.assert_called()
        assert "evicted" in logger.info.call_args[0][0].lower()

    def test_no_eviction_log_when_none_stale(self, logger, conn_url):
        conn = _make_conn(active_count=0, evicted_str="DELETE 0")
        self._checkin(conn, logger, conn_url)
        info_msgs = [str(c) for c in logger.info.call_args_list]
        assert not any("evicted" in m.lower() for m in info_msgs)

    def test_fail_open_on_connect_error(self, logger, conn_url):
        """DB connection failure → returns True (never blocks GPU work)."""
        settings_mock = MagicMock()
        settings_mock.GPU_JOB_TIMEOUT_SEC = 3600
        settings_mock.WORKER_ID = "w"
        with patch.object(_asyncpg_mod, "connect", new=AsyncMock(side_effect=OSError("refused"))):
            with patch.object(wm, "settings", settings_mock):
                result = wm._gpu_checkin("job-1", "finetune", conn_url, logger)
        assert result is True
        logger.warning.assert_called_once()
        assert "non-fatal" in logger.warning.call_args[0][0].lower()

    def test_connection_closed_after_checkin(self, logger, conn_url):
        conn = _make_conn(active_count=0)
        self._checkin(conn, logger, conn_url)
        conn.close.assert_called_once()

    def test_stale_cutoff_reflects_timeout_sec(self, logger, conn_url):
        """The DELETE timestamp = now − GPU_JOB_TIMEOUT_SEC (within 2s tolerance)."""
        timeout_sec = 7200
        conn = _make_conn(active_count=0)
        before = time.time()
        self._checkin(conn, logger, conn_url, timeout_sec=timeout_sec)
        after = time.time()

        delete_call = conn.execute.call_args_list[0]
        cutoff = delete_call[0][1]  # first positional arg after query
        if hasattr(cutoff, "timestamp"):
            cutoff = cutoff.timestamp()
        assert before - timeout_sec - 2 <= cutoff <= after - timeout_sec + 2

    def test_worker_id_stored_in_insert(self, logger, conn_url):
        conn = _make_conn(active_count=0)
        self._checkin(conn, logger, conn_url, worker_id="drone-worker-7")
        insert_str = str(conn.execute.call_args_list[-1])
        assert "drone-worker-7" in insert_str


# ── _gpu_checkout ─────────────────────────────────────────────────────────────


class TestGpuCheckout:
    def _checkout(self, conn, logger, conn_url, job_id="job-1"):
        with patch.object(_asyncpg_mod, "connect", new=AsyncMock(return_value=conn)):
            wm._gpu_checkout(job_id, conn_url, logger)

    def test_deletes_correct_job_row(self, logger, conn_url):
        conn = _make_conn()
        self._checkout(conn, logger, conn_url, job_id="job-xyz")
        delete_calls = [
            c for c in conn.execute.call_args_list if "DELETE FROM gpu_jobs WHERE job_id" in str(c)
        ]
        assert len(delete_calls) == 1
        assert "job-xyz" in str(delete_calls[0])

    def test_connection_closed_after_checkout(self, logger, conn_url):
        conn = _make_conn()
        self._checkout(conn, logger, conn_url)
        conn.close.assert_called_once()

    def test_fail_open_on_connect_error(self, logger, conn_url):
        """DB failure on checkout is non-fatal — no exception raised."""
        with patch.object(_asyncpg_mod, "connect", new=AsyncMock(side_effect=OSError("refused"))):
            wm._gpu_checkout("job-1", conn_url, logger)  # must not raise
        logger.warning.assert_called_once()
        assert "non-fatal" in logger.warning.call_args[0][0].lower()

    def test_fail_open_on_execute_error(self, logger, conn_url):
        conn = AsyncMock()
        conn.close = AsyncMock()
        conn.execute = AsyncMock(side_effect=RuntimeError("tx aborted"))
        with patch.object(_asyncpg_mod, "connect", new=AsyncMock(return_value=conn)):
            wm._gpu_checkout("job-1", conn_url, logger)  # must not raise
        logger.warning.assert_called_once()


# ── Round trip ────────────────────────────────────────────────────────────────


class TestCheckinCheckoutRoundTrip:
    def test_full_round_trip(self, logger, conn_url):
        """Check-in inserts a row; check-out deletes it."""
        checkin_conn = _make_conn(active_count=0)
        checkout_conn = _make_conn()
        settings_mock = MagicMock()
        settings_mock.GPU_JOB_TIMEOUT_SEC = 3600
        settings_mock.WORKER_ID = "worker-1"

        with patch.object(_asyncpg_mod, "connect", new=AsyncMock(return_value=checkin_conn)):
            with patch.object(wm, "settings", settings_mock):
                wm._gpu_checkin("round-trip-job", "finetune", conn_url, logger)

        with patch.object(_asyncpg_mod, "connect", new=AsyncMock(return_value=checkout_conn)):
            wm._gpu_checkout("round-trip-job", conn_url, logger)

        inserts = [
            c for c in checkin_conn.execute.call_args_list if "INSERT INTO gpu_jobs" in str(c)
        ]
        deletes = [
            c
            for c in checkout_conn.execute.call_args_list
            if "DELETE FROM gpu_jobs WHERE job_id" in str(c)
        ]
        assert len(inserts) == 1
        assert len(deletes) == 1
