"""GPU job isolation via a PostgreSQL advisory semaphore table.

The ``gpu_jobs`` table acts as an advisory semaphore: workers register
before allocating GPU memory and deregister on completion.  Stale entries
older than ``GPU_JOB_TIMEOUT_SEC`` are evicted on every check-in so a
crashed worker does not permanently block GPU access.

The semaphore is fail-open: DB errors are logged but never block GPU work.
"""

from datetime import timedelta

import asyncpg

from selfsuvis.pipeline.core import settings, utcnow
from selfsuvis.worker._run import _run


class GPULock:
    """Context manager that registers/deregisters a job in the gpu_jobs table."""

    def __init__(self, job_id: str, job_type: str, conn_url: str, logger):
        self.job_id = job_id
        self.job_type = job_type
        self.conn_url = conn_url
        self.logger = logger

    async def _checkin(self) -> None:
        conn = await asyncpg.connect(self.conn_url)
        try:
            now = utcnow()
            stale_cutoff = now - timedelta(seconds=settings.GPU_JOB_TIMEOUT_SEC)
            evicted = await conn.execute("DELETE FROM gpu_jobs WHERE started_at < $1", stale_cutoff)
            evicted_count = int(evicted.split()[-1])
            if evicted_count:
                self.logger.info("GPU isolation: evicted %d stale gpu_jobs entry(s)", evicted_count)
            active = await conn.fetchval("SELECT COUNT(*) FROM gpu_jobs")
            if active > 0:
                holder = await conn.fetchrow("SELECT job_id, job_type FROM gpu_jobs LIMIT 1")
                self.logger.warning(
                    "GPU isolation: GPU busy (job_id=%s type=%s) -- job %s will proceed anyway "
                    "(consider scaling down concurrent GPU jobs)",
                    holder["job_id"] if holder else "?",
                    holder["job_type"] if holder else "?",
                    self.job_id,
                )
            await conn.execute(
                "INSERT INTO gpu_jobs (job_id, job_type, worker_id, started_at) "
                "VALUES ($1, $2, $3, $4) ON CONFLICT (job_id) DO NOTHING",
                self.job_id,
                self.job_type,
                settings.WORKER_ID,
                now,
            )
        finally:
            await conn.close()

    async def _checkout(self) -> None:
        conn = await asyncpg.connect(self.conn_url)
        try:
            await conn.execute("DELETE FROM gpu_jobs WHERE job_id = $1", self.job_id)
        finally:
            await conn.close()

    def __enter__(self):
        try:
            _run(self._checkin())
        except Exception as exc:
            self.logger.warning("GPU isolation: check-in failed (non-fatal): %s", exc)
        return self

    def __exit__(self, *_):
        try:
            _run(self._checkout())
        except Exception as exc:
            self.logger.warning("GPU isolation: check-out failed (non-fatal): %s", exc)


def _gpu_checkin(job_id: str, job_type: str, conn_url: str, logger) -> bool:
    """Synchronous wrapper around GPULock._checkin. Always returns True (fail-open)."""
    lock = GPULock(job_id, job_type, conn_url, logger)
    try:
        _run(lock._checkin())
    except Exception as exc:
        logger.warning("GPU isolation: check-in failed (non-fatal): %s", exc)
    return True


def _gpu_checkout(job_id: str, conn_url: str, logger) -> None:
    """Synchronous wrapper around GPULock._checkout. Fail-open on error."""
    lock = GPULock(job_id, "unknown", conn_url, logger)
    try:
        _run(lock._checkout())
    except Exception as exc:
        logger.warning("GPU isolation: check-out failed (non-fatal): %s", exc)
