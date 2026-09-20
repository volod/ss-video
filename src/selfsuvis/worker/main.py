import os
import time
from enum import Enum

import asyncpg

from selfsuvis.pipeline.core import (
    get_logger,
    log_preflight,
    run_production_preflight,
    settings,
    validate_settings,
)
from selfsuvis.pipeline.storage import fetch_and_claim_next_pending
from selfsuvis.pipeline.storage.processed import ainit_db as init_processed_db_async
from selfsuvis.worker._run import _run, _update_job_sync, init_loop
from selfsuvis.worker.handlers import (
    handle_finetune_job,
    handle_index_job,
    handle_postflight_mapping_job,
    handle_postflight_semantic_graph_job,
    handle_reembed_job,
)

logger = get_logger(__name__)


class JobType(str, Enum):
    INDEX = "index"
    FINETUNE = "supervised_finetune"
    REEMBED = "reembed"
    POSTFLIGHT_MAPPING = "postflight_mapping"
    POSTFLIGHT_SEMANTIC_GRAPH = "postflight_semantic_graph"


def _claim_next_job(pool) -> dict | None:
    """Atomically claim the next pending job using SELECT FOR UPDATE SKIP LOCKED."""

    async def _claim():
        async with pool.acquire() as conn:
            async with conn.transaction():
                return await fetch_and_claim_next_pending(conn)

    try:
        return _run(_claim())
    except Exception as exc:
        sqlstate = getattr(exc, "sqlstate", None)
        if sqlstate == "42P01" or type(exc).__name__ == "UndefinedTableError":
            logger.warning("jobs table not ready: %s", exc)
            return None
        raise


def main() -> None:
    init_loop()

    validate_settings()
    report = run_production_preflight("worker")
    log_preflight(report)
    if os.getenv("STARTUP_PREFLIGHT_STRICT", "false").lower() == "true":
        report.raise_for_errors()
    logger.info("Worker started")

    conn_url = settings.DATABASE_URL
    if not conn_url:
        logger.error("DATABASE_URL not configured -- worker cannot start")
        return

    async def _bootstrap():
        from selfsuvis.pipeline.storage.migrate_video import migrate as migrate_video

        try:
            await migrate_video(conn_url, verbose=False)
        except Exception as exc:
            if type(exc).__name__ != "UniqueViolationError":
                raise
            logger.warning("schema bootstrap race, continuing: %s", exc)
        await init_processed_db_async()
        return await asyncpg.create_pool(
            dsn=conn_url,
            min_size=1,
            max_size=10,
            timeout=10,
        )

    pool = _run(_bootstrap())

    try:
        while True:
            job = _claim_next_job(pool)
            if not job:
                time.sleep(settings.WORKER_POLL_INTERVAL)
                continue

            job_id = job["id"]
            job_type = job.get("type")
            payload = job["payload"]
            logger.info("Job claimed id=%s type=%s", job_id, job_type)

            if job_type == JobType.FINETUNE:
                handle_finetune_job(job_id, payload, pool, conn_url, logger)
                continue

            if job_type == JobType.REEMBED:
                handle_reembed_job(job_id, payload, conn_url, logger)
                continue

            if job_type == JobType.POSTFLIGHT_MAPPING:
                handle_postflight_mapping_job(job_id, payload, pool, logger)
                continue

            if job_type == JobType.POSTFLIGHT_SEMANTIC_GRAPH:
                handle_postflight_semantic_graph_job(job_id, payload, pool, logger)
                continue

            if job_type not in (None, JobType.INDEX):
                logger.warning("Unknown job type=%s id=%s -- marking error", job_type, job_id)
                _update_job_sync(
                    pool,
                    job_id,
                    status="error",
                    error=f"unknown job type: {job_type}",
                    finished_at=time.time(),
                )
                continue

            handle_index_job(job_id, payload, job, pool, logger)
    finally:
        _run(pool.close())


if __name__ == "__main__":
    main()
