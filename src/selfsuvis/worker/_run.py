"""Shared event loop runner for the worker process.

All handler modules import ``_run`` from here so that every async coroutine
runs on the same persistent loop that the asyncpg pool was created on.
``init_loop`` must be called once in ``main()`` before the pool is created.
"""

import asyncio

from selfsuvis.pipeline.storage import update_job

_loop: asyncio.AbstractEventLoop | None = None


def _run(coro):
    """Run *coro* on the worker's persistent event loop."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro)


def init_loop() -> asyncio.AbstractEventLoop:
    """Create and register a fresh event loop; call once at worker start."""
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    return _loop


def _update_job_sync(pool, job_id: str, **kwargs) -> None:
    """Synchronous helper to update a job row via the pool."""

    async def _upd():
        async with pool.acquire() as conn:
            await update_job(conn, job_id, **kwargs)

    _run(_upd())
