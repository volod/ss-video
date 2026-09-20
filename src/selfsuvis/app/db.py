"""Database pool utilities for the video FastAPI app lifecycle."""

import asyncpg
from fastapi import FastAPI, HTTPException, Request

from selfsuvis.pipeline.core import get_logger, settings

logger = get_logger(__name__)


async def init_db_pool(app: FastAPI) -> None:
    """Initialize the video asyncpg pool and attach it to app state."""
    db_url = settings.DATABASE_URL
    if not db_url:
        app.state.db_pool = None
        logger.warning("DATABASE_URL not configured; API DB operations unavailable")
        return
    app.state.db_pool = await asyncpg.create_pool(
        dsn=db_url,
        min_size=1,
        max_size=10,
        timeout=10,
    )


async def close_db_pool(app: FastAPI) -> None:
    """Close the asyncpg pool if it exists."""
    pool: asyncpg.Pool | None = getattr(app.state, "db_pool", None)
    if pool is not None:
        await pool.close()


def get_db_pool(request: Request) -> asyncpg.Pool:
    """Return the video DB pool from request app state or raise 503."""
    pool: asyncpg.Pool | None = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")
    return pool


def get_db_pool_optional(request: Request) -> asyncpg.Pool | None:
    """Return the video DB pool from request app state, or None if not configured."""
    return getattr(request.app.state, "db_pool", None)
