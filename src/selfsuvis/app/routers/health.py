"""Video API /health — postgres, qdrant, redis."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from selfsuvis.app.api_utils import ERROR_RESPONSES
from selfsuvis.pipeline.core import get_logger, settings

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", responses={503: ERROR_RESPONSES[503]})
async def health(request: Request):
    """Health check for the video API."""
    status = "ok"
    details: dict = {}

    pool = getattr(request.app.state, "db_pool", None)
    if pool:
        try:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            details["postgres"] = "ok"
        except Exception as exc:
            logger.warning("Health: postgres error: %s", exc)
            details["postgres"] = "error"
            status = "down"
    else:
        details["postgres"] = "unconfigured"

    try:
        from selfsuvis.app.state import app_state

        app_state.store.client.get_collections()
        details["qdrant"] = "connected"
    except Exception:
        details["qdrant"] = "error"

    details["redis"] = "unconfigured"
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.HEALTH_REDIS_URL, socket_connect_timeout=2)
        await r.ping()
        details["redis"] = "ok"
        await r.aclose()
    except Exception as exc:
        logger.warning("Health: redis error: %s", exc)
        details["redis"] = "error"
        status = "down"

    http_status = 200 if status != "down" else 503
    return JSONResponse({"status": status, **details}, status_code=http_status)
