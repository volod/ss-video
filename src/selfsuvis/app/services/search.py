import time
from typing import TYPE_CHECKING, Any

import asyncpg
from PIL import Image
from pydantic import BaseModel

from selfsuvis.app.state import dino_model, store
from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger
from selfsuvis.pipeline.core.optional_deps import require_qdrant_models

logger = get_logger(__name__)

_REEMBED_STATUS_CACHE_TTL_SEC = 2.0
_reembed_status_cache: dict = {"value": False, "checked_at": 0.0}

# Blend weights for CLIP + DINOv3 reranking: clip_score * W_CLIP + dino_score * W_DINO.
# Must sum to 1.0.  DINOv3 weight is intentionally small since DINOv3 is a
# discriminative backbone while CLIP scores encode cross-modal text alignment.
_RERANK_W_CLIP = 0.7
_RERANK_W_DINO = 0.3


class SearchResult(BaseModel):
    id: int
    score: float
    type: str
    video_id: str
    segment_id: int
    t_sec: float
    thumbnail_path: str
    frame_path: str | None = None
    tile_path: str | None = None
    bbox: dict | None = None


if TYPE_CHECKING:
    from qdrant_client.http import models as qmodels  # pragma: no cover


def payload_filter(search_type: str) -> "qmodels.Filter | None":
    qmodels = require_qdrant_models()
    if search_type == "both":
        return None
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="type",
                match=qmodels.MatchValue(value=search_type),
            )
        ]
    )


async def _reembed_is_active(db_pool: Any | None = None) -> bool:
    """Return True if a reembed job is currently running.

    Uses a short TTL cache to avoid querying DB for every image query.
    Fails fast on DB errors (returns False) so search is never blocked.
    """
    now = time.monotonic()
    if now - _reembed_status_cache["checked_at"] < _REEMBED_STATUS_CACHE_TTL_SEC:
        return bool(_reembed_status_cache["value"])

    db_url = settings.DATABASE_URL
    if db_pool is None and not db_url:
        _reembed_status_cache["value"] = False
        _reembed_status_cache["checked_at"] = now
        return False
    try:
        if db_pool is not None:
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT 1 FROM jobs WHERE type = 'reembed' AND status = 'running' LIMIT 1"
                )
                active = row is not None
        else:
            conn = await asyncpg.connect(db_url, timeout=3)
            try:
                row = await conn.fetchrow(
                    "SELECT 1 FROM jobs WHERE type = 'reembed' AND status = 'running' LIMIT 1"
                )
                active = row is not None
            finally:
                await conn.close()
        _reembed_status_cache["value"] = active
        _reembed_status_cache["checked_at"] = now
        return active
    except Exception:
        _reembed_status_cache["value"] = False
        _reembed_status_cache["checked_at"] = now
        return False


async def search_vectors(
    vector_space: str,
    query_vec,
    search_type: str,
    top_k: int,
    enable_rerank: bool,
    image_query: Image.Image | None,
    db_pool: Any | None = None,
) -> list[dict]:
    k_retrieve = max(top_k, settings.K_RETRIEVE)
    filter_obj = payload_filter(search_type)

    scored = store.search(vector_space, query_vec, k_retrieve, filter_obj)
    results = format_results(scored)

    if enable_rerank and image_query is not None and vector_space == "clip" and dino_model:
        # Suppress dino reranking while a reembed sweep is active: Qdrant holds a
        # mix of old-model and new-model dino vectors during the sweep, so cosine
        # similarity between them is meaningless and would corrupt result ranking.
        if await _reembed_is_active(db_pool=db_pool):
            logger.info("Dino reranking suppressed: reembed sweep is active (falling back to clip)")
        else:
            dino_vec = dino_model.encode_images([image_query], batch_size=1)[0]
            dino_scored = store.search("dino", dino_vec, k_retrieve, filter_obj)
            dino_map = {p.id: p.score for p in dino_scored}
            for r in results:
                if r["id"] in dino_map:
                    r["score"] = _RERANK_W_CLIP * r["score"] + _RERANK_W_DINO * dino_map[r["id"]]
            results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


def _tile_bbox(payload: dict) -> dict | None:
    return {
        "x": payload.get("x"),
        "y": payload.get("y"),
        "w": payload.get("w"),
        "h": payload.get("h"),
    }


def format_results(scored: list["qmodels.ScoredPoint"]) -> list[dict]:
    results = []
    for p in scored:
        payload = p.payload or {}
        result_type = payload.get("type") or "frame"
        bbox = _tile_bbox(payload) if result_type == "tile" else None
        result = SearchResult(
            id=p.id,
            score=float(p.score),
            type=result_type,
            video_id=payload.get("video_id") or "",
            segment_id=payload.get("segment_id") or 0,
            t_sec=payload.get("t_sec") or 0.0,
            thumbnail_path=payload.get("tile_path") or payload.get("frame_path") or "",
            frame_path=payload.get("frame_path"),
            tile_path=payload.get("tile_path"),
            bbox=bbox,
        )
        results.append(result.model_dump())
    return results
