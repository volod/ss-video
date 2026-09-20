"""Scene Intelligence API — POST /query/scene

Structured JSONB query over indexed frame_facts_json (vehicle/road scene data).

Supports:
  - Optional text query (re-ranked by CLIP vector similarity)
  - vehicle_count_min / vehicle_count_max  (jsonb_array_length on vehicle_groups)
  - road_condition  keyword filter (exact match on frame_facts_json->>'road_condition')
  - gps_bbox  [min_lat, min_lon, max_lat, max_lon]  spatial bounding box
  - time_range  [start_sec, end_sec]  mission frame timestamp window
  - top_k  maximum results (1–100, default 20)

Auth: X-API-Key header required.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from selfsuvis.app.api_utils import ERROR_RESPONSES
from selfsuvis.app.db import get_db_pool
from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.app.state import clip_model
from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.storage.common import qdrant_id_from_pg

logger = get_logger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["scene"],
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)


# -- Request / Response models -------------------------------------------------


class GpsBbox(BaseModel):
    """Bounding box for GPS filtering."""

    min_lat: float = Field(description="Minimum latitude (decimal degrees, WGS-84)")
    min_lon: float = Field(description="Minimum longitude")
    max_lat: float = Field(description="Maximum latitude")
    max_lon: float = Field(description="Maximum longitude")

    @model_validator(mode="after")
    def _validate_bbox(self) -> "GpsBbox":
        if self.min_lat > self.max_lat:
            raise ValueError("min_lat must be ≤ max_lat")
        if self.min_lon > self.max_lon:
            raise ValueError("min_lon must be ≤ max_lon")
        return self


class TimeRange(BaseModel):
    """Frame timestamp range within a mission."""

    start_sec: float = Field(
        ge=0.0, description="Earliest frame timestamp (seconds from mission start)"
    )
    end_sec: float = Field(
        ge=0.0, description="Latest frame timestamp (seconds from mission start)"
    )

    @model_validator(mode="after")
    def _validate_range(self) -> "TimeRange":
        if self.start_sec > self.end_sec:
            raise ValueError("start_sec must be ≤ end_sec")
        return self


class SceneQuery(BaseModel):
    """POST /query/scene request body."""

    text: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional text query — results re-ranked by CLIP similarity when provided",
    )
    vehicle_count_min: int | None = Field(
        default=None,
        ge=0,
        description="Minimum total vehicle count across all vehicle_groups",
    )
    vehicle_count_max: int | None = Field(
        default=None,
        ge=0,
        description="Maximum total vehicle count across all vehicle_groups",
    )
    road_condition: str | None = Field(
        default=None,
        max_length=50,
        description="Exact road_condition keyword (e.g. clear, wet, snow, ice, debris, unknown)",
    )
    gps_bbox: GpsBbox | None = Field(
        default=None,
        description="GPS bounding box filter",
    )
    time_range: TimeRange | None = Field(
        default=None,
        description="Frame timestamp range within missions",
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )


class SceneMatch(BaseModel):
    """Single frame result from a scene query."""

    frame_id: str
    mission_id: str
    score: float
    t_sec: float
    lat: float | None
    lon: float | None
    caption: str | None
    vehicle_count: int | None
    road_condition: str | None
    road_surface: str | None
    scene_summary: str | None
    frame_path: str


class SceneQueryResponse(BaseModel):
    results: list[SceneMatch]
    total_matched: int
    filters_applied: list[str]


# -- Helpers -------------------------------------------------------------------


def _count_vehicles(frame_facts: dict[str, Any] | None) -> int | None:
    """Sum the 'count' field across all vehicle_groups entries."""
    if not frame_facts or not isinstance(frame_facts, dict):
        return None
    groups = frame_facts.get("vehicle_groups")
    if not isinstance(groups, list):
        return None
    total = 0
    for g in groups:
        if isinstance(g, dict):
            total += int(g.get("count", 0))
    return total


def _build_sql_filters(body: SceneQuery) -> tuple[str, list]:
    """Return (WHERE clause fragment, positional params list) for asyncpg query."""
    clauses: list[str] = [
        "frame_facts_json IS NOT NULL",
    ]
    params: list = []
    n = 0  # param counter

    def _p(val: Any) -> str:
        nonlocal n
        n += 1
        params.append(val)
        return f"${n}"

    # road_condition exact match
    if body.road_condition:
        clauses.append(f"frame_facts_json->>'road_condition' = {_p(body.road_condition)}")

    # GPS bbox (gps_json stores lat/lon as numbers)
    if body.gps_bbox:
        bbox = body.gps_bbox
        clauses.append(f"(gps_json->>'lat')::double precision >= {_p(bbox.min_lat)}")
        clauses.append(f"(gps_json->>'lat')::double precision <= {_p(bbox.max_lat)}")
        clauses.append(f"(gps_json->>'lon')::double precision >= {_p(bbox.min_lon)}")
        clauses.append(f"(gps_json->>'lon')::double precision <= {_p(bbox.max_lon)}")

    # time_range
    if body.time_range:
        clauses.append(f"t_sec >= {_p(body.time_range.start_sec)}")
        clauses.append(f"t_sec <= {_p(body.time_range.end_sec)}")

    # vehicle_count filters: compute in SQL using jsonb_path_query_array
    # vehicle_total = sum of count fields (integer) across vehicle_groups array
    # Requires postgres 12+ jsonb_path_query support
    if body.vehicle_count_min is not None or body.vehicle_count_max is not None:
        # Build a sub-expression that sums vehicle counts
        # We cast the result of jsonb_path_query_array to an aggregate in SQL
        vcount_expr = (
            "COALESCE(("
            "  SELECT SUM((elem->>'count')::integer)"
            "  FROM jsonb_array_elements(frame_facts_json->'vehicle_groups') AS elem"
            "), 0)"
        )
        if body.vehicle_count_min is not None:
            clauses.append(f"{vcount_expr} >= {_p(body.vehicle_count_min)}")
        if body.vehicle_count_max is not None:
            clauses.append(f"{vcount_expr} <= {_p(body.vehicle_count_max)}")

    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


# -- Endpoint ------------------------------------------------------------------


@router.post(
    "/scene",
    response_model=SceneQueryResponse,
    summary="Structured scene query — filter by vehicle count, road condition, GPS bbox, time range",
    responses={
        400: ERROR_RESPONSES[400],
        503: ERROR_RESPONSES[503],
    },
)
async def query_scene(body: SceneQuery, request: Request) -> SceneQueryResponse:
    """Query indexed frames by structured JSONB predicates.

    Filters are applied in PostgreSQL against ``frame_facts_json`` (produced by
    Phase 2 Gemma/Qwen scene extraction). An optional text query triggers CLIP
    embedding and re-ranks the filtered results by cosine similarity.

    All filters are AND-combined. Omit a filter to leave it unconstrained.
    """
    db_pool = get_db_pool(request)

    # -- Build SQL -------------------------------------------------------------
    where, params = _build_sql_filters(body)

    # Fetch more rows than top_k when a text query will re-rank
    fetch_limit = body.top_k if body.text is None else min(body.top_k * 5, 500)
    fetch_param_idx = len(params) + 1
    params.append(fetch_limit)

    sql = f"""
        SELECT
            f.id            AS frame_id,
            f.mission_id,
            f.frame_path,
            f.t_sec,
            f.caption,
            f.frame_facts_json,
            f.gps_json,
            f.qdrant_id
        FROM frames f
        {where}
        ORDER BY f.created_at DESC
        LIMIT ${fetch_param_idx}
    """

    try:
        rows = await db_pool.fetch(sql, *params)
    except Exception as exc:
        logger.error("query/scene DB error: %s", exc)
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")

    total_matched = len(rows)

    # -- Optional CLIP re-ranking -----------------------------------------------
    frame_id_to_score: dict[str, float] = {}

    if body.text and rows:
        try:
            text_vec = clip_model.encode_texts([body.text])[0]
        except Exception as exc:
            logger.warning("CLIP encode failed for scene query: %s", exc)
            text_vec = None

        if text_vec is not None:
            import numpy as np

            # Fetch clip embeddings for matched frames from Qdrant
            qdrant_ids = [
                qdrant_id_from_pg(r["qdrant_id"]) for r in rows if r["qdrant_id"] is not None
            ]

            if qdrant_ids:
                try:
                    from selfsuvis.app.state import qdrant_store

                    points = qdrant_store.client.retrieve(
                        collection_name=qdrant_store.collection_name,
                        ids=qdrant_ids,
                        with_vectors=["clip"],
                    )
                    for pt in points:
                        vec = None
                        if isinstance(pt.vector, dict):
                            vec = pt.vector.get("clip")
                        elif pt.vector is not None:
                            vec = pt.vector
                        if vec is not None:
                            fv = np.array(vec, dtype=np.float32)
                            score = float(
                                np.dot(text_vec, fv)
                                / (np.linalg.norm(text_vec) * np.linalg.norm(fv) + 1e-8)
                            )
                            payload = pt.payload or {}
                            fid = payload.get("frame_id") or str(pt.id)
                            frame_id_to_score[fid] = score
                except Exception as exc:
                    logger.warning("Qdrant retrieve for re-rank failed: %s", exc)

            # Sort rows by CLIP score descending; unscored rows go to the end
            rows = sorted(
                rows,
                key=lambda r: frame_id_to_score.get(r["frame_id"], -1.0),
                reverse=True,
            )

    # -- Build response --------------------------------------------------------
    results: list[SceneMatch] = []
    for i, row in enumerate(rows[: body.top_k]):
        facts: dict = row["frame_facts_json"] or {}
        gps: dict = row["gps_json"] or {}
        vehicle_count = _count_vehicles(facts)

        # Score: cosine similarity when available, else rank-order (1.0 → 0.0)
        if body.text and row["frame_id"] in frame_id_to_score:
            score = frame_id_to_score[row["frame_id"]]
        else:
            score = 1.0 - (i / max(body.top_k, 1))

        results.append(
            SceneMatch(
                frame_id=row["frame_id"],
                mission_id=row["mission_id"],
                score=round(float(score), 4),
                t_sec=float(row["t_sec"]),
                lat=gps.get("lat"),
                lon=gps.get("lon"),
                caption=row["caption"],
                vehicle_count=vehicle_count,
                road_condition=facts.get("road_condition"),
                road_surface=facts.get("road_surface"),
                scene_summary=facts.get("scene_summary"),
                frame_path=row["frame_path"],
            )
        )

    # Summarise which filters were applied
    filters_applied: list[str] = []
    if body.text:
        filters_applied.append("text_rerank")
    if body.road_condition:
        filters_applied.append("road_condition")
    if body.gps_bbox:
        filters_applied.append("gps_bbox")
    if body.time_range:
        filters_applied.append("time_range")
    if body.vehicle_count_min is not None:
        filters_applied.append("vehicle_count_min")
    if body.vehicle_count_max is not None:
        filters_applied.append("vehicle_count_max")

    logger.info(
        "query/scene: filters=%s matched=%d returned=%d",
        filters_applied,
        total_matched,
        len(results),
    )

    return SceneQueryResponse(
        results=results,
        total_matched=total_matched,
        filters_applied=filters_applied,
    )
