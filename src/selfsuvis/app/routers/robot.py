"""Robot advisory API — POST /query/pose

Spatial query for robots in the field. Given either a GPS position (lat, lon) or
metric ENU coordinates (tx, ty, tz), returns the top-K most visually similar indexed
frames from within the search radius.

GPS path (v1):
  - Provide lat + lon (alt optional). Uses GPS bounding-box Qdrant filter.
  - GPS_FILTER_2D=false (default): lat-only Qdrant filter + Python lon post-filter.
  - GPS_FILTER_2D=true: 2D Qdrant filter (requires validated payload indexes).

ENU path (v2):
  - Provide tx + ty + tz (metric ENU). Requires Phase 2 ICP global map to be useful.
  - Uses enu.tx / enu.ty Qdrant payload filter + Python 3D ENU distance post-filter.
  - GPS coordinates are not required in this path.

Phase 5 extension: when a DB pool is available and the scene_timeline table is
populated, the response includes a ``last_visits`` summary — the last 3 visits
to the nearest GPS waypoint, with their captions and scene facts.

Auth: X-API-Key header required.
Latency target: p99 < 200ms (advisory use only — does not block robot motion).
"""

import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from selfsuvis.app.db import get_db_pool_optional
from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.app.state import clip_model, qdrant_store
from selfsuvis.pipeline.analysis.change_detection import latlon_bbox
from selfsuvis.pipeline.core import get_logger, settings

logger = get_logger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["robot"],
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)

# Approximate metres per degree latitude (flat-earth, valid for small radii)
_M_PER_DEG_LAT = 111_320.0


class PoseQuery(BaseModel):
    """Robot pose query request.

    Requires either GPS (lat + lon) or metric ENU (tx + ty + tz).
    """

    lat: float | None = Field(default=None, description="Latitude (decimal degrees, WGS-84)")
    lon: float | None = Field(default=None, description="Longitude (decimal degrees, WGS-84)")
    alt: float = Field(default=0.0, description="Altitude in metres (optional)")
    heading_deg: float | None = Field(
        default=None, description="Robot heading in degrees (0=North, 90=East)"
    )
    radius_m: float = Field(
        default=50.0,
        ge=1.0,
        le=5000.0,
        description="Search radius in metres (GPS bbox or ENU sphere)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of results to return",
    )
    tx: float | None = Field(default=None, description="ENU East (m)")
    ty: float | None = Field(default=None, description="ENU North (m)")
    tz: float | None = Field(default=None, description="ENU Up (m)")
    robot_ids: list[str] | None = Field(
        default=None,
        description="Filter results to frames contributed by these robot IDs (all robots if omitted)",
    )
    global_map_id: int | None = Field(
        default=None,
        description="Restrict search to frames from a specific site (global_map id). "
        "Required for ENU queries to be meaningful across multiple sites.",
    )

    @model_validator(mode="after")
    def _require_gps_or_enu(self) -> "PoseQuery":
        has_gps = self.lat is not None and self.lon is not None
        has_enu = self.tx is not None and self.ty is not None and self.tz is not None
        if not has_gps and not has_enu:
            raise ValueError(
                "Provide either GPS coordinates (lat + lon) or ENU coordinates (tx + ty + tz)"
            )
        return self


class PoseMatch(BaseModel):
    """Single frame match in a pose query response."""

    frame_id: str | None
    mission_id: str | None
    score: float
    t_sec: float
    lat: float | None
    lon: float | None
    alt: float | None
    distance_m: float | None  # approximate GPS distance from query point
    frame_path: str | None
    global_pose_json: dict[str, Any] | None


class TimelineVisit(BaseModel):
    """One historical visit entry from the scene_timeline table."""

    mission_id: str
    frame_id: str
    t_sec: float | None
    lat: float | None
    lon: float | None
    caption: str | None
    road_condition: str | None
    vehicle_count: int | None
    created_at: str | None  # ISO-8601 string


class PoseQueryResponse(BaseModel):
    results: list[PoseMatch]
    query_lat: float | None
    query_lon: float | None
    query_tx: float | None
    query_ty: float | None
    query_tz: float | None
    radius_m: float
    filter_strategy: str  # "2d", "1d+python", or "enu+python"
    global_map_id: int | None
    last_visits: list[TimelineVisit] | None = None  # Phase 5: last 3 visits near query point


def _gps_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Flat-earth approximate distance in metres (valid for radius_m << 50 km)."""
    dlat = (lat2 - lat1) * _M_PER_DEG_LAT
    dlon = (lon2 - lon1) * _M_PER_DEG_LAT * abs(math.cos(math.radians(lat1)))
    return math.sqrt(dlat * dlat + dlon * dlon)


def _enu_distance_m(
    tx1: float, ty1: float, tz1: float, tx2: float, ty2: float, tz2: float
) -> float:
    """3D Euclidean distance in the ENU frame (metres)."""
    return math.sqrt((tx2 - tx1) ** 2 + (ty2 - ty1) ** 2 + (tz2 - tz1) ** 2)


async def _get_last_visits(
    db_pool,
    lat: float,
    lon: float,
    radius_m: float,
    limit: int = 3,
) -> list[TimelineVisit]:
    """Query scene_timeline for the last N visits near a GPS point.

    Returns visits sorted by created_at DESC (most recent first).
    Returns empty list if the table is empty or DB unavailable.
    """
    min_lat, max_lat, min_lon, max_lon = latlon_bbox(lat, lon, radius_m)
    try:
        rows = await db_pool.fetch(
            """
            SELECT mission_id, frame_id, t_sec, gps_lat, gps_lon,
                   caption, facts_json, created_at
            FROM scene_timeline
            WHERE gps_lat BETWEEN $1 AND $2
              AND gps_lon BETWEEN $3 AND $4
            ORDER BY created_at DESC
            LIMIT $5
            """,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            limit,
        )
    except Exception as exc:
        logger.debug("scene_timeline query failed (table may not exist yet): %s", exc)
        return []

    visits: list[TimelineVisit] = []
    for row in rows:
        facts = row["facts_json"] or {}
        # Count vehicles
        vehicle_count: int | None = None
        groups = facts.get("vehicle_groups") if isinstance(facts, dict) else None
        if isinstance(groups, list):
            vehicle_count = sum(int(g.get("count", 0)) for g in groups if isinstance(g, dict))
        visits.append(
            TimelineVisit(
                mission_id=row["mission_id"],
                frame_id=row["frame_id"],
                t_sec=row["t_sec"],
                lat=row["gps_lat"],
                lon=row["gps_lon"],
                caption=row["caption"],
                road_condition=facts.get("road_condition") if isinstance(facts, dict) else None,
                vehicle_count=vehicle_count,
                created_at=str(row["created_at"]) if row["created_at"] else None,
            )
        )
    return visits


@router.post(
    "/pose",
    response_model=PoseQueryResponse,
    summary="Robot advisory pose query — find nearest indexed frames by GPS or ENU position",
)
async def query_pose(body: PoseQuery, request: Request) -> PoseQueryResponse:
    """Find the top-K indexed frames nearest to the robot's position.

    Supports two query modes:
    - GPS (lat + lon): uses GPS bounding-box filter on indexed frame payloads.
    - ENU (tx + ty + tz): uses ENU bounding-box filter; requires Phase 2 global map.

    Auth: X-API-Key header required.
    """
    import numpy as np

    try:
        from selfsuvis.pipeline.core.optional_deps import require_qdrant_models

        qmodels = require_qdrant_models()
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    radius_m = body.radius_m
    has_gps = body.lat is not None and body.lon is not None

    # Optional robot_id filter — add to every query strategy
    robot_must: list = []
    if body.robot_ids:
        robot_must.append(
            qmodels.FieldCondition(
                key="robot_id",
                match=qmodels.MatchAny(any=body.robot_ids),
            )
        )
    if body.global_map_id is not None:
        robot_must.append(
            qmodels.FieldCondition(
                key="global_map_id",
                match=qmodels.MatchValue(value=body.global_map_id),
            )
        )

    if has_gps:
        lat, lon = body.lat, body.lon  # type: ignore[assignment]
        min_lat, max_lat, min_lon, max_lon = latlon_bbox(lat, lon, radius_m)

        if settings.GPS_FILTER_2D:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="gps.lat", range=qmodels.Range(gte=min_lat, lte=max_lat)
                    ),
                    qmodels.FieldCondition(
                        key="gps.lon", range=qmodels.Range(gte=min_lon, lte=max_lon)
                    ),
                ]
                + robot_must
            )
            filter_strategy = "2d"
        else:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="gps.lat", range=qmodels.Range(gte=min_lat, lte=max_lat)
                    ),
                ]
                + robot_must
            )
            filter_strategy = "1d+python"
    else:
        # ENU path: filter on enu.tx and enu.ty bounding box (2D), 3D post-filter in Python
        tx, ty, tz = body.tx, body.ty, body.tz  # type: ignore[assignment]
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="enu.tx", range=qmodels.Range(gte=tx - radius_m, lte=tx + radius_m)
                ),
                qmodels.FieldCondition(
                    key="enu.ty", range=qmodels.Range(gte=ty - radius_m, lte=ty + radius_m)
                ),
            ]
            + robot_must
        )
        filter_strategy = "enu+python"

    dummy_vec = np.zeros(clip_model.embed_dim, dtype=np.float32)

    try:
        fetch_k = body.top_k * 4 if filter_strategy != "2d" else body.top_k
        response = qdrant_store.client.query_points(
            collection_name=qdrant_store.collection_name,
            query=dummy_vec.tolist(),
            using="clip",
            query_filter=query_filter,
            limit=min(fetch_k, 200),
            with_payload=True,
            with_vectors=False,
        )
        raw_results = response.points
    except Exception as exc:
        logger.error("Robot API: Qdrant search failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Vector store error: {exc}")

    # Python post-filters
    if has_gps and not settings.GPS_FILTER_2D:
        raw_results = [
            r
            for r in raw_results
            if min_lon <= (r.payload or {}).get("gps", {}).get("lon", 999) <= max_lon
        ]
    elif not has_gps:
        # 3D ENU distance post-filter
        def _within_enu(r) -> bool:
            enu = (r.payload or {}).get("enu") or {}
            ftx, fty, ftz = enu.get("tx"), enu.get("ty"), enu.get("tz")
            if ftx is None or fty is None or ftz is None:
                return False
            return _enu_distance_m(tx, ty, tz, ftx, fty, ftz) <= radius_m

        raw_results = [r for r in raw_results if _within_enu(r)]

    matches: list[PoseMatch] = []
    for r in raw_results[: body.top_k]:
        payload = r.payload or {}
        gps = payload.get("gps") or {}
        enu = payload.get("enu") or {}
        frame_lat = gps.get("lat")
        frame_lon = gps.get("lon")

        if has_gps and frame_lat is not None and frame_lon is not None:
            dist_m = _gps_distance_m(lat, lon, frame_lat, frame_lon)
        elif not has_gps and enu.get("tx") is not None:
            dist_m = _enu_distance_m(tx, ty, tz, enu["tx"], enu["ty"], enu.get("tz", 0.0))
        else:
            dist_m = None

        matches.append(
            PoseMatch(
                frame_id=payload.get("frame_id"),
                mission_id=payload.get("mission_id"),
                score=float(r.score),
                t_sec=float(payload.get("t_sec", 0.0)),
                lat=frame_lat,
                lon=frame_lon,
                alt=gps.get("alt"),
                distance_m=dist_m,
                frame_path=payload.get("frame_path"),
                global_pose_json=payload.get("global_pose_json"),
            )
        )

    matches.sort(key=lambda m: m.distance_m if m.distance_m is not None else 1e9)

    if has_gps:
        logger.info(
            "Robot API: lat=%.6f lon=%.6f radius=%.0fm filter=%s results=%d",
            body.lat,
            body.lon,
            radius_m,
            filter_strategy,
            len(matches),
        )
    else:
        logger.info(
            "Robot API: tx=%.2f ty=%.2f tz=%.2f radius=%.0fm filter=%s results=%d",
            body.tx,
            body.ty,
            body.tz,
            radius_m,
            filter_strategy,
            len(matches),
        )

    # Phase 5: fetch last-3-visits summary from scene_timeline (GPS queries only)
    last_visits: list[TimelineVisit] | None = None
    if has_gps:
        db_pool = get_db_pool_optional(request)
        if db_pool is not None:
            last_visits = await _get_last_visits(
                db_pool,
                body.lat,
                body.lon,
                radius_m,
                limit=3,  # type: ignore[arg-type]
            )

    return PoseQueryResponse(
        results=matches,
        query_lat=body.lat,
        query_lon=body.lon,
        query_tx=body.tx,
        query_ty=body.ty,
        query_tz=body.tz,
        radius_m=radius_m,
        filter_strategy=filter_strategy,
        global_map_id=body.global_map_id,
        last_visits=last_visits,
    )
